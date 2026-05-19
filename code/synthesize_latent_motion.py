import argparse
import json
import os

import numpy as np
import torch

import latent_lstm_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_folder", type=str, required=True)
    parser.add_argument("--read_weight_path", type=str, required=True)
    parser.add_argument("--write_latent_motion_folder", type=str, required=True)
    parser.add_argument("--write_bvh_motion_folder", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--use_ik", action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=["train", "eval", "all"],
    )
    parser.add_argument("--seed_file", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--initial_seq_len", type=int, default=16)
    parser.add_argument("--generate_latent_steps", type=int, default=40)
    parser.add_argument(
        "--seed_start_index",
        type=int,
        default=0,
        help="Use -1 to choose a random valid start per sampled sequence",
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    latent_lstm_utils.seed_all(args.seed)
    latent_context = latent_lstm_utils.load_latent_context(
        latent_folder=args.latent_folder,
        manifest_path=args.manifest_path,
        data_path=args.data_path,
        model_path=args.model_path,
    )
    latent_entries = latent_lstm_utils.load_latent_entries(
        latent_context,
        split=args.split,
    )
    seed_entries = latent_lstm_utils.choose_seed_entries(
        latent_entries,
        seed_file=args.seed_file,
        num_samples=args.num_samples,
        random_seed=args.seed,
    )

    os.makedirs(args.write_latent_motion_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_checkpoint_path, checkpoint_payload = latent_lstm_utils.load_lstm_checkpoint(
        args.read_weight_path,
        device,
    )

    train_config = checkpoint_payload.get("train_config", {})
    resolved_model_type = latent_lstm_utils.resolve_model_type(
        train_config,
        args.model_type,
    )
    latent_width = int(
        train_config.get(
            "latent_width",
            latent_lstm_utils.infer_latent_width(latent_context, seed_entries),
        )
    )
    model = latent_lstm_utils.build_latent_sequence_model(
        model_type=resolved_model_type,
        latent_width=latent_width,
        hidden_size=int(train_config.get("hidden_size", args.hidden_size)),
        num_layers=int(train_config.get("num_layers", args.num_layers)),
        dropout=float(train_config.get("dropout", args.dropout)),
    ).to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()

    runtime = latent_lstm_utils.create_autoencoder_runtime(
        latent_context,
        device,
        load_ik=args.use_ik,
    )
    dataset_cache, dataset_index_cache = latent_lstm_utils.build_runtime_dataset_caches(
        runtime,
        [entry["source_split"] for entry in seed_entries],
    )

    generation_records = []
    rng = np.random.default_rng(args.seed)

    for sample_index, latent_entry in enumerate(seed_entries):
        full_latent = np.load(latent_entry["latent_file_path"]).astype(np.float32)
        if full_latent.ndim != 2:
            raise ValueError(
                "Expected latent array [T, C] in {}, got {}".format(
                    latent_entry["latent_file_path"],
                    full_latent.shape,
                )
            )
        if full_latent.shape[1] != latent_width:
            raise ValueError(
                "Latent width {} does not match LSTM checkpoint width {} for {}".format(
                    full_latent.shape[1],
                    latent_width,
                    latent_entry["latent_file_path"],
                )
            )
        if full_latent.shape[0] < args.initial_seq_len:
            raise ValueError(
                "Latent file {} is too short for initial_seq_len {}".format(
                    latent_entry["latent_file_path"],
                    args.initial_seq_len,
                )
            )

        seed_start_index = int(args.seed_start_index)
        if seed_start_index < 0:
            max_start_index = full_latent.shape[0] - args.initial_seq_len
            seed_start_index = int(rng.integers(0, max_start_index + 1))

        seed_end_index = seed_start_index + args.initial_seq_len
        if seed_end_index > full_latent.shape[0]:
            raise ValueError(
                "Seed slice [{}:{}) exceeds latent length {} for {}".format(
                    seed_start_index,
                    seed_end_index,
                    full_latent.shape[0],
                    latent_entry["latent_file_path"],
                )
            )

        seed_latent = full_latent[seed_start_index:seed_end_index]
        seed_tensor = torch.tensor(
            seed_latent,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            generated_latent = (
                model.generate(seed_tensor, args.generate_latent_steps)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        source_stem = os.path.splitext(latent_entry["source_file_name"])[0]
        output_stem = "{:02d}_{}_seed{:04d}_gen{:04d}".format(
            sample_index,
            source_stem,
            seed_start_index,
            args.generate_latent_steps,
        )

        latent_output_path = os.path.join(
            args.write_latent_motion_folder,
            output_stem + ".npy",
        )
        np.save(latent_output_path, generated_latent)

        bvh_output_path = latent_lstm_utils.decode_latent_sequence_for_entry(
            runtime,
            dataset_cache,
            dataset_index_cache,
            latent_entry,
            generated_latent,
            output_stem + ".bvh",
            args.write_bvh_motion_folder,
        )

        metadata_path = os.path.join(
            args.write_latent_motion_folder,
            output_stem + ".json",
        )
        generation_metadata = latent_lstm_utils.build_generation_metadata(
            latent_entry,
            resolved_checkpoint_path,
            args.initial_seq_len,
            args.generate_latent_steps,
            seed_start_index,
            runtime=runtime,
        )
        generation_metadata["latent_output_path"] = os.path.abspath(latent_output_path)
        generation_metadata["bvh_output_path"] = os.path.abspath(bvh_output_path)
        generation_metadata["checkpoint_iteration"] = int(
            checkpoint_payload.get("iteration", -1)
        )
        generation_metadata["model_type"] = resolved_model_type
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(generation_metadata, handle, indent=2, sort_keys=True)

        print(
            "Generated {} -> {}".format(
                latent_entry["source_relative_path"],
                bvh_output_path,
            )
        )
        generation_records.append(generation_metadata)

    summary_path = os.path.join(
        args.write_latent_motion_folder,
        "generation_summary.json",
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(generation_records, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()