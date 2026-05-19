import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import latent_lstm_utils

PRINT_EVERY_ITERATIONS = 500
SAVE_EVERY_ITERATIONS = 5000
EVAL_EVERY_ITERATIONS = 5000


def resolve_condition_settings(train_config, condition_num, groundtruth_num):
    checkpoint_condition_num = 0
    checkpoint_groundtruth_num = 0
    if train_config is not None:
        checkpoint_condition_num = int(train_config.get("condition_num", 0))
        checkpoint_groundtruth_num = int(train_config.get("groundtruth_num", 0))

    if condition_num is None:
        condition_num = checkpoint_condition_num
    if groundtruth_num is None:
        groundtruth_num = checkpoint_groundtruth_num

    condition_num = int(condition_num)
    groundtruth_num = int(groundtruth_num)

    if condition_num < 0 or groundtruth_num < 0:
        raise ValueError("condition_num and groundtruth_num must be non-negative")
    if (condition_num == 0) != (groundtruth_num == 0):
        raise ValueError(
            "condition_num and groundtruth_num must both be zero, or both be positive"
        )

    if train_config is not None:
        if (
            condition_num != checkpoint_condition_num
            or groundtruth_num != checkpoint_groundtruth_num
        ):
            raise ValueError(
                "Requested condition schedule {}-{} does not match checkpoint schedule {}-{}".format(
                    groundtruth_num,
                    condition_num,
                    checkpoint_groundtruth_num,
                    checkpoint_condition_num,
                )
            )

    return condition_num, groundtruth_num


def compute_batch_loss(
    model,
    latent_batch,
    device,
    criterion,
    condition_num,
    groundtruth_num,
):
    latent_batch = latent_batch.to(device=device, dtype=torch.float32)
    input_seq = latent_batch[:, :-1, :]
    target_seq = latent_batch[:, 1:, :]
    if condition_num > 0 and groundtruth_num > 0:
        predicted_seq, _ = model.forward_conditioned(
            input_seq,
            condition_num=condition_num,
            groundtruth_num=groundtruth_num,
        )
    else:
        predicted_seq, _ = model(input_seq)
    return criterion(predicted_seq, target_seq)


def evaluate_model(
    model,
    dataloader,
    device,
    criterion,
    max_batches,
    condition_num,
    groundtruth_num,
):
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch_index, latent_batch in enumerate(dataloader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            loss = compute_batch_loss(
                model,
                latent_batch,
                device,
                criterion,
                condition_num,
                groundtruth_num,
            )
            total_loss += float(loss.detach().cpu().item())
            batch_count += 1

    model.train()
    if batch_count == 0:
        return None
    return total_loss / float(batch_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_folder", type=str, required=True)
    parser.add_argument("--write_weight_folder", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, default="")
    parser.add_argument("--read_weight_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model_type",
        type=str,
        default=latent_lstm_utils.DEFAULT_MODEL_TYPE,
        choices=latent_lstm_utils.LATENT_MODEL_TYPES,
    )
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--condition_num", type=int, default=None)
    parser.add_argument("--groundtruth_num", type=int, default=None)
    parser.add_argument("--total_iterations", type=int, default=10000)
    parser.add_argument(
        "--print_every_iterations",
        type=int,
        default=PRINT_EVERY_ITERATIONS,
    )
    parser.add_argument(
        "--save_every_iterations",
        type=int,
        default=SAVE_EVERY_ITERATIONS,
    )
    parser.add_argument(
        "--eval_every_iterations",
        type=int,
        default=EVAL_EVERY_ITERATIONS,
    )
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=8,
        help="Number of eval batches to average when eval split exists",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    latent_lstm_utils.seed_all(args.seed)
    latent_context = latent_lstm_utils.load_latent_context(
        latent_folder=args.latent_folder,
        manifest_path=args.manifest_path,
    )

    os.makedirs(args.write_weight_folder, exist_ok=True)

    train_entries = latent_lstm_utils.load_latent_entries(latent_context, split="train")
    eval_entries = latent_lstm_utils.load_latent_entries(latent_context, split="eval")

    sample_length = int(args.seq_len) + 1
    train_dataset = latent_lstm_utils.LatentWindowDataset(
        train_entries,
        sample_length=sample_length,
        window_stride=args.window_stride,
    )
    eval_dataset = None
    if len(eval_entries) > 0:
        eval_dataset = latent_lstm_utils.LatentWindowDataset(
            eval_entries,
            sample_length=sample_length,
            window_stride=args.window_stride,
        )

    latent_width = train_dataset.latent_width
    manifest_width = None
    if latent_context.manifest is not None:
        manifest_width = latent_context.manifest.get("expected_latent_width")
    if manifest_width is not None and int(manifest_width) != latent_width:
        raise ValueError(
            "Latent export width {} does not match manifest expectation {}".format(
                latent_width,
                manifest_width,
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_type = latent_lstm_utils.validate_model_type(args.model_type)
    model = latent_lstm_utils.build_latent_sequence_model(
        model_type=resolved_model_type,
        latent_width=latent_width,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    resolved_condition_num, resolved_groundtruth_num = resolve_condition_settings(
        None,
        args.condition_num,
        args.groundtruth_num,
    )

    start_iteration = 0
    if args.read_weight_path != "":
        resolved_checkpoint_path, checkpoint_payload = (
            latent_lstm_utils.load_lstm_checkpoint(
                args.read_weight_path,
                device,
            )
        )
        train_config = checkpoint_payload.get("train_config", {})
        resolved_model_type = latent_lstm_utils.resolve_model_type(
            train_config,
            args.model_type,
        )
        resolved_condition_num, resolved_groundtruth_num = resolve_condition_settings(
            train_config,
            args.condition_num,
            args.groundtruth_num,
        )
        checkpoint_latent_width = train_config.get("latent_width")
        if (
            checkpoint_latent_width is not None
            and int(checkpoint_latent_width) != latent_width
        ):
            raise ValueError(
                "Checkpoint latent width {} does not match latent dataset width {}".format(
                    checkpoint_latent_width,
                    latent_width,
                )
            )
        model.load_state_dict(checkpoint_payload["model_state_dict"])
        optimizer_state_dict = checkpoint_payload.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = int(checkpoint_payload.get("iteration", -1)) + 1
        print(
            "Resumed latent LSTM from {} at iteration {}".format(
                resolved_checkpoint_path, start_iteration
            )
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    print(
        "Training latent LSTM with width {} using model_type {}".format(
            latent_width,
            resolved_model_type,
        )
    )
    if resolved_condition_num > 0 and resolved_groundtruth_num > 0:
        print(
            "Training condition schedule: {} ground-truth / {} feedback".format(
                resolved_groundtruth_num,
                resolved_condition_num,
            )
        )
    else:
        print("Training condition schedule: full teacher forcing")
    print("Train windows: {}".format(len(train_dataset)))
    if eval_dataset is not None:
        print("Eval windows: {}".format(len(eval_dataset)))

    train_iterator = iter(train_loader)
    model.train()

    for iteration in range(start_iteration, args.total_iterations):
        try:
            latent_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            latent_batch = next(train_iterator)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_batch_loss(
            model,
            latent_batch,
            device,
            criterion,
            resolved_condition_num,
            resolved_groundtruth_num,
        )
        loss.backward()
        optimizer.step()

        if iteration % args.print_every_iterations == 0:
            print("###########iter {:07d}######################".format(iteration))
            print("loss_mse: {}".format(float(loss.detach().cpu().item())))

        if eval_loader is not None and iteration % args.eval_every_iterations == 0:
            eval_loss = evaluate_model(
                model,
                eval_loader,
                device,
                criterion,
                args.eval_batches,
                resolved_condition_num,
                resolved_groundtruth_num,
            )
            if eval_loss is not None:
                print("eval_loss_mse: {}".format(eval_loss))

        if iteration % args.save_every_iterations == 0:
            checkpoint_path = os.path.join(
                args.write_weight_folder,
                "{:07d}.weight".format(iteration),
            )
            torch.save(
                latent_lstm_utils.build_lstm_checkpoint_payload(
                    model,
                    optimizer,
                    iteration,
                    {
                        "latent_width": latent_width,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "model_type": resolved_model_type,
                        "seq_len": args.seq_len,
                        "window_stride": args.window_stride,
                        "condition_num": resolved_condition_num,
                        "groundtruth_num": resolved_groundtruth_num,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "seed": args.seed,
                    },
                    latent_context,
                ),
                checkpoint_path,
            )

    final_checkpoint_path = os.path.join(args.write_weight_folder, "latest.weight")
    torch.save(
        latent_lstm_utils.build_lstm_checkpoint_payload(
            model,
            optimizer,
            args.total_iterations - 1,
            {
                "latent_width": latent_width,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "model_type": resolved_model_type,
                "seq_len": args.seq_len,
                "window_stride": args.window_stride,
                "condition_num": resolved_condition_num,
                "groundtruth_num": resolved_groundtruth_num,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
            latent_context,
        ),
        final_checkpoint_path,
    )
    print("Saved final latent LSTM checkpoint to {}".format(final_checkpoint_path))


if __name__ == "__main__":
    main()
