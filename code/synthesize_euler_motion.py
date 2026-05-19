import os
import argparse
import random

import numpy as np
import torch

import read_bvh
import pytorch_train_euler_aclstm as euler_train


def resolve_weight_path(read_weight_path):
    if os.path.isfile(read_weight_path):
        return read_weight_path

    if os.path.isdir(read_weight_path):
        weight_files = []
        for file_name in sorted(os.listdir(read_weight_path)):
            if file_name.endswith(".weight"):
                weight_files.append(file_name)

        if len(weight_files) == 0:
            raise ValueError("No .weight files found in {}".format(read_weight_path))

        resolved_path = os.path.join(read_weight_path, weight_files[-1])
        print("Using latest checkpoint: {}".format(resolved_path))
        return resolved_path

    raise ValueError("Checkpoint path does not exist: {}".format(read_weight_path))


def generate_seq(
    initial_seq_np, generate_frames_number, model, save_dance_folder, normalizer
):
    device = next(model.parameters()).device
    # reuse the training-time root differencing before handing the seed to the recurrent model.
    initial_seq_model_np = euler_train.build_model_input_sequence(initial_seq_np)
    initial_seq = torch.tensor(initial_seq_model_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        predict_seq = model.generate(initial_seq, generate_frames_number)

    batch = initial_seq_np.shape[0]
    for batch_index in range(batch):
        out_seq = predict_seq[batch_index].detach().cpu().numpy()
        # convert the model output back to absolute root motion before writing BVH.
        out_motion = euler_train.reconstruct_absolute_motion(out_seq, normalizer)
        output_path = os.path.join(
            save_dance_folder, "out" + "%02d" % batch_index + ".bvh"
        )
        read_bvh.write_euler_traindata_to_bvh(output_path, out_motion)

    return predict_seq.detach().cpu().numpy()


def test(
    dances,
    frame_rate,
    batch,
    initial_seq_len,
    generate_frames_number,
    read_weight_path,
    write_bvh_motion_folder,
    in_frame_size=euler_train.IN_FRAME_SIZE,
    hidden_size=euler_train.HIDDEN_SIZE,
    out_frame_size=euler_train.IN_FRAME_SIZE,
    recenter_root=False,
    recenter_y=False,
    normalization_stats_path="",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = None
    if normalization_stats_path != "":
        normalizer = euler_train.EulerMotionNormalizer.from_file(
            normalization_stats_path, device
        )

    model = euler_train.acLSTM(
        in_frame_size=in_frame_size,
        hidden_size=hidden_size,
        out_frame_size=out_frame_size,
        normalizer=normalizer,
    )
    resolved_weight_path = resolve_weight_path(read_weight_path)
    model.load_state_dict(torch.load(resolved_weight_path, map_location=device))
    model.to(device)
    model.eval()

    dance_len_lst = euler_train.get_dance_len_lst(dances)
    random_range = len(dance_len_lst)
    speed = frame_rate / 30

    dance_batch = []
    for batch_index in range(batch):
        dance_id = dance_len_lst[np.random.randint(0, random_range)]
        dance = dances[dance_id].copy()
        dance_len = dance.shape[0]

        max_start_id = int(dance_len - initial_seq_len * speed - 10)
        if max_start_id <= 10:
            raise ValueError(
                "Motion {} is too short for initial_seq_len {} at frame rate {}".format(
                    dance_id, initial_seq_len, frame_rate
                )
            )

        start_id = random.randint(10, max_start_id)
        sample_seq = []
        for frame_index in range(initial_seq_len):
            sample_seq = sample_seq + [dance[int(frame_index * speed + start_id)]]

        # keep seed preprocessing aligned with the training checkpoint, minus any extra random yaw augmentation.
        sample_seq_prepared = euler_train.prepare_sequence_for_model(
            sample_seq,
            normalizer=normalizer,
            recenter_root=recenter_root,
            recenter_y=recenter_y,
            augment_yaw_range_degrees=0.0,
        )
        dance_batch = dance_batch + [sample_seq_prepared]

    dance_batch_np = np.array(dance_batch)
    generate_seq(
        dance_batch_np,
        generate_frames_number,
        model,
        write_bvh_motion_folder,
        normalizer,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dances_folder", type=str, required=True, help="Path for the training data"
    )
    parser.add_argument(
        "--read_weight_path",
        type=str,
        required=True,
        help="Checkpoint model path or folder containing .weight files",
    )
    parser.add_argument(
        "--write_bvh_motion_folder",
        type=str,
        required=True,
        help="Path to store generated bvh",
    )
    parser.add_argument(
        "--dance_frame_rate", type=int, default=60, help="Dance frame rate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Number of motion seeds to sample"
    )
    parser.add_argument(
        "--initial_seq_len", type=int, default=15, help="Number of seed frames"
    )
    parser.add_argument(
        "--generate_frames_number",
        type=int,
        default=400,
        help="Number of frames to generate after the seed sequence",
    )
    parser.add_argument("--in_frame", type=int, default=132, help="Input channel")
    parser.add_argument("--out_frame", type=int, default=132, help="Output channels")
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden size of the network"
    )
    parser.add_argument(
        "--recenter_root",
        action="store_true",
        help="Recenter root translation around the seed mean before synthesis",
    )
    parser.add_argument(
        "--recenter_y",
        action="store_true",
        help="Include root height when recentering the seed sequence",
    )
    parser.add_argument(
        "--normalization_stats_path",
        type=str,
        default="",
        help="Optional normalization stats produced by normalize_representation_data.py",
    )

    args = parser.parse_args()

    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances = euler_train.load_dances(args.dances_folder)

    test(
        dances,
        args.dance_frame_rate,
        args.batch_size,
        args.initial_seq_len,
        args.generate_frames_number,
        args.read_weight_path,
        args.write_bvh_motion_folder,
        args.in_frame,
        args.hidden_size,
        args.out_frame,
        recenter_root=args.recenter_root,
        recenter_y=args.recenter_y,
        normalization_stats_path=args.normalization_stats_path,
    )


if __name__ == "__main__":
    main()
