import os
import argparse
import random

import numpy as np
import torch

import read_bvh
import pytorch_train_quad_aclstm as quad_train

TRANSLATION_X_INDEX = quad_train.TRANSLATION_X_INDEX
TRANSLATION_Z_INDEX = quad_train.TRANSLATION_Z_INDEX
TRANSLATION_SIZE = quad_train.TRANSLATION_SIZE
QUATERNION_SIZE = quad_train.QUATERNION_SIZE
HIDDEN_SIZE = quad_train.HIDDEN_SIZE
IN_FRAME_SIZE = quad_train.IN_FRAME_SIZE
NON_END_BONES = read_bvh.non_end_bones


class acLSTM(quad_train.acLSTM):
    def generate(self, initial_seq, generate_frames_number):
        batch = initial_seq.size(0)
        vec_h, vec_c = self.init_hidden(batch)

        out_frames = []
        out_frame = torch.zeros(batch, self.out_frame_size, device=initial_seq.device)
        previous_frame = None

        for frame_index in range(initial_seq.size(1)):
            in_frame = initial_seq[:, frame_index]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)
            previous_frame = in_frame

        for frame_index in range(generate_frames_number):
            # normalize and sign-stabilize each feedback frame before rolling out the next step.
            in_frame = self.sanitize_quaternion_frame(out_frame, previous_frame)
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)
            previous_frame = in_frame

        return torch.stack(out_frames, dim=1)


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


def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):
    device = next(model.parameters()).device
    # the model expects root x/z displacements, but BVH export later needs absolute root motion again.
    initial_seq_dif_np = quad_train.build_model_input_sequence(initial_seq_np)
    initial_seq = torch.tensor(initial_seq_dif_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        predict_seq = model.generate(initial_seq, generate_frames_number)

    batch = initial_seq_np.shape[0]
    frame_size = initial_seq_np.shape[2]

    for batch_index in range(batch):
        out_seq = (
            predict_seq[batch_index].detach().cpu().numpy().reshape(-1, frame_size)
        )
        out_motion = quad_train.reconstruct_absolute_motion_np(out_seq)
        output_path = os.path.join(
            save_dance_folder, "out" + "%02d" % batch_index + ".bvh"
        )
        read_bvh.write_quaternion_traindata_to_bvh(
            output_path, out_motion, NON_END_BONES
        )

    return predict_seq.detach().cpu().numpy().reshape(batch, -1, frame_size)


def test(
    dances,
    frame_rate,
    batch,
    initial_seq_len,
    generate_frames_number,
    read_weight_path,
    write_bvh_motion_folder,
    in_frame_size=IN_FRAME_SIZE,
    hidden_size=HIDDEN_SIZE,
    out_frame_size=IN_FRAME_SIZE,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    resolved_weight_path = resolve_weight_path(read_weight_path)
    quad_train.load_model_state_dict_compatible(
        model,
        torch.load(resolved_weight_path, map_location=device),
        resolved_weight_path,
    )
    model.to(device)
    model.eval()

    dance_len_lst = quad_train.get_dance_len_lst(dances)
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

        dance_batch = dance_batch + [
            # lock the seed onto one quaternion sign branch before the autoregressive rollout starts.
            read_bvh.enforce_quaternion_sequence_continuity(np.array(sample_seq))
        ]

    dance_batch_np = np.array(dance_batch)
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)


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
    parser.add_argument("--in_frame", type=int, default=175, help="Input channel")
    parser.add_argument("--out_frame", type=int, default=175, help="Output channels")
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden size of the network"
    )

    args = parser.parse_args()

    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances = quad_train.load_dances(args.dances_folder)

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
    )


if __name__ == "__main__":
    main()
