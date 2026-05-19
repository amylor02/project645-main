###########################################################################
###########################################################################
## ATTENTION:
##
##
## The actual file we use is pytorch_train_quad_aclstm_no_fk.py
## Executing this file will just redirect to that one!
##
##
##
###########################################################################
###########################################################################

import os
import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn

import read_bvh

TRANSLATION_X_INDEX = 0
TRANSLATION_Z_INDEX = 2
TRANSLATION_SIZE = 3
QUATERNION_SIZE = 4
HIDDEN_SIZE = 1024
CONDITION_NUM = 5
GROUNDTRUTH_NUM = 5
IN_FRAME_SIZE = 175
PRINT_EVERY_ITERATIONS = 100
SAVE_EVERY_ITERATIONS = 500
NON_END_BONES = read_bvh.non_end_bones
ROOT_JOINT_NAME = next(iter(read_bvh.skeleton.keys()))
JOINT_NAMES = list(read_bvh.skeleton.keys())
JOINT_NAME_TO_INDEX = {
    joint_name: index for index, joint_name in enumerate(JOINT_NAMES)
}
JOINT_PARENT_INDICES = [
    (
        -1
        if read_bvh.skeleton[joint_name]["parent"] is None
        else JOINT_NAME_TO_INDEX[read_bvh.skeleton[joint_name]["parent"]]
    )
    for joint_name in JOINT_NAMES
]
JOINT_OFFSETS = (
    np.array(
        [read_bvh.skeleton[joint_name]["offsets"] for joint_name in JOINT_NAMES],
        dtype=np.float32,
    )
    * read_bvh.weight_translation
)
BACKWARD_COMPATIBLE_STATE_KEYS = {"joint_offsets", "identity_quaternion"}
JOINT_QUATERNION_START_INDICES = []
for joint_name in JOINT_NAMES:
    if joint_name == ROOT_JOINT_NAME:
        JOINT_QUATERNION_START_INDICES.append(TRANSLATION_SIZE)
    elif joint_name in NON_END_BONES:
        JOINT_QUATERNION_START_INDICES.append(
            TRANSLATION_SIZE
            + QUATERNION_SIZE
            + NON_END_BONES.index(joint_name) * QUATERNION_SIZE
        )
    else:
        JOINT_QUATERNION_START_INDICES.append(-1)


class acLSTM(nn.Module):
    def __init__(
        self,
        in_frame_size=IN_FRAME_SIZE,
        hidden_size=HIDDEN_SIZE,
        out_frame_size=IN_FRAME_SIZE,
        translation_loss_weight=1.0,
        quaternion_loss_weight=1.0,
        fk_loss_weight=1.0,
    ):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size
        self.translation_loss_weight = translation_loss_weight
        self.quaternion_loss_weight = quaternion_loss_weight
        self.fk_loss_weight = fk_loss_weight
        self.joint_parent_indices = JOINT_PARENT_INDICES
        self.joint_quaternion_start_indices = JOINT_QUATERNION_START_INDICES

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

        self.register_buffer(
            "joint_offsets", torch.tensor(JOINT_OFFSETS, dtype=torch.float32)
        )
        self.register_buffer(
            "identity_quaternion",
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        )

    def init_hidden(self, batch):
        device = next(self.parameters()).device
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        c1 = torch.zeros(batch, self.hidden_size, device=device)
        c2 = torch.zeros(batch, self.hidden_size, device=device)
        h0 = torch.zeros(batch, self.hidden_size, device=device)
        h1 = torch.zeros(batch, self.hidden_size, device=device)
        h2 = torch.zeros(batch, self.hidden_size, device=device)
        return [h0, h1, h2], [c0, c1, c2]

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]
        return out_frame, vec_h_new, vec_c_new

    def normalize_quaternions(self, quaternions):
        return quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def sanitize_quaternion_frame(self, frame, reference_frame=None):
        translation = frame[:, 0:TRANSLATION_SIZE]
        quaternions = self.normalize_quaternions(
            frame[:, TRANSLATION_SIZE:].reshape(frame.size(0), -1, QUATERNION_SIZE)
        )

        if reference_frame is not None:
            reference_quaternions = self.normalize_quaternions(
                reference_frame[:, TRANSLATION_SIZE:].reshape(
                    reference_frame.size(0), -1, QUATERNION_SIZE
                )
            )
            continuity_sign = torch.where(
                torch.sum(quaternions * reference_quaternions, dim=2, keepdim=True)
                < 0.0,
                -torch.ones_like(quaternions[:, :, 0:1]),
                torch.ones_like(quaternions[:, :, 0:1]),
            )
            quaternions = quaternions * continuity_sign

        return torch.cat(
            (translation, quaternions.reshape(frame.size(0), -1)),
            dim=1,
        )

    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        gt_lst = np.ones((100, groundtruth_num))
        con_lst = np.zeros((100, condition_num))
        lst = np.concatenate((gt_lst, con_lst), 1).reshape(-1)
        return lst[0:seq_len]

    def forward(
        self, real_seq, condition_num=CONDITION_NUM, groundtruth_num=GROUNDTRUTH_NUM
    ):
        batch = real_seq.size(0)
        seq_len = real_seq.size(1)

        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        vec_h, vec_c = self.init_hidden(batch)

        out_frames = []
        out_frame = torch.zeros(batch, self.out_frame_size, device=real_seq.device)
        previous_frame = None

        for i in range(seq_len):
            if condition_lst[i] == 1:
                in_frame = real_seq[:, i]
            else:
                in_frame = self.sanitize_quaternion_frame(out_frame, previous_frame)

            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)
            previous_frame = in_frame

        return torch.stack(out_frames, dim=1)

    def reconstruct_absolute_motion_torch(self, motion_sequence):
        absolute_motion = motion_sequence.clone()
        absolute_motion[:, :, TRANSLATION_X_INDEX] = torch.cumsum(
            motion_sequence[:, :, TRANSLATION_X_INDEX], dim=1
        )
        absolute_motion[:, :, TRANSLATION_Z_INDEX] = torch.cumsum(
            motion_sequence[:, :, TRANSLATION_Z_INDEX], dim=1
        )
        return absolute_motion

    def quaternion_multiply(self, left_quaternion, right_quaternion):
        left_w, left_x, left_y, left_z = torch.unbind(left_quaternion, dim=-1)
        right_w, right_x, right_y, right_z = torch.unbind(right_quaternion, dim=-1)

        return torch.stack(
            (
                left_w * right_w
                - left_x * right_x
                - left_y * right_y
                - left_z * right_z,
                left_w * right_x
                + left_x * right_w
                + left_y * right_z
                - left_z * right_y,
                left_w * right_y
                - left_x * right_z
                + left_y * right_w
                + left_z * right_x,
                left_w * right_z
                + left_x * right_y
                - left_y * right_x
                + left_z * right_w,
            ),
            dim=-1,
        )

    def rotate_vector_by_quaternion(self, vectors, quaternions):
        quaternion_xyz = quaternions[..., 1:4]
        quaternion_w = quaternions[..., 0:1]
        t = 2.0 * torch.cross(quaternion_xyz, vectors, dim=-1)
        return vectors + quaternion_w * t + torch.cross(quaternion_xyz, t, dim=-1)

    def quaternion_sequence_to_joint_positions(self, motion_sequence):
        batch, seq_len, _ = motion_sequence.size()
        flattened_motion = motion_sequence.reshape(-1, self.out_frame_size)
        frame_count = flattened_motion.size(0)

        joint_positions = []
        world_quaternions = []
        identity_quaternion = self.identity_quaternion.unsqueeze(0).expand(
            frame_count, -1
        )

        for joint_index, joint_name in enumerate(JOINT_NAMES):
            parent_index = self.joint_parent_indices[joint_index]
            quaternion_start_index = self.joint_quaternion_start_indices[joint_index]

            if quaternion_start_index >= 0:
                local_quaternion = self.normalize_quaternions(
                    flattened_motion[
                        :,
                        quaternion_start_index : quaternion_start_index
                        + QUATERNION_SIZE,
                    ]
                )
            else:
                local_quaternion = identity_quaternion

            if parent_index < 0:
                joint_position = flattened_motion[:, 0:TRANSLATION_SIZE]
                world_quaternion = local_quaternion
            else:
                parent_position = joint_positions[parent_index]
                parent_world_quaternion = world_quaternions[parent_index]
                joint_offset = (
                    self.joint_offsets[joint_index].unsqueeze(0).expand(frame_count, -1)
                )
                joint_position = parent_position + self.rotate_vector_by_quaternion(
                    joint_offset, parent_world_quaternion
                )
                world_quaternion = self.normalize_quaternions(
                    self.quaternion_multiply(parent_world_quaternion, local_quaternion)
                )

            joint_positions.append(joint_position)
            world_quaternions.append(world_quaternion)

        stacked_positions = torch.stack(joint_positions, dim=1)
        return stacked_positions.reshape(batch, seq_len, len(JOINT_NAMES), 3)

    def calculate_fk_position_loss(self, out_seq, groundtruth_seq):
        predicted_motion = self.reconstruct_absolute_motion_torch(out_seq)
        groundtruth_motion = self.reconstruct_absolute_motion_torch(groundtruth_seq)

        predicted_positions = self.quaternion_sequence_to_joint_positions(
            predicted_motion
        )
        groundtruth_positions = self.quaternion_sequence_to_joint_positions(
            groundtruth_motion
        )

        return nn.MSELoss()(
            predicted_positions[:, :, 1:, :],
            groundtruth_positions[:, :, 1:, :],
        )

    def calculate_loss_components(self, out_seq, groundtruth_seq):
        translation_loss = nn.MSELoss()(
            out_seq[:, :, 0:TRANSLATION_SIZE],
            groundtruth_seq[:, :, 0:TRANSLATION_SIZE],
        )

        pred_quaternions = self.normalize_quaternions(
            out_seq[:, :, TRANSLATION_SIZE:].reshape(-1, QUATERNION_SIZE)
        )
        gt_quaternions = self.normalize_quaternions(
            groundtruth_seq[:, :, TRANSLATION_SIZE:].reshape(-1, QUATERNION_SIZE)
        )

        quaternion_dot = (
            torch.sum(pred_quaternions * gt_quaternions, dim=1)
            .abs()
            .clamp(0.0, 1.0 - 1e-7)
        )
        quaternion_loss = (2.0 * torch.acos(quaternion_dot)).mean()
        if self.fk_loss_weight == 0.0:
            fk_position_loss = translation_loss.new_zeros(())
        else:
            fk_position_loss = self.calculate_fk_position_loss(out_seq, groundtruth_seq)

        weighted_translation_loss = self.translation_loss_weight * translation_loss
        weighted_quaternion_loss = self.quaternion_loss_weight * quaternion_loss
        weighted_fk_position_loss = self.fk_loss_weight * fk_position_loss
        total_loss = (
            weighted_translation_loss
            + weighted_quaternion_loss
            + weighted_fk_position_loss
        )

        return {
            "total_loss": total_loss,
            "translation_loss": translation_loss,
            "quaternion_loss": quaternion_loss,
            "fk_position_loss": fk_position_loss,
            "weighted_translation_loss": weighted_translation_loss,
            "weighted_quaternion_loss": weighted_quaternion_loss,
            "weighted_fk_position_loss": weighted_fk_position_loss,
        }

    def calculate_loss(self, out_seq, groundtruth_seq):
        return self.calculate_loss_components(out_seq, groundtruth_seq)["total_loss"]


def load_model_state_dict_compatible(model, state_dict, checkpoint_path=""):
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key
        for key in incompatible_keys.missing_keys
        if key not in BACKWARD_COMPATIBLE_STATE_KEYS
    ]
    unexpected_keys = [
        key
        for key in incompatible_keys.unexpected_keys
        if key not in BACKWARD_COMPATIBLE_STATE_KEYS
    ]

    if missing_keys or unexpected_keys:
        error_parts = []
        if missing_keys:
            error_parts.append("missing keys: {}".format(missing_keys))
        if unexpected_keys:
            error_parts.append("unexpected keys: {}".format(unexpected_keys))
        checkpoint_label = checkpoint_path if checkpoint_path != "" else "checkpoint"
        raise RuntimeError(
            "Incompatible {} ({})".format(
                checkpoint_label,
                "; ".join(error_parts),
            )
        )

    ignored_missing_keys = [
        key
        for key in incompatible_keys.missing_keys
        if key in BACKWARD_COMPATIBLE_STATE_KEYS
    ]
    if ignored_missing_keys:
        print(
            "Loaded checkpoint {} with backward-compatible defaults for {}".format(
                checkpoint_path if checkpoint_path != "" else "",
                ignored_missing_keys,
            )
        )


def build_model_input_sequence(real_seq_np):
    dif = (
        real_seq_np[:, 1 : real_seq_np.shape[1]]
        - real_seq_np[:, 0 : real_seq_np.shape[1] - 1]
    )
    real_seq_dif_np = real_seq_np[:, 0 : real_seq_np.shape[1] - 1].copy()
    real_seq_dif_np[:, :, TRANSLATION_X_INDEX] = dif[:, :, TRANSLATION_X_INDEX]
    real_seq_dif_np[:, :, TRANSLATION_Z_INDEX] = dif[:, :, TRANSLATION_Z_INDEX]
    return real_seq_dif_np


def reconstruct_absolute_motion_np(model_motion):
    reconstructed_motion = np.array(model_motion, dtype=np.float64, copy=True)
    last_x = 0.0
    last_z = 0.0
    for frame in range(reconstructed_motion.shape[0]):
        reconstructed_motion[frame, TRANSLATION_X_INDEX] = (
            reconstructed_motion[frame, TRANSLATION_X_INDEX] + last_x
        )
        last_x = reconstructed_motion[frame, TRANSLATION_X_INDEX]

        reconstructed_motion[frame, TRANSLATION_Z_INDEX] = (
            reconstructed_motion[frame, TRANSLATION_Z_INDEX] + last_z
        )
        last_z = reconstructed_motion[frame, TRANSLATION_Z_INDEX]

    return read_bvh.enforce_quaternion_sequence_continuity(reconstructed_motion)


def train_one_iteration(
    real_seq_np,
    model,
    optimizer,
    iteration,
    save_dance_folder,
    print_loss=False,
    save_bvh_motion=True,
):
    device = next(model.parameters()).device

    real_seq_dif_np = build_model_input_sequence(real_seq_np)
    real_seq = torch.tensor(real_seq_dif_np, dtype=torch.float32, device=device)

    seq_len = real_seq.size(1) - 1
    in_real_seq = real_seq[:, 0:seq_len]
    predict_groundtruth_seq = real_seq[:, 1 : seq_len + 1]

    predict_seq = model.forward(in_real_seq, CONDITION_NUM, GROUNDTRUTH_NUM)

    optimizer.zero_grad()
    loss_components = model.calculate_loss_components(
        predict_seq, predict_groundtruth_seq
    )
    loss = loss_components["total_loss"]
    loss.backward()
    optimizer.step()

    if print_loss == True:
        print("###########" + "iter %07d" % iteration + "######################")
        print("loss_total: " + str(loss_components["total_loss"].detach().cpu().item()))
        print(
            "loss_translation: {} (weighted: {})".format(
                loss_components["translation_loss"].detach().cpu().item(),
                loss_components["weighted_translation_loss"].detach().cpu().item(),
            )
        )
        print(
            "loss_quaternion: {} (weighted: {})".format(
                loss_components["quaternion_loss"].detach().cpu().item(),
                loss_components["weighted_quaternion_loss"].detach().cpu().item(),
            )
        )
        print(
            "loss_fk_position: {} (weighted: {})".format(
                loss_components["fk_position_loss"].detach().cpu().item(),
                loss_components["weighted_fk_position_loss"].detach().cpu().item(),
            )
        )

    if save_bvh_motion == True:
        gt_seq = predict_groundtruth_seq[0].detach().cpu().numpy()
        out_seq = predict_seq[0].detach().cpu().numpy()

        gt_motion = reconstruct_absolute_motion_np(gt_seq)
        out_motion = reconstruct_absolute_motion_np(out_seq)

        read_bvh.write_quaternion_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + "_gt.bvh"),
            gt_motion,
            NON_END_BONES,
        )
        read_bvh.write_quaternion_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + "_out.bvh"),
            out_motion,
            NON_END_BONES,
        )


def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        length = 10
        if length < 1:
            length = 1
        len_lst = len_lst + [length]

    index_lst = []
    index = 0
    for length in len_lst:
        for i in range(length):
            index_lst = index_lst + [index]
        index = index + 1
    return index_lst


def load_dances(dance_folder):
    dance_files = sorted(os.listdir(dance_folder))
    dances = []
    print("Loading motion files...")
    for dance_file in dance_files:
        if dance_file.endswith(".npy") == False:
            continue
        dance = np.load(os.path.join(dance_folder, dance_file))
        dance = read_bvh.enforce_quaternion_sequence_continuity(dance)
        dances = dances + [dance]
    print(len(dances), " Motion files loaded")

    return dances


def train(
    dances,
    frame_rate,
    batch,
    seq_len,
    read_weight_path,
    write_weight_folder,
    write_bvh_motion_folder,
    in_frame,
    out_frame,
    hidden_size=HIDDEN_SIZE,
    total_iter=500000,
    translation_loss_weight=1.0,
    quaternion_loss_weight=1.0,
    fk_loss_weight=1.0,
):
    seq_len = seq_len + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = acLSTM(
        in_frame_size=in_frame,
        hidden_size=hidden_size,
        out_frame_size=out_frame,
        translation_loss_weight=translation_loss_weight,
        quaternion_loss_weight=quaternion_loss_weight,
        fk_loss_weight=fk_loss_weight,
    )

    if read_weight_path != "":
        load_model_state_dict_compatible(
            model,
            torch.load(read_weight_path, map_location=device),
            read_weight_path,
        )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    dance_len_lst = get_dance_len_lst(dances)
    random_range = len(dance_len_lst)
    speed = frame_rate / 30

    for iteration in range(total_iter):
        dance_batch = []
        for b in range(batch):
            dance_id = dance_len_lst[np.random.randint(0, random_range)]
            dance = dances[dance_id].copy()
            dance_len = dance.shape[0]

            max_start_id = int(dance_len - seq_len * speed - 10)
            if max_start_id <= 10:
                raise ValueError(
                    "Motion {} is too short for seq_len {} at frame rate {}".format(
                        dance_id, seq_len, frame_rate
                    )
                )

            start_id = random.randint(10, max_start_id)
            sample_seq = []
            for i in range(seq_len):
                sample_seq = sample_seq + [dance[int(i * speed + start_id)]]

            T = [0.1 * (random.random() - 0.5), 0.0, 0.1 * (random.random() - 0.5)]
            R = [0, 1, 0, (random.random() - 0.5) * np.pi * 2]
            sample_seq_augmented = read_bvh.augment_quaternion_train_data(
                sample_seq, T, R
            )
            dance_batch = dance_batch + [sample_seq_augmented]

        dance_batch_np = np.array(dance_batch)

        print_loss = False
        save_bvh_motion = False
        if iteration % PRINT_EVERY_ITERATIONS == 0:
            print_loss = True
        if iteration % SAVE_EVERY_ITERATIONS == 0:
            save_bvh_motion = True
            path = os.path.join(write_weight_folder, "%07d" % iteration + ".weight")
            torch.save(model.state_dict(), path)

        train_one_iteration(
            dance_batch_np,
            model,
            optimizer,
            iteration,
            write_bvh_motion_folder,
            print_loss,
            save_bvh_motion,
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dances_folder", type=str, required=True, help="Path for the training data"
    )
    parser.add_argument(
        "--write_weight_folder",
        type=str,
        required=True,
        help="Path to store checkpoints",
    )
    parser.add_argument(
        "--write_bvh_motion_folder",
        type=str,
        required=True,
        help="Path to store test generated bvh",
    )
    parser.add_argument(
        "--read_weight_path", type=str, default="", help="Checkpoint model path"
    )
    parser.add_argument(
        "--dance_frame_rate", type=int, default=60, help="Dance frame rate"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--in_frame", type=int, default=175, help="Input channel")
    parser.add_argument("--out_frame", type=int, default=175, help="Output channels")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length")
    parser.add_argument(
        "--total_iterations", type=int, default=100000, help="Total iterations"
    )
    parser.add_argument(
        "--translation_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the translation MSE term",
    )
    parser.add_argument(
        "--quaternion_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the quaternion angular loss term",
    )
    parser.add_argument(
        "--fk_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the FK joint-position loss term",
    )

    args = parser.parse_args()

    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances = load_dances(args.dances_folder)

    train(
        dances,
        args.dance_frame_rate,
        args.batch_size,
        args.seq_len,
        args.read_weight_path,
        args.write_weight_folder,
        args.write_bvh_motion_folder,
        args.in_frame,
        args.out_frame,
        args.hidden_size,
        total_iter=args.total_iterations,
        translation_loss_weight=args.translation_loss_weight,
        quaternion_loss_weight=args.quaternion_loss_weight,
        fk_loss_weight=args.fk_loss_weight,
    )


def _filter_legacy_cli_args(argv):
    filtered_argv = []
    skip_next = False

    for argument_index, argument in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if argument == "--fk_loss_weight":
            skip_next = True
            continue

        if argument.startswith("--fk_loss_weight="):
            continue

        filtered_argv.append(argument)

    return filtered_argv


def main_redirect_to_no_fk(argv=None):
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    filtered_argv = _filter_legacy_cli_args(effective_argv)

    if len(filtered_argv) != len(effective_argv):
        print(
            "pytorch_train_quad_aclstm.py now delegates to pytorch_train_quad_aclstm_no_fk.py; ignoring --fk_loss_weight."
        )

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]] + filtered_argv
        import pytorch_train_quad_aclstm_no_fk as quad_no_fk_train

        quad_no_fk_train.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main_redirect_to_no_fk()
