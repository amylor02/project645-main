import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import read_bvh

TRANSLATION_X_INDEX = 0
TRANSLATION_Y_INDEX = 1
TRANSLATION_Z_INDEX = 2
TRANSLATION_SIZE = 3
ROTATION_START_INDEX = 3
HIDDEN_SIZE = 1024
CONDITION_NUM = 5
GROUNDTRUTH_NUM = 5
IN_FRAME_SIZE = 132
SAVE_EVERY_ITERATIONS = 4000
PRINT_EVERY_ITERATIONS = 500


class EulerMotionNormalizer:
    def __init__(self, channel_mean, safe_std, device):
        self.channel_mean_np = np.array(channel_mean, dtype=np.float64, copy=True)
        self.safe_std_np = np.array(safe_std, dtype=np.float64, copy=True)
        self.channel_mean = torch.tensor(
            self.channel_mean_np, dtype=torch.float32, device=device
        )
        self.safe_std = torch.tensor(
            self.safe_std_np, dtype=torch.float32, device=device
        )

    @classmethod
    def from_file(cls, stats_path, device):
        stats = np.load(stats_path)
        representation = str(stats["representation"].item())
        if representation != "euler":
            raise ValueError(
                "Expected Euler normalization stats, found {} in {}".format(
                    representation, stats_path
                )
            )
        return cls(stats["channel_mean"], stats["safe_std"], device)

    def normalize_absolute_motion_np(self, motion):
        return (motion - self.channel_mean_np) / self.safe_std_np

    def denormalize_absolute_motion_np(self, motion):
        return motion * self.safe_std_np + self.channel_mean_np

    def denormalize_model_motion_torch(self, motion):
        denormalized_motion = motion * self.safe_std
        denormalized_motion[..., TRANSLATION_Y_INDEX] = (
            denormalized_motion[..., TRANSLATION_Y_INDEX]
            + self.channel_mean[TRANSLATION_Y_INDEX]
        )
        denormalized_motion[..., ROTATION_START_INDEX:] = (
            denormalized_motion[..., ROTATION_START_INDEX:]
            + self.channel_mean[ROTATION_START_INDEX:]
        )
        return denormalized_motion

    def normalize_model_motion_torch(self, motion):
        normalized_motion = motion / self.safe_std
        normalized_motion[..., TRANSLATION_Y_INDEX] = (
            motion[..., TRANSLATION_Y_INDEX] - self.channel_mean[TRANSLATION_Y_INDEX]
        ) / self.safe_std[TRANSLATION_Y_INDEX]
        normalized_motion[..., ROTATION_START_INDEX:] = (
            motion[..., ROTATION_START_INDEX:]
            - self.channel_mean[ROTATION_START_INDEX:]
        ) / self.safe_std[ROTATION_START_INDEX:]
        return normalized_motion


def wrap_euler_angles_torch(angle_values):
    return torch.remainder(angle_values + 180.0, 360.0) - 180.0


class acLSTM(nn.Module):
    def __init__(
        self,
        in_frame_size=IN_FRAME_SIZE,
        hidden_size=HIDDEN_SIZE,
        out_frame_size=IN_FRAME_SIZE,
        translation_loss_weight=1.0,
        rotation_loss_weight=1.0,
        normalizer=None,
    ):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size
        self.translation_loss_weight = translation_loss_weight
        self.rotation_loss_weight = rotation_loss_weight
        self.normalizer = normalizer

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

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

    def sanitize_frame(self, frame):
        # wrap feedback in physical Euler space so autoregressive inputs stay on the same angle branch as training.
        if self.normalizer is not None:
            frame_physical = self.normalizer.denormalize_model_motion_torch(frame)
        else:
            frame_physical = frame

        frame_physical[..., ROTATION_START_INDEX:] = wrap_euler_angles_torch(
            frame_physical[..., ROTATION_START_INDEX:]
        )

        if self.normalizer is not None:
            return self.normalizer.normalize_model_motion_torch(frame_physical)
        return frame_physical

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

        for i in range(seq_len):
            if condition_lst[i] == 1:
                in_frame = real_seq[:, i]
            else:
                in_frame = self.sanitize_frame(out_frame)

            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)

        return torch.stack(out_frames, dim=1)

    def generate(self, initial_seq, generate_frames_number):
        batch = initial_seq.size(0)
        vec_h, vec_c = self.init_hidden(batch)

        out_frames = []
        out_frame = torch.zeros(batch, self.out_frame_size, device=initial_seq.device)

        for i in range(initial_seq.size(1)):
            in_frame = initial_seq[:, i]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)

        for i in range(generate_frames_number):
            in_frame = self.sanitize_frame(out_frame)
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)

        return torch.stack(out_frames, dim=1)

    def calculate_loss_components(self, out_seq, groundtruth_seq):
        if self.normalizer is not None:
            pred_motion = self.normalizer.denormalize_model_motion_torch(out_seq)
            gt_motion = self.normalizer.denormalize_model_motion_torch(groundtruth_seq)
        else:
            pred_motion = out_seq
            gt_motion = groundtruth_seq

        translation_loss = nn.MSELoss()(
            pred_motion[..., 0:TRANSLATION_SIZE],
            gt_motion[..., 0:TRANSLATION_SIZE],
        )

        angle_delta = wrap_euler_angles_torch(
            pred_motion[..., ROTATION_START_INDEX:]
            - gt_motion[..., ROTATION_START_INDEX:]
        )
        angle_delta_radians = angle_delta * (np.pi / 180.0)
        rotation_loss = torch.mean(1.0 - torch.cos(angle_delta_radians))

        weighted_translation_loss = self.translation_loss_weight * translation_loss
        weighted_rotation_loss = self.rotation_loss_weight * rotation_loss
        total_loss = weighted_translation_loss + weighted_rotation_loss

        return {
            "total_loss": total_loss,
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "weighted_translation_loss": weighted_translation_loss,
            "weighted_rotation_loss": weighted_rotation_loss,
        }

    def calculate_loss(self, out_seq, groundtruth_seq):
        return self.calculate_loss_components(out_seq, groundtruth_seq)["total_loss"]


def build_model_input_sequence(real_seq_np):
    # predict root x/z as frame-to-frame displacement, but keep root y and joint rotations in-place.
    dif = real_seq_np[:, 1:] - real_seq_np[:, 0 : real_seq_np.shape[1] - 1]
    if torch.is_tensor(real_seq_np):
        model_seq_np = real_seq_np[:, 0 : real_seq_np.shape[1] - 1].clone()
    else:
        model_seq_np = real_seq_np[:, 0 : real_seq_np.shape[1] - 1].copy()
    model_seq_np[:, :, TRANSLATION_X_INDEX] = dif[:, :, TRANSLATION_X_INDEX]
    model_seq_np[:, :, TRANSLATION_Z_INDEX] = dif[:, :, TRANSLATION_Z_INDEX]
    return model_seq_np


def reconstruct_absolute_motion(model_motion, normalizer=None):
    # preview/export needs absolute root x/z again, so undo the differencing used for the model input.
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

    if normalizer is not None:
        reconstructed_motion = normalizer.denormalize_absolute_motion_np(
            reconstructed_motion
        )

    return read_bvh.wrap_euler_train_data(reconstructed_motion)


def prepare_sequence_for_model(
    motion_sequence,
    normalizer=None,
    recenter_root=False,
    recenter_y=False,
    augment_yaw_range_degrees=0.0,
):
    # do geometry edits in physical angle space first, then re-normalize if this checkpoint was trained on normalized data.
    prepared_sequence = np.array(motion_sequence, dtype=np.float64, copy=True)

    if normalizer is not None:
        prepared_sequence = normalizer.denormalize_absolute_motion_np(prepared_sequence)

    prepared_sequence = read_bvh.wrap_euler_train_data(prepared_sequence)

    if recenter_root == True:
        prepared_sequence = read_bvh.recenter_euler_root_translation(
            prepared_sequence, recenter_y
        )

    if augment_yaw_range_degrees > 0.0:
        yaw_angle_radians = np.deg2rad(
            random.uniform(-augment_yaw_range_degrees, augment_yaw_range_degrees)
        )
        prepared_sequence = read_bvh.augment_euler_train_data(
            prepared_sequence,
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, yaw_angle_radians],
        )

    prepared_sequence = read_bvh.wrap_euler_train_data(prepared_sequence)

    if normalizer is not None:
        return normalizer.normalize_absolute_motion_np(prepared_sequence)
    return prepared_sequence


class EulerWindowDataset(Dataset):
    def __init__(
        self,
        dances,
        frame_rate,
        seq_len,
        normalizer=None,
        recenter_root=False,
        recenter_y=False,
        augment_yaw_range_degrees=0.0,
    ):
        self.dances = dances
        self.seq_len = seq_len
        self.normalizer = normalizer
        self.recenter_root = recenter_root
        self.recenter_y = recenter_y
        self.augment_yaw_range_degrees = augment_yaw_range_degrees
        self.sample_offsets = np.array(
            [int(frame_index * frame_rate / 30) for frame_index in range(seq_len)],
            dtype=np.int32,
        )
        self.sample_locations = []

        for dance_id, dance in enumerate(self.dances):
            max_start_id = int(dance.shape[0] - seq_len * frame_rate / 30 - 10)
            if max_start_id <= 10:
                continue

            for start_id in range(10, max_start_id + 1):
                self.sample_locations.append((dance_id, start_id))

        if len(self.sample_locations) == 0:
            raise ValueError(
                "No valid Euler training windows found for seq_len {} at frame rate {}".format(
                    seq_len, frame_rate
                )
            )

    def __len__(self):
        return len(self.sample_locations)

    def __getitem__(self, index):
        dance_id, start_id = self.sample_locations[index]
        dance = self.dances[dance_id]
        sample_seq = dance[start_id + self.sample_offsets]
        sample_seq_prepared = prepare_sequence_for_model(
            sample_seq,
            normalizer=self.normalizer,
            recenter_root=self.recenter_root,
            recenter_y=self.recenter_y,
            augment_yaw_range_degrees=self.augment_yaw_range_degrees,
        )
        return np.asarray(sample_seq_prepared, dtype=np.float32)


def create_training_dataloader(
    dances,
    frame_rate,
    batch_size,
    seq_len,
    normalizer=None,
    recenter_root=False,
    recenter_y=False,
    augment_yaw_range_degrees=0.0,
    pin_memory=False,
):
    dataset = EulerWindowDataset(
        dances,
        frame_rate,
        seq_len,
        normalizer=normalizer,
        recenter_root=recenter_root,
        recenter_y=recenter_y,
        augment_yaw_range_degrees=augment_yaw_range_degrees,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


def train_one_iteration(
    real_seq_np,
    model,
    optimizer,
    iteration,
    save_dance_folder,
    normalizer=None,
    print_loss=False,
    save_bvh_motion=True,
):
    device = next(model.parameters()).device

    if torch.is_tensor(real_seq_np):
        real_seq = real_seq_np.to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        real_seq = torch.tensor(real_seq_np, dtype=torch.float32, device=device)

    real_seq = build_model_input_sequence(real_seq)

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
            "loss_rotation: {} (weighted: {})".format(
                loss_components["rotation_loss"].detach().cpu().item(),
                loss_components["weighted_rotation_loss"].detach().cpu().item(),
            )
        )

    if save_bvh_motion == True:
        gt_seq = predict_groundtruth_seq[0].detach().cpu().numpy()
        out_seq = predict_seq[0].detach().cpu().numpy()

        gt_motion = reconstruct_absolute_motion(gt_seq, normalizer)
        out_motion = reconstruct_absolute_motion(out_seq, normalizer)

        read_bvh.write_euler_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + "_gt.bvh"),
            gt_motion,
        )
        read_bvh.write_euler_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + "_out.bvh"),
            out_motion,
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
    recenter_root=False,
    recenter_y=False,
    augment_yaw_range_degrees=0.0,
    normalization_stats_path="",
    translation_loss_weight=1.0,
    rotation_loss_weight=1.0,
):
    seq_len = seq_len + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = None
    if normalization_stats_path != "":
        normalizer = EulerMotionNormalizer.from_file(normalization_stats_path, device)

    model = acLSTM(
        in_frame_size=in_frame,
        hidden_size=hidden_size,
        out_frame_size=out_frame,
        translation_loss_weight=translation_loss_weight,
        rotation_loss_weight=rotation_loss_weight,
        normalizer=normalizer,
    )

    if read_weight_path != "":
        model.load_state_dict(torch.load(read_weight_path, map_location=device))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()

    training_dataloader = create_training_dataloader(
        dances,
        frame_rate,
        batch,
        seq_len,
        normalizer=normalizer,
        recenter_root=recenter_root,
        recenter_y=recenter_y,
        augment_yaw_range_degrees=augment_yaw_range_degrees,
        pin_memory=device.type == "cuda",
    )
    training_dataloader_iterator = iter(training_dataloader)

    for iteration in range(total_iter):
        try:
            dance_batch = next(training_dataloader_iterator)
        except StopIteration:
            training_dataloader_iterator = iter(training_dataloader)
            dance_batch = next(training_dataloader_iterator)

        print_loss = False
        save_bvh_motion = False
        if iteration % PRINT_EVERY_ITERATIONS == 0:
            print_loss = True
        if iteration % SAVE_EVERY_ITERATIONS == 0:
            save_bvh_motion = True
            path = os.path.join(write_weight_folder, "%07d" % iteration + ".weight")
            torch.save(model.state_dict(), path)

        train_one_iteration(
            dance_batch,
            model,
            optimizer,
            iteration,
            write_bvh_motion_folder,
            normalizer=normalizer,
            print_loss=print_loss,
            save_bvh_motion=save_bvh_motion,
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
    parser.add_argument("--in_frame", type=int, default=132, help="Input channel")
    parser.add_argument("--out_frame", type=int, default=132, help="Output channels")
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden size of the network"
    )
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length")
    parser.add_argument(
        "--total_iterations", type=int, default=100000, help="Total iterations"
    )
    parser.add_argument(
        "--recenter_root",
        action="store_true",
        help="Recenter root translation around the sequence mean before training",
    )
    parser.add_argument(
        "--recenter_y",
        action="store_true",
        help="Include root height when recentering the sequence",
    )
    parser.add_argument(
        "--augment_yaw_range_degrees",
        type=float,
        default=0.0,
        help="Optional world-yaw augmentation range in degrees",
    )
    parser.add_argument(
        "--normalization_stats_path",
        type=str,
        default="",
        help="Optional normalization stats produced by normalize_representation_data.py",
    )
    parser.add_argument(
        "--translation_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the translation MSE term",
    )
    parser.add_argument(
        "--rotation_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the Euler angle distance term",
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
        recenter_root=args.recenter_root,
        recenter_y=args.recenter_y,
        augment_yaw_range_degrees=args.augment_yaw_range_degrees,
        normalization_stats_path=args.normalization_stats_path,
        translation_loss_weight=args.translation_loss_weight,
        rotation_loss_weight=args.rotation_loss_weight,
    )


if __name__ == "__main__":
    main()
