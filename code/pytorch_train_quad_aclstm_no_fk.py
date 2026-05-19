import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import read_bvh
import pytorch_train_quad_aclstm as quad_train

TRANSLATION_X_INDEX = quad_train.TRANSLATION_X_INDEX
TRANSLATION_Z_INDEX = quad_train.TRANSLATION_Z_INDEX
HIDDEN_SIZE = quad_train.HIDDEN_SIZE
CONDITION_NUM = quad_train.CONDITION_NUM
GROUNDTRUTH_NUM = quad_train.GROUNDTRUTH_NUM
IN_FRAME_SIZE = quad_train.IN_FRAME_SIZE
PRINT_EVERY_ITERATIONS = 500  # quad_train.PRINT_EVERY_ITERATIONS
CHECKPOINT_SAVE_EVERY_ITERATIONS = 4000  # quad_train.SAVE_EVERY_ITERATIONS
PREVIEW_SAVE_EVERY_ITERATIONS = 5000
NON_END_BONES = quad_train.NON_END_BONES


def create_autocast_context(device_type, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def create_grad_scaler(device_type, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device=device_type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class QuaternionNoFKacLSTM(quad_train.acLSTM):
    def __init__(
        self,
        in_frame_size=IN_FRAME_SIZE,
        hidden_size=HIDDEN_SIZE,
        out_frame_size=IN_FRAME_SIZE,
        translation_loss_weight=1.0,
        quaternion_loss_weight=1.0,
        fk_loss_weight=0.0,
        use_quaternion_mse_loss=False,
    ):
        super(QuaternionNoFKacLSTM, self).__init__(
            in_frame_size=in_frame_size,
            hidden_size=hidden_size,
            out_frame_size=out_frame_size,
            translation_loss_weight=translation_loss_weight,
            quaternion_loss_weight=quaternion_loss_weight,
            fk_loss_weight=fk_loss_weight,
        )
        self.use_quaternion_mse_loss = use_quaternion_mse_loss

    def forward(
        self, real_seq, condition_num=CONDITION_NUM, groundtruth_num=GROUNDTRUTH_NUM
    ):
        batch = real_seq.size(0)
        seq_len = real_seq.size(1)
        cycle_length = condition_num + groundtruth_num

        vec_h, vec_c = self.init_hidden(batch)

        out_frames = []
        out_frame = torch.zeros(batch, self.out_frame_size, device=real_seq.device)
        previous_frame = None

        for frame_index in range(seq_len):
            # keep the original 5-gt / 5-feedback schedule, but sanitize feedback so quaternions stay normalized and sign-consistent.
            if frame_index % cycle_length < groundtruth_num:
                in_frame = real_seq[:, frame_index]
            else:
                in_frame = self.sanitize_quaternion_frame(out_frame, previous_frame)

            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_frames.append(out_frame)
            previous_frame = in_frame

        return torch.stack(out_frames, dim=1)

    def calculate_loss_components(self, out_seq, groundtruth_seq):
        translation_loss = nn.MSELoss()(
            out_seq[:, :, 0 : quad_train.TRANSLATION_SIZE],
            groundtruth_seq[:, :, 0 : quad_train.TRANSLATION_SIZE],
        )

        pred_quaternions = self.normalize_quaternions(
            out_seq[:, :, quad_train.TRANSLATION_SIZE :].reshape(
                -1, quad_train.QUATERNION_SIZE
            )
        )
        gt_quaternions = self.normalize_quaternions(
            groundtruth_seq[:, :, quad_train.TRANSLATION_SIZE :].reshape(
                -1, quad_train.QUATERNION_SIZE
            )
        )

        if self.use_quaternion_mse_loss == True:
            # align signs before mse because q and -q represent the same rotation.
            sign_alignment = torch.where(
                torch.sum(pred_quaternions * gt_quaternions, dim=1, keepdim=True) < 0.0,
                -torch.ones_like(pred_quaternions[:, 0:1]),
                torch.ones_like(pred_quaternions[:, 0:1]),
            )
            aligned_pred_quaternions = pred_quaternions * sign_alignment
            quaternion_loss = nn.MSELoss()(
                aligned_pred_quaternions,
                gt_quaternions,
            )
        else:
            quaternion_dot = (
                torch.sum(pred_quaternions * gt_quaternions, dim=1)
                .abs()
                .clamp(0.0, 1.0 - 1e-7)
            )
            quaternion_loss = (2.0 * torch.acos(quaternion_dot)).mean()

        weighted_translation_loss = self.translation_loss_weight * translation_loss
        weighted_quaternion_loss = self.quaternion_loss_weight * quaternion_loss
        total_loss = weighted_translation_loss + weighted_quaternion_loss

        return {
            "total_loss": total_loss,
            "translation_loss": translation_loss,
            "quaternion_loss": quaternion_loss,
            "weighted_translation_loss": weighted_translation_loss,
            "weighted_quaternion_loss": weighted_quaternion_loss,
        }


class QuaternionNoFKWindowDataset(Dataset):
    def __init__(
        self,
        dances,
        frame_rate,
        seq_len,
        augment_translation_range=0.0,
        augment_yaw_range_degrees=0.0,
    ):
        self.dances = dances
        self.seq_len = seq_len
        self.augment_translation_range = augment_translation_range
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
                "No valid quaternion training windows found for seq_len {} at frame rate {}".format(
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
            augment_translation_range=self.augment_translation_range,
            augment_yaw_range_degrees=self.augment_yaw_range_degrees,
        )
        return np.asarray(sample_seq_prepared, dtype=np.float32)


def create_training_dataloader(
    dances,
    frame_rate,
    batch_size,
    seq_len,
    augment_translation_range=0.0,
    augment_yaw_range_degrees=0.0,
    pin_memory=False,
):
    dataset = QuaternionNoFKWindowDataset(
        dances,
        frame_rate,
        seq_len,
        augment_translation_range=augment_translation_range,
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


def build_model_input_sequence(real_seq):
    # as in the other baselines, train the root x/z channels as displacements rather than absolute positions.
    dif = real_seq[:, 1:] - real_seq[:, 0 : real_seq.shape[1] - 1]
    model_seq = real_seq[:, 0 : real_seq.shape[1] - 1].clone()
    model_seq[:, :, TRANSLATION_X_INDEX] = dif[:, :, TRANSLATION_X_INDEX]
    model_seq[:, :, TRANSLATION_Z_INDEX] = dif[:, :, TRANSLATION_Z_INDEX]
    return model_seq


def prepare_sequence_for_model(
    motion_sequence,
    augment_translation_range=0.0,
    augment_yaw_range_degrees=0.0,
):
    prepared_sequence = np.array(motion_sequence, dtype=np.float64, copy=True)

    if augment_translation_range > 0.0 or augment_yaw_range_degrees > 0.0:
        translation_augment = [
            random.uniform(-augment_translation_range, augment_translation_range),
            0.0,
            random.uniform(-augment_translation_range, augment_translation_range),
        ]
        yaw_angle_radians = np.deg2rad(
            random.uniform(-augment_yaw_range_degrees, augment_yaw_range_degrees)
        )
        prepared_sequence = read_bvh.augment_quaternion_train_data(
            prepared_sequence,
            translation_augment,
            [0.0, 1.0, 0.0, yaw_angle_radians],
        )

    return prepared_sequence


def train_one_iteration(
    real_seq_batch,
    model,
    optimizer,
    iteration,
    save_dance_folder,
    grad_scaler=None,
    use_mixed_precision=False,
    print_loss=False,
    save_bvh_motion=True,
):
    device = next(model.parameters()).device

    if torch.is_tensor(real_seq_batch):
        real_seq = real_seq_batch.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
    else:
        real_seq = torch.tensor(real_seq_batch, dtype=torch.float32, device=device)

    real_seq = build_model_input_sequence(real_seq)

    seq_len = real_seq.size(1) - 1
    in_real_seq = real_seq[:, 0:seq_len]
    predict_groundtruth_seq = real_seq[:, 1 : seq_len + 1]

    with create_autocast_context(device.type, use_mixed_precision):
        predict_seq = model.forward(in_real_seq, CONDITION_NUM, GROUNDTRUTH_NUM)

    optimizer.zero_grad(set_to_none=True)
    # keep the actual loss evaluation in float32 even when the recurrent pass used autocast.
    loss_components = model.calculate_loss_components(
        predict_seq.float(), predict_groundtruth_seq.float()
    )
    loss = loss_components["total_loss"]
    if grad_scaler is not None and grad_scaler.is_enabled():
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
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

    if save_bvh_motion == True:
        gt_seq = predict_groundtruth_seq[0].detach().float().cpu().numpy()
        out_seq = predict_seq[0].detach().float().cpu().numpy()

        gt_motion = quad_train.reconstruct_absolute_motion_np(gt_seq)
        out_motion = quad_train.reconstruct_absolute_motion_np(out_seq)

        read_bvh.write_quaternion_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + "_gt.bvh"),
            gt_motion,
            NON_END_BONES,
        )
        read_bvh.write_quaternion_traindata_to_bvh(
            os.path.join(save_dance_folder, "%07d" % iteration + ".bvh"),
            out_motion,
            NON_END_BONES,
        )


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
    augment_translation_range=0.0,
    augment_yaw_range_degrees=0.0,
    use_quaternion_mse_loss=False,
):
    seq_len = seq_len + 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mixed_precision = device.type == "cuda"

    model = QuaternionNoFKacLSTM(
        in_frame_size=in_frame,
        hidden_size=hidden_size,
        out_frame_size=out_frame,
        translation_loss_weight=translation_loss_weight,
        quaternion_loss_weight=quaternion_loss_weight,
        fk_loss_weight=0.0,
        use_quaternion_mse_loss=use_quaternion_mse_loss,
    )

    if read_weight_path != "":
        quad_train.load_model_state_dict_compatible(
            model,
            torch.load(read_weight_path, map_location=device),
            read_weight_path,
        )

    model.to(device)
    train_model = model

    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
    grad_scaler = create_grad_scaler(device.type, use_mixed_precision)
    train_model.train()

    if use_quaternion_mse_loss == True:
        print("Using sign-aligned unit-quaternion MSE loss in the no-FK trainer")

    training_dataloader = create_training_dataloader(
        dances,
        frame_rate,
        batch,
        seq_len,
        augment_translation_range=augment_translation_range,
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
        if iteration % CHECKPOINT_SAVE_EVERY_ITERATIONS == 0:
            path = os.path.join(write_weight_folder, "%07d" % iteration + ".weight")
            torch.save(model.state_dict(), path)
        if iteration % PREVIEW_SAVE_EVERY_ITERATIONS == 0:
            save_bvh_motion = True

        train_one_iteration(
            dance_batch,
            train_model,
            optimizer,
            iteration,
            write_bvh_motion_folder,
            grad_scaler=grad_scaler,
            use_mixed_precision=use_mixed_precision,
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
        "--augment_translation_range",
        type=float,
        default=0.0,
        help="Optional max absolute x/z translation jitter applied before training",
    )
    parser.add_argument(
        "--augment_yaw_range_degrees",
        type=float,
        default=0.0,
        help="Optional max absolute world-yaw augmentation angle in degrees",
    )
    parser.add_argument(
        "--use_quaternion_mse_loss",
        action="store_true",
        help="Use MSE on sign-aligned unit quaternions instead of the default angular quaternion loss",
    )

    args = parser.parse_args()

    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances = quad_train.load_dances(args.dances_folder)

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
        augment_translation_range=args.augment_translation_range,
        augment_yaw_range_degrees=args.augment_yaw_range_degrees,
        use_quaternion_mse_loss=args.use_quaternion_mse_loss,
    )


if __name__ == "__main__":
    main()
