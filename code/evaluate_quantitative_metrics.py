import argparse
import csv
import json
import os
import random
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import transforms3d.quaternions as quat

import pytorch_train_euler_aclstm as euler_train
import pytorch_train_pos_aclstm as pos_train
import pytorch_train_quad_aclstm as quad_train
import read_bvh
import rotation2xyz
import synthesize_pos_motion as pos_synth
import synthesize_quad_motion as quad_synth

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_ROOT_DEFAULT = os.path.join(WORKSPACE_ROOT, "synthesis", "quant")

REPRESENTATION_CHOICES = ["quat_no_fk_mse", "euler", "pos"]
STYLE_CHOICES = ["indian", "martial", "salsa"]

COMMON_JOINT_NAMES = list(read_bvh.skeleton.keys())
COMMON_ROOT_JOINT_NAME = "hip"
COMMON_ROOT_INDEX = COMMON_JOINT_NAMES.index(COMMON_ROOT_JOINT_NAME)
END_EFFECTOR_NAMES = ["lHand", "rHand", "lFoot", "rFoot", "head"]
FOOT_JOINT_NAMES = ["lFoot", "rFoot"]
ROTATION_JOINT_NAMES = [COMMON_ROOT_JOINT_NAME] + [
    bone_name
    for bone_name in read_bvh.non_end_bones
    if bone_name != COMMON_ROOT_JOINT_NAME
]

TRANSLATION_X_INDEX = 0
TRANSLATION_Y_INDEX = 1
TRANSLATION_Z_INDEX = 2
POSITION_ROOT_X_INDEX = read_bvh.joint_index[COMMON_ROOT_JOINT_NAME] * 3
POSITION_ROOT_Y_INDEX = POSITION_ROOT_X_INDEX + 1
POSITION_ROOT_Z_INDEX = POSITION_ROOT_X_INDEX + 2


@dataclass(frozen=True)
class ExperimentConfig:
    representation: str
    style: str
    dances_folder: str
    checkpoint_path: str
    in_frame_size: int
    hidden_size: int
    out_frame_size: int

    @property
    def experiment_id(self):
        return "{}/{}".format(self.representation, self.style)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run assignment-aligned quantitative motion evaluation across the "
            "project's positional, Euler, and quaternion checkpoints."
        )
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=REPRESENTATION_CHOICES,
        choices=REPRESENTATION_CHOICES,
        help="Representation groups to evaluate.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=STYLE_CHOICES,
        choices=STYLE_CHOICES,
        help="Dance styles to evaluate.",
    )
    parser.add_argument(
        "--initial_seq_len",
        type=int,
        default=20,
        help="Number of real seed frames provided to the model.",
    )
    parser.add_argument(
        "--generate_frames_number",
        type=int,
        default=20,
        help="Number of future frames to autoregressively generate per seed.",
    )
    parser.add_argument(
        "--metric_horizons",
        nargs="+",
        type=int,
        default=[20],
        help="Future horizons to score. The default is the assignment-required 20 steps.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of deterministic seed windows to sample per style.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=60,
        help="Source frame rate used by the stored training arrays.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic seed-window selection.",
    )
    parser.add_argument(
        "--full_suite",
        action="store_true",
        help=(
            "Also compute the broader common-space metric suite: root-relative MPJPE, "
            "root ADE/FDE, end-effector MPJPE, rotation error when available, foot sliding, "
            "and MPJPE-vs-horizon curves."
        ),
    )
    parser.add_argument(
        "--export_bvh",
        action="store_true",
        help=(
            "Export rollout BVHs plus side-by-side seed, predicted, and ground-truth "
            "horizon clips under synthesis/quant/bvh/."
        ),
    )
    parser.add_argument(
        "--output_root",
        default=OUTPUT_ROOT_DEFAULT,
        help="Output folder for JSON/CSV summaries and optional BVH exports.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise immediately instead of skipping missing or failing experiments.",
    )
    parser.add_argument(
        "--start_buffer",
        type=int,
        default=10,
        help="Minimum start-frame buffer, matching the existing synthesis scripts.",
    )
    parser.add_argument(
        "--foot_height_tolerance",
        type=float,
        default=0.02,
        help="Height tolerance used for stance detection in the optional foot sliding metric.",
    )
    parser.add_argument(
        "--foot_stance_percentile",
        type=float,
        default=20.0,
        help="Ground-truth foot-speed percentile used to mark stance frames for foot sliding.",
    )

    args = parser.parse_args()
    args.metric_horizons = sorted(set(args.metric_horizons))

    if args.initial_seq_len < 2:
        raise ValueError("initial_seq_len must be at least 2.")
    if args.generate_frames_number < 1:
        raise ValueError("generate_frames_number must be at least 1.")
    if len(args.metric_horizons) == 0 or min(args.metric_horizons) < 1:
        raise ValueError("metric_horizons must contain positive integers.")
    if args.generate_frames_number < max(args.metric_horizons):
        raise ValueError(
            "generate_frames_number ({}) must be >= max(metric_horizons) ({}).".format(
                args.generate_frames_number, max(args.metric_horizons)
            )
        )
    if args.num_seeds < 1:
        raise ValueError("num_seeds must be at least 1.")
    if args.frame_rate % 30 != 0:
        raise ValueError(
            "frame_rate must be divisible by 30 for the current evaluation logic."
        )
    if not (0.0 <= args.foot_stance_percentile <= 100.0):
        raise ValueError("foot_stance_percentile must be between 0 and 100.")

    return args


def build_default_experiments(workspace_root):
    return [
        ExperimentConfig(
            representation="quat_no_fk_mse",
            style="indian",
            dances_folder=os.path.join(workspace_root, "train_data_quad", "indian"),
            checkpoint_path=os.path.join(
                workspace_root, "weights", "quad_no_fk_mse", "indian"
            ),
            in_frame_size=quad_train.IN_FRAME_SIZE,
            hidden_size=quad_train.HIDDEN_SIZE,
            out_frame_size=quad_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="quat_no_fk_mse",
            style="martial",
            dances_folder=os.path.join(workspace_root, "train_data_quad", "martial"),
            checkpoint_path=os.path.join(
                workspace_root, "weights", "quad_no_fk_mse", "martial"
            ),
            in_frame_size=quad_train.IN_FRAME_SIZE,
            hidden_size=quad_train.HIDDEN_SIZE,
            out_frame_size=quad_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="quat_no_fk_mse",
            style="salsa",
            dances_folder=os.path.join(workspace_root, "train_data_quad", "salsa"),
            checkpoint_path=os.path.join(
                workspace_root, "weights", "quad_no_fk_mse", "salsa"
            ),
            in_frame_size=quad_train.IN_FRAME_SIZE,
            hidden_size=quad_train.HIDDEN_SIZE,
            out_frame_size=quad_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="euler",
            style="indian",
            dances_folder=os.path.join(workspace_root, "train_data_euler", "indian"),
            checkpoint_path=os.path.join(workspace_root, "weights", "euler", "indian"),
            in_frame_size=euler_train.IN_FRAME_SIZE,
            hidden_size=euler_train.HIDDEN_SIZE,
            out_frame_size=euler_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="euler",
            style="martial",
            dances_folder=os.path.join(workspace_root, "train_data_euler", "martial"),
            checkpoint_path=os.path.join(workspace_root, "weights", "euler", "martial"),
            in_frame_size=euler_train.IN_FRAME_SIZE,
            hidden_size=euler_train.HIDDEN_SIZE,
            out_frame_size=euler_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="euler",
            style="salsa",
            dances_folder=os.path.join(workspace_root, "train_data_euler", "salsa"),
            checkpoint_path=os.path.join(workspace_root, "weights", "euler", "salsa"),
            in_frame_size=euler_train.IN_FRAME_SIZE,
            hidden_size=euler_train.HIDDEN_SIZE,
            out_frame_size=euler_train.IN_FRAME_SIZE,
        ),
        ExperimentConfig(
            representation="pos",
            style="indian",
            dances_folder=os.path.join(workspace_root, "train_data_pos", "indian"),
            checkpoint_path=os.path.join(workspace_root, "weights", "pos", "indian"),
            in_frame_size=pos_synth.In_frame_size,
            hidden_size=pos_synth.Hidden_size,
            out_frame_size=pos_synth.In_frame_size,
        ),
        ExperimentConfig(
            representation="pos",
            style="martial",
            dances_folder=os.path.join(workspace_root, "train_data_pos", "martial"),
            checkpoint_path=os.path.join(workspace_root, "weights", "pos", "martial"),
            in_frame_size=pos_synth.In_frame_size,
            hidden_size=pos_synth.Hidden_size,
            out_frame_size=pos_synth.In_frame_size,
        ),
        ExperimentConfig(
            representation="pos",
            style="salsa",
            dances_folder=os.path.join(workspace_root, "train_data_pos", "salsa"),
            checkpoint_path=os.path.join(workspace_root, "weights", "pos", "salsa"),
            in_frame_size=pos_synth.In_frame_size,
            hidden_size=pos_synth.Hidden_size,
            out_frame_size=pos_synth.In_frame_size,
        ),
    ]


def filter_experiments(experiments, args):
    return [
        experiment
        for experiment in experiments
        if experiment.representation in args.representations
        and experiment.style in args.styles
    ]


def resolve_checkpoint_path(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if os.path.isdir(checkpoint_path):
        weight_files = [
            file_name
            for file_name in sorted(os.listdir(checkpoint_path))
            if file_name.endswith(".weight")
        ]
        if len(weight_files) == 0:
            raise ValueError("No .weight files found in {}".format(checkpoint_path))
        return os.path.join(checkpoint_path, weight_files[-1])

    raise ValueError("Checkpoint path does not exist: {}".format(checkpoint_path))


def load_named_dances(representation, dances_folder):
    if os.path.isdir(dances_folder) == False:
        raise ValueError("Dance folder does not exist: {}".format(dances_folder))

    named_dances = {}
    for file_name in sorted(os.listdir(dances_folder)):
        if file_name.endswith(".npy") == False:
            continue
        dance = np.load(os.path.join(dances_folder, file_name))
        if representation == "quat_no_fk_mse":
            dance = read_bvh.enforce_quaternion_sequence_continuity(dance)
        named_dances[file_name] = np.array(dance, dtype=np.float64, copy=True)

    if len(named_dances) == 0:
        raise ValueError("No .npy motion files found in {}".format(dances_folder))

    return named_dances


def build_style_seed_specs(style, style_datasets, args, required_gt_horizon):
    common_clip_names = None
    for named_dances in style_datasets.values():
        clip_names = set(named_dances.keys())
        if common_clip_names is None:
            common_clip_names = clip_names
        else:
            common_clip_names = common_clip_names.intersection(clip_names)

    common_clip_names = sorted(common_clip_names or [])
    if len(common_clip_names) == 0:
        raise ValueError(
            "No aligned .npy files found across the selected representations for style {}.".format(
                style
            )
        )

    speed = args.frame_rate // 30
    candidate_specs = []

    for clip_name in common_clip_names:
        min_length = min(
            named_dances[clip_name].shape[0] for named_dances in style_datasets.values()
        )
        max_start_id = int(
            min_length
            - (args.initial_seq_len + required_gt_horizon) * speed
            - args.start_buffer
        )
        if max_start_id <= args.start_buffer:
            continue

        for start_id in range(args.start_buffer, max_start_id + 1):
            candidate_specs.append(
                {
                    "clip_name": clip_name,
                    "start_id": int(start_id),
                }
            )

    if len(candidate_specs) == 0:
        raise ValueError(
            "No valid seed windows found for style {} with initial_seq_len={} and horizon={}.".format(
                style, args.initial_seq_len, required_gt_horizon
            )
        )

    rng = random.Random(args.seed + zlib.crc32(style.encode("utf-8")))
    sample_count = min(args.num_seeds, len(candidate_specs))
    selected_specs = rng.sample(candidate_specs, sample_count)
    selected_specs = sorted(
        selected_specs, key=lambda spec: (spec["clip_name"], spec["start_id"])
    )

    return selected_specs


def sample_motion_sequence(dance, start_id, frame_count, speed):
    frames = []
    for frame_index in range(frame_count):
        frames.append(dance[int(frame_index * speed + start_id)])
    return np.array(frames, dtype=np.float64)


def rebase_root_translation_xz(motion_sequence, x_index, z_index):
    rebased_motion = np.array(motion_sequence, dtype=np.float64, copy=True)
    rebased_motion[:, x_index] = rebased_motion[:, x_index] - rebased_motion[0, x_index]
    rebased_motion[:, z_index] = rebased_motion[:, z_index] - rebased_motion[0, z_index]
    return rebased_motion


def reconstruct_root_translation_xz(model_motion, x_index, z_index):
    reconstructed_motion = np.array(model_motion, dtype=np.float64, copy=True)
    last_x = 0.0
    last_z = 0.0
    for frame_index in range(reconstructed_motion.shape[0]):
        reconstructed_motion[frame_index, x_index] = (
            reconstructed_motion[frame_index, x_index] + last_x
        )
        last_x = reconstructed_motion[frame_index, x_index]

        reconstructed_motion[frame_index, z_index] = (
            reconstructed_motion[frame_index, z_index] + last_z
        )
        last_z = reconstructed_motion[frame_index, z_index]

    return reconstructed_motion


def normalize_quaternion_array(quaternion_array, epsilon=1e-8):
    normalized = np.array(quaternion_array, dtype=np.float64, copy=True)
    norms = np.linalg.norm(normalized, axis=-1, keepdims=True)
    safe_norms = np.where(norms > epsilon, norms, 1.0)
    normalized = normalized / safe_norms

    identity_quaternion = np.zeros_like(normalized)
    identity_quaternion[..., 0] = 1.0
    return np.where(norms > epsilon, normalized, identity_quaternion)


def position_motion_to_joint_positions(motion_sequence):
    joint_positions = []
    for frame in motion_sequence:
        position_dict = read_bvh.data_vec_to_position_dic(frame, read_bvh.skeleton)
        joint_positions.append(
            [position_dict[joint_name] for joint_name in COMMON_JOINT_NAMES]
        )
    return np.array(joint_positions, dtype=np.float64)


def euler_motion_to_joint_positions(motion_sequence):
    joint_positions = []
    for frame in motion_sequence:
        raw_frame = np.array(frame, dtype=np.float64, copy=True)
        raw_frame[0:3] = raw_frame[0:3] / read_bvh.weight_translation
        position_dict = rotation2xyz.get_skeleton_position(
            raw_frame, read_bvh.non_end_bones, read_bvh.skeleton
        )
        joint_positions.append(
            [
                np.asarray(position_dict[joint_name], dtype=np.float64).reshape(3)
                for joint_name in COMMON_JOINT_NAMES
            ]
        )
    return np.array(joint_positions, dtype=np.float64)


def quaternion_motion_to_joint_positions(model, motion_sequence):
    device = next(model.parameters()).device
    motion_tensor = torch.tensor(
        motion_sequence[np.newaxis, ...], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        positions = model.quaternion_sequence_to_joint_positions(motion_tensor)
    return positions[0].detach().cpu().numpy().astype(np.float64)


def euler_motion_to_local_quaternions(motion_sequence):
    frame_count = motion_sequence.shape[0]
    local_quaternions = np.zeros(
        (frame_count, len(ROTATION_JOINT_NAMES), quad_train.QUATERNION_SIZE),
        dtype=np.float64,
    )

    for frame_index, frame in enumerate(motion_sequence):
        root_angles = np.array([frame[5], frame[4], frame[3]], dtype=np.float64)
        local_quaternions[frame_index, 0] = quat.mat2quat(
            rotation2xyz.eulerAnglesToRotationMatrix_hip(root_angles)
        )

        for bone_index, joint_name in enumerate(read_bvh.non_end_bones):
            base_index = 6 + bone_index * 3
            joint_angles = np.array(
                [frame[base_index + 1], frame[base_index + 2], frame[base_index]],
                dtype=np.float64,
            )
            target_index = ROTATION_JOINT_NAMES.index(joint_name)
            local_quaternions[frame_index, target_index] = quat.mat2quat(
                rotation2xyz.eulerAnglesToRotationMatrix(joint_angles)
            )

    return normalize_quaternion_array(local_quaternions)


def quaternion_motion_to_local_quaternions(motion_sequence):
    frame_count = motion_sequence.shape[0]
    local_quaternions = np.zeros(
        (frame_count, len(ROTATION_JOINT_NAMES), quad_train.QUATERNION_SIZE),
        dtype=np.float64,
    )

    for joint_name in ROTATION_JOINT_NAMES:
        joint_index = quad_train.JOINT_NAME_TO_INDEX[joint_name]
        quaternion_start_index = quad_train.JOINT_QUATERNION_START_INDICES[joint_index]
        target_index = ROTATION_JOINT_NAMES.index(joint_name)
        local_quaternions[:, target_index, :] = motion_sequence[
            :,
            quaternion_start_index : quaternion_start_index
            + quad_train.QUATERNION_SIZE,
        ]

    return normalize_quaternion_array(local_quaternions)


def generate_position_future(seed_motion, generate_frames_number, model):
    dif_motion = seed_motion[1:] - seed_motion[0 : seed_motion.shape[0] - 1]
    model_input = seed_motion[0 : seed_motion.shape[0] - 1].copy()
    model_input[:, POSITION_ROOT_X_INDEX] = dif_motion[:, POSITION_ROOT_X_INDEX]
    model_input[:, POSITION_ROOT_Z_INDEX] = dif_motion[:, POSITION_ROOT_Z_INDEX]

    initial_seq = torch.tensor(
        model_input[np.newaxis, ...], dtype=torch.float32, device="cuda"
    )
    with torch.no_grad():
        predicted_sequence = model.forward(initial_seq, generate_frames_number)

    predicted_motion = (
        predicted_sequence[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        .reshape(-1, model.out_frame_size)
    )
    predicted_motion = reconstruct_root_translation_xz(
        predicted_motion, POSITION_ROOT_X_INDEX, POSITION_ROOT_Z_INDEX
    )
    warmup_length = seed_motion.shape[0] - 1
    return predicted_motion[warmup_length:]


def generate_euler_future(seed_motion, generate_frames_number, model):
    prepared_seed_motion = euler_train.prepare_sequence_for_model(seed_motion)
    model_input = euler_train.build_model_input_sequence(
        prepared_seed_motion[np.newaxis, ...]
    )
    device = next(model.parameters()).device
    initial_seq = torch.tensor(model_input, dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted_sequence = model.generate(initial_seq, generate_frames_number)

    predicted_motion = predicted_sequence[0].detach().cpu().numpy().astype(np.float64)
    predicted_motion = euler_train.reconstruct_absolute_motion(predicted_motion)
    warmup_length = seed_motion.shape[0] - 1
    return predicted_motion[warmup_length:]


def generate_quaternion_future(seed_motion, generate_frames_number, model):
    model_input = quad_train.build_model_input_sequence(seed_motion[np.newaxis, ...])
    device = next(model.parameters()).device
    initial_seq = torch.tensor(model_input, dtype=torch.float32, device=device)

    with torch.no_grad():
        predicted_sequence = model.generate(initial_seq, generate_frames_number)

    predicted_motion = predicted_sequence[0].detach().cpu().numpy().astype(np.float64)
    predicted_motion = quad_train.reconstruct_absolute_motion_np(predicted_motion)
    warmup_length = seed_motion.shape[0] - 1
    return predicted_motion[warmup_length:]


def compute_mpjpe(predicted_positions, groundtruth_positions):
    return float(
        np.mean(np.linalg.norm(predicted_positions - groundtruth_positions, axis=-1))
    )


def compute_root_relative_mpjpe(predicted_positions, groundtruth_positions):
    predicted_root_relative = (
        predicted_positions
        - predicted_positions[:, COMMON_ROOT_INDEX : COMMON_ROOT_INDEX + 1, :]
    )
    groundtruth_root_relative = (
        groundtruth_positions
        - groundtruth_positions[:, COMMON_ROOT_INDEX : COMMON_ROOT_INDEX + 1, :]
    )
    return compute_mpjpe(predicted_root_relative, groundtruth_root_relative)


def compute_root_trajectory_metrics(predicted_positions, groundtruth_positions):
    predicted_root = predicted_positions[:, COMMON_ROOT_INDEX, :]
    groundtruth_root = groundtruth_positions[:, COMMON_ROOT_INDEX, :]
    root_errors = np.linalg.norm(predicted_root - groundtruth_root, axis=-1)
    return {
        "root_ade": float(np.mean(root_errors)),
        "root_fde": float(root_errors[-1]),
    }


def compute_end_effector_mpjpe(predicted_positions, groundtruth_positions):
    end_effector_indices = [
        COMMON_JOINT_NAMES.index(joint_name)
        for joint_name in END_EFFECTOR_NAMES
        if joint_name in COMMON_JOINT_NAMES
    ]
    predicted_effectors = predicted_positions[:, end_effector_indices, :]
    groundtruth_effectors = groundtruth_positions[:, end_effector_indices, :]
    return compute_mpjpe(predicted_effectors, groundtruth_effectors)


def compute_rotation_error_degrees(predicted_quaternions, groundtruth_quaternions):
    alignment = np.abs(np.sum(predicted_quaternions * groundtruth_quaternions, axis=-1))
    alignment = np.clip(alignment, -1.0, 1.0)
    angle_errors = 2.0 * np.arccos(alignment)
    return float(np.degrees(np.mean(angle_errors)))


def compute_foot_sliding(predicted_positions, groundtruth_positions, args):
    if predicted_positions.shape[0] < 2 or groundtruth_positions.shape[0] < 2:
        return None

    foot_indices = [
        COMMON_JOINT_NAMES.index(joint_name)
        for joint_name in FOOT_JOINT_NAMES
        if joint_name in COMMON_JOINT_NAMES
    ]
    if len(foot_indices) == 0:
        return None

    predicted_feet = predicted_positions[:, foot_indices, :]
    groundtruth_feet = groundtruth_positions[:, foot_indices, :]

    groundtruth_horizontal_velocity = np.linalg.norm(
        np.diff(groundtruth_feet[:, :, [0, 2]], axis=0), axis=-1
    )
    groundtruth_foot_heights = groundtruth_feet[:-1, :, 1]
    groundtruth_floor_heights = np.min(groundtruth_feet[:, :, 1], axis=0, keepdims=True)

    speed_threshold = np.percentile(
        groundtruth_horizontal_velocity, args.foot_stance_percentile
    )
    stance_mask = (groundtruth_horizontal_velocity <= speed_threshold) & (
        groundtruth_foot_heights
        <= groundtruth_floor_heights + args.foot_height_tolerance
    )

    if np.any(stance_mask) == False:
        return None

    predicted_horizontal_velocity = np.linalg.norm(
        np.diff(predicted_feet[:, :, [0, 2]], axis=0), axis=-1
    )
    return float(np.mean(predicted_horizontal_velocity[stance_mask]))


def compute_stepwise_mpjpe(predicted_positions, groundtruth_positions):
    return np.linalg.norm(predicted_positions - groundtruth_positions, axis=-1).mean(
        axis=1
    )


def summarize_values(values):
    if len(values) == 0:
        return None
    array = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def load_position_model(experiment):
    if torch.cuda.is_available() == False:
        raise RuntimeError(
            "The positional synthesis model requires CUDA because its current implementation hard-codes .cuda() tensors."
        )

    model = pos_synth.acLSTM(
        experiment.in_frame_size,
        experiment.hidden_size,
        experiment.out_frame_size,
    )
    checkpoint_path = resolve_checkpoint_path(experiment.checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    model.cuda()
    model.eval()
    return model, checkpoint_path


def load_euler_model(experiment):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = euler_train.acLSTM(
        experiment.in_frame_size,
        experiment.hidden_size,
        experiment.out_frame_size,
    )
    checkpoint_path = resolve_checkpoint_path(experiment.checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, checkpoint_path


def load_quaternion_model(experiment):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = quad_synth.acLSTM(
        experiment.in_frame_size,
        experiment.hidden_size,
        experiment.out_frame_size,
    )
    checkpoint_path = resolve_checkpoint_path(experiment.checkpoint_path)
    quad_train.load_model_state_dict_compatible(
        model,
        torch.load(checkpoint_path, map_location=device),
        checkpoint_path,
    )
    model.to(device)
    model.eval()
    return model, checkpoint_path


def load_model(experiment):
    if experiment.representation == "pos":
        return load_position_model(experiment)
    if experiment.representation == "euler":
        return load_euler_model(experiment)
    if experiment.representation == "quat_no_fk_mse":
        return load_quaternion_model(experiment)
    raise ValueError("Unsupported representation: {}".format(experiment.representation))


def get_representation_root_indices(representation):
    if representation == "pos":
        return POSITION_ROOT_X_INDEX, POSITION_ROOT_Z_INDEX
    return TRANSLATION_X_INDEX, TRANSLATION_Z_INDEX


def convert_motion_to_positions(representation, model, motion_sequence):
    if representation == "pos":
        return position_motion_to_joint_positions(motion_sequence)
    if representation == "euler":
        return euler_motion_to_joint_positions(motion_sequence)
    if representation == "quat_no_fk_mse":
        return quaternion_motion_to_joint_positions(model, motion_sequence)
    raise ValueError("Unsupported representation: {}".format(representation))


def convert_motion_to_local_quaternions(representation, motion_sequence):
    if representation == "euler":
        return euler_motion_to_local_quaternions(motion_sequence)
    if representation == "quat_no_fk_mse":
        return quaternion_motion_to_local_quaternions(motion_sequence)
    return None


def export_rollout_bvh(representation, rollout_motion, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if representation == "pos":
        read_bvh.write_traindata_to_bvh(output_path, rollout_motion)
    elif representation == "euler":
        read_bvh.write_euler_traindata_to_bvh(output_path, rollout_motion)
    elif representation == "quat_no_fk_mse":
        read_bvh.write_quaternion_traindata_to_bvh(
            output_path, rollout_motion, read_bvh.non_end_bones
        )
    else:
        raise ValueError("Unsupported representation: {}".format(representation))


def build_bvh_export_relative_path(experiment, seed_spec, suffix):
    return os.path.join(
        "bvh",
        experiment.representation,
        experiment.style,
        "{}__start_{:04d}__{}.bvh".format(
            os.path.splitext(seed_spec["clip_name"])[0],
            seed_spec["start_id"],
            suffix,
        ),
    )


def evaluate_seed(experiment, model, dance, seed_spec, args, required_gt_horizon):
    speed = args.frame_rate // 30
    total_required_frames = args.initial_seq_len + required_gt_horizon
    sampled_motion = sample_motion_sequence(
        dance, seed_spec["start_id"], total_required_frames, speed
    )

    root_x_index, root_z_index = get_representation_root_indices(
        experiment.representation
    )
    sampled_motion = rebase_root_translation_xz(
        sampled_motion, root_x_index, root_z_index
    )
    seed_motion = sampled_motion[: args.initial_seq_len]
    groundtruth_future_motion = sampled_motion[
        args.initial_seq_len : args.initial_seq_len + required_gt_horizon
    ]

    if experiment.representation == "pos":
        predicted_future_motion = generate_position_future(
            seed_motion, args.generate_frames_number, model
        )
    elif experiment.representation == "euler":
        predicted_future_motion = generate_euler_future(
            seed_motion, args.generate_frames_number, model
        )
    elif experiment.representation == "quat_no_fk_mse":
        predicted_future_motion = generate_quaternion_future(
            seed_motion, args.generate_frames_number, model
        )
    else:
        raise ValueError(
            "Unsupported representation: {}".format(experiment.representation)
        )

    predicted_eval_motion = predicted_future_motion[:required_gt_horizon]
    predicted_positions = convert_motion_to_positions(
        experiment.representation, model, predicted_eval_motion
    )
    groundtruth_positions = convert_motion_to_positions(
        experiment.representation, model, groundtruth_future_motion
    )

    horizon_metrics = {}
    for horizon in args.metric_horizons:
        predicted_horizon_positions = predicted_positions[:horizon]
        groundtruth_horizon_positions = groundtruth_positions[:horizon]
        metrics = {
            "mpjpe": compute_mpjpe(
                predicted_horizon_positions, groundtruth_horizon_positions
            )
        }

        if args.full_suite:
            metrics["root_relative_mpjpe"] = compute_root_relative_mpjpe(
                predicted_horizon_positions, groundtruth_horizon_positions
            )
            metrics.update(
                compute_root_trajectory_metrics(
                    predicted_horizon_positions, groundtruth_horizon_positions
                )
            )
            metrics["end_effector_mpjpe"] = compute_end_effector_mpjpe(
                predicted_horizon_positions, groundtruth_horizon_positions
            )

            predicted_quaternions = convert_motion_to_local_quaternions(
                experiment.representation, predicted_eval_motion[:horizon]
            )
            groundtruth_quaternions = convert_motion_to_local_quaternions(
                experiment.representation, groundtruth_future_motion[:horizon]
            )
            if (
                predicted_quaternions is not None
                and groundtruth_quaternions is not None
            ):
                metrics["rotation_error_degrees"] = compute_rotation_error_degrees(
                    predicted_quaternions, groundtruth_quaternions
                )

        horizon_metrics["h{}".format(horizon)] = metrics

    result = {
        "clip_name": seed_spec["clip_name"],
        "start_id": int(seed_spec["start_id"]),
        "horizons": horizon_metrics,
    }

    if args.full_suite:
        rollout_positions = convert_motion_to_positions(
            experiment.representation,
            model,
            predicted_future_motion[:required_gt_horizon],
        )
        result["foot_sliding"] = compute_foot_sliding(
            rollout_positions, groundtruth_positions, args
        )
        result["mpjpe_curve"] = [
            float(value)
            for value in compute_stepwise_mpjpe(
                predicted_positions,
                groundtruth_positions,
            )
        ]

    if args.export_bvh:
        rollout_motion = np.concatenate((seed_motion, predicted_future_motion), axis=0)
        rollout_relative_output_path = os.path.join(
            "bvh",
            experiment.representation,
            experiment.style,
            "{}__start_{:04d}.bvh".format(
                os.path.splitext(seed_spec["clip_name"])[0], seed_spec["start_id"]
            ),
        )
        export_rollout_bvh(
            experiment.representation,
            rollout_motion,
            os.path.join(args.output_root, rollout_relative_output_path),
        )
        result["export_bvh_path"] = rollout_relative_output_path.replace("\\", "/")
        result["export_rollout_bvh_path"] = result["export_bvh_path"]

        seed_relative_output_path = build_bvh_export_relative_path(
            experiment, seed_spec, "seed"
        )
        export_rollout_bvh(
            experiment.representation,
            seed_motion,
            os.path.join(args.output_root, seed_relative_output_path),
        )
        result["export_seed_bvh_path"] = seed_relative_output_path.replace("\\", "/")

        for horizon in args.metric_horizons:
            predicted_horizon_relative_output_path = build_bvh_export_relative_path(
                experiment, seed_spec, "pred_h{}".format(horizon)
            )
            export_rollout_bvh(
                experiment.representation,
                predicted_future_motion[:horizon],
                os.path.join(args.output_root, predicted_horizon_relative_output_path),
            )
            result["export_pred_h{}_bvh_path".format(horizon)] = (
                predicted_horizon_relative_output_path.replace("\\", "/")
            )

            groundtruth_horizon_relative_output_path = build_bvh_export_relative_path(
                experiment, seed_spec, "gt_h{}".format(horizon)
            )
            export_rollout_bvh(
                experiment.representation,
                groundtruth_future_motion[:horizon],
                os.path.join(
                    args.output_root, groundtruth_horizon_relative_output_path
                ),
            )
            result["export_gt_h{}_bvh_path".format(horizon)] = (
                groundtruth_horizon_relative_output_path.replace("\\", "/")
            )

    return result


def aggregate_seed_results(seed_results, args):
    aggregate = {
        "seed_count": len(seed_results),
        "horizons": {},
    }

    for horizon in args.metric_horizons:
        horizon_key = "h{}".format(horizon)
        metric_names = sorted(
            {
                metric_name
                for seed_result in seed_results
                for metric_name in seed_result["horizons"][horizon_key].keys()
            }
        )
        aggregate["horizons"][horizon_key] = {}
        for metric_name in metric_names:
            values = []
            for seed_result in seed_results:
                metric_value = seed_result["horizons"][horizon_key].get(metric_name)
                if metric_value is not None:
                    values.append(metric_value)
            summary = summarize_values(values)
            if summary is not None:
                aggregate["horizons"][horizon_key][metric_name] = summary

    if args.full_suite:
        foot_sliding_values = [
            seed_result["foot_sliding"]
            for seed_result in seed_results
            if seed_result.get("foot_sliding") is not None
        ]
        foot_sliding_summary = summarize_values(foot_sliding_values)
        if foot_sliding_summary is not None:
            aggregate["foot_sliding"] = foot_sliding_summary

        curve_values = [
            np.array(seed_result["mpjpe_curve"], dtype=np.float64)
            for seed_result in seed_results
            if "mpjpe_curve" in seed_result
        ]
        if len(curve_values) > 0:
            stacked_curves = np.stack(curve_values, axis=0)
            aggregate["mpjpe_curve"] = {
                "mean": stacked_curves.mean(axis=0).tolist(),
                "std": stacked_curves.std(axis=0).tolist(),
            }

    return aggregate


def flatten_experiment_summary(experiment_result, args):
    row = {
        "experiment_id": experiment_result["experiment_id"],
        "representation": experiment_result["representation"],
        "style": experiment_result["style"],
        "status": experiment_result["status"],
    }

    if experiment_result["status"] != "ok":
        row["reason"] = experiment_result.get("reason", "")
        return row

    aggregate = experiment_result["aggregate"]
    row["seed_count"] = aggregate["seed_count"]
    for horizon in args.metric_horizons:
        horizon_key = "h{}".format(horizon)
        for metric_name, summary in aggregate["horizons"].get(horizon_key, {}).items():
            row["{}_{}_mean".format(metric_name, horizon_key)] = summary["mean"]
            row["{}_{}_std".format(metric_name, horizon_key)] = summary["std"]

    if args.full_suite and "foot_sliding" in aggregate:
        row["foot_sliding_mean"] = aggregate["foot_sliding"]["mean"]
        row["foot_sliding_std"] = aggregate["foot_sliding"]["std"]

    return row


def flatten_seed_rows(experiment_result, args):
    rows = []
    if experiment_result["status"] != "ok":
        return rows

    for seed_index, seed_result in enumerate(experiment_result["seed_results"]):
        row = {
            "experiment_id": experiment_result["experiment_id"],
            "representation": experiment_result["representation"],
            "style": experiment_result["style"],
            "seed_index": seed_index,
            "clip_name": seed_result["clip_name"],
            "start_id": seed_result["start_id"],
        }
        for horizon in args.metric_horizons:
            horizon_key = "h{}".format(horizon)
            for metric_name, metric_value in seed_result["horizons"][
                horizon_key
            ].items():
                row["{}_{}".format(metric_name, horizon_key)] = metric_value
        if args.full_suite:
            row["foot_sliding"] = seed_result.get("foot_sliding")
        for key, value in seed_result.items():
            if key.startswith("export_") and value is not None:
                row[key] = value
        rows.append(row)
    return rows


def build_curve_rows(experiment_result):
    rows = []
    if experiment_result["status"] != "ok":
        return rows

    curve_summary = experiment_result["aggregate"].get("mpjpe_curve")
    if curve_summary is None:
        return rows

    for step_index, mean_value in enumerate(curve_summary["mean"], start=1):
        rows.append(
            {
                "experiment_id": experiment_result["experiment_id"],
                "representation": experiment_result["representation"],
                "style": experiment_result["style"],
                "step": step_index,
                "mpjpe_mean": mean_value,
                "mpjpe_std": curve_summary["std"][step_index - 1],
            }
        )
    return rows


def write_csv_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if len(rows) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_file.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_experiment(experiment, style_context, args, required_gt_horizon):
    model, checkpoint_path = load_model(experiment)
    dance_map = style_context["datasets"][experiment.representation]
    seed_specs = style_context["seed_specs"]

    seed_results = []
    for seed_spec in seed_specs:
        dance = dance_map[seed_spec["clip_name"]]
        seed_results.append(
            evaluate_seed(
                experiment, model, dance, seed_spec, args, required_gt_horizon
            )
        )

    aggregate = aggregate_seed_results(seed_results, args)
    return {
        "experiment_id": experiment.experiment_id,
        "representation": experiment.representation,
        "style": experiment.style,
        "status": "ok",
        "checkpoint_path": os.path.relpath(checkpoint_path, WORKSPACE_ROOT).replace(
            "\\", "/"
        ),
        "seed_specs": seed_specs,
        "seed_results": seed_results,
        "aggregate": aggregate,
    }


def print_experiment_summary(experiment_result, args):
    if experiment_result["status"] != "ok":
        print(
            "[{}] skipped: {}".format(
                experiment_result["experiment_id"], experiment_result.get("reason", "")
            )
        )
        return

    horizon_key = "h{}".format(args.metric_horizons[0])
    mpjpe_summary = experiment_result["aggregate"]["horizons"][horizon_key]["mpjpe"]
    print(
        "[{}] MPJPE@{} mean={:.6f} std={:.6f} over {} seeds".format(
            experiment_result["experiment_id"],
            args.metric_horizons[0],
            mpjpe_summary["mean"],
            mpjpe_summary["std"],
            experiment_result["aggregate"]["seed_count"],
        )
    )


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    all_experiments = build_default_experiments(WORKSPACE_ROOT)
    selected_experiments = filter_experiments(all_experiments, args)
    if len(selected_experiments) == 0:
        raise ValueError("No experiments matched the selected representations/styles.")

    required_gt_horizon = max(args.metric_horizons)
    if args.full_suite:
        required_gt_horizon = max(required_gt_horizon, args.generate_frames_number)

    experiments_by_style = {}
    for experiment in selected_experiments:
        experiments_by_style.setdefault(experiment.style, []).append(experiment)

    style_contexts = {}
    for style, style_experiments in experiments_by_style.items():
        style_datasets = {}
        for experiment in style_experiments:
            style_datasets[experiment.representation] = load_named_dances(
                experiment.representation, experiment.dances_folder
            )
        style_contexts[style] = {
            "datasets": style_datasets,
            "seed_specs": build_style_seed_specs(
                style, style_datasets, args, required_gt_horizon
            ),
        }

    experiment_results = []
    for experiment in selected_experiments:
        try:
            experiment_result = run_experiment(
                experiment,
                style_contexts[experiment.style],
                args,
                required_gt_horizon,
            )
        except Exception as error:
            if args.strict:
                raise
            experiment_result = {
                "experiment_id": experiment.experiment_id,
                "representation": experiment.representation,
                "style": experiment.style,
                "status": "skipped",
                "reason": str(error),
            }
        experiment_results.append(experiment_result)
        print_experiment_summary(experiment_result, args)

    summary_payload = {
        "args": {
            "representations": args.representations,
            "styles": args.styles,
            "initial_seq_len": args.initial_seq_len,
            "generate_frames_number": args.generate_frames_number,
            "metric_horizons": args.metric_horizons,
            "num_seeds": args.num_seeds,
            "frame_rate": args.frame_rate,
            "seed": args.seed,
            "full_suite": args.full_suite,
            "export_bvh": args.export_bvh,
            "output_root": os.path.relpath(args.output_root, WORKSPACE_ROOT).replace(
                "\\", "/"
            ),
        },
        "results": experiment_results,
    }

    summary_json_path = os.path.join(args.output_root, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary_payload, summary_file, indent=2)

    summary_rows = [
        flatten_experiment_summary(experiment_result, args)
        for experiment_result in experiment_results
    ]
    write_csv_rows(os.path.join(args.output_root, "summary.csv"), summary_rows)

    seed_rows = []
    for experiment_result in experiment_results:
        seed_rows.extend(flatten_seed_rows(experiment_result, args))
    write_csv_rows(os.path.join(args.output_root, "seed_metrics.csv"), seed_rows)

    if args.full_suite:
        curve_rows = []
        for experiment_result in experiment_results:
            curve_rows.extend(build_curve_rows(experiment_result))
        write_csv_rows(os.path.join(args.output_root, "horizon_curve.csv"), curve_rows)

    print("Wrote {}".format(os.path.relpath(summary_json_path, WORKSPACE_ROOT)))


if __name__ == "__main__":
    main()
