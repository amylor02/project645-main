import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymotion.rotations.ortho6d as ortho6d
import torch
from pymotion.ops.skeleton import from_root_dual_quat, translation_each_joint

from generator_architecture import Generator_Model
from latent_diffusion_prior import ContinuousLatentDiffusionPrior
from motion_data import (
    LMA_KEYS,
    TestMotionData,
    compute_contact_signals,
    integrate_root_translation_np,
    split_motion_joints,
)
from train_data import Train_Data
from train_latent_diffusion import (
    DEFAULT_PARAM,
    build_eval_sample,
    freeze_module,
    load_lma_annotation,
    load_style_resources,
    prepare_train_batch,
    resolve_style_for_filename,
    seed_all,
)
from train_vq_vae import (
    get_bvh_from_disk,
    get_info_from_bvh,
    load_model,
    param as base_param,
    result_to_bvh,
)


DEFAULT_SOURCE_LATENT_EDIT_STRENGTH = 0.35


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample from the continuous latent diffusion prior"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Dataset root containing train/, eval/, and annotations/",
    )
    parser.add_argument(
        "--vae-model-path", required=True, help="Path to pretrained VAE generator.pt"
    )
    parser.add_argument(
        "--diffusion-model-path",
        required=True,
        help="Path to latent_diffusion_prior.pt or latent_diffusion_prior_best.pt",
    )
    parser.add_argument(
        "--source-split",
        choices=["auto", "train", "eval"],
        default="auto",
        help="Dataset split used as the condition source when --bvh-path is not provided. Use auto to infer from --bvh-path.",
    )
    parser.add_argument(
        "--clip-index",
        type=int,
        default=0,
        help="Clip index within the selected split used as the condition source",
    )
    parser.add_argument(
        "--eval-index",
        dest="clip_index",
        type=int,
        help="Backward-compatible alias for --clip-index",
    )
    parser.add_argument(
        "--bvh-path",
        default=None,
        help="Optional direct BVH path. If provided, it overrides --source-split and --clip-index",
    )
    parser.add_argument(
        "--mode", choices=["full", "lma_only", "traj_only", "uncond"], default="full"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Override CFG scale from the saved diffusion config",
    )
    parser.add_argument(
        "--lma-cfg-scale",
        type=float,
        default=None,
        help="Optional separate guidance scale for the LMA contribution",
    )
    parser.add_argument(
        "--traj-cfg-scale",
        type=float,
        default=None,
        help="Optional separate guidance scale for the trajectory contribution",
    )
    parser.add_argument(
        "--style-cfg-scale",
        type=float,
        default=None,
        help="Optional separate guidance scale for the style contribution",
    )
    parser.add_argument(
        "--full-mode-lma-prefix",
        action="store_true",
        help="When mode=full, use lma_only guidance for the first denoising steps and switch to full guidance for the remainder.",
    )
    parser.add_argument(
        "--full-mode-lma-prefix-fraction",
        type=float,
        default=0.5,
        help="Fraction of denoising steps that use lma_only before switching back to full when --full-mode-lma-prefix is enabled.",
    )
    parser.add_argument(
        "--full-mode-lma-suffix",
        action="store_true",
        help="When mode=full, use full guidance first and switch to lma_only guidance for the final denoising steps.",
    )
    parser.add_argument(
        "--full-mode-lma-suffix-fraction",
        type=float,
        default=0.5,
        help="Fraction of final denoising steps that use lma_only when --full-mode-lma-suffix is enabled.",
    )
    parser.add_argument(
        "--style-label",
        default=None,
        help="Optional target style label override. Defaults to the source clip style if known.",
    )
    parser.add_argument(
        "--style-id",
        type=int,
        default=None,
        help="Optional target style id override. Uses the checkpoint vocabulary.",
    )
    parser.add_argument(
        "--sample-steps", type=int, default=None, help="Override diffusion sample steps"
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="Override latent chunk length for long sampling",
    )
    parser.add_argument(
        "--overlap-len", type=int, default=None, help="Override latent overlap length"
    )
    parser.add_argument(
        "--halo-len", type=int, default=None, help="Override latent halo length"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override sampling temperature"
    )
    parser.add_argument("--eta", type=float, default=None, help="Override DDIM eta")
    parser.add_argument("--latent-target", choices=["mean", "sample"], default="mean")
    parser.add_argument(
        "--source-latent-edit",
        action="store_true",
        help="Initialize sampling from a partially noised source latent instead of pure noise",
    )
    parser.add_argument(
        "--edit-strength",
        type=float,
        default=None,
        help="Source-latent edit strength in [0, 1]. Lower stays closer to the source clip.",
    )
    parser.add_argument(
        "--source-latent-noise-timestep",
        type=int,
        default=None,
        help="Advanced override for the source-latent forward-noise timestep",
    )
    parser.add_argument(
        "--frame-len",
        type=int,
        default=None,
        help="Optional output frame length override. Defaults to the eval condition length.",
    )
    parser.add_argument(
        "--lma-override-path",
        default=None,
        help="Optional path to an external LMA signal (.csv or .npy). If provided, it overrides the source LMA condition.",
    )
    parser.add_argument(
        "--root-override-blend",
        type=float,
        default=1.0,
        help="Blend factor for predicted root override in lma_only/uncond modes",
    )
    parser.add_argument(
        "--disable-predicted-root-override",
        action="store_true",
        help="Do not inject the predicted rough root trajectory into decode-time root channels",
    )
    parser.add_argument(
        "--save-bvh",
        action="store_true",
        help="Export a BVH using the eval sample skeleton as reference",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for tensors and optional BVH export",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAM["seed"])
    return parser.parse_args()


def load_diffusion_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_param = copy.deepcopy(base_param)
    saved_param.update(checkpoint.get("param", {}))
    return checkpoint, saved_param


def extract_style_metadata(checkpoint: dict):
    style_vocab = checkpoint.get("style_vocab") or []
    style_to_id = checkpoint.get("style_to_id") or {
        label: index for index, label in enumerate(style_vocab)
    }
    return list(style_vocab), dict(style_to_id)


def split_root_diagnostics(root_traj):
    if root_traj is None:
        return {
            "root_rotation": None,
            "root_translation": None,
            "root_translation_velocity": None,
        }

    root_rotation = root_traj[..., :6]
    root_translation = root_traj[..., 6:9]
    root_translation_velocity = None
    if root_translation.size(1) > 1:
        root_translation_velocity = root_translation[:, 1:] - root_translation[:, :-1]
    return {
        "root_rotation": root_rotation,
        "root_translation": root_translation,
        "root_translation_velocity": root_translation_velocity,
    }


def _resample_columns(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [T, C] array, got shape {values.shape}")
    target_len = max(int(target_len), 1)
    if values.shape[0] == target_len:
        return values.astype(np.float32)
    if values.shape[0] == 0:
        return np.zeros((target_len, values.shape[1]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values.astype(np.float32), target_len, axis=0)

    source_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    columns = [
        np.interp(target_x, source_x, values[:, column])
        for column in range(values.shape[1])
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def _coerce_lma_array(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        if values.size % len(LMA_KEYS) != 0:
            raise ValueError(
                f"1D LMA override length must be divisible by {len(LMA_KEYS)}, got {values.size}"
            )
        values = values.reshape(-1, len(LMA_KEYS))
    if values.ndim != 2:
        raise ValueError(f"Expected 2D LMA override array, got shape {values.shape}")
    if values.shape[1] < len(LMA_KEYS):
        padded = np.zeros((values.shape[0], len(LMA_KEYS)), dtype=np.float32)
        padded[:, : values.shape[1]] = values
        values = padded
    elif values.shape[1] > len(LMA_KEYS):
        values = values[:, : len(LMA_KEYS)]
    return values.astype(np.float32)


def load_lma_override_array(path_like: str) -> np.ndarray:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"LMA override file does not exist: {path}")

    if path.suffix.lower() == ".npy":
        values = np.load(path)
        return _coerce_lma_array(values)

    if path.suffix.lower() in {".csv", ".txt"}:
        dataframe = pd.read_csv(path)
        numeric = dataframe.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size == 0:
            raise ValueError(f"No numeric columns found in {path}")
        return _coerce_lma_array(numeric)

    raise ValueError(f"Unsupported LMA override file type: {path.suffix}")


def source_latent_edit_requested(args) -> bool:
    return bool(
        getattr(args, "source_latent_edit", False)
        or getattr(args, "edit_strength", None) is not None
        or getattr(args, "source_latent_noise_timestep", None) is not None
    )


def resolve_source_latent_edit_schedule(args, num_train_timesteps: int):
    max_timestep = max(int(num_train_timesteps) - 1, 0)
    timestep_override = getattr(args, "source_latent_noise_timestep", None)
    if timestep_override is not None:
        timestep = int(timestep_override)
        if timestep < 0 or timestep > max_timestep:
            raise ValueError(
                "source-latent-noise-timestep must be in "
                f"[0, {max_timestep}], got {timestep}"
            )
        effective_strength = (
            float(timestep) / float(max_timestep) if max_timestep > 0 else 0.0
        )
        return timestep, effective_strength, True

    strength = getattr(args, "edit_strength", None)
    if strength is None:
        strength = DEFAULT_SOURCE_LATENT_EDIT_STRENGTH
    strength = float(strength)
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f"edit-strength must be in [0, 1], got {strength}")
    timestep = int(round(strength * max_timestep))
    return timestep, strength, False


def _unique_ordered_indices(indices):
    ordered = []
    seen = set()
    for index in indices:
        value = int(index)
        if value < 0 or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _decoded_motion_to_world_positions(decoded_motion, means, stds, reference_bvh):
    frame_major = (
        decoded_motion.detach()
        .permute(1, 0)
        .contiguous()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    mean_dqs = means["dqs"].detach().cpu().numpy()
    std_dqs = stds["dqs"].detach().cpu().numpy()
    dqs = frame_major * std_dqs + mean_dqs
    dqs = dqs.reshape(dqs.shape[0], -1, 9)
    skeletal_dqs, _ = split_motion_joints(
        dqs,
        synthetic_joint_count=(
            int(means.get("synthetic_contact_joint_count", 1))
            if isinstance(means, dict)
            else 1
        ),
    )
    root_positions = integrate_root_translation_np(
        skeletal_dqs[:, 0, :],
        reference_bvh.data["positions"][:, 0, :3],
    )
    dual_quats = ortho6d.to_dual_quat(skeletal_dqs).reshape(
        skeletal_dqs.shape[0], -1, 8
    )

    bvh_parents = np.array(reference_bvh.data["parents"])
    _, rotations = from_root_dual_quat(dual_quats, bvh_parents)

    fk_parents = list(reference_bvh.data["parents"])
    fk_parents[0] = 0
    offsets = np.array(reference_bvh.data["offsets"], dtype=np.float32).copy()
    offsets[0] = 0.0
    joint_positions = translation_each_joint(
        rotations,
        root_positions.copy(),
        fk_parents,
        offsets,
    )
    return root_positions.astype(np.float32), joint_positions.astype(np.float32)


def _summarize_foot_skating(
    joint_positions, feet_idx, height_thresh=0.2, vel_thresh=0.02
):
    foot_positions = np.asarray(joint_positions[:, feet_idx, :], dtype=np.float32)
    if foot_positions.size == 0:
        return {
            "skating_ratio": 0.0,
            "avg_skating_dist": 0.0,
            "per_foot_skating": [],
        }

    ground_y = float(np.percentile(foot_positions[:, :, 1].reshape(-1), 2.0))
    foot_height = foot_positions[:, :, 1] - ground_y
    foot_vel_xz = np.linalg.norm(
        np.diff(
            foot_positions[:, :, [0, 2]],
            axis=0,
            prepend=foot_positions[0:1, :, [0, 2]],
        ),
        axis=-1,
    )
    is_contact = foot_height < float(height_thresh)
    is_skating = np.logical_and(is_contact, foot_vel_xz > float(vel_thresh))
    skating_values = foot_vel_xz[is_skating]
    return {
        "skating_ratio": float(np.mean(np.any(is_skating, axis=1))),
        "avg_skating_dist": (
            float(np.mean(skating_values)) if skating_values.size > 0 else 0.0
        ),
        "per_foot_skating": is_skating.mean(axis=0).astype(np.float32).tolist(),
    }


def _apply_contact_anchored_root_correction(
    root_positions,
    joint_positions,
    contact_binary,
    contact_probs,
    feet_idx,
    strength=1.0,
    max_step=0.05,
):
    corrected_root = np.array(root_positions, dtype=np.float32, copy=True)
    corrected_positions = np.array(joint_positions, dtype=np.float32, copy=True)
    num_frames = corrected_root.shape[0]
    root_offset_xz = np.zeros(2, dtype=np.float32)
    offset_trace = np.zeros((num_frames, 2), dtype=np.float32)
    anchors = [None for _ in feet_idx]
    previous_active = np.zeros(len(feet_idx), dtype=bool)
    strength = float(np.clip(strength, 0.0, 1.0))
    max_step = max(float(max_step), 0.0)

    for frame_index in range(num_frames):
        desired_offsets = []
        weights = []
        for local_index, foot_index in enumerate(feet_idx):
            is_active = contact_binary[frame_index, local_index] > 0.5
            foot_position_xz = joint_positions[frame_index, foot_index, [0, 2]]
            corrected_foot_xz = foot_position_xz + root_offset_xz

            if is_active and not previous_active[local_index]:
                anchors[local_index] = corrected_foot_xz.copy()
            elif not is_active:
                anchors[local_index] = None

            if is_active and anchors[local_index] is not None:
                desired_offsets.append(anchors[local_index] - foot_position_xz)
                weights.append(
                    max(float(contact_probs[frame_index, local_index]), 1e-4)
                )

            previous_active[local_index] = is_active

        if desired_offsets:
            desired_offset = np.average(
                np.stack(desired_offsets, axis=0),
                axis=0,
                weights=np.asarray(weights, dtype=np.float32),
            ).astype(np.float32)
            delta = desired_offset - root_offset_xz
            delta_norm = float(np.linalg.norm(delta))
            if max_step > 0.0 and delta_norm > max_step:
                delta = delta * (max_step / max(delta_norm, 1e-8))
            root_offset_xz = root_offset_xz + strength * delta

        corrected_root[frame_index, 0] = (
            root_positions[frame_index, 0] + root_offset_xz[0]
        )
        corrected_root[frame_index, 2] = (
            root_positions[frame_index, 2] + root_offset_xz[1]
        )
        offset_trace[frame_index] = root_offset_xz.copy()

    corrected_positions[:, :, 0] = joint_positions[:, :, 0] + offset_trace[:, None, 0]
    corrected_positions[:, :, 2] = joint_positions[:, :, 2] + offset_trace[:, None, 1]
    return corrected_root, corrected_positions, offset_trace


def _prepare_root_translation_stats(root, mean_root=None, std_root=None):
    if mean_root is None or std_root is None:
        return None, None

    if not torch.is_tensor(mean_root):
        mean_root = torch.as_tensor(mean_root, dtype=root.dtype, device=root.device)
    else:
        mean_root = mean_root.to(device=root.device, dtype=root.dtype)

    if not torch.is_tensor(std_root):
        std_root = torch.as_tensor(std_root, dtype=root.dtype, device=root.device)
    else:
        std_root = std_root.to(device=root.device, dtype=root.dtype)

    translation_mean = mean_root[..., 6:9]
    translation_std = std_root[..., 6:9].clamp_min(1e-8)
    if translation_mean.dim() == 1:
        broadcast_shape = [1] * (root.dim() - 1) + [3]
        translation_mean = translation_mean.view(*broadcast_shape)
        translation_std = translation_std.view(*broadcast_shape)
    return translation_mean, translation_std


def _denormalize_root_translation(root, mean_root=None, std_root=None):
    translation = root[..., 6:9].clone()
    translation_mean, translation_std = _prepare_root_translation_stats(
        root,
        mean_root=mean_root,
        std_root=std_root,
    )
    if translation_mean is None or translation_std is None:
        return translation
    return translation * translation_std + translation_mean


def _set_normalized_root_translation(root, translation, mean_root=None, std_root=None):
    updated_root = root.clone()
    if not torch.is_tensor(translation):
        translation = torch.as_tensor(
            translation,
            dtype=updated_root.dtype,
            device=updated_root.device,
        )
    else:
        translation = translation.to(
            device=updated_root.device, dtype=updated_root.dtype
        )

    translation_mean, translation_std = _prepare_root_translation_stats(
        updated_root,
        mean_root=mean_root,
        std_root=std_root,
    )
    if translation_mean is None or translation_std is None:
        updated_root[..., 6:9] = translation
    else:
        updated_root[..., 6:9] = (translation - translation_mean) / translation_std
    return updated_root


def apply_planted_foot_grounding(
    decoded_motion,
    mean_root,
    std_root,
    means,
    stds,
    reference_bvh,
    feet_idx,
    strength=1.0,
    frame_rate=30.0,
    min_speed_threshold=0.02,
):
    feet_idx = _unique_ordered_indices(feet_idx)
    if not feet_idx:
        normalized_root = decoded_motion[:, :9].clone().permute(0, 2, 1).contiguous()
        return decoded_motion, normalized_root, None

    decoded_motion = decoded_motion.clone()
    normalized_root = decoded_motion[:, :9].clone().permute(0, 2, 1).contiguous()

    diagnostics = {
        "feet_idx": feet_idx,
        "contact_probs": [],
        "contact_binary": [],
        "root_offset_xz": [],
        "root_translation_before": [],
        "root_translation_after": [],
        "skating_before": [],
        "skating_after": [],
    }

    for batch_index in range(decoded_motion.size(0)):
        root_positions, joint_positions = _decoded_motion_to_world_positions(
            decoded_motion[batch_index],
            means,
            stds,
            reference_bvh,
        )
        contact_signals = compute_contact_signals(
            joint_positions,
            idxes=feet_idx,
            frame_rate=frame_rate,
            min_speed_threshold=max(float(min_speed_threshold), 1e-4),
        )
        corrected_root_positions, corrected_positions, offset_trace = (
            _apply_contact_anchored_root_correction(
                root_positions,
                joint_positions,
                contact_signals["contact_binary"],
                contact_signals["contact_probs"],
                feet_idx,
                strength=strength,
            )
        )

        normalized_root[batch_index] = _set_normalized_root_translation(
            normalized_root[batch_index],
            corrected_root_positions,
            mean_root=mean_root,
            std_root=std_root,
        )

        diagnostics["contact_probs"].append(
            torch.from_numpy(contact_signals["contact_probs"])
        )
        diagnostics["contact_binary"].append(
            torch.from_numpy(contact_signals["contact_binary"])
        )
        diagnostics["root_offset_xz"].append(torch.from_numpy(offset_trace))
        diagnostics["root_translation_before"].append(torch.from_numpy(root_positions))
        diagnostics["root_translation_after"].append(
            torch.from_numpy(corrected_root_positions)
        )
        diagnostics["skating_before"].append(
            _summarize_foot_skating(joint_positions, feet_idx)
        )
        diagnostics["skating_after"].append(
            _summarize_foot_skating(corrected_positions, feet_idx)
        )

    decoded_motion[:, :9] = normalized_root.permute(0, 2, 1).contiguous()

    for key in [
        "contact_probs",
        "contact_binary",
        "root_offset_xz",
        "root_translation_before",
        "root_translation_after",
    ]:
        diagnostics[key] = torch.stack(diagnostics[key], dim=0)

    return decoded_motion, normalized_root, diagnostics


def apply_lma_override_to_tags(
    tags: dict, lma_override_array, means, stds, device: torch.device
):
    if lma_override_array is None:
        return tags

    values = _coerce_lma_array(lma_override_array)
    target_len = None
    for key in LMA_KEYS:
        existing = tags.get(key)
        if existing is not None:
            target_len = existing.shape[1]
            break

    if target_len is None:
        target_len = values.shape[0]
    values = _resample_columns(values, target_len)

    for column_index, key in enumerate(LMA_KEYS):
        channel = torch.tensor(
            values[:, column_index], dtype=torch.float32, device=device
        ).view(1, target_len, 1)
        mean = means.get(key)
        std = stds.get(key)
        if mean is not None and std is not None:
            mean = mean.to(device).reshape(1, 1, -1)
            std = std.to(device).clamp_min(1e-8).reshape(1, 1, -1)
            channel = (channel - mean) / std
        tags[key] = channel
    return tags


def resolve_generator_checkpoint(path_like: str) -> str:
    candidate = Path(path_like)
    if candidate.is_dir():
        generator_path = candidate / "generator.pt"
        if not generator_path.exists():
            raise FileNotFoundError(f"Could not find generator.pt under {candidate}")
        return str(generator_path)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Could not find VAE checkpoint at {candidate}")


def list_eval_bvhs(data_path: str):
    eval_dir = Path(data_path) / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")
    return sorted(
        [filename for filename in os.listdir(eval_dir) if filename.endswith(".bvh")]
    )


def list_split_bvhs(data_path: str, split_name: str):
    split_dir = Path(data_path) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_name} directory does not exist: {split_dir}")
    return sorted(
        [filename for filename in os.listdir(split_dir) if filename.endswith(".bvh")]
    )


def infer_annotation_split(data_path: str, explicit_split: str, bvh_path: Path):
    if explicit_split in {"train", "eval"}:
        return explicit_split
    bvh_parts = {part.lower() for part in bvh_path.parts}
    if "train" in bvh_parts:
        return "train"
    if "eval" in bvh_parts:
        return "eval"
    return "eval"


def resolve_source_split(source_split: str, bvh_path: str = None):
    if source_split in {"train", "eval"}:
        return source_split
    if bvh_path is not None:
        return infer_annotation_split("", source_split, Path(bvh_path))
    return "eval"


def build_single_motion_dataset(
    data_path: str,
    param: dict,
    device: torch.device,
    means,
    stds,
    style_lookup: dict,
    style_to_id: dict,
    split_name: str = "eval",
    clip_index: int = 0,
    bvh_path: str = None,
):
    split_name = resolve_source_split(split_name, bvh_path=bvh_path)
    if bvh_path is not None:
        bvh_file = Path(bvh_path)
        if not bvh_file.exists():
            raise FileNotFoundError(f"BVH file does not exist: {bvh_file}")
        filename = bvh_file.name
        source_dir = bvh_file.parent
        annotation_split = infer_annotation_split(data_path, split_name, bvh_file)
    else:
        split_files = list_split_bvhs(data_path, split_name)
        if not split_files:
            raise RuntimeError(
                f"No BVH files found under {Path(data_path) / split_name}"
            )
        if clip_index < 0 or clip_index >= len(split_files):
            raise IndexError(
                f"clip-index must be in [0, {len(split_files) - 1}], got {clip_index}"
            )
        filename = split_files[clip_index]
        source_dir = Path(data_path) / split_name
        annotation_split = split_name

    bvh = get_bvh_from_disk(str(source_dir), filename)
    rots, pos, parents, offsets, _, og_rots = get_info_from_bvh(
        bvh, get_missing_frames=False
    )
    pos_all_joints = bvh.compute_global_pos()
    lma_data = load_lma_annotation(Path(data_path), annotation_split, filename)
    style_label, style_id = resolve_style_for_filename(
        filename, style_lookup, style_to_id
    )

    eval_dataset = TestMotionData(param, 1.0, device)
    eval_dataset.set_means_stds(means, stds)
    eval_dataset.add_motion(
        offsets,
        pos[:, 0, :],
        rots,
        parents,
        bvh,
        filename,
        pos_all_joints,
        og_rots=og_rots,
        end_sites=bvh.data["end_sites"],
        end_sites_parents=bvh.data["end_sites_parents"],
        lma_features=lma_data,
        style_id=style_id if style_id >= 0 else None,
        style_label=style_label,
    )
    eval_dataset.normalize()
    return eval_dataset, parents, filename, annotation_split, style_label, style_id


def run_sampling(args, lma_override_array=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_target_mode = getattr(args, "latent_target", "mean")
    if latent_target_mode not in {"mean", "sample"}:
        raise ValueError(
            f"latent_target must be 'mean' or 'sample', got {latent_target_mode}"
        )
    use_full_mode_lma_prefix = bool(getattr(args, "full_mode_lma_prefix", False))
    full_mode_lma_prefix_fraction = float(
        getattr(args, "full_mode_lma_prefix_fraction", 0.5)
    )
    use_full_mode_lma_suffix = bool(getattr(args, "full_mode_lma_suffix", False))
    full_mode_lma_suffix_fraction = float(
        getattr(args, "full_mode_lma_suffix_fraction", 0.5)
    )
    if use_full_mode_lma_prefix and use_full_mode_lma_suffix:
        raise ValueError(
            "--full-mode-lma-prefix and --full-mode-lma-suffix cannot both be enabled"
        )

    print(f"Using device: {device}")
    print(f"Loading diffusion checkpoint: {args.diffusion_model_path}")
    checkpoint, param = load_diffusion_checkpoint(args.diffusion_model_path, device)
    style_vocab, style_to_id = extract_style_metadata(checkpoint)
    style_resources = load_style_resources(args.data_path)
    vae_checkpoint_path = resolve_generator_checkpoint(args.vae_model_path)
    print(f"Resolved VAE checkpoint: {vae_checkpoint_path}")

    if args.style_label is not None and args.style_id is not None:
        raise ValueError("Use only one of --style-label or --style-id")

    train_data = Train_Data(device, param)
    if args.bvh_path is not None:
        reference_bvh_path = Path(args.bvh_path)
        if not reference_bvh_path.exists():
            raise FileNotFoundError(f"BVH file does not exist: {reference_bvh_path}")
        print(f"Using direct BVH: {reference_bvh_path}")
        reference_bvh = get_bvh_from_disk(
            str(reference_bvh_path.parent), reference_bvh_path.name
        )
    else:
        effective_source_split = resolve_source_split(args.source_split)
        split_files = list_split_bvhs(args.data_path, effective_source_split)
        if args.clip_index < 0 or args.clip_index >= len(split_files):
            raise IndexError(
                f"clip-index must be in [0, {len(split_files) - 1}], got {args.clip_index}"
            )
        print(
            f"Using {effective_source_split} clip {args.clip_index}: {split_files[args.clip_index]}"
        )
        reference_dir = Path(args.data_path) / effective_source_split
        reference_bvh = get_bvh_from_disk(
            str(reference_dir), split_files[args.clip_index]
        )

    _, _, reference_parents, _, _, _ = get_info_from_bvh(
        reference_bvh, get_missing_frames=False
    )
    generator_model = Generator_Model(
        device, param, reference_parents, train_data, is_vae=True, is_vq_vae=False
    ).to(device)
    print("Loading pretrained VAE weights and normalization stats")
    means, stds = load_model(generator_model, vae_checkpoint_path, train_data, device)
    freeze_module(generator_model.static_encoder)
    freeze_module(generator_model.autoencoder)

    print("Preparing selected conditioning clip only")
    source_split_for_styles = resolve_source_split(args.source_split, args.bvh_path)
    source_style_lookup = style_resources.get(
        f"{source_split_for_styles}_lookup", style_resources.get("eval_lookup", {})
    )
    (
        eval_dataset,
        _,
        source_filename,
        annotation_split,
        source_style_label,
        source_style_id,
    ) = build_single_motion_dataset(
        args.data_path,
        param,
        device,
        means,
        stds,
        style_lookup=source_style_lookup,
        style_to_id=style_to_id,
        split_name=args.source_split,
        clip_index=args.clip_index,
        bvh_path=args.bvh_path,
    )
    print(
        f"Condition source ready: {source_filename} | annotation split: {annotation_split}"
    )
    if source_style_label is not None:
        print(f"Source clip style: {source_style_label} (id={source_style_id})")

    prior = ContinuousLatentDiffusionPrior(
        latent_dim=int(param.get("vae_latent_dim", 504)),
        hidden_dim=int(param.get("diffusion_hidden_dim", 512)),
        num_layers=int(param.get("diffusion_num_layers", 8)),
        num_heads=int(param.get("diffusion_num_heads", 8)),
        dropout=float(param.get("diffusion_dropout", 0.1)),
        lma_dim=len(LMA_KEYS),
        traj_dim=int(param.get("rough_root_traj_dim", 9)),
        num_train_timesteps=int(param.get("diffusion_num_train_timesteps", 1000)),
        use_per_lma_channel_encoders=bool(
            param.get("diffusion_use_per_lma_channel_encoders", False)
        ),
        per_lma_channel_dropout=float(
            param.get("diffusion_per_lma_channel_dropout", 0.1)
        ),
        latent_velocity_weight=float(
            param.get("diffusion_latent_velocity_weight", 0.0)
        ),
        root_traj_loss_weight=float(param.get("diffusion_root_traj_loss_weight", 0.0)),
        root_traj_velocity_weight=float(
            param.get("diffusion_root_traj_velocity_weight", 0.0)
        ),
        root_rot_loss_scale=float(param.get("diffusion_root_rot_loss_scale", 1.0)),
        root_pos_loss_scale=float(param.get("diffusion_root_pos_loss_scale", 3.0)),
        p_drop_lma=float(param.get("diffusion_condition_drop_lma", 0.15)),
        p_drop_traj=float(param.get("diffusion_condition_drop_traj", 0.45)),
        p_drop_both=float(param.get("diffusion_condition_drop_both", 0.05)),
        lma_condition_scale=float(param.get("diffusion_lma_condition_scale", 1.15)),
        traj_condition_scale=float(param.get("diffusion_traj_condition_scale", 1.0)),
        num_styles=len(style_vocab),
        style_drop=float(param.get("diffusion_style_drop", 0.0)),
        style_condition_scale=float(param.get("diffusion_style_condition_scale", 1.0)),
    ).to(device)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(
            "Diffusion checkpoint did not contain ema_state_dict or model_state_dict"
        )
    load_result = prior.load_state_dict(state_dict, strict=False)
    prior.eval()
    has_predicted_root_head = not any(
        key.startswith("root_traj_head.") for key in load_result.missing_keys
    )
    if load_result.missing_keys:
        print(f"Loaded checkpoint with missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Loaded checkpoint with unexpected keys: {load_result.unexpected_keys}")
    if not has_predicted_root_head:
        print(
            "Checkpoint does not include the new root prediction head; decode-time root override will be disabled."
        )
    style_condition_enabled = prior.style_embedding is not None
    if style_condition_enabled:
        print(f"Style conditioning enabled with {len(style_vocab)} labels")
    elif args.style_label is not None or args.style_id is not None:
        print(
            "Checkpoint does not include style conditioning; requested style override will be ignored."
        )
    print("Diffusion prior loaded")

    eval_motion = build_eval_sample(eval_dataset.get_item(0), device)
    prepare_train_batch(train_data, eval_motion, eval_motion)
    if lma_override_array is None and args.lma_override_path is not None:
        print(f"Loading external LMA override: {args.lma_override_path}")
        lma_override_array = load_lma_override_array(args.lma_override_path)
    if lma_override_array is not None:
        print("Applying external LMA override to conditioning tags")
        apply_lma_override_to_tags(
            train_data.tags, lma_override_array, means, stds, device
        )

    selected_style_label = source_style_label
    selected_style_id = source_style_id if source_style_id >= 0 else None
    if style_condition_enabled:
        if args.style_id is not None:
            if args.style_id < 0 or args.style_id >= len(style_vocab):
                raise ValueError(
                    f"style-id must be in [0, {len(style_vocab) - 1}], got {args.style_id}"
                )
            selected_style_id = int(args.style_id)
            selected_style_label = style_vocab[selected_style_id]
        elif args.style_label is not None:
            if args.style_label not in style_to_id:
                raise ValueError(
                    f"Unknown style label '{args.style_label}'. Known labels: {sorted(style_to_id)}"
                )
            selected_style_label = args.style_label
            selected_style_id = int(style_to_id[args.style_label])

        if selected_style_id is not None:
            train_data.tags["style_id"] = torch.tensor(
                [selected_style_id], dtype=torch.long, device=device
            )
            print(
                f"Using style condition: {selected_style_label} (id={selected_style_id})"
            )
        else:
            print(
                "No style label found for the source clip; sampling without style conditioning"
            )

    tags = train_data.tags
    lma_seq, rough_root_traj, style_id = (
        generator_model.autoencoder.build_diffusion_condition_tensors(tags)
    )

    default_frame_len = int(eval_motion["dqs"].size(1))
    target_frame_len = (
        int(args.frame_len) if args.frame_len is not None else default_frame_len
    )
    latent_len = max(int(math.ceil(target_frame_len / 8.0)), 1)
    print(
        f"Sampling mode={args.mode} | target_frames={target_frame_len} | latent_steps={latent_len}"
    )
    if use_full_mode_lma_prefix and args.mode == "full":
        print(
            "Full-mode LMA prefix enabled | "
            f"prefix_fraction={full_mode_lma_prefix_fraction:.3f}"
        )
    if use_full_mode_lma_suffix and args.mode == "full":
        print(
            "Full-mode LMA suffix enabled | "
            f"suffix_fraction={full_mode_lma_suffix_fraction:.3f}"
        )

    source_latent = None
    source_latent_noise_timestep = None
    source_latent_effective_strength = None
    source_latent_timestep_overridden = False
    source_latent_edit_enabled = source_latent_edit_requested(args)
    if source_latent_edit_enabled:
        source_latent = generator_model.autoencoder.encode_latent_sequence(
            train_data.sparse_motion,
            use_mean=latent_target_mode == "mean",
        ).detach()
        (
            source_latent_noise_timestep,
            source_latent_effective_strength,
            source_latent_timestep_overridden,
        ) = resolve_source_latent_edit_schedule(
            args,
            num_train_timesteps=prior.num_train_timesteps,
        )
        if source_latent_timestep_overridden:
            print(
                "Source-latent editing enabled | "
                f"latent_target={latent_target_mode} | "
                f"noise_timestep={source_latent_noise_timestep}/{prior.num_train_timesteps - 1}"
            )
        else:
            print(
                "Source-latent editing enabled | "
                f"latent_target={latent_target_mode} | "
                f"edit_strength={source_latent_effective_strength:.3f} | "
                f"noise_timestep={source_latent_noise_timestep}/{prior.num_train_timesteps - 1}"
            )

    sampled_latent, predicted_root_traj = prior.sample_long(
        total_seq_len=latent_len,
        batch_size=1,
        lma_seq=lma_seq,
        traj_seq=rough_root_traj,
        style_id=style_id,
        source_latent=source_latent,
        source_noise_timestep=source_latent_noise_timestep,
        mode=args.mode,
        cfg_scale=float(
            args.cfg_scale
            if args.cfg_scale is not None
            else param.get("diffusion_cfg_scale", 2.5)
        ),
        lma_cfg_scale=args.lma_cfg_scale,
        traj_cfg_scale=args.traj_cfg_scale,
        style_cfg_scale=args.style_cfg_scale,
        use_full_mode_lma_prefix=use_full_mode_lma_prefix,
        full_mode_lma_prefix_fraction=full_mode_lma_prefix_fraction,
        use_full_mode_lma_suffix=use_full_mode_lma_suffix,
        full_mode_lma_suffix_fraction=full_mode_lma_suffix_fraction,
        num_steps=int(
            args.sample_steps
            if args.sample_steps is not None
            else param.get("diffusion_sample_steps", 50)
        ),
        chunk_len=int(
            args.chunk_len
            if args.chunk_len is not None
            else param.get("diffusion_chunk_len", 128)
        ),
        overlap_len=int(
            args.overlap_len
            if args.overlap_len is not None
            else param.get("diffusion_overlap_len", 32)
        ),
        halo_len=int(
            args.halo_len
            if args.halo_len is not None
            else param.get("diffusion_halo_len", 8)
        ),
        eta=float(
            args.eta if args.eta is not None else param.get("diffusion_eta", 0.0)
        ),
        temperature=float(
            args.temperature
            if args.temperature is not None
            else param.get("diffusion_temperature", 1.0)
        ),
        device=device,
        return_aux=True,
    )

    root_override = None
    if (
        not args.disable_predicted_root_override
        and has_predicted_root_head
        and args.mode in {"lma_only", "uncond"}
    ):
        root_override = predicted_root_traj
        print("Applying predicted rough root trajectory as a decode-time root override")

    print("Decoding sampled latent with the pretrained VAE decoder")
    ae_offsets = generator_model.static_encoder(train_data.offsets)
    decoded_motion, normalized_root = (
        generator_model.autoencoder.decode_latent_sequence(
            sampled_latent,
            ae_offsets,
            train_data.mean_dqs,
            train_data.std_dqs,
            train_data.denorm_offsets,
            mean_root=train_data.mean_root,
            std_root=train_data.std_root,
            tags=tags,
            root_override=root_override,
            root_override_blend=args.root_override_blend,
        )
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.diffusion_model_path).resolve().parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.bvh_path is not None:
        stem_prefix = Path(args.bvh_path).stem
        stem = f"sample_{stem_prefix}_{args.mode}"
    else:
        stem = f"sample_{args.source_split}{args.clip_index:03d}_{args.mode}"
    if source_latent_edit_enabled:
        stem = f"{stem}_sourceedit_t{source_latent_noise_timestep:04d}"
    tensor_path = output_dir / f"{stem}.pt"
    print(f"Saving tensor outputs to {tensor_path}")
    torch.save(
        {
            "sampled_latent": sampled_latent.detach().cpu(),
            "source_latent": (
                None if source_latent is None else source_latent.detach().cpu()
            ),
            "decoded_motion": decoded_motion.detach().cpu(),
            "normalized_root": normalized_root.detach().cpu(),
            "predicted_rough_root_traj": predicted_root_traj.detach().cpu(),
            "predicted_root_rotation": split_root_diagnostics(predicted_root_traj)[
                "root_rotation"
            ]
            .detach()
            .cpu(),
            "predicted_root_translation": split_root_diagnostics(predicted_root_traj)[
                "root_translation"
            ]
            .detach()
            .cpu(),
            "predicted_root_translation_velocity": (
                None
                if split_root_diagnostics(predicted_root_traj)[
                    "root_translation_velocity"
                ]
                is None
                else split_root_diagnostics(predicted_root_traj)[
                    "root_translation_velocity"
                ]
                .detach()
                .cpu()
            ),
            "mode": args.mode,
            "clip_index": args.clip_index,
            "source_split": args.source_split,
            "source_filename": source_filename,
            "target_frame_len": target_frame_len,
            "lma_seq": None if lma_seq is None else lma_seq.detach().cpu(),
            "rough_root_traj": (
                None if rough_root_traj is None else rough_root_traj.detach().cpu()
            ),
            "source_root_rotation": (
                None
                if rough_root_traj is None
                else split_root_diagnostics(rough_root_traj)["root_rotation"]
                .detach()
                .cpu()
            ),
            "source_root_translation": (
                None
                if rough_root_traj is None
                else split_root_diagnostics(rough_root_traj)["root_translation"]
                .detach()
                .cpu()
            ),
            "source_root_translation_velocity": (
                None
                if rough_root_traj is None
                or split_root_diagnostics(rough_root_traj)["root_translation_velocity"]
                is None
                else split_root_diagnostics(rough_root_traj)[
                    "root_translation_velocity"
                ]
                .detach()
                .cpu()
            ),
            "predicted_root_rotation_mse_vs_source": (
                None
                if rough_root_traj is None
                else float(
                    torch.mean(
                        (
                            split_root_diagnostics(predicted_root_traj)["root_rotation"]
                            - split_root_diagnostics(rough_root_traj)["root_rotation"]
                        )
                        ** 2
                    ).item()
                )
            ),
            "predicted_root_translation_mse_vs_source": (
                None
                if rough_root_traj is None
                else float(
                    torch.mean(
                        (
                            split_root_diagnostics(predicted_root_traj)[
                                "root_translation"
                            ]
                            - split_root_diagnostics(rough_root_traj)[
                                "root_translation"
                            ]
                        )
                        ** 2
                    ).item()
                )
            ),
            "predicted_root_translation_velocity_mse_vs_source": (
                None
                if rough_root_traj is None
                or split_root_diagnostics(predicted_root_traj)[
                    "root_translation_velocity"
                ]
                is None
                or split_root_diagnostics(rough_root_traj)["root_translation_velocity"]
                is None
                else float(
                    torch.mean(
                        (
                            split_root_diagnostics(predicted_root_traj)[
                                "root_translation_velocity"
                            ]
                            - split_root_diagnostics(rough_root_traj)[
                                "root_translation_velocity"
                            ]
                        )
                        ** 2
                    ).item()
                )
            ),
            "lma_override_applied": lma_override_array is not None,
            "predicted_root_override_applied": root_override is not None,
            "source_latent_edit_applied": source_latent_edit_enabled,
            "source_latent_edit_strength": source_latent_effective_strength,
            "source_latent_noise_timestep": source_latent_noise_timestep,
            "source_latent_timestep_overridden": source_latent_timestep_overridden,
            "source_latent_target": latent_target_mode,
            "full_mode_lma_prefix_applied": use_full_mode_lma_prefix,
            "full_mode_lma_prefix_fraction": full_mode_lma_prefix_fraction,
            "full_mode_lma_suffix_applied": use_full_mode_lma_suffix,
            "full_mode_lma_suffix_fraction": full_mode_lma_suffix_fraction,
            "foot_grounding_applied": False,
            "lma_cfg_scale": args.lma_cfg_scale,
            "traj_cfg_scale": args.traj_cfg_scale,
            "style_cfg_scale": args.style_cfg_scale,
            "style_id": None if style_id is None else style_id.detach().cpu(),
            "source_style_label": source_style_label,
            "source_style_id": source_style_id,
            "selected_style_label": selected_style_label,
            "selected_style_id": selected_style_id,
            "style_vocab": style_vocab,
        },
        tensor_path,
    )

    bvh_path = None
    if args.save_bvh:
        bvh, _ = eval_dataset.get_bvh(0)
        save_dir = Path("data")
        save_dir.mkdir(parents=True, exist_ok=True)
        print("Exporting BVH")
        result_to_bvh(
            decoded_motion,
            means,
            stds,
            bvh,
            f"{stem}.bvh",
            save=True,
        )
        bvh_path = save_dir / f"eval_{stem}.bvh"

    _, sampled_joint_positions = _decoded_motion_to_world_positions(
        decoded_motion[0],
        means,
        stds,
        reference_bvh,
    )
    source_joint_positions = np.asarray(
        reference_bvh.compute_global_pos(),
        dtype=np.float32,
    )

    print(f"Saved sample tensors to {tensor_path}")
    if bvh_path is not None:
        print(f"Saved BVH to {bvh_path}")
    return {
        "tensor_path": tensor_path,
        "bvh_path": bvh_path,
        "source_filename": source_filename,
        "source_joint_positions": source_joint_positions,
        "sampled_joint_positions": sampled_joint_positions,
        "parents": [int(parent) for parent in reference_parents],
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    lma_override_array = None
    if args.lma_override_path is not None:
        lma_override_array = load_lma_override_array(args.lma_override_path)
    run_sampling(args, lma_override_array=lma_override_array)


if __name__ == "__main__":
    main()
