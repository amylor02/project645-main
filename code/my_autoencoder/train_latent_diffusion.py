import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pymotion.rotations.ortho6d_torch as ortho6d
import torch
import torch.nn.functional as F
from pymotion.ops.skeleton import compute_global_pos_torch
from pymotion.rotations.dual_quat_torch import to_rotation_translation
from torch.utils.data.dataloader import DataLoader

from generator_architecture import Generator_Model
from motion_data import (
    LMA_KEYS,
    TestMotionData,
    TrainMotionData,
    integrate_root_translation_torch,
    split_motion_joints,
)
from train_data import Train_Data
from train_vq_vae import (
    get_bvh_from_disk,
    get_info_from_bvh,
    load_model,
    param as base_param,
)
from latent_diffusion_prior import ContinuousLatentDiffusionPrior


DEFAULT_PARAM = copy.deepcopy(base_param)
DEFAULT_PARAM.update(
    {
        "random_yaw_augmentation": False,
        "random_yaw_aug_max_degrees": float(
            base_param.get("random_yaw_aug_max_degrees", 180.0)
        ),
        "window_size": 1024,
        "window_step": 256,
        "batch_size": 8,
        "epochs": 400,
        "rough_root_avg_window": 3,
        "diffusion_hidden_dim": 512,
        "diffusion_num_layers": 8,
        "diffusion_num_heads": 8,
        "diffusion_dropout": 0.1,
        "diffusion_num_train_timesteps": 1000,
        "diffusion_latent_velocity_weight": 0.01,
        "diffusion_root_traj_loss_weight": 0.6,
        "diffusion_root_traj_velocity_weight": 0.1,
        "diffusion_decoded_foot_position_weight": 0,  # 0.10,
        "diffusion_decoded_foot_velocity_weight": 0,  # 0.1,
        "diffusion_decoded_foot_acceleration_weight": 0.00,
        "diffusion_decoded_foot_every_n_batches": 4,
        "diffusion_decoded_foot_start_epoch": 500,  # 50,
        "diffusion_root_rot_loss_scale": 0,  # 10.00,
        "diffusion_root_pos_loss_scale": 0,  # 0.75,
        "diffusion_condition_drop_lma": 0.1,
        "diffusion_condition_drop_traj": 0.7,
        "diffusion_condition_drop_both": 0.05,
        "diffusion_final_condition_drop_lma": 0.05,
        "diffusion_final_condition_drop_traj": 0.1,
        "diffusion_final_condition_drop_both": 0.0,
        "diffusion_use_explicit_condition_mode_probs": False,
        "diffusion_condition_mode_prob_full": 0.55,
        "diffusion_condition_mode_prob_lma_only": 0.20,
        "diffusion_condition_mode_prob_traj_only": 0.20,
        "diffusion_condition_mode_prob_uncond": 0.05,
        "diffusion_use_per_lma_channel_encoders": False,
        "diffusion_per_lma_channel_dropout": 0.1,
        "diffusion_lma_condition_scale": 1.5,
        "diffusion_traj_condition_scale": 1.0,
        "diffusion_num_styles": 0,
        "diffusion_style_drop": 0.1,
        "diffusion_final_style_drop": 0.025,
        "diffusion_style_condition_scale": 1.0,
        "diffusion_condition_focus_start": 0.45,
        "diffusion_cfg_scale": 2.5,
        "diffusion_sample_steps": 50,
        "diffusion_chunk_len": 128,
        "diffusion_overlap_len": 32,
        "diffusion_halo_len": 8,
        "diffusion_eta": 0.0,
        "diffusion_temperature": 1.0,
    }
)


def _find_style_file(styles_root: Path, split_name: str):
    for suffix in (".json", ".txt"):
        candidate = styles_root / f"{split_name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_style_lookup_file(style_path: Path):
    if style_path is None or not style_path.exists():
        return {}

    text = style_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    raw_data = None
    try:
        raw_data = json.loads(text)
    except json.JSONDecodeError:
        raw_data = None

    if raw_data is None:
        mapping = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            filename, label = line.split(":", 1)
            mapping[Path(filename.strip()).name] = label.strip()
        return mapping

    if not isinstance(raw_data, dict):
        raise ValueError(
            f"Expected JSON object in {style_path}, got {type(raw_data)!r}"
        )

    return {
        Path(str(filename)).name: str(label).strip()
        for filename, label in raw_data.items()
        if str(label).strip()
    }


def load_style_resources(data_path: str):
    styles_root = Path(data_path) / "styles"
    train_lookup = _load_style_lookup_file(_find_style_file(styles_root, "train"))
    eval_lookup = _load_style_lookup_file(_find_style_file(styles_root, "eval"))
    style_vocab = sorted({label for label in train_lookup.values() if label})
    style_to_id = {label: index for index, label in enumerate(style_vocab)}
    return {
        "train_lookup": train_lookup,
        "eval_lookup": eval_lookup,
        "style_vocab": style_vocab,
        "style_to_id": style_to_id,
    }


def resolve_style_for_filename(filename: str, style_lookup: dict, style_to_id: dict):
    filename = Path(filename).name
    style_label = style_lookup.get(filename)
    if style_label is None:
        return None, -1
    return style_label, int(style_to_id.get(style_label, -1))


def seed_all(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_condition_mode_probs(args) -> Dict[str, float]:
    return {
        "full": float(args.condition_mode_prob_full),
        "lma_only": float(args.condition_mode_prob_lma_only),
        "traj_only": float(args.condition_mode_prob_traj_only),
        "uncond": float(args.condition_mode_prob_uncond),
    }


def load_lma_annotation(split_root: Path, split_name: str, filename: str):
    try:
        split_path = (
            split_root / "annotations" / split_name / f"{Path(filename).stem}.csv"
        )
        if not split_path.exists():
            split_path = split_root / "annotations" / f"{Path(filename).stem}.csv"
        if not split_path.exists():
            return None
        dataframe = pd.read_csv(split_path)
        numeric = dataframe.select_dtypes(include=["number"]).to_numpy()
        if numeric.size == 0:
            return None
        if numeric.shape[1] < len(LMA_KEYS):
            return numeric
        return numeric[:, : len(LMA_KEYS)]
    except Exception:
        return None


def build_datasets(data_path: str, param: dict, device: torch.device):
    train_root = Path(data_path)
    train_dir = train_root / "train"
    eval_dir = train_root / "eval"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")

    train_dataset = TrainMotionData(param, 1.0, device)
    eval_dataset = TestMotionData(param, 1.0, device)
    style_metadata = load_style_resources(data_path)
    reference_parents = None

    for filename in sorted(os.listdir(train_dir)):
        if not filename.endswith(".bvh"):
            continue
        bvh_from_disk = get_bvh_from_disk(str(train_dir), filename)
        rots, pos, parents, offsets, _, og_rots = get_info_from_bvh(
            bvh_from_disk,
            get_missing_frames=False,
        )
        if reference_parents is None:
            reference_parents = parents.copy()
        assert reference_parents == parents
        pos_all_joints = bvh_from_disk.compute_global_pos()
        lma_data = load_lma_annotation(train_root, "train", filename)
        style_label, style_id = resolve_style_for_filename(
            filename,
            style_metadata["train_lookup"],
            style_metadata["style_to_id"],
        )
        train_dataset.add_motion(
            offsets,
            pos[:, 0, :],
            rots,
            parents,
            pos_all_joints,
            og_rots=og_rots,
            end_sites=bvh_from_disk.data["end_sites"],
            end_sites_parents=bvh_from_disk.data["end_sites_parents"],
            lma_features=lma_data,
            style_id=style_id if style_id >= 0 else None,
            style_label=style_label,
        )

    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)

    for filename in sorted(os.listdir(eval_dir)):
        if not filename.endswith(".bvh"):
            continue
        bvh = get_bvh_from_disk(str(eval_dir), filename)
        rots, pos, parents, offsets, _, og_rots = get_info_from_bvh(
            bvh, get_missing_frames=False
        )
        assert reference_parents == parents
        pos_all_joints = bvh.compute_global_pos()
        lma_data = load_lma_annotation(train_root, "eval", filename)
        style_label, style_id = resolve_style_for_filename(
            filename,
            style_metadata["eval_lookup"],
            style_metadata["style_to_id"],
        )
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
    return train_dataset, eval_dataset, reference_parents, style_metadata


def freeze_module(module: torch.nn.Module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def prepare_train_batch(
    train_data: Train_Data,
    denorm_motion,
    norm_motion,
    apply_yaw_augmentation: bool = False,
):
    denorm_offsets = denorm_motion.get("denorm_offsets", denorm_motion["offsets"])
    train_data.set_offsets(
        norm_motion["offsets"],
        denorm_offsets,
    )
    train_data.set_motions(
        norm_motion["dqs"],
        norm_motion["displacement"],
    )
    train_data.set_tags(norm_motion["tags"])
    train_data.set_rots(norm_motion["rots"])
    train_data.set_global_pos(denorm_motion["global_pos"])
    if "foot_positions" in denorm_motion:
        foot_positions = denorm_motion["foot_positions"]
        if not torch.is_tensor(foot_positions):
            foot_positions = torch.as_tensor(
                foot_positions,
                dtype=torch.float32,
                device=train_data.device,
            )
        else:
            foot_positions = foot_positions.to(
                device=train_data.device,
                dtype=torch.float32,
            )
        train_data.set_foot_positions(foot_positions)
    else:
        train_data.foot_positions = None
    if "end_sites" in denorm_motion:
        train_data.set_end_sites(
            denorm_motion["end_sites"], denorm_motion["end_sites_parents"]
        )
    if apply_yaw_augmentation:
        train_data.apply_random_yaw_augmentation()


def update_ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float):
    with torch.no_grad():
        source_state = model.state_dict()
        target_state = ema_model.state_dict()
        for key, value in source_state.items():
            target_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def create_model_dir(name: str, data_path: str) -> Path:
    model_dir = Path("models") / f"diffusion_{name}_{Path(data_path).name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_checkpoint(
    model_dir: Path,
    prior: ContinuousLatentDiffusionPrior,
    ema_prior: ContinuousLatentDiffusionPrior,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_eval_loss: float,
    train_dataset: TrainMotionData,
    style_metadata: dict,
    param: dict,
    args,
    is_best: bool = False,
):
    checkpoint_path = model_dir / "latent_diffusion_prior.pt"
    data_path = model_dir / "latent_diffusion_data.pt"
    if is_best:
        checkpoint_path = model_dir / "latent_diffusion_prior_best.pt"
        data_path = model_dir / "latent_diffusion_data_best.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": prior.state_dict(),
            "ema_state_dict": ema_prior.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eval_loss": best_eval_loss,
            "param": param,
            "args": vars(args),
            "style_vocab": style_metadata.get("style_vocab", []),
            "style_to_id": style_metadata.get("style_to_id", {}),
        },
        checkpoint_path,
    )
    torch.save(
        {
            "means": train_dataset.means,
            "stds": train_dataset.stds,
            "param": param,
            "style_vocab": style_metadata.get("style_vocab", []),
            "style_to_id": style_metadata.get("style_to_id", {}),
        },
        data_path,
    )


def load_training_checkpoint(
    checkpoint_path: str,
    prior: ContinuousLatentDiffusionPrior,
    ema_prior: ContinuousLatentDiffusionPrior,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reset_optimizer: bool = False,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    prior.load_state_dict(checkpoint["model_state_dict"])

    ema_state_dict = checkpoint.get("ema_state_dict")
    if ema_state_dict is not None:
        ema_prior.load_state_dict(ema_state_dict)
    else:
        ema_prior.load_state_dict(prior.state_dict())

    if not reset_optimizer and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_eval_score = float(checkpoint.get("best_eval_loss", float("inf")))
    return checkpoint, start_epoch, best_eval_score


def build_eval_sample(eval_motion, device: torch.device):
    batch = {}
    for key, value in eval_motion.items():
        if isinstance(value, dict):
            batch[key] = {
                inner_key: (
                    inner_value.unsqueeze(0).to(device)
                    if torch.is_tensor(inner_value)
                    else inner_value
                )
                for inner_key, inner_value in value.items()
            }
            continue
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value
    return batch


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


def get_contact_condition_tensors(tags: dict):
    if tags is None:
        return None, None
    return tags.get("foot_contact_latent"), tags.get("support_contact_latent")


def _unique_feet_indices(feet_idx):
    return list(dict.fromkeys(int(index) for index in feet_idx if int(index) >= 0))


def _align_temporal_tensor(sequence: torch.Tensor, target_len: int, mode: str):
    if sequence is None:
        return None
    if sequence.dim() == 2:
        sequence = sequence.unsqueeze(0)
    if sequence.dim() != 3:
        raise ValueError(f"Expected [B, T, C] sequence, got {sequence.shape}")
    if sequence.size(1) == target_len:
        return sequence

    interpolate_kwargs = {}
    if mode in {"linear", "bilinear", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    return F.interpolate(
        sequence.transpose(1, 2),
        size=target_len,
        mode=mode,
        **interpolate_kwargs,
    ).transpose(1, 2)


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if weights is None:
        return F.mse_loss(prediction, target)

    weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    while weights.dim() < prediction.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(prediction)
    denom = weights.sum().clamp_min(1e-6)
    return ((prediction - target) ** 2 * weights).sum() / denom


def _normalize_fk_end_sites(end_sites, batch_size: int, device: torch.device):
    if end_sites is None:
        return None

    if not torch.is_tensor(end_sites):
        end_sites = torch.as_tensor(end_sites, dtype=torch.float32, device=device)
    else:
        end_sites = end_sites.to(device=device, dtype=torch.float32)

    # Some dataset paths can hand us time-expanded end-site offsets even though they
    # are static skeleton data. Reduce those to the per-sample offsets FK expects.
    while end_sites.dim() > 3:
        if end_sites.size(0) == batch_size:
            end_sites = end_sites[:, 0]
        else:
            end_sites = end_sites[0]

    if end_sites.dim() == 3 and end_sites.size(0) not in {1, batch_size}:
        end_sites = end_sites[0]

    return end_sites


def _normalize_fk_end_site_parents(
    end_sites_parents,
    batch_size: int,
    device: torch.device,
):
    if end_sites_parents is None:
        return None

    if not torch.is_tensor(end_sites_parents):
        end_sites_parents = torch.as_tensor(
            end_sites_parents,
            dtype=torch.long,
            device=device,
        )
    else:
        end_sites_parents = end_sites_parents.to(device=device, dtype=torch.long)

    while end_sites_parents.dim() > 2:
        if end_sites_parents.size(0) == batch_size:
            end_sites_parents = end_sites_parents[:, 0]
        else:
            end_sites_parents = end_sites_parents[0]

    if end_sites_parents.dim() == 2 and end_sites_parents.size(0) not in {
        1,
        batch_size,
    }:
        end_sites_parents = end_sites_parents[:1]

    if end_sites_parents.dim() == 1:
        end_sites_parents = end_sites_parents.unsqueeze(0)

    return end_sites_parents


def decode_motion_to_global_positions(
    motion: torch.Tensor,
    train_data: Train_Data,
    parents,
) -> torch.Tensor:
    batch_size, _, frame_count = motion.shape
    safe_std = train_data.std_dqs.clamp_min(1e-8).view(1, 1, -1)
    mean_dqs = train_data.mean_dqs.view(1, 1, -1)
    denormalized = motion.permute(0, 2, 1).contiguous() * safe_std + mean_dqs
    denormalized = denormalized.view(batch_size, frame_count, -1, 9)
    skeletal_motion, _ = split_motion_joints(
        denormalized,
        synthetic_joint_count=int(
            train_data.param.get("synthetic_contact_joint_count", 0)
        ),
    )
    dual_quats = ortho6d.to_dual_quat(skeletal_motion)
    rotations, _ = to_rotation_translation(dual_quats)
    root_positions = integrate_root_translation_torch(
        skeletal_motion[:, :, 0, :],
        train_data.global_pos,
    )

    fk_kwargs = {}
    if hasattr(train_data, "end_sites") and hasattr(train_data, "end_sites_parents"):
        fk_kwargs["end_sites"] = _normalize_fk_end_sites(
            train_data.end_sites,
            batch_size=batch_size,
            device=motion.device,
        )
        fk_kwargs["end_sites_parents"] = _normalize_fk_end_site_parents(
            train_data.end_sites_parents,
            batch_size=batch_size,
            device=motion.device,
        )

    joint_positions, _ = compute_global_pos_torch(
        rotations,
        root_positions,
        train_data.denorm_offsets,
        parents,
        **fk_kwargs,
    )
    return joint_positions


def compute_decoded_foot_kinematics_losses(
    generator_model: Generator_Model,
    train_data: Train_Data,
    latent_prediction: torch.Tensor,
    ae_offsets,
    target_motion: torch.Tensor = None,
    target_positions: torch.Tensor = None,
    tags: dict = None,
) -> Dict[str, torch.Tensor]:
    target_motion = train_data.motion if target_motion is None else target_motion
    tags = train_data.tags if tags is None else tags
    device = latent_prediction.device
    zero = latent_prediction.new_zeros(())
    feet_idx = _unique_feet_indices(generator_model.param.get("feet_idxs", []))
    foot_contact_binary = None if tags is None else tags.get("foot_contact_binary")

    diagnostics = {
        "decoded_foot_position_loss": zero,
        "decoded_foot_velocity_loss": zero,
        "decoded_foot_acceleration_loss": zero,
        "decoded_planted_foot_speed": zero,
    }
    if not feet_idx or foot_contact_binary is None:
        return diagnostics

    decoded_motion, _ = generator_model.autoencoder.decode_latent_sequence(
        latent_prediction,
        ae_offsets,
        train_data.mean_dqs,
        train_data.std_dqs,
        train_data.denorm_offsets,
        mean_root=train_data.mean_root,
        std_root=train_data.std_root,
        tags=tags,
    )
    predicted_positions = decode_motion_to_global_positions(
        decoded_motion,
        train_data,
        generator_model.parents,
    )
    predicted_feet = predicted_positions[:, :, feet_idx, :]

    if target_positions is None:
        target_positions = getattr(train_data, "foot_positions", None)

    if target_positions is None:
        with torch.no_grad():
            target_positions = decode_motion_to_global_positions(
                target_motion.detach(),
                train_data,
                generator_model.parents,
            )
        target_feet = target_positions[:, :, feet_idx, :]
    else:
        if not torch.is_tensor(target_positions):
            target_positions = torch.as_tensor(
                target_positions,
                dtype=predicted_feet.dtype,
                device=device,
            )
        else:
            target_positions = target_positions.to(
                device=device,
                dtype=predicted_feet.dtype,
            )

        if target_positions.dim() == 3:
            target_positions = target_positions.unsqueeze(0)

        if target_positions.size(2) == len(feet_idx):
            target_feet = target_positions
        else:
            target_feet = target_positions[:, :, feet_idx, :]

    foot_contact_binary = _align_temporal_tensor(
        foot_contact_binary.to(device=device, dtype=latent_prediction.dtype),
        predicted_positions.size(1),
        mode="nearest",
    )
    foot_contact_binary = foot_contact_binary[..., : len(feet_idx)].clamp(0.0, 1.0)

    diagnostics["decoded_foot_position_loss"] = _weighted_mse(
        predicted_feet,
        target_feet,
        foot_contact_binary,
    )

    if predicted_feet.size(1) > 1:
        predicted_velocity = predicted_feet[:, 1:] - predicted_feet[:, :-1]
        target_velocity = target_feet[:, 1:] - target_feet[:, :-1]
        contact_velocity = 0.5 * (
            foot_contact_binary[:, 1:, :] + foot_contact_binary[:, :-1, :]
        )
        diagnostics["decoded_foot_velocity_loss"] = _weighted_mse(
            predicted_velocity,
            target_velocity,
            contact_velocity,
        )

        planted_speed = torch.norm(predicted_velocity, dim=-1)
        diagnostics["decoded_planted_foot_speed"] = (
            planted_speed * contact_velocity
        ).sum() / contact_velocity.sum().clamp_min(1e-6)

        if predicted_feet.size(1) > 2:
            predicted_acceleration = (
                predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
            )
            target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
            contact_acceleration = 0.5 * (
                contact_velocity[:, 1:, :] + contact_velocity[:, :-1, :]
            )
            diagnostics["decoded_foot_acceleration_loss"] = _weighted_mse(
                predicted_acceleration,
                target_acceleration,
                contact_acceleration,
            )

    return diagnostics


def apply_decoded_foot_kinematics_losses(
    loss_dict: dict,
    foot_loss_dict: dict,
    param: dict,
) -> dict:
    position_weight = float(param.get("diffusion_decoded_foot_position_weight", 0.0))
    velocity_weight = float(param.get("diffusion_decoded_foot_velocity_weight", 0.0))
    acceleration_weight = float(
        param.get("diffusion_decoded_foot_acceleration_weight", 0.0)
    )

    loss_dict = dict(loss_dict)
    loss_dict.update(foot_loss_dict)
    loss_dict["loss"] = (
        loss_dict["loss"]
        + position_weight * foot_loss_dict["decoded_foot_position_loss"]
        + velocity_weight * foot_loss_dict["decoded_foot_velocity_loss"]
        + acceleration_weight * foot_loss_dict["decoded_foot_acceleration_loss"]
    )
    return loss_dict


def make_zero_decoded_foot_loss_dict(
    reference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    zero = reference.new_zeros(())
    return {
        "decoded_foot_position_loss": zero,
        "decoded_foot_velocity_loss": zero,
        "decoded_foot_acceleration_loss": zero,
        "decoded_planted_foot_speed": zero,
    }


def should_apply_decoded_foot_losses(epoch: int, batch_index: int, param: dict) -> bool:
    if (
        float(param.get("diffusion_decoded_foot_position_weight", 0.0)) <= 0.0
        and float(param.get("diffusion_decoded_foot_velocity_weight", 0.0)) <= 0.0
        and float(param.get("diffusion_decoded_foot_acceleration_weight", 0.0)) <= 0.0
    ):
        return False

    start_epoch = max(int(param.get("diffusion_decoded_foot_start_epoch", 1)), 1)
    if epoch < start_epoch:
        return False

    every_n_batches = max(
        int(param.get("diffusion_decoded_foot_every_n_batches", 1)),
        1,
    )
    return batch_index % every_n_batches == 0


@torch.no_grad()
def compute_condition_adherence_metrics(
    prior: ContinuousLatentDiffusionPrior,
    clean_latent: torch.Tensor,
    lma_seq,
    traj_seq,
    style_id,
):
    batch_size = clean_latent.size(0)
    device = clean_latent.device
    timestep_value = max(int(prior.num_train_timesteps * 0.6), 1)
    timesteps = torch.full(
        (batch_size,), timestep_value, device=device, dtype=torch.long
    )
    noise = torch.randn_like(clean_latent)
    noisy_latent = prior.q_sample(clean_latent, timesteps, noise=noise)

    x0_uncond, root_uncond = prior._guided_prediction(
        noisy_latent,
        timesteps,
        lma_seq=lma_seq,
        traj_seq=traj_seq,
        style_id=style_id,
        mode="uncond",
        cfg_scale=1.0,
    )
    metrics = {
        "traj_condition_response": 0.0,
        "lma_condition_response": 0.0,
        "full_condition_response": 0.0,
        "traj_root_translation_mse": 0.0,
        "full_root_translation_mse": 0.0,
    }

    if traj_seq is not None:
        traj_target = prior._ensure_sequence_tensor(
            traj_seq, feature_dim=prior.traj_dim
        )
        x0_traj, root_traj = prior._guided_prediction(
            noisy_latent,
            timesteps,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
            mode="traj_only",
            cfg_scale=1.0,
        )
        metrics["traj_condition_response"] = torch.mean(
            (x0_traj - x0_uncond) ** 2
        ).item()
        metrics["traj_root_translation_mse"] = F.mse_loss(
            root_traj[..., 6:9], traj_target[..., 6:9]
        ).item()

    if lma_seq is not None:
        x0_lma, _ = prior._guided_prediction(
            noisy_latent,
            timesteps,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
            mode="lma_only",
            cfg_scale=1.0,
        )
        metrics["lma_condition_response"] = torch.mean((x0_lma - x0_uncond) ** 2).item()

    if lma_seq is not None or traj_seq is not None:
        x0_full, root_full = prior._guided_prediction(
            noisy_latent,
            timesteps,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
            mode="full",
            cfg_scale=1.0,
        )
        metrics["full_condition_response"] = torch.mean(
            (x0_full - x0_uncond) ** 2
        ).item()
        if traj_seq is not None:
            traj_target = prior._ensure_sequence_tensor(
                traj_seq, feature_dim=prior.traj_dim
            )
            metrics["full_root_translation_mse"] = F.mse_loss(
                root_full[..., 6:9], traj_target[..., 6:9]
            ).item()

    return metrics


def combine_eval_score(metrics: dict) -> float:
    return float(
        metrics["loss"]
        + 0.35 * metrics.get("traj_root_translation_mse", 0.0)
        + 0.25 * metrics.get("full_root_translation_mse", 0.0)
        + 0.40 * metrics.get("decoded_foot_velocity_loss", 0.0)
        + 0.25 * metrics.get("decoded_planted_foot_speed", 0.0)
        - 0.1 * metrics.get("traj_condition_response", 0.0)
        - 0.1 * metrics.get("lma_condition_response", 0.0)
        - 0.1 * metrics.get("full_condition_response", 0.0)
    )


@torch.no_grad()
def evaluate_prior(
    prior: ContinuousLatentDiffusionPrior,
    generator_model: Generator_Model,
    train_data: Train_Data,
    eval_dataset: TestMotionData,
    latent_target_mode: str,
    max_items: int = 8,
):
    if eval_dataset.get_len() == 0:
        return {
            "loss": 0.0,
            "x0_loss": 0.0,
            "latent_velocity_loss": 0.0,
            "root_traj_loss": 0.0,
            "root_traj_velocity_loss": 0.0,
            "decoded_foot_position_loss": 0.0,
            "decoded_foot_velocity_loss": 0.0,
            "decoded_foot_acceleration_loss": 0.0,
            "decoded_planted_foot_speed": 0.0,
            "traj_condition_response": 0.0,
            "lma_condition_response": 0.0,
            "full_condition_response": 0.0,
            "traj_root_translation_mse": 0.0,
            "full_root_translation_mse": 0.0,
            "score": 0.0,
        }

    prior.eval()
    metric_sums = {
        "loss": 0.0,
        "x0_loss": 0.0,
        "latent_velocity_loss": 0.0,
        "root_traj_loss": 0.0,
        "root_traj_velocity_loss": 0.0,
        "decoded_foot_position_loss": 0.0,
        "decoded_foot_velocity_loss": 0.0,
        "decoded_foot_acceleration_loss": 0.0,
        "decoded_planted_foot_speed": 0.0,
        "traj_condition_response": 0.0,
        "lma_condition_response": 0.0,
        "full_condition_response": 0.0,
        "traj_root_translation_mse": 0.0,
        "full_root_translation_mse": 0.0,
        "score": 0.0,
    }
    item_count = min(max_items, eval_dataset.get_len())
    for index in range(min(max_items, eval_dataset.get_len())):
        eval_motion = build_eval_sample(eval_dataset.get_item(index), train_data.device)
        prepare_train_batch(train_data, eval_motion, eval_motion)
        latent_target = generator_model.autoencoder.encode_latent_sequence(
            train_data.sparse_motion,
            use_mean=latent_target_mode == "mean",
        ).detach()
        lma_seq, traj_seq, style_id = (
            generator_model.autoencoder.build_diffusion_condition_tensors(
                train_data.tags
            )
        )
        loss_dict = prior.compute_loss(
            latent_target,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
        )
        ae_offsets = generator_model.static_encoder(train_data.offsets)
        foot_loss_dict = compute_decoded_foot_kinematics_losses(
            generator_model,
            train_data,
            loss_dict["x0_pred"],
            ae_offsets,
        )
        loss_dict = apply_decoded_foot_kinematics_losses(
            loss_dict,
            foot_loss_dict,
            train_data.param,
        )
        adherence_metrics = compute_condition_adherence_metrics(
            prior,
            latent_target,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
        )
        metric_sums["loss"] += loss_dict["loss"].item()
        metric_sums["x0_loss"] += loss_dict["x0_loss"].item()
        metric_sums["latent_velocity_loss"] += loss_dict["latent_velocity_loss"].item()
        metric_sums["root_traj_loss"] += loss_dict["root_traj_loss"].item()
        metric_sums["root_traj_velocity_loss"] += loss_dict[
            "root_traj_velocity_loss"
        ].item()
        metric_sums["decoded_foot_position_loss"] += loss_dict[
            "decoded_foot_position_loss"
        ].item()
        metric_sums["decoded_foot_velocity_loss"] += loss_dict[
            "decoded_foot_velocity_loss"
        ].item()
        metric_sums["decoded_foot_acceleration_loss"] += loss_dict[
            "decoded_foot_acceleration_loss"
        ].item()
        metric_sums["decoded_planted_foot_speed"] += loss_dict[
            "decoded_planted_foot_speed"
        ].item()
        for key, value in adherence_metrics.items():
            metric_sums[key] += float(value)
    prior.train()
    averaged = {key: value / max(item_count, 1) for key, value in metric_sums.items()}
    averaged["score"] = combine_eval_score(averaged)
    return averaged


@torch.no_grad()
def save_preview(
    model_dir: Path,
    epoch: int,
    ema_prior: ContinuousLatentDiffusionPrior,
    generator_model: Generator_Model,
    eval_dataset: TestMotionData,
    device: torch.device,
    param: dict,
):
    if eval_dataset.get_len() == 0:
        return

    eval_motion = build_eval_sample(eval_dataset.get_item(0), device)
    offsets = eval_motion["offsets"]
    denorm_offsets = eval_motion["denorm_offsets"]
    tags = eval_motion["tags"]
    lma_seq, rough_root_traj, style_id = (
        generator_model.autoencoder.build_diffusion_condition_tensors(tags)
    )
    foot_contact_latent, support_contact_latent = get_contact_condition_tensors(tags)
    latent_len = (
        rough_root_traj.size(1)
        if rough_root_traj is not None
        else max(int(math.ceil(eval_motion["dqs"].size(1) / 8)), 1)
    )
    sampled_latent, predicted_root_traj = ema_prior.sample_long(
        total_seq_len=latent_len,
        batch_size=1,
        lma_seq=lma_seq,
        traj_seq=rough_root_traj,
        style_id=style_id,
        mode="full",
        cfg_scale=float(param.get("diffusion_cfg_scale", 2.5)),
        num_steps=int(param.get("diffusion_sample_steps", 50)),
        chunk_len=int(param.get("diffusion_chunk_len", 128)),
        overlap_len=int(param.get("diffusion_overlap_len", 32)),
        halo_len=int(param.get("diffusion_halo_len", 8)),
        eta=float(param.get("diffusion_eta", 0.0)),
        temperature=float(param.get("diffusion_temperature", 1.0)),
        device=device,
        return_aux=True,
    )
    ae_offsets = generator_model.static_encoder(offsets)
    decoded_motion, normalized_root = (
        generator_model.autoencoder.decode_latent_sequence(
            sampled_latent,
            ae_offsets,
            generator_model.data.mean_dqs,
            generator_model.data.std_dqs,
            denorm_offsets,
            mean_root=generator_model.data.mean_root,
            std_root=generator_model.data.std_root,
            tags=tags,
        )
    )
    torch.save(
        {
            "sampled_latent": sampled_latent.detach().cpu(),
            "decoded_motion": decoded_motion.detach().cpu(),
            "normalized_root": normalized_root.detach().cpu(),
            "rough_root_traj": (
                None if rough_root_traj is None else rough_root_traj.detach().cpu()
            ),
            "predicted_rough_root_traj": predicted_root_traj.detach().cpu(),
            "lma_seq": None if lma_seq is None else lma_seq.detach().cpu(),
            "foot_contact_latent": (
                None
                if foot_contact_latent is None
                else foot_contact_latent.detach().cpu()
            ),
            "support_contact_latent": (
                None
                if support_contact_latent is None
                else support_contact_latent.detach().cpu()
            ),
            "style_id": None if style_id is None else style_id.detach().cpu(),
            "style_label": eval_motion.get("style_label"),
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
            "target_root_rotation": (
                None
                if rough_root_traj is None
                else split_root_diagnostics(rough_root_traj)["root_rotation"]
                .detach()
                .cpu()
            ),
            "target_root_translation": (
                None
                if rough_root_traj is None
                else split_root_diagnostics(rough_root_traj)["root_translation"]
                .detach()
                .cpu()
            ),
            "target_root_translation_velocity": (
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
            "root_rotation_mse": (
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
            "root_translation_mse": (
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
            "root_translation_velocity_mse": (
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
        },
        model_dir / f"preview_epoch_{epoch:04d}.pt",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a continuous latent MDM-style diffusion prior"
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Dataset root containing train/, eval/, and annotations/",
    )
    parser.add_argument(
        "--vae-model-path",
        required=True,
        help="Path to pretrained generator.pt checkpoint",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional path to a saved latent_diffusion_prior.pt or latent_diffusion_prior_best.pt checkpoint to continue training from.",
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="When resuming, keep model weights but reinitialize the optimizer using the current learning-rate and weight-decay arguments.",
    )
    parser.add_argument("--name", default="latent_mdm", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=DEFAULT_PARAM["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PARAM["batch_size"])
    parser.add_argument("--window-size", type=int, default=DEFAULT_PARAM["window_size"])
    parser.add_argument("--window-step", type=int, default=DEFAULT_PARAM["window_step"])
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_PARAM["diffusion_hidden_dim"]
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_PARAM["diffusion_num_layers"]
    )
    parser.add_argument(
        "--num-heads", type=int, default=DEFAULT_PARAM["diffusion_num_heads"]
    )
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_PARAM["diffusion_dropout"]
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=DEFAULT_PARAM["diffusion_num_train_timesteps"],
    )
    parser.add_argument(
        "--sample-steps", type=int, default=DEFAULT_PARAM["diffusion_sample_steps"]
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=DEFAULT_PARAM["diffusion_cfg_scale"]
    )
    parser.add_argument(
        "--chunk-len", type=int, default=DEFAULT_PARAM["diffusion_chunk_len"]
    )
    parser.add_argument(
        "--overlap-len", type=int, default=DEFAULT_PARAM["diffusion_overlap_len"]
    )
    parser.add_argument(
        "--halo-len", type=int, default=DEFAULT_PARAM["diffusion_halo_len"]
    )
    parser.add_argument("--latent-target", choices=["mean", "sample"], default="mean")
    parser.add_argument(
        "--latent-velocity-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_latent_velocity_weight"],
    )
    parser.add_argument(
        "--root-loss-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_root_traj_loss_weight"],
    )
    parser.add_argument(
        "--root-velocity-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_root_traj_velocity_weight"],
    )
    parser.add_argument(
        "--decoded-foot-position-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_decoded_foot_position_weight"],
    )
    parser.add_argument(
        "--decoded-foot-velocity-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_decoded_foot_velocity_weight"],
    )
    parser.add_argument(
        "--decoded-foot-acceleration-weight",
        type=float,
        default=DEFAULT_PARAM["diffusion_decoded_foot_acceleration_weight"],
    )
    parser.add_argument(
        "--decoded-foot-every-n-batches",
        type=int,
        default=DEFAULT_PARAM["diffusion_decoded_foot_every_n_batches"],
        help="Compute decoded-foot losses every N training batches.",
    )
    parser.add_argument(
        "--decoded-foot-start-epoch",
        type=int,
        default=DEFAULT_PARAM["diffusion_decoded_foot_start_epoch"],
        help="First epoch that applies decoded-foot losses; use 16 to start after epoch 15.",
    )
    parser.add_argument(
        "--root-rot-scale",
        type=float,
        default=DEFAULT_PARAM["diffusion_root_rot_loss_scale"],
    )
    parser.add_argument(
        "--root-pos-scale",
        type=float,
        default=DEFAULT_PARAM["diffusion_root_pos_loss_scale"],
    )
    parser.add_argument(
        "--condition-drop-lma",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_drop_lma"],
    )
    parser.add_argument(
        "--condition-drop-traj",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_drop_traj"],
    )
    parser.add_argument(
        "--condition-drop-both",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_drop_both"],
    )
    parser.add_argument(
        "--final-condition-drop-lma",
        type=float,
        default=DEFAULT_PARAM["diffusion_final_condition_drop_lma"],
    )
    parser.add_argument(
        "--final-condition-drop-traj",
        type=float,
        default=DEFAULT_PARAM["diffusion_final_condition_drop_traj"],
    )
    parser.add_argument(
        "--final-condition-drop-both",
        type=float,
        default=DEFAULT_PARAM["diffusion_final_condition_drop_both"],
    )
    parser.add_argument(
        "--lma-condition-scale",
        type=float,
        default=DEFAULT_PARAM["diffusion_lma_condition_scale"],
    )
    parser.add_argument(
        "--traj-condition-scale",
        type=float,
        default=DEFAULT_PARAM["diffusion_traj_condition_scale"],
    )
    parser.add_argument(
        "--style-drop",
        type=float,
        default=DEFAULT_PARAM["diffusion_style_drop"],
    )
    parser.add_argument(
        "--final-style-drop",
        type=float,
        default=DEFAULT_PARAM["diffusion_final_style_drop"],
    )
    parser.add_argument(
        "--style-condition-scale",
        type=float,
        default=DEFAULT_PARAM["diffusion_style_condition_scale"],
    )
    parser.add_argument(
        "--condition-focus-start",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_focus_start"],
        help="Training progress fraction after which conditioning dropout is annealed toward its final values.",
    )
    parser.add_argument(
        "--use-explicit-condition-mode-probs",
        action="store_true",
        help="Train with explicit sampled full/lma_only/traj_only/uncond modes instead of independent condition dropout masks.",
    )
    parser.add_argument(
        "--condition-mode-prob-full",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_mode_prob_full"],
    )
    parser.add_argument(
        "--condition-mode-prob-lma-only",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_mode_prob_lma_only"],
    )
    parser.add_argument(
        "--condition-mode-prob-traj-only",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_mode_prob_traj_only"],
    )
    parser.add_argument(
        "--condition-mode-prob-uncond",
        type=float,
        default=DEFAULT_PARAM["diffusion_condition_mode_prob_uncond"],
    )
    parser.add_argument(
        "--use-per-lma-channel-encoders",
        action="store_true",
        help="Use one temporal encoder per LMA channel instead of a single shared LMA encoder.",
    )
    parser.add_argument(
        "--per-lma-channel-dropout",
        type=float,
        default=DEFAULT_PARAM["diffusion_per_lma_channel_dropout"],
        help="When per-channel LMA encoders are enabled, randomly drop whole LMA channel streams during training before fusion.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--num-eval-items", type=int, default=8)
    parser.add_argument(
        "--prior-random-yaw-augmentation",
        action="store_true",
        help="Apply the same random yaw augmentation used for VAE training before encoding latent targets and root conditions for prior training batches.",
    )
    parser.add_argument(
        "--prior-random-yaw-max-degrees",
        type=float,
        default=DEFAULT_PARAM["random_yaw_aug_max_degrees"],
        help="Maximum absolute yaw rotation in degrees when prior random yaw augmentation is enabled.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAM["seed"])
    return parser.parse_args()


def format_epoch_summary(
    epoch: int,
    was_best: bool,
    train_metrics: dict,
    eval_metrics: dict,
    dropout_state: dict,
    condition_strategy: str,
    best_eval_score: float,
):
    best_marker = " *" if was_best else ""
    return [
        (
            f"Epoch {epoch:04d}{best_marker} | train={train_metrics['loss']:.6f} | "
            f"eval_loss={eval_metrics['loss']:.6f} | eval_score={eval_metrics['score']:.6f} | "
            f"best_score={best_eval_score:.6f}"
        ),
        (
            f"  train: x0={train_metrics['x0_loss']:.6f} | lat_vel={train_metrics['latent_velocity_loss']:.6f} | "
            f"root={train_metrics['root_traj_loss']:.6f} | root_vel={train_metrics['root_traj_velocity_loss']:.6f} | "
            f"root_rot={train_metrics['root_rotation_loss']:.6f} | root_pos={train_metrics['root_translation_loss']:.6f}"
        ),
        (
            f"  feet : pos={train_metrics['decoded_foot_position_loss']:.6f} | "
            f"vel={train_metrics['decoded_foot_velocity_loss']:.6f} | "
            f"acc={train_metrics['decoded_foot_acceleration_loss']:.6f} | "
            f"plant_speed={eval_metrics['decoded_planted_foot_speed']:.6f} | "
            f"traj_mse={eval_metrics['traj_root_translation_mse']:.6f} | "
            f"full_mse={eval_metrics['full_root_translation_mse']:.6f}"
        ),
        (
            f"  cond : lma_rsp={eval_metrics['lma_condition_response']:.6f} | "
            f"traj_rsp={eval_metrics['traj_condition_response']:.6f} | "
            f"full_rsp={eval_metrics['full_condition_response']:.6f} | "
            f"mask={condition_strategy} | "
            f"drop(lma/traj/style)={dropout_state['p_drop_lma']:.3f}/"
            f"{dropout_state['p_drop_traj']:.3f}/{dropout_state['style_drop']:.3f}"
        ),
    ]


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    param = copy.deepcopy(DEFAULT_PARAM)
    param.update(
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "diffusion_hidden_dim": args.hidden_dim,
            "diffusion_num_layers": args.num_layers,
            "diffusion_num_heads": args.num_heads,
            "diffusion_dropout": args.dropout,
            "diffusion_num_train_timesteps": args.num_train_timesteps,
            "diffusion_sample_steps": args.sample_steps,
            "diffusion_cfg_scale": args.cfg_scale,
            "diffusion_chunk_len": args.chunk_len,
            "diffusion_overlap_len": args.overlap_len,
            "diffusion_halo_len": args.halo_len,
            "diffusion_latent_velocity_weight": args.latent_velocity_weight,
            "diffusion_root_traj_loss_weight": args.root_loss_weight,
            "diffusion_root_traj_velocity_weight": args.root_velocity_weight,
            "diffusion_decoded_foot_position_weight": args.decoded_foot_position_weight,
            "diffusion_decoded_foot_velocity_weight": args.decoded_foot_velocity_weight,
            "diffusion_decoded_foot_acceleration_weight": args.decoded_foot_acceleration_weight,
            "diffusion_decoded_foot_every_n_batches": args.decoded_foot_every_n_batches,
            "diffusion_decoded_foot_start_epoch": args.decoded_foot_start_epoch,
            "diffusion_root_rot_loss_scale": args.root_rot_scale,
            "diffusion_root_pos_loss_scale": args.root_pos_scale,
            "diffusion_condition_drop_lma": args.condition_drop_lma,
            "diffusion_condition_drop_traj": args.condition_drop_traj,
            "diffusion_condition_drop_both": args.condition_drop_both,
            "diffusion_final_condition_drop_lma": args.final_condition_drop_lma,
            "diffusion_final_condition_drop_traj": args.final_condition_drop_traj,
            "diffusion_final_condition_drop_both": args.final_condition_drop_both,
            "diffusion_use_explicit_condition_mode_probs": args.use_explicit_condition_mode_probs,
            "diffusion_condition_mode_prob_full": args.condition_mode_prob_full,
            "diffusion_condition_mode_prob_lma_only": args.condition_mode_prob_lma_only,
            "diffusion_condition_mode_prob_traj_only": args.condition_mode_prob_traj_only,
            "diffusion_condition_mode_prob_uncond": args.condition_mode_prob_uncond,
            "diffusion_use_per_lma_channel_encoders": args.use_per_lma_channel_encoders,
            "diffusion_per_lma_channel_dropout": args.per_lma_channel_dropout,
            "diffusion_lma_condition_scale": args.lma_condition_scale,
            "diffusion_traj_condition_scale": args.traj_condition_scale,
            "diffusion_style_drop": args.style_drop,
            "diffusion_final_style_drop": args.final_style_drop,
            "diffusion_style_condition_scale": args.style_condition_scale,
            "diffusion_condition_focus_start": args.condition_focus_start,
            "random_yaw_augmentation": args.prior_random_yaw_augmentation,
            "random_yaw_aug_max_degrees": args.prior_random_yaw_max_degrees,
        }
    )

    train_dataset, eval_dataset, reference_parents, style_metadata = build_datasets(
        args.data_path, param, device
    )
    param["diffusion_num_styles"] = len(style_metadata.get("style_vocab", []))
    train_loader = DataLoader(
        train_dataset, batch_size=param["batch_size"], shuffle=True
    )

    train_data = Train_Data(device, param)
    generator_model = Generator_Model(
        device, param, reference_parents, train_data, is_vae=True, is_vq_vae=False
    ).to(device)
    load_model(generator_model, args.vae_model_path, train_data, device)
    train_data.set_displacement_means_stds(
        train_dataset.means["displacement"],
        train_dataset.stds["displacement"],
    )
    train_data.set_tag_root_means_stds(
        train_dataset.means["smooth_root_pos"],
        train_dataset.stds["smooth_root_pos"],
        train_dataset.means.get("rough_root_traj"),
        train_dataset.stds.get("rough_root_traj"),
    )
    freeze_module(generator_model.static_encoder)
    freeze_module(generator_model.autoencoder)

    prior = ContinuousLatentDiffusionPrior(
        latent_dim=int(param.get("vae_latent_dim", 504)),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lma_dim=len(LMA_KEYS),
        traj_dim=int(param.get("rough_root_traj_dim", 9)),
        num_train_timesteps=args.num_train_timesteps,
        condition_mode_probs=build_condition_mode_probs(args),
        use_explicit_condition_mode_probs=args.use_explicit_condition_mode_probs,
        use_per_lma_channel_encoders=args.use_per_lma_channel_encoders,
        per_lma_channel_dropout=args.per_lma_channel_dropout,
        latent_velocity_weight=args.latent_velocity_weight,
        root_traj_loss_weight=args.root_loss_weight,
        root_traj_velocity_weight=args.root_velocity_weight,
        root_rot_loss_scale=args.root_rot_scale,
        root_pos_loss_scale=args.root_pos_scale,
        p_drop_lma=args.condition_drop_lma,
        p_drop_traj=args.condition_drop_traj,
        p_drop_both=args.condition_drop_both,
        final_p_drop_lma=args.final_condition_drop_lma,
        final_p_drop_traj=args.final_condition_drop_traj,
        final_p_drop_both=args.final_condition_drop_both,
        lma_condition_scale=args.lma_condition_scale,
        traj_condition_scale=args.traj_condition_scale,
        num_styles=len(style_metadata.get("style_vocab", [])),
        style_drop=args.style_drop,
        final_style_drop=args.final_style_drop,
        style_condition_scale=args.style_condition_scale,
        condition_focus_start=args.condition_focus_start,
    ).to(device)
    ema_prior = copy.deepcopy(prior).eval()
    optimizer = torch.optim.AdamW(
        prior.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.resume_checkpoint is not None:
        checkpoint_path = Path(args.resume_checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint, start_epoch, best_eval_score = load_training_checkpoint(
            str(checkpoint_path),
            prior,
            ema_prior,
            optimizer,
            device=device,
            reset_optimizer=args.resume_reset_optimizer,
        )
        model_dir = checkpoint_path.parent
        print(
            f"Resuming prior training from {checkpoint_path} | "
            f"next_epoch={start_epoch} | best_score={best_eval_score:.6f}"
        )
        if args.resume_reset_optimizer:
            print(
                "Optimizer state was reset on resume; using current learning-rate and weight-decay arguments."
            )
    else:
        checkpoint = None
        start_epoch = 1
        model_dir = create_model_dir(args.name, args.data_path)
        best_eval_score = float("inf")

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}, which is beyond requested total epochs {args.epochs}."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        prior.train()
        progress = epoch / max(args.epochs, 1)
        prior.set_training_progress(progress)
        ema_prior.set_training_progress(progress)
        epoch_loss = 0.0
        epoch_x0_loss = 0.0
        epoch_latent_vel = 0.0
        epoch_root_loss = 0.0
        epoch_root_vel = 0.0
        epoch_root_rot = 0.0
        epoch_root_pos = 0.0
        epoch_decoded_foot_pos = 0.0
        epoch_decoded_foot_vel = 0.0
        epoch_decoded_foot_acc = 0.0
        epoch_decoded_foot_updates = 0
        for batch_index, (denorm_motion, norm_motion) in enumerate(
            train_loader,
            start=1,
        ):
            prepare_train_batch(
                train_data,
                denorm_motion,
                norm_motion,
                apply_yaw_augmentation=args.prior_random_yaw_augmentation,
            )

            with torch.no_grad():
                latent_target = generator_model.autoencoder.encode_latent_sequence(
                    train_data.sparse_motion,
                    use_mean=args.latent_target == "mean",
                ).detach()
                lma_seq, rough_root_traj, style_id = (
                    generator_model.autoencoder.build_diffusion_condition_tensors(
                        train_data.tags
                    )
                )
                ae_offsets = generator_model.static_encoder(train_data.offsets)

            loss_dict = prior.compute_loss(
                latent_target,
                lma_seq=lma_seq,
                traj_seq=rough_root_traj,
                style_id=style_id,
            )
            if should_apply_decoded_foot_losses(epoch, batch_index, param):
                foot_loss_dict = compute_decoded_foot_kinematics_losses(
                    generator_model,
                    train_data,
                    loss_dict["x0_pred"],
                    ae_offsets,
                )
                epoch_decoded_foot_updates += 1
            else:
                foot_loss_dict = make_zero_decoded_foot_loss_dict(loss_dict["loss"])
            loss_dict = apply_decoded_foot_kinematics_losses(
                loss_dict,
                foot_loss_dict,
                param,
            )
            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(prior, ema_prior, decay=args.ema_decay)

            epoch_loss += loss_dict["loss"].item()
            epoch_x0_loss += loss_dict["x0_loss"].item()
            epoch_latent_vel += loss_dict["latent_velocity_loss"].item()
            epoch_root_loss += loss_dict["root_traj_loss"].item()
            epoch_root_vel += loss_dict["root_traj_velocity_loss"].item()
            epoch_root_rot += loss_dict["root_rotation_loss"].item()
            epoch_root_pos += loss_dict["root_translation_loss"].item()
            epoch_decoded_foot_pos += loss_dict["decoded_foot_position_loss"].item()
            epoch_decoded_foot_vel += loss_dict["decoded_foot_velocity_loss"].item()
            epoch_decoded_foot_acc += loss_dict["decoded_foot_acceleration_loss"].item()

        denom = max(len(train_loader), 1)
        epoch_loss /= denom
        epoch_x0_loss /= denom
        epoch_latent_vel /= denom
        epoch_root_loss /= denom
        epoch_root_vel /= denom
        epoch_root_rot /= denom
        epoch_root_pos /= denom
        decoded_foot_denom = max(epoch_decoded_foot_updates, 1)
        epoch_decoded_foot_pos /= decoded_foot_denom
        epoch_decoded_foot_vel /= decoded_foot_denom
        epoch_decoded_foot_acc /= decoded_foot_denom

        eval_metrics = {
            "loss": float("nan"),
            "score": float("nan"),
            "decoded_foot_position_loss": float("nan"),
            "decoded_foot_velocity_loss": float("nan"),
            "decoded_foot_acceleration_loss": float("nan"),
            "decoded_planted_foot_speed": float("nan"),
            "traj_root_translation_mse": float("nan"),
            "full_root_translation_mse": float("nan"),
            "traj_condition_response": float("nan"),
            "lma_condition_response": float("nan"),
            "full_condition_response": float("nan"),
        }
        was_best = False
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            eval_metrics = evaluate_prior(
                ema_prior,
                generator_model,
                train_data,
                eval_dataset,
                latent_target_mode=args.latent_target,
                max_items=args.num_eval_items,
            )
            if eval_metrics["score"] < best_eval_score:
                best_eval_score = eval_metrics["score"]
                was_best = True
                save_checkpoint(
                    model_dir,
                    prior,
                    ema_prior,
                    optimizer,
                    epoch,
                    best_eval_score,
                    train_dataset,
                    style_metadata,
                    param,
                    args,
                    is_best=True,
                )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                model_dir,
                prior,
                ema_prior,
                optimizer,
                epoch,
                best_eval_score,
                train_dataset,
                style_metadata,
                param,
                args,
            )

        if args.preview_every > 0 and epoch % args.preview_every == 0:
            save_preview(
                model_dir,
                epoch,
                ema_prior,
                generator_model,
                eval_dataset,
                device,
                param,
            )

        dropout_state = prior.get_current_dropout_state()
        train_metrics = {
            "loss": epoch_loss,
            "x0_loss": epoch_x0_loss,
            "latent_velocity_loss": epoch_latent_vel,
            "root_traj_loss": epoch_root_loss,
            "root_traj_velocity_loss": epoch_root_vel,
            "root_rotation_loss": epoch_root_rot,
            "root_translation_loss": epoch_root_pos,
            "decoded_foot_position_loss": epoch_decoded_foot_pos,
            "decoded_foot_velocity_loss": epoch_decoded_foot_vel,
            "decoded_foot_acceleration_loss": epoch_decoded_foot_acc,
        }
        for line in format_epoch_summary(
            epoch,
            was_best,
            train_metrics,
            eval_metrics,
            dropout_state,
            "explicit-modes" if args.use_explicit_condition_mode_probs else "dropout",
            best_eval_score,
        ):
            print(line)


if __name__ == "__main__":
    main()
