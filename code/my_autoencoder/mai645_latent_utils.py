from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from generator_architecture import Generator_Model
from ik_architecture import IK_Model
from motion_data import TestMotionData
from train_data import Train_Data
import train_vq_vae


@dataclass
class FileRecord:
    split_name: str
    filename: str
    file_path: str

    @property
    def source_relative_path(self) -> str:
        return os.path.join(self.split_name, self.filename)


@dataclass
class LatentFileRecord:
    split_name: str
    filename: str
    file_path: str

    @property
    def latent_relative_path(self) -> str:
        return os.path.join(self.split_name, self.filename)

    @property
    def source_file_name(self) -> str:
        if self.filename.endswith(".npy"):
            return self.filename[:-4]
        return self.filename

    @property
    def source_relative_path(self) -> str:
        return os.path.join(self.split_name, self.source_file_name)


@dataclass
class AutoencoderRuntime:
    param: dict[str, Any]
    device: torch.device
    data_path: str
    generator_path: str
    ik_path: str | None
    data_stats_path: str
    downsampling_factor: int
    train_data: Train_Data
    generator_model: Generator_Model
    ik_model: IK_Model | None
    means: dict[str, Any]
    stds: dict[str, Any]
    reference_parents: list[int]
    param_source: str


def clone_default_param() -> dict[str, Any]:
    return copy.deepcopy(train_vq_vae.param)


def apply_param_overrides(
    param: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    if overrides is None:
        return param
    for key, value in overrides.items():
        if value is None:
            continue
        param[key] = value
    return param


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


RUNTIME_PARAM_SUMMARY_KEYS = (
    "autoencoder_module",
    "training_stage",
    "use_vae",
    "vae_latent_dim",
    "vae_hidden_dim",
    "stride_encoder_conv",
    "bvh_scale_factor",
)


def extract_runtime_param_summary(param: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in RUNTIME_PARAM_SUMMARY_KEYS:
        if key in param:
            summary[key] = copy.deepcopy(param[key])
    return summary


def _load_checkpoint_payload(path: str) -> dict[str, Any] | None:
    if os.path.exists(path) is False:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) is False:
        return None
    return payload


def load_saved_model_param(
    data_stats_path: str,
    generator_path: str,
) -> tuple[dict[str, Any] | None, str | None]:
    data_payload = _load_checkpoint_payload(data_stats_path)
    if isinstance(data_payload, dict) and isinstance(
        data_payload.get("model_param"), dict
    ):
        return copy.deepcopy(data_payload["model_param"]), "data.pt"

    generator_payload = _load_checkpoint_payload(generator_path)
    if isinstance(generator_payload, dict) and isinstance(
        generator_payload.get("model_param"), dict
    ):
        return copy.deepcopy(generator_payload["model_param"]), "generator.pt"

    return None, None


def infer_param_overrides_from_checkpoint(
    generator_path: str,
) -> tuple[dict[str, Any], str | None]:
    checkpoint = _load_checkpoint_payload(generator_path)
    if checkpoint is None:
        return {}, None

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict) is False:
        return {}, None

    has_vae_posterior = any(
        key.startswith("autoencoder.encoder.posterior_mean_head.")
        for key in state_dict.keys()
    )
    if has_vae_posterior:
        return {}, None

    return {"use_vae": False}, "checkpoint_state_dict"


def build_checkpoint_mismatch_message(
    generator_path: str,
    param_source: str,
    param: dict[str, Any],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> str:
    detail_parts = []
    if len(missing_keys) > 0:
        detail_parts.append("missing keys: {}".format(", ".join(missing_keys[:8])))
    if len(unexpected_keys) > 0:
        detail_parts.append(
            "unexpected keys: {}".format(", ".join(unexpected_keys[:8]))
        )

    return (
        "Checkpoint {} does not match the reconstructed autoencoder configuration "
        "resolved from {} with param summary {}. {}. Re-export latents only after "
        "loading the checkpoint with the matching model configuration."
    ).format(
        os.path.abspath(generator_path),
        param_source,
        extract_runtime_param_summary(param),
        "; ".join(detail_parts) if detail_parts else "Unknown mismatch",
    )


def get_expected_latent_width(runtime: AutoencoderRuntime) -> int:
    autoencoder = runtime.generator_model.autoencoder
    if bool(getattr(autoencoder, "is_vae", False)):
        return int(getattr(autoencoder.encoder, "vae_latent_dim", 0))

    channel_list = getattr(autoencoder.encoder, "channel_list", None)
    if channel_list is None or len(channel_list) == 0:
        raise ValueError("Could not determine latent width for the loaded autoencoder")
    return int(channel_list[-1])


def resolve_generator_checkpoint_path(model_path: str) -> tuple[str, str]:
    normalized_path = os.path.abspath(model_path)
    if os.path.isdir(normalized_path):
        generator_path = os.path.join(normalized_path, "generator.pt")
        if os.path.exists(generator_path) is False:
            raise FileNotFoundError(
                "Could not find generator.pt under {}".format(normalized_path)
            )
        return generator_path, os.path.join(normalized_path, "data.pt")

    if os.path.exists(normalized_path) is False:
        raise FileNotFoundError(normalized_path)
    if os.path.basename(normalized_path) != "generator.pt":
        raise ValueError(
            "model_path must point to generator.pt or to its containing directory"
        )
    return normalized_path, os.path.join(os.path.dirname(normalized_path), "data.pt")


def resolve_ik_checkpoint_path(generator_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(generator_path)), "ik.pt")


def collect_bvh_file_records(
    data_path: str,
    split: str,
    max_files: int = 0,
) -> list[FileRecord]:
    split = split.lower()
    if split not in {"train", "eval", "all"}:
        raise ValueError("split must be one of train, eval, or all")

    split_names = [split] if split != "all" else ["train", "eval"]
    file_records: list[FileRecord] = []
    for split_name in split_names:
        split_dir = os.path.join(data_path, split_name)
        if os.path.exists(split_dir) is False:
            raise ValueError("{} directory does not exist".format(split_dir))
        for filename in sorted(os.listdir(split_dir)):
            if filename.endswith(".bvh") is False:
                continue
            file_records.append(
                FileRecord(
                    split_name=split_name,
                    filename=filename,
                    file_path=os.path.join(split_dir, filename),
                )
            )
            if max_files > 0 and len(file_records) >= max_files:
                return file_records
    return file_records


def collect_latent_file_records(
    latent_folder: str,
    split: str,
    max_files: int = 0,
) -> list[LatentFileRecord]:
    split = split.lower()
    if split not in {"train", "eval", "all"}:
        raise ValueError("split must be one of train, eval, or all")

    normalized_folder = os.path.abspath(latent_folder)
    if os.path.exists(normalized_folder) is False:
        raise FileNotFoundError(
            "Latent folder does not exist: {}".format(normalized_folder)
        )

    split_names = [split] if split != "all" else ["train", "eval"]
    file_records: list[LatentFileRecord] = []
    for split_name in split_names:
        split_dir = os.path.join(normalized_folder, split_name)
        if os.path.exists(split_dir) is False:
            continue
        for filename in sorted(os.listdir(split_dir)):
            if filename.endswith(".npy") is False:
                continue
            file_records.append(
                LatentFileRecord(
                    split_name=split_name,
                    filename=filename,
                    file_path=os.path.join(split_dir, filename),
                )
            )
            if max_files > 0 and len(file_records) >= max_files:
                return file_records
    return file_records


def load_optional_lma_features(
    data_path: str,
    split_name: str,
    filename: str,
) -> np.ndarray | None:
    try:
        from pathlib import Path
        import pandas as pd
    except Exception:
        return None

    stem = Path(filename).stem
    annotation_candidates = [
        Path(data_path) / "annotations" / split_name / (stem + ".csv"),
        Path(data_path) / "annotations" / (stem + ".csv"),
    ]
    for candidate in annotation_candidates:
        if candidate.exists() is False:
            continue
        dataframe = pd.read_csv(candidate)
        numeric = dataframe.select_dtypes(include=["number"]).to_numpy()
        if numeric.size == 0:
            return None
        if numeric.shape[1] >= 6:
            return numeric[:, :6]
        return numeric
    return None


def get_reference_parents(file_record: FileRecord) -> list[int]:
    directory = os.path.dirname(file_record.file_path)
    bvh = train_vq_vae.get_bvh_from_disk(directory, file_record.filename)
    _, _, parents, _, _, _ = train_vq_vae.get_info_from_bvh(
        bvh,
        incremental_rots=False,
        get_missing_frames=False,
    )
    return list(parents)


def create_runtime(
    data_path: str,
    model_path: str,
    device: torch.device,
    param_overrides: dict[str, Any] | None = None,
    load_ik: bool = False,
) -> AutoencoderRuntime:
    generator_path, data_stats_path = resolve_generator_checkpoint_path(model_path)
    ik_path = resolve_ik_checkpoint_path(generator_path)
    if os.path.exists(ik_path) is False:
        ik_path = None

    param = clone_default_param()
    saved_param, param_source = load_saved_model_param(data_stats_path, generator_path)
    if saved_param is not None:
        param = apply_param_overrides(param, saved_param)
    else:
        inferred_param, inferred_source = infer_param_overrides_from_checkpoint(
            generator_path
        )
        param = apply_param_overrides(param, inferred_param)
        param_source = inferred_source

    if param_source is None:
        param_source = "default_param"

    param = apply_param_overrides(param, param_overrides)

    file_records = collect_bvh_file_records(data_path, split="all", max_files=1)
    if len(file_records) == 0:
        raise ValueError("No BVH files found under {}".format(data_path))
    reference_parents = get_reference_parents(file_records[0])

    train_data = Train_Data(device, param)
    generator_model = Generator_Model(
        device,
        param,
        reference_parents,
        train_data,
        is_vae=bool(param.get("use_vae", False)),
        is_vq_vae=False,
    ).to(device)
    means, stds, missing_keys, unexpected_keys = train_vq_vae.load_model(
        generator_model,
        generator_path,
        train_data,
        device,
        return_incompatible_keys=True,
    )
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            build_checkpoint_mismatch_message(
                generator_path,
                param_source,
                param,
                missing_keys,
                unexpected_keys,
            )
        )
    generator_model.eval()

    ik_model = None
    if load_ik:
        if ik_path is None:
            raise FileNotFoundError(
                "Could not find ik.pt next to {}. Train the my_autoencoder pipeline with --train_mode all so it saves generator.pt, ik.pt, and data.pt in the same model folder before using --use_ik.".format(
                    os.path.abspath(generator_path)
                )
            )

        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
        train_vq_vae.load_model(ik_model, ik_path, train_data, device)
        ik_model.eval()

    return AutoencoderRuntime(
        param=param,
        device=device,
        data_path=os.path.abspath(data_path),
        generator_path=generator_path,
        ik_path=ik_path,
        data_stats_path=data_stats_path,
        downsampling_factor=int(param["stride_encoder_conv"]) ** 3,
        train_data=train_data,
        generator_model=generator_model,
        ik_model=ik_model,
        means=means,
        stds=stds,
        reference_parents=reference_parents,
        param_source=param_source,
    )


def build_test_dataset(
    runtime: AutoencoderRuntime,
    split: str,
    max_files: int = 0,
) -> tuple[TestMotionData, list[FileRecord]]:
    file_records = collect_bvh_file_records(
        runtime.data_path, split=split, max_files=max_files
    )
    if len(file_records) == 0:
        raise ValueError("No BVH files found for split {}".format(split))

    dataset = TestMotionData(runtime.param, train_vq_vae.scale, runtime.device)
    dataset.set_means_stds(runtime.means, runtime.stds)

    for file_record in file_records:
        directory = os.path.dirname(file_record.file_path)
        bvh = train_vq_vae.get_bvh_from_disk(directory, file_record.filename)
        rots, pos, parents, offsets, bvh, og_rots = train_vq_vae.get_info_from_bvh(
            bvh,
            incremental_rots=False,
            get_missing_frames=False,
        )
        if list(parents) != runtime.reference_parents:
            raise ValueError(
                "Skeleton mismatch while loading {}".format(file_record.file_path)
            )

        pos_all_joints = bvh.compute_global_pos()
        dataset.add_motion(
            offsets,
            pos[:, 0, :],
            rots,
            parents,
            bvh,
            file_record.filename,
            pos_all_joints,
            og_rots=og_rots,
            end_sites=bvh.data["end_sites"],
            end_sites_parents=bvh.data["end_sites_parents"],
            lma_features=load_optional_lma_features(
                runtime.data_path,
                file_record.split_name,
                file_record.filename,
            ),
        )

    dataset.normalize()
    return dataset, file_records


def _build_tags_batch(tags: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone().detach().unsqueeze(0) for key, value in tags.items()}


def prepare_train_data_from_dataset_item(
    runtime: AutoencoderRuntime,
    dataset: TestMotionData,
    index: int,
) -> tuple[dict[str, Any], Any, str]:
    motion = dataset.get_item(index)
    bvh, filename = dataset.get_bvh(index)

    runtime.train_data.set_offsets(
        motion["offsets"].unsqueeze(0),
        motion["denorm_offsets"].unsqueeze(0),
    )
    runtime.train_data.set_rot_order(bvh.data["rot_order"])
    runtime.train_data.set_end_sites(
        torch.tensor(
            motion["end_sites"], dtype=torch.float32, device=runtime.device
        ).unsqueeze(0),
        torch.tensor(
            motion["end_sites_parents"], dtype=torch.long, device=runtime.device
        ).unsqueeze(0),
    )
    runtime.train_data.set_motions(
        motion["dqs"].unsqueeze(0),
        motion["displacement"].unsqueeze(0),
    )
    runtime.train_data.set_tags(_build_tags_batch(motion["tags"]))
    runtime.train_data.set_rots(motion["rots"].unsqueeze(0))
    runtime.train_data.set_global_pos(
        torch.tensor(
            bvh.data["positions"][:, 0], dtype=torch.float32, device=runtime.device
        ).unsqueeze(0)
    )
    runtime.train_data.set_foot_positions(
        torch.tensor(
            motion["foot_positions"], dtype=torch.float32, device=runtime.device
        ).unsqueeze(0)
    )
    return motion, bvh, filename


def encode_dataset_item(
    runtime: AutoencoderRuntime,
    dataset: TestMotionData,
    index: int,
    use_mean: bool = True,
) -> dict[str, Any]:
    motion, bvh, filename = prepare_train_data_from_dataset_item(
        runtime, dataset, index
    )
    with torch.no_grad():
        latent = runtime.generator_model.autoencoder.encode_latent_sequence(
            runtime.train_data.sparse_motion,
            use_mean=use_mean,
        )
        latent_mean, latent_logvar, latent_kl = (
            runtime.generator_model.autoencoder.get_variational_stats()
        )

    latent_np = latent[0].detach().cpu().numpy().astype(np.float32)
    result = {
        "filename": filename,
        "latent": latent_np,
        "original_frame_count": int(bvh.data["rotations"].shape[0]),
        "prepared_frame_count": int(motion["dqs"].shape[0]),
        "latent_length": int(latent_np.shape[0]),
        "latent_width": int(latent_np.shape[1]),
    }
    if latent_mean is not None:
        result["latent_mean_shape"] = list(latent_mean.shape[1:])
    if latent_logvar is not None:
        result["latent_logvar_shape"] = list(latent_logvar.shape[1:])
    if latent_kl is not None:
        result["latent_kl"] = float(latent_kl.detach().cpu().item())
    return result


def refine_motion_with_ik(
    runtime: AutoencoderRuntime,
    decoded_motion: torch.Tensor,
) -> torch.Tensor:
    if runtime.ik_model is None:
        return decoded_motion
    return runtime.ik_model.forward(decoded_motion)


def reconstruct_dataset_item_to_bvh(
    runtime: AutoencoderRuntime,
    dataset: TestMotionData,
    index: int,
    output_filename: str,
    output_dir: str,
) -> str:
    _, bvh, _ = prepare_train_data_from_dataset_item(runtime, dataset, index)

    with torch.no_grad():
        reconstructed_motion = runtime.generator_model.forward()
        reconstructed_motion = refine_motion_with_ik(runtime, reconstructed_motion)
        _, saved_filename = train_vq_vae.result_to_bvh(
            reconstructed_motion,
            runtime.means,
            runtime.stds,
            bvh,
            output_filename,
            save=True,
            output_dir=output_dir,
            filename_prefix="",
        )

    return os.path.join(output_dir, saved_filename)


def decode_dataset_item_to_bvh(
    runtime: AutoencoderRuntime,
    dataset: TestMotionData,
    index: int,
    latent: np.ndarray,
    output_filename: str,
    output_dir: str,
) -> str:
    _, bvh, _ = prepare_train_data_from_dataset_item(runtime, dataset, index)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError("Expected latent array [T, C], got {}".format(latent.shape))

    expected_latent_width = get_expected_latent_width(runtime)
    if latent.shape[1] != expected_latent_width:
        raise ValueError(
            "Latent width {} does not match checkpoint expectation {} resolved from {}. "
            "These latents were likely exported with a different autoencoder configuration. "
            "Re-run export_latent_dataset.py with the same --model_path used for decoding.".format(
                latent.shape[1], expected_latent_width, runtime.param_source
            )
        )

    latent_tensor = torch.tensor(
        latent, dtype=torch.float32, device=runtime.device
    ).unsqueeze(0)

    with torch.no_grad():
        ae_offsets = runtime.generator_model.static_encoder(runtime.train_data.offsets)
        decoded_motion, _ = runtime.generator_model.autoencoder.decode_latent_sequence(
            latent_tensor,
            ae_offsets,
            runtime.train_data.mean_dqs,
            runtime.train_data.std_dqs,
            runtime.train_data.denorm_offsets,
            mean_root=runtime.train_data.mean_root,
            std_root=runtime.train_data.std_root,
            tags=runtime.train_data.tags,
        )
        decoded_motion = refine_motion_with_ik(runtime, decoded_motion)
        _, saved_filename = train_vq_vae.result_to_bvh(
            decoded_motion,
            runtime.means,
            runtime.stds,
            bvh,
            output_filename,
            save=True,
            output_dir=output_dir,
            filename_prefix="",
        )
    return os.path.join(output_dir, saved_filename)


def summarize_runtime(runtime: AutoencoderRuntime) -> dict[str, Any]:
    return {
        "generator_path": runtime.generator_path,
        "ik_path": runtime.ik_path,
        "use_ik": runtime.ik_model is not None,
        "data_stats_path": runtime.data_stats_path,
        "autoencoder_module": runtime.param.get("autoencoder_module"),
        "training_stage": runtime.param.get("training_stage"),
        "use_vae": bool(runtime.param.get("use_vae", False)),
        "vae_latent_dim": int(runtime.param.get("vae_latent_dim", 0)),
        "bvh_scale_factor": float(
            runtime.param.get("bvh_scale_factor", train_vq_vae.scale)
        ),
        "downsampling_factor": runtime.downsampling_factor,
        "expected_latent_width": get_expected_latent_width(runtime),
        "resolved_param_source": runtime.param_source,
        "resolved_model_param": extract_runtime_param_summary(runtime.param),
    }
