import argparse
from pathlib import Path

import numpy as np
import torch

from motion_data import TrainMotionData
import train_vq_vae


EPS = 1e-8
STATIC_FEATURE_KEYS = {"offsets"}
PERCENTILES = (0, 5, 25, 50, 75, 95, 100)


def resolve_data_path(model_arg: str) -> Path:
    model_path = Path(model_arg)
    if model_path.is_dir():
        direct_data = model_path / "data.pt"
        autosave_data = model_path / "autosave" / "data.pt"
        if direct_data.exists():
            return direct_data
        if autosave_data.exists():
            return autosave_data
        raise FileNotFoundError(
            f"Could not find data.pt inside model directory: {model_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_arg}")

    if model_path.name == "data.pt":
        return model_path
    if model_path.name in {"generator.pt", "ik.pt"}:
        data_path = model_path.with_name("data.pt")
        if data_path.exists():
            return data_path
        raise FileNotFoundError(f"Could not find sibling data.pt for: {model_path}")

    raise ValueError(
        "Model path must be a model directory, data.pt, generator.pt, or ik.pt"
    )


def resolve_bvh_files(
    dataset_arg: str, split: str, max_files: int | None
) -> list[Path]:
    dataset_path = Path(dataset_arg)
    if dataset_path.is_file():
        if dataset_path.suffix.lower() != ".bvh":
            raise ValueError(f"Expected a .bvh file, got: {dataset_path}")
        return [dataset_path]

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_arg}")

    if split == "all":
        search_dirs = [
            path
            for path in [dataset_path / "train", dataset_path / "eval"]
            if path.is_dir()
        ]
        if not search_dirs:
            search_dirs = [dataset_path]
    elif split in {"train", "eval"}:
        split_dir = dataset_path / split
        if split_dir.is_dir():
            search_dirs = [split_dir]
        else:
            raise FileNotFoundError(
                f"Requested split directory does not exist: {split_dir}"
            )
    else:
        if (dataset_path / "train").is_dir():
            search_dirs = [dataset_path / "train"]
        else:
            search_dirs = [dataset_path]

    files = []
    for search_dir in search_dirs:
        files.extend(sorted(search_dir.glob("*.bvh")))

    if not files:
        raise FileNotFoundError(
            f"No .bvh files found under {[str(path) for path in search_dirs]}"
        )

    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def load_lma_annotations(bvh_path: Path):
    try:
        import pandas as pd
    except Exception:
        return None

    stem = bvh_path.stem
    parents_to_try = [
        bvh_path.parent,
        bvh_path.parent.parent,
        bvh_path.parent.parent.parent,
    ]

    for base in parents_to_try:
        if not base.exists():
            continue
        candidates = [
            base / "annotations" / "train" / f"{stem}.csv",
            base / "annotations" / "eval" / f"{stem}.csv",
            base / "annotations" / f"{stem}.csv",
        ]
        for annotation_path in candidates:
            if not annotation_path.exists():
                continue
            dataframe = pd.read_csv(annotation_path)
            numeric = dataframe.select_dtypes(include=["number"]).to_numpy()
            if numeric.size == 0:
                return None
            return numeric[:, :6] if numeric.shape[1] >= 6 else numeric
    return None


def to_cpu_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def build_dataset(bvh_files: list[Path], device: torch.device) -> TrainMotionData:
    dataset = TrainMotionData(train_vq_vae.param, train_vq_vae.scale, device)
    reference_parents = None

    for bvh_path in bvh_files:
        bvh = train_vq_vae.get_bvh_from_disk(str(bvh_path.parent), bvh_path.name)
        rots, pos, parents, offsets, _, og_rots = train_vq_vae.get_info_from_bvh(
            bvh,
            get_missing_frames=False,
        )

        if reference_parents is None:
            reference_parents = parents.copy()
        elif reference_parents != parents:
            raise ValueError(
                f"Skeleton parent structure mismatch for {bvh_path.name}; "
                "this script expects a consistent skeleton across the inspected dataset."
            )

        dataset.add_motion(
            offsets,
            pos[:, 0, :],
            rots,
            parents,
            bvh.compute_global_pos(),
            og_rots=og_rots,
            end_sites=bvh.data["end_sites"],
            end_sites_parents=bvh.data["end_sites_parents"],
            lma_features=load_lma_annotations(bvh_path),
        )

    return dataset


def extract_feature_from_motion(motion: dict, key: str):
    if key in motion:
        return motion[key]
    return motion.get("tags", {}).get(key)


def tensor_percentiles(
    tensor: torch.Tensor, percentiles=PERCENTILES
) -> dict[int, float]:
    array = tensor.detach().cpu().reshape(-1).float().numpy()
    if array.size == 0:
        return {percentile: 0.0 for percentile in percentiles}
    return {
        percentile: float(np.percentile(array, percentile))
        for percentile in percentiles
    }


def format_percentiles(
    label: str, tensor: torch.Tensor, percentiles=PERCENTILES
) -> str:
    stats = tensor_percentiles(tensor, percentiles=percentiles)
    parts = [f"p{percentile}={value:.6f}" for percentile, value in stats.items()]
    return f"{label}: " + ", ".join(parts)


def top_flat_indices(values: torch.Tensor, top_k: int):
    flat = values.reshape(-1)
    if flat.numel() == 0 or top_k <= 0:
        return []
    count = min(top_k, flat.numel())
    top_values, top_indices = torch.topk(flat, count)
    shape = tuple(values.shape)
    results = []
    for value, flat_index in zip(top_values.tolist(), top_indices.tolist()):
        unraveled = np.unravel_index(flat_index, shape)
        results.append((unraveled, float(value)))
    return results


def format_top_entries(label: str, values: torch.Tensor, top_k: int) -> str:
    entries = top_flat_indices(values, top_k)
    if not entries:
        return f"{label}: none"
    parts = [f"idx={index} val={value:.6f}" for index, value in entries]
    return f"{label}: " + " | ".join(parts)


class NormalizedAccumulator:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        mean = to_cpu_tensor(mean)
        std = to_cpu_tensor(std)
        std = std.clone()
        std[std.abs() < EPS] = 1.0

        self.mean_flat = mean.reshape(1, -1)
        self.std_flat = std.reshape(1, -1)
        self.sum = torch.zeros_like(self.mean_flat, dtype=torch.float64)
        self.sumsq = torch.zeros_like(self.mean_flat, dtype=torch.float64)
        self.prefix_count = 0
        self.abs_gt_3 = 0
        self.abs_gt_5 = 0
        self.total_values = 0

    def update(self, values):
        values_tensor = to_cpu_tensor(values).reshape(-1, self.mean_flat.shape[1])
        normalized = (values_tensor - self.mean_flat) / self.std_flat
        normalized64 = normalized.to(dtype=torch.float64)
        self.sum += normalized64.sum(dim=0, keepdim=True)
        self.sumsq += (normalized64 * normalized64).sum(dim=0, keepdim=True)
        self.prefix_count += normalized64.shape[0]
        abs_normalized = normalized64.abs()
        self.abs_gt_3 += int((abs_normalized > 3.0).sum().item())
        self.abs_gt_5 += int((abs_normalized > 5.0).sum().item())
        self.total_values += int(abs_normalized.numel())

    def finalize(self):
        if self.prefix_count == 0:
            zeros = torch.zeros_like(self.sum.squeeze(0), dtype=torch.float32)
            return {
                "channel_mean": zeros,
                "channel_std": zeros,
                "gt3_frac": 0.0,
                "gt5_frac": 0.0,
            }

        mean = (self.sum / float(self.prefix_count)).squeeze(0)
        var = (self.sumsq / float(self.prefix_count)).squeeze(0) - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        return {
            "channel_mean": mean.to(dtype=torch.float32),
            "channel_std": std.to(dtype=torch.float32),
            "gt3_frac": self.abs_gt_3 / max(1, self.total_values),
            "gt5_frac": self.abs_gt_5 / max(1, self.total_values),
        }


def normalized_stats_from_dataset(
    dataset: TrainMotionData,
    means: dict,
    stds: dict,
) -> dict[str, dict]:
    accumulators = {
        key: NormalizedAccumulator(means[key], stds[key])
        for key in means.keys()
        if key in stds
    }

    for motion in dataset.motions:
        for key, accumulator in accumulators.items():
            values = extract_feature_from_motion(motion, key)
            if values is None:
                continue
            accumulator.update(values)

    return {key: accumulator.finalize() for key, accumulator in accumulators.items()}


def summarize_feature(
    key: str,
    model_mean: torch.Tensor,
    model_std: torch.Tensor,
    fresh_mean: torch.Tensor,
    fresh_std: torch.Tensor,
    normalized_stats: dict,
    top_k: int,
):
    model_mean = to_cpu_tensor(model_mean)
    model_std = to_cpu_tensor(model_std)
    fresh_mean = to_cpu_tensor(fresh_mean)
    fresh_std = to_cpu_tensor(fresh_std)

    safe_fresh_std = fresh_std.clone()
    safe_fresh_std[safe_fresh_std.abs() < EPS] = 1.0
    std_ratio = model_std / safe_fresh_std
    abs_mean_delta = (fresh_mean - model_mean).abs()
    abs_channel_mean = normalized_stats["channel_mean"].abs()
    channel_std = normalized_stats["channel_std"]

    tiny_model_std = int((model_std < 1e-6).sum().item())
    tiny_fresh_std = int((fresh_std < 1e-6).sum().item())

    print(f"\n[{key}] shape={tuple(model_mean.shape)}")
    print(format_percentiles("  model |mean|", model_mean.abs()))
    print(format_percentiles("  model std", model_std))
    print(f"  model tiny std count: {tiny_model_std}/{model_std.numel()}")
    print(format_percentiles("  fresh |mean|", fresh_mean.abs()))
    print(format_percentiles("  fresh std", fresh_std))
    print(f"  fresh tiny std count: {tiny_fresh_std}/{fresh_std.numel()}")
    print(format_percentiles("  |fresh_mean - model_mean|", abs_mean_delta))
    print(format_percentiles("  model_std / fresh_std", std_ratio))
    print(format_percentiles("  |normalized channel mean|", abs_channel_mean))
    print(format_percentiles("  normalized channel std", channel_std))
    print(
        f"  normalized values beyond 3 sigma: {normalized_stats['gt3_frac'] * 100.0:.4f}%"
    )
    print(
        f"  normalized values beyond 5 sigma: {normalized_stats['gt5_frac'] * 100.0:.4f}%"
    )

    if top_k > 0:
        print(format_top_entries("  top |mean delta| dims", abs_mean_delta, top_k))
        print(
            format_top_entries(
                "  top |normalized mean| dims",
                abs_channel_mean,
                top_k,
            )
        )
        print(
            format_top_entries(
                "  top |normalized std-1| dims",
                (channel_std - 1.0).abs(),
                top_k,
            )
        )

    warnings = []
    if tiny_model_std > 0:
        warnings.append("checkpoint contains tiny std channels")
    if (
        key not in STATIC_FEATURE_KEYS
        and tensor_percentiles(abs_channel_mean)[95] > 0.25
    ):
        warnings.append("normalized mean is noticeably shifted")
    channel_std_percentiles = tensor_percentiles(channel_std)
    if key not in STATIC_FEATURE_KEYS and (
        channel_std_percentiles[5] < 0.5 or channel_std_percentiles[95] > 1.5
    ):
        warnings.append("normalized scale is materially off")
    if normalized_stats["gt5_frac"] > 0.01:
        warnings.append("heavy tails after normalization")

    if warnings:
        print("  warnings: " + "; ".join(warnings))
    else:
        print("  warnings: none")


def load_checkpoint_stats(data_path: Path):
    data = torch.load(data_path, map_location="cpu")
    means = data.get("means")
    stds = data.get("stds")
    if means is None or stds is None:
        raise KeyError(f"Expected means/stds inside checkpoint data file: {data_path}")
    return means, stds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether a model checkpoint's stored means/stds are sane for a given dataset."
        )
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Model directory, data.pt, generator.pt, or ik.pt",
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Dataset root, split folder, or single .bvh file",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "train", "eval", "all"],
        default="auto",
        help="Which dataset split to inspect when dataset_path is a dataset root",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for dataset feature construction",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit the number of BVH files scanned",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many worst offending dimensions to print per feature group",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_data_path(args.model_path)
    bvh_files = resolve_bvh_files(args.dataset_path, args.split, args.max_files)

    print(f"Using device: {device}")
    print(f"Checkpoint stats file: {data_path}")
    print(f"Inspecting {len(bvh_files)} BVH files")
    if bvh_files:
        print(f"First file: {bvh_files[0]}")
        print(f"Last file:  {bvh_files[-1]}")

    model_means, model_stds = load_checkpoint_stats(data_path)
    dataset = build_dataset(bvh_files, device)
    raw_motion_count = len(dataset.motions)
    dataset.normalize()

    print(f"Constructed {raw_motion_count} motion windows")
    print(f"Checkpoint stat groups: {len(model_means)}")
    print(f"Fresh stat groups: {len(dataset.means)}")

    shared_keys = [
        key
        for key in sorted(model_means.keys())
        if key in model_stds and key in dataset.means and key in dataset.stds
    ]
    only_in_checkpoint = sorted(set(model_means.keys()) - set(dataset.means.keys()))
    only_in_dataset = sorted(set(dataset.means.keys()) - set(model_means.keys()))
    shape_mismatches = []
    compatible_keys = []

    for key in shared_keys:
        checkpoint_shape = tuple(to_cpu_tensor(model_means[key]).shape)
        dataset_shape = tuple(to_cpu_tensor(dataset.means[key]).shape)
        if checkpoint_shape != dataset_shape:
            shape_mismatches.append((key, checkpoint_shape, dataset_shape))
            continue
        compatible_keys.append(key)

    print(f"Shared feature groups: {len(shared_keys)}")
    print(f"Shape-compatible groups: {len(compatible_keys)}")
    if only_in_checkpoint:
        print("Only in checkpoint stats: " + ", ".join(only_in_checkpoint))
    if only_in_dataset:
        print("Only in fresh dataset stats: " + ", ".join(only_in_dataset))
    if shape_mismatches:
        print("Shape mismatches:")
        for key, checkpoint_shape, dataset_shape in shape_mismatches:
            print(
                f"  - {key}: checkpoint {checkpoint_shape}, fresh dataset {dataset_shape}"
            )

    normalized_stats = normalized_stats_from_dataset(
        dataset,
        {key: model_means[key] for key in compatible_keys},
        {key: model_stds[key] for key in compatible_keys},
    )

    for key in compatible_keys:
        summarize_feature(
            key,
            model_means[key],
            model_stds[key],
            dataset.means[key],
            dataset.stds[key],
            normalized_stats[key],
            args.top_k,
        )

    print("\nInterpretation guide:")
    print(
        "  - Fresh means/stds come from rebuilding the dataset with the same feature pipeline as VAE training."
    )
    print(
        "  - If |normalized channel mean| stays near 0 and normalized channel std stays near 1, the checkpoint stats match this dataset well."
    )
    print(
        "  - Tiny checkpoint std counts are suspicious unless the feature is intentionally static, like offsets."
    )
    print(
        "  - Large mean deltas or large std-ratio drift suggest dataset mismatch, changed preprocessing, or stale checkpoint stats."
    )
    print(
        "  - High >5 sigma fractions indicate heavy tails or a bad scale fit for some channels."
    )


if __name__ == "__main__":
    main()
