"""
VAE posterior diagnostic script.

Usage examples:

python diagnose_vae_posterior.py \
    .\models\model_dif_sp_vae_DanceDB \
    .\data\DanceDB \
    --split eval

python diagnose_vae_posterior.py \
    .\models\model_dif_sp_vae_DanceDB\generator.pt \
    .\data\DanceDB\eval\Andria_Angry_v2_C3D_poses.bvh

Interpretation guide:

1. Posterior mean near zero is good.
   The VAE prior is standard normal, so a dataset-wide latent mean centered near 0
   usually means the posterior stays aligned with the prior.

2. Posterior mean standard deviation tells you whether the latent is active.
   If the mean std is extremely small everywhere, the posterior may be collapsing.
   If it is clearly above 0, the latent carries information.

3. Log-variance mean controls average uncertainty.
   A negative logvar mean means variance below 1. That is normal.
   Very large positive logvar means the posterior is too noisy.
   Very large negative logvar means the posterior is becoming too deterministic.

4. Clamp hit fractions are important.
   If many logvar values hit `vae_logvar_min`, the model wants even smaller variance
   than allowed, which often means a more deterministic posterior.
   If many values hit `vae_logvar_max`, the model wants much larger variance, which
   can indicate unstable or overly noisy uncertainty modeling.

5. KL percentiles show where the regularization cost really lives.
   Low overall KL with low mean-std can indicate collapse.
   High KL concentrated in a small number of latent dimensions means only a subset of
   dimensions are doing most of the work.
   Broadly distributed KL across dimensions means the latent load is more evenly shared.

6. Compare with reconstruction quality.
   These diagnostics do not tell you whether the VAE is good for your task by
   themselves. They must be read together with the actual eval metric.
   If reconstruction is much worse than the deterministic AE, the posterior may be
   healthy as a VAE but still too costly for a pure reconstruction objective.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path

import numpy as np
import torch

from motion_data import TestMotionData
from train_data import Train_Data
from generator_architecture import Generator_Model
import train_vq_vae


PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def resolve_model_path(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.is_dir():
        generator_path = model_path / "generator.pt"
        if generator_path.exists():
            return str(generator_path)
        raise FileNotFoundError(f"Could not find generator.pt inside {model_path}")

    if model_path.is_file():
        return str(model_path)

    raise FileNotFoundError(f"Model path does not exist: {model_arg}")


def resolve_bvh_files(dataset_arg: str, split: str | None) -> list[Path]:
    dataset_path = Path(dataset_arg)
    if dataset_path.is_file():
        if dataset_path.suffix.lower() != ".bvh":
            raise ValueError(f"Expected a .bvh file, got: {dataset_path}")
        return [dataset_path]

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_arg}")

    scan_dir = dataset_path
    if split:
        candidate = dataset_path / split
        if candidate.is_dir():
            scan_dir = candidate

    files = sorted(scan_dir.glob("*.bvh"))
    if not files:
        raise FileNotFoundError(f"No .bvh files found in: {scan_dir}")
    return files


def load_lma_annotations(bvh_path: Path):
    try:
        import pandas as pd
    except Exception:
        return None

    stem = bvh_path.stem
    parents_to_try = [bvh_path.parent, bvh_path.parent.parent, bvh_path.parent.parent.parent]

    for base in parents_to_try:
        if not base or not base.exists():
            continue
        candidates = [
            base / "annotations" / "eval" / f"{stem}.csv",
            base / "annotations" / "train" / f"{stem}.csv",
            base / "annotations" / f"{stem}.csv",
        ]
        for ann_path in candidates:
            if not ann_path.exists():
                continue
            df = pd.read_csv(ann_path)
            numeric = df.select_dtypes(include=["number"]).to_numpy()
            if numeric.size == 0:
                return None
            return numeric[:, :6] if numeric.shape[1] >= 6 else numeric

    return None


def build_eval_dataset(bvh_files: list[Path], device):
    eval_dataset = TestMotionData(train_vq_vae.param, train_vq_vae.scale, device)
    reference_parents = None

    for bvh_path in bvh_files:
        filename = bvh_path.name
        rots, pos, parents, offsets, bvh, og_rots = train_vq_vae.get_info_from_bvh(
            train_vq_vae.get_bvh_from_disk(str(bvh_path.parent), filename),
            incremental_rots=False,
            get_missing_frames=False,
        )

        if reference_parents is None:
            reference_parents = parents.copy()
        else:
            assert reference_parents == parents

        pos_all_joints = bvh.compute_global_pos()
        lma_features = load_lma_annotations(bvh_path)

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
            lma_features=lma_features,
        )

    return eval_dataset, reference_parents


def set_eval_sample(train_data, motion, bvh):
    device = train_data.device
    train_data.set_offsets(
        motion["offsets"].unsqueeze(0),
        motion["denorm_offsets"].unsqueeze(0),
    )
    train_data.set_end_sites(
        torch.tensor(motion["end_sites"], dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(motion["end_sites_parents"], dtype=torch.float32, device=device).unsqueeze(0),
    )
    train_data.set_motions(
        motion["dqs"].unsqueeze(0),
        motion["displacement"].unsqueeze(0),
    )
    train_data.set_rots(motion["rots"].unsqueeze(0))
    train_data.set_tags({
        key: value.clone().detach().unsqueeze(0)
        for key, value in motion["tags"].items()
    })
    train_data.set_rot_order(bvh.data["rot_order"])
    train_data.set_global_pos(torch.tensor(bvh.data["positions"][:, 0], dtype=torch.float32, device=device).unsqueeze(0))


def tensor_percentiles(values: torch.Tensor, percentiles: list[int]) -> dict[int, float]:
    array = values.detach().reshape(-1).float().cpu().numpy()
    return {percentile: float(np.percentile(array, percentile)) for percentile in percentiles}


def format_percentiles(name: str, stats: dict[int, float]) -> str:
    parts = [f"p{percentile}={value:.6f}" for percentile, value in stats.items()]
    return f"{name}: " + ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Diagnose VAE posterior statistics over a BVH dataset")
    parser.add_argument("model_path", type=str, help="Path to generator.pt or a model directory containing generator.pt")
    parser.add_argument("dataset_path", type=str, help="Path to a .bvh file, a folder of .bvh files, or a dataset root")
    parser.add_argument("--split", choices=["train", "eval"], default=None, help="Optional split subfolder to scan when dataset_path is a dataset root")
    parser.add_argument("--suppress-model-prints", action="store_true", help="Suppress debug prints emitted by the model forward pass")
    parser.add_argument("--top-k-dims", type=int, default=10, help="How many highest-KL latent dimensions to print")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model_path(args.model_path)
    bvh_files = resolve_bvh_files(args.dataset_path, args.split)

    print(f"Using device: {device}")
    print(f"Scanning {len(bvh_files)} BVH files")

    eval_dataset, reference_parents = build_eval_dataset(bvh_files, device)

    train_data = Train_Data(device, train_vq_vae.param)
    generator_model = Generator_Model(
        device,
        train_vq_vae.param,
        reference_parents,
        train_data,
        is_vq_vae=False,
    ).to(device)

    means, stds = train_vq_vae.load_model(generator_model, model_path, train_data, device)
    eval_dataset.set_means_stds(means, stds)
    eval_dataset.normalize()

    generator_model.eval()

    logvar_min = float(train_vq_vae.param.get("vae_logvar_min", -6.0))
    logvar_max = float(train_vq_vae.param.get("vae_logvar_max", 2.0))
    clamp_eps = 1e-5

    mu_all = []
    logvar_all = []
    kl_all = []
    dim_kl_sum = None
    dim_count = 0
    per_file_rows = []

    for index in range(eval_dataset.get_len()):
        motion = eval_dataset.get_item(index)
        bvh, filename = eval_dataset.get_bvh(index)
        set_eval_sample(train_data, motion, bvh)

        with torch.no_grad():
            if args.suppress_model_prints:
                with contextlib.redirect_stdout(io.StringIO()):
                    generator_model.forward()
            else:
                generator_model.forward()

        mu, logvar, _ = generator_model.autoencoder.get_variational_stats()
        if mu is None or logvar is None:
            raise RuntimeError("Model did not expose VAE posterior stats. Ensure use_vae=True and the VAE autoencoder module is loaded.")

        raw_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

        mu_cpu = mu.detach().cpu()
        logvar_cpu = logvar.detach().cpu()
        raw_kl_cpu = raw_kl.detach().cpu()

        mu_all.append(mu_cpu.reshape(-1))
        logvar_all.append(logvar_cpu.reshape(-1))
        kl_all.append(raw_kl_cpu.reshape(-1))

        per_dim_kl = raw_kl_cpu.mean(dim=(0, 1))
        if dim_kl_sum is None:
            dim_kl_sum = per_dim_kl.clone() * (mu_cpu.shape[0] * mu_cpu.shape[1])
        else:
            dim_kl_sum += per_dim_kl * (mu_cpu.shape[0] * mu_cpu.shape[1])
        dim_count += mu_cpu.shape[0] * mu_cpu.shape[1]

        hit_min_frac = float((logvar_cpu <= (logvar_min + clamp_eps)).float().mean().item())
        hit_max_frac = float((logvar_cpu >= (logvar_max - clamp_eps)).float().mean().item())

        per_file_rows.append({
            "filename": filename,
            "frames": int(motion["dqs"].shape[0]),
            "mu_mean": float(mu_cpu.mean().item()),
            "mu_std": float(mu_cpu.std().item()),
            "logvar_mean": float(logvar_cpu.mean().item()),
            "logvar_std": float(logvar_cpu.std().item()),
            "kl_mean": float(raw_kl_cpu.mean().item()),
            "logvar_hit_min_frac": hit_min_frac,
            "logvar_hit_max_frac": hit_max_frac,
        })

    mu_all = torch.cat(mu_all, dim=0)
    logvar_all = torch.cat(logvar_all, dim=0)
    kl_all = torch.cat(kl_all, dim=0)
    dim_mean_kl = dim_kl_sum / max(dim_count, 1)

    global_hit_min_frac = float((logvar_all <= (logvar_min + clamp_eps)).float().mean().item())
    global_hit_max_frac = float((logvar_all >= (logvar_max - clamp_eps)).float().mean().item())

    print()
    print("=== Aggregate posterior summary ===")
    print(f"mu mean={mu_all.mean().item():.6f}, mu std={mu_all.std().item():.6f}")
    print(f"logvar mean={logvar_all.mean().item():.6f}, logvar std={logvar_all.std().item():.6f}")
    print(f"raw KL mean={kl_all.mean().item():.6f}, raw KL std={kl_all.std().item():.6f}")
    print(f"logvar hit min fraction={global_hit_min_frac:.6%} at threshold {logvar_min}")
    print(f"logvar hit max fraction={global_hit_max_frac:.6%} at threshold {logvar_max}")
    print(format_percentiles("mu", tensor_percentiles(mu_all, PERCENTILES)))
    print(format_percentiles("logvar", tensor_percentiles(logvar_all, PERCENTILES)))
    print(format_percentiles("raw_kl", tensor_percentiles(kl_all, PERCENTILES)))
    print(format_percentiles("per_dim_mean_kl", tensor_percentiles(dim_mean_kl, PERCENTILES)))

    top_k = min(args.top_k_dims, int(dim_mean_kl.numel()))
    top_values, top_indices = torch.topk(dim_mean_kl, k=top_k)
    print(f"top {top_k} latent dimensions by mean KL:")
    for rank, (dim_index, value) in enumerate(zip(top_indices.tolist(), top_values.tolist()), start=1):
        print(f"  {rank:02d}. dim={dim_index} mean_kl={value:.6f}")

    print()
    print("=== Per-file summary ===")
    for row in per_file_rows:
        print(
            f"{row['filename']}: frames={row['frames']} "
            f"mu(mean/std)=({row['mu_mean']:.4f}/{row['mu_std']:.4f}) "
            f"logvar(mean/std)=({row['logvar_mean']:.4f}/{row['logvar_std']:.4f}) "
            f"kl_mean={row['kl_mean']:.4f} "
            f"hit_min={row['logvar_hit_min_frac']:.4%} "
            f"hit_max={row['logvar_hit_max_frac']:.4%}"
        )


if __name__ == "__main__":
    main()