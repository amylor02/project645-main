import os
import argparse
import torch
import torch.nn.functional as F
from motion_data import TestMotionData, compute_tags
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import eval_metrics
import numpy as np
import train_vq_vae
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as ortho6d_np
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from matplotlib.ticker import MaxNLocator
from train_vq_vae import plot_generated_motion_signals

# Evaluation Modes
GENERATOR = 1
IK = 2
use_quaternion_predictor = False
use_second_file = False
calc_phase = False
use_phase_predictor = False
foot_1_idx = 15
foot_2_idx = 18
use_incr_rots = False
lma_only = True
# Which LMA channels to take from file 2 (1=BODY, 2=EFFORT_WEIGHT_STRONG,
# 3=EFFORT_TIME_SUDDEN, 4=EFFORT_FLOW_BOUND, 5=SHAPE, 6=SPACE).
# Listed channels come from file 2; all others fall back to file 1's values.# Special value "mix": average file 1 and file 2 for all channels.# E.g. "1" keeps BODY from file 2, rest from file 1.
# Empty string "" uses all LMA from file 1.
lma_transfer_channels = "123456"


def pad_to_window(arr, window_size):
    """Pad a numpy array along axis 0 by repeating its last frame until
    the length is a multiple of window_size.  Matches the padding used in
    TrainMotionData / TestMotionData.add_motion."""
    n = arr.shape[0]
    if n == 0:
        return arr
    pad = (-n) % window_size
    if pad > 0:
        tail = np.tile(arr[-1:], (pad,) + (1,) * (arr.ndim - 1))
        arr = np.concatenate([arr, tail], axis=0)
    return arr


def to_float_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def align_time_length(value, target_len, device):
    tensor = to_float_tensor(value, device)
    if tensor.dim() == 0:
        return tensor.repeat(target_len)
    if tensor.size(0) == target_len:
        return tensor.clone()
    if tensor.size(0) == 0:
        return torch.zeros(
            (target_len, *tensor.shape[1:]), device=device, dtype=tensor.dtype
        )

    trailing_shape = tensor.shape[1:]
    flat = tensor.reshape(tensor.size(0), -1).transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(flat, size=target_len, mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1).reshape(target_len, *trailing_shape)


def visualize_contacts_for_bvh(bvh_path, bvh_filename, title=None):
    try:
        from visualizer import visualize_motion_and_tags

        eval_bvh = train_vq_vae.get_bvh_from_disk(bvh_path, bvh_filename)
        eval_rots, _, eval_parents, _, eval_bvh, _ = train_vq_vae.get_info_from_bvh(
            eval_bvh,
            incremental_rots=False,
            get_missing_frames=False,
        )
        rot_order = np.tile(
            ["y", "x", "z"], (eval_rots.shape[0], eval_rots.shape[1], 1)
        )
        rots_euler = quat.to_euler(eval_rots, rot_order)
        eval_pos = eval_bvh.compute_global_pos()
        tags = compute_tags(
            eval_pos,
            downsample_factor=8,
            rots=rots_euler[:, 0],
            is_deg=False,
            quats=eval_rots,
            skeleton_height=train_vq_vae.param["skeleton_height"],
            head_idx=train_vq_vae.param["head_idx"],
            head_height=train_vq_vae.param["head_height"],
            feet_idx=train_vq_vae.param["feet_idxs"],
            not_dog=train_vq_vae.param["not_dog"],
            feet_contact_threshold=train_vq_vae.param["feet_contact_threshold"],
            window_size=train_vq_vae.param["window_size"],
            parents=eval_parents,
            sparse_joints=train_vq_vae.param.get("sparse_joints"),
        )
        visualize_motion_and_tags(
            eval_pos,
            tags,
            parents=eval_parents,
            title=title,
        )
    except Exception as exc:
        label = title or bvh_filename
        print(f"[visualizer] Failed to visualize contacts for {label}: {exc}")


def visualize_requested_contacts(
    mode, input_dir, input_filename, eval_path, eval_filename
):
    if mode in {"original", "both"}:
        visualize_contacts_for_bvh(
            input_dir,
            input_filename,
            title=f"Original Contacts: {input_filename}",
        )

    if mode in {"generated", "both"} and eval_path is not None:
        visualize_contacts_for_bvh(
            eval_path,
            eval_filename,
            title=f"Generated Contacts: {eval_filename}",
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = TestMotionData(train_vq_vae.param, train_vq_vae.scale, device)

    # Load BVH
    filename = os.path.basename(args.input_path)
    dir = args.input_path[: -len(filename)]
    rots, pos, parents, offsets, bvh, og_rots = train_vq_vae.get_info_from_bvh(
        train_vq_vae.get_bvh_from_disk(dir, filename),
        incremental_rots=False,
        get_missing_frames=False,
    )
    root_rots_euler_y = bvh.data["rotations"][:, 0, 0]  # y x z
    sin = np.sin(np.deg2rad(root_rots_euler_y))
    cos = np.cos(np.deg2rad(root_rots_euler_y))
    sin_cos = np.concatenate([sin[:, None], cos[:, None]], axis=1)

    initial_frame = rots[:, 0, :]  # store for later
    initial_ortho = ortho6d_np.from_quat(initial_frame).reshape(
        initial_frame.shape[0], -1
    )
    # initial_frame = None
    # pos_all_joints = translation_each_joint(rots.copy(), pos[:,0,:].copy(), parents.copy(), offsets.copy())
    pos_all_joints = bvh.compute_global_pos()
    if use_incr_rots:
        rots = quat.compute_incremental_quaternions(rots)

    # try to load LMA annotations for this eval BVH (search from the data root) and pass them into add_motion
    lma_main = None
    try:
        from pathlib import Path
        import pandas as pd

        stem = Path(filename).stem
        data_root = Path(dir).parent
        ann_path = data_root / "annotations" / "eval" / (stem + ".csv")
        if not ann_path.exists():
            ann_path = data_root / "annotations" / "train" / (stem + ".csv")
        if not ann_path.exists():
            ann_path = data_root / "annotations" / (stem + ".csv")
        if ann_path.exists():
            df = pd.read_csv(ann_path)
            numeric = df.select_dtypes(include=["number"]).to_numpy()
            if numeric.size > 0:
                if numeric.shape[1] >= 6:
                    lma_main = numeric[:, :6]
                    print(f"[LMA] Loaded annotation for {filename}: {lma_main.shape}")
                else:
                    print(f"[LMA] No annotation found for {filename} - using zeros")
                    lma_main = numeric
    except Exception:
        lma_main = None
    ###############################################

    if args.input_path2:
        filename2 = os.path.basename(args.input_path2)
        dir2 = args.input_path2[: -len(filename2)]
        rots2, pos2, parents2, offsets2, bvh2, og_rots2 = (
            train_vq_vae.get_info_from_bvh(
                train_vq_vae.get_bvh_from_disk(dir2, filename2),
                incremental_rots=False,
            )
        )
        root_rots_euler_y = bvh2.data["rotations"][:, 0, 0]  # y x z
        sin = np.sin(np.deg2rad(root_rots_euler_y))
        cos = np.cos(np.deg2rad(root_rots_euler_y))
        sin_cos = np.concatenate([sin[:, None], cos[:, None]], axis=1)
        # rots2 = gaussian_filter1d(rots2, sigma=8.0, axis=0)
        initial_frame = rots2[0:, 0, :].copy()  # store for later
        initial_ortho = ortho6d_np.from_quat(initial_frame).reshape(
            initial_frame.shape[0], -1
        )
        rots2 = quat.compute_incremental_quaternions(rots2)
        root_rots_euler = bvh2.data["rotations"][:, 0, :]  # y x z
        pos2[0] = np.copy(pos2[1])

        # noise = np.random.randn(*rots.shape) * 0.05
        # rots = rots + noise

        # TestMotionData
        # eval_dataset2.set_means_stds(means, stds)
        # pos2[:,0,:] = gaussian_filter1d(pos2[:,0,:], sigma=4.0, axis=0)

        # pos_all_joints2 = translation_each_joint(rots2, pos2[:,0,:], parents2, offsets2)
        pos_all_joints2 = bvh2.compute_global_pos()
        min_len = min(rots.shape[0], rots2.shape[0])
        print("Original lengths:", rots.shape[0], rots2.shape[0])

        if min_len % train_vq_vae.param["window_size"] != 0:
            pad_size = (
                train_vq_vae.param["window_size"]
                - min_len % train_vq_vae.param["window_size"]
            )
            min_len += pad_size  # temporary solution, should be fixed in the future

        # pos_all_joints2[:,0,:]  = pos_all_joints2[:,0,:] * 5
        # pos_all_joints2[:,0,:1] /= 2
        scalar = 1
        param2 = train_vq_vae.param2
        # pos_all_joints2[:,0,0] -= pos_all_joints2[0,0,0]
        # pos_all_joints2[:,0,2] -= pos_all_joints2[0,0,2]
        # plt.plot(pos_all_joints2[:1000,0,0],pos_all_joints2[:1000,0,2])

        def plot_trajectory(
            ax, positions, n=None, smooth_sigma=3, cmap="viridis", step_arrow=40
        ):
            """
            positions: (F, J, 3) or (F, 3) - will use root joint at index 0 if 3D-per-joint
            """
            if positions.ndim == 3:
                root = positions[:, 0, :]  # (F,3)
            else:
                root = positions
            if n is None:
                n = root.shape[0]
            root = root[:n]

            # smooth X,Z optionally
            x = gaussian_filter1d(root[:, 0], sigma=smooth_sigma)
            z = gaussian_filter1d(root[:, 2], sigma=smooth_sigma)

            # line + color by time
            t = np.arange(len(x))
            points = np.stack([x, z], axis=1)
            ax.plot(x, z, color="0.2", lw=1.5, alpha=0.6)
            sc = ax.scatter(x, z, c=t, cmap=cmap, s=12, lw=0, alpha=0.9)
            cbar = plt.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label("frame")

            # start / end markers
            ax.scatter(x[0], z[0], color="green", s=80, marker="*", label="start")
            ax.scatter(x[-1], z[-1], color="red", s=60, marker="X", label="end")

            # # arrows showing motion direction (subsample)
            # if len(x) > step_arrow:
            #     diffs = np.stack([np.diff(x), np.diff(z)], axis=1)
            #     norms = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-6)
            #     dirs = diffs / norms
            #     idxs = np.arange(0, len(dirs), step_arrow)
            #     ax.quiver(x[idxs], z[idxs], dirs[idxs,0], dirs[idxs,1],
            #               angles='xy', scale_units='xy', scale=0.5, color='tab:blue', width=0.0008, alpha=0.9)

            ax.set_xlabel("X (world)")
            ax.set_ylabel("Z (world)")
            ax.set_title("Root trajectory (XZ plane)")
            # ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            ax.set_aspect("equal", adjustable="box")
            return ax

        # fig, ax = plt.subplots(figsize=(6,6))
        # plot_trajectory(ax, pos_all_joints2, n=1800, smooth_sigma=1, step_arrow=64)
        # plt.tight_layout()
        # plt.show()

        # plt.show()
        # try to load matching LMA annotations for the second file
        lma2 = None
        try:
            from pathlib import Path
            import pandas as pd

            data_root2 = Path(dir2).parent
            print(data_root2)
            ann_path = (
                data_root2 / "annotations" / "eval" / (Path(filename2).stem + ".csv")
            )
            if not ann_path.exists():
                ann_path = (
                    data_root2
                    / "annotations"
                    / "train"
                    / (Path(filename2).stem + ".csv")
                )
            if not ann_path.exists():
                ann_path = data_root2 / "annotations" / (Path(filename2).stem + ".csv")
            if ann_path.exists():
                df = pd.read_csv(ann_path)
                numeric = df.select_dtypes(include=["number"]).to_numpy()
                if numeric.size > 0:
                    if numeric.shape[1] >= 6:
                        lma2 = numeric[:, :6]
                        print(f"[LMA] Loaded annotation for {filename2}: {lma2.shape}")
                    else:
                        print(
                            f"[LMA] No annotation found for {filename2} - using zeros"
                        )
                        lma2 = numeric
        except Exception:
            lma2 = None

        print(min_len)
        tags2 = compute_tags(
            pos_all_joints2[:min_len],
            downsample_factor=8,
            is_human=True,
            skeleton_height=param2["skeleton_height"],
            head_idx=param2["head_idx"],
            head_height=param2["head_height"],
            rots=np.deg2rad(root_rots_euler[:min_len]),
            scalar=scalar,
            quats=rots2[:min_len],
            feet_idx=param2["feet_idxs"],
            shoulder_idx=[5, 9],
            not_dog=param2["not_dog"],
            feet_contact_threshold=param2["feet_contact_threshold"],
            window_size=param2["window_size"],
            lma_features=lma2,
            sparse_joints=param2.get("sparse_joints"),
        )

        for key in tags2:
            if not isinstance(tags2[key], torch.Tensor):
                tags2[key] = torch.tensor(tags2[key]).float().to(device)

        for _k, _v in tags2.items():
            _shape = (
                tuple(_v.shape)
                if isinstance(_v, (torch.Tensor, np.ndarray))
                else type(_v)
            )
            print(f"[tags2] {_k}: {_shape}")

        print("============")
        if lma_only is True:

            _ws = 64  # train_vq_vae.param["window_size"]
            _pos_main = pad_to_window(pos_all_joints, _ws)
            _quat_main = pad_to_window(rots, _ws)
            _euler_main = pad_to_window(
                bvh.data["rotations"][:, 0, :], _ws
            )  # (F, 3) y x z degrees
            tags_main = compute_tags(
                _pos_main,
                downsample_factor=8,
                rots=np.deg2rad(_euler_main),
                quats=_quat_main,
                skeleton_height=train_vq_vae.param["skeleton_height"],
                head_idx=train_vq_vae.param["head_idx"],
                head_height=train_vq_vae.param["head_height"],
                feet_idx=train_vq_vae.param["feet_idxs"],
                not_dog=train_vq_vae.param["not_dog"],
                feet_contact_threshold=train_vq_vae.param["feet_contact_threshold"],
                window_size=_ws,
                lma_features=lma_main,
                sparse_joints=train_vq_vae.param.get("sparse_joints"),
            )

            for _k, _v in tags_main.items():
                _shape = (
                    tuple(_v.shape)
                    if isinstance(_v, (torch.Tensor, np.ndarray))
                    else type(_v)
                )
                print(f"[tags_main] {_k}: {_shape}")

            # ===== selective LMA channel transfer =====
            # Frame-rate tags always come from file 1 because the evaluated
            # motion and decoder conditioning length are tied to file 1.
            # LMA tags can come from file 2, but are aligned to file 1's
            # LMA timeline so we don't keep partial tails from a longer clip.
            _LMA_CHANNEL_NAMES = [
                "BODY",  # 1
                "EFFORT_WEIGHT_STRONG",  # 2
                "EFFORT_TIME_SUDDEN",  # 3
                "EFFORT_FLOW_BOUND",  # 4
                "SHAPE",  # 5
                "SPACE",  # 6
            ]
            tags_main = {
                _k: to_float_tensor(_v, device) for _k, _v in tags_main.items()
            }
            tags_second_lma = {
                _k: to_float_tensor(_v, device)
                for _k, _v in tags2.items()
                if _k in _LMA_CHANNEL_NAMES
            }

            frame_target_len = next(
                (
                    tags_main[_k].shape[0]
                    for _k in tags_main
                    if _k not in _LMA_CHANNEL_NAMES
                ),
                None,
            )
            lma_target_len = next(
                (
                    tags_main[_k].shape[0]
                    for _k in _LMA_CHANNEL_NAMES
                    if _k in tags_main
                ),
                None,
            )
            if lma_target_len is None:
                lma_target_len = next(
                    (
                        tags_second_lma[_k].shape[0]
                        for _k in _LMA_CHANNEL_NAMES
                        if _k in tags_second_lma
                    ),
                    (
                        max(1, frame_target_len // 4)
                        if frame_target_len is not None
                        else 1
                    ),
                )

            tags2 = {
                _k: _v.clone()
                for _k, _v in tags_main.items()
                if _k not in _LMA_CHANNEL_NAMES
            }

            if lma_transfer_channels == "mix":
                for _lma_key in _LMA_CHANNEL_NAMES:
                    v_main = tags_main.get(_lma_key)
                    v_second = tags_second_lma.get(_lma_key)
                    if v_main is not None and v_second is not None:
                        tags2[_lma_key] = (
                            align_time_length(v_main, lma_target_len, device)
                            + align_time_length(v_second, lma_target_len, device)
                        ) * 0.5
                    elif v_second is not None:
                        tags2[_lma_key] = align_time_length(
                            v_second, lma_target_len, device
                        )
                    elif v_main is not None:
                        tags2[_lma_key] = align_time_length(
                            v_main, lma_target_len, device
                        )
            else:
                _selected = set(lma_transfer_channels)  # digits as strings
                for _i, _lma_key in enumerate(_LMA_CHANNEL_NAMES, start=1):
                    use_file2 = str(_i) in _selected and _lma_key in tags_second_lma
                    source = (
                        tags_second_lma.get(_lma_key)
                        if use_file2
                        else tags_main.get(_lma_key)
                    )
                    if source is None:
                        source = tags_second_lma.get(_lma_key)
                    if source is not None:
                        tags2[_lma_key] = align_time_length(
                            source, lma_target_len, device
                        )

            # ===== plot LMA: tags_main vs tags2 after assignment =====
            import matplotlib.pyplot as plt

            _n_lma = len(_LMA_CHANNEL_NAMES)
            _fig, _axes = plt.subplots(
                _n_lma, 1, figsize=(14, 2.5 * _n_lma), sharex=True
            )
            for _i, _lma_key in enumerate(_LMA_CHANNEL_NAMES):
                _ax = _axes[_i]
                if _lma_key in tags_main:
                    _v1 = tags_main[_lma_key]
                    _v1 = (
                        _v1.cpu().numpy()
                        if isinstance(_v1, torch.Tensor)
                        else np.asarray(_v1)
                    )
                    _ax.plot(
                        _v1.flatten(),
                        label=f"file1 ({_lma_key})",
                        color="steelblue",
                        linewidth=1.2,
                    )
                if _lma_key in tags2:
                    _v2 = tags2[_lma_key]
                    _v2 = (
                        _v2.cpu().numpy()
                        if isinstance(_v2, torch.Tensor)
                        else np.asarray(_v2)
                    )
                    _ax.plot(
                        _v2.flatten(),
                        label=f"file2 ({_lma_key})",
                        color="tomato",
                        linewidth=1.2,
                        linestyle="--",
                    )
                _ax.set_ylabel(_lma_key, fontsize=8)
                _ax.legend(loc="upper right", fontsize=7)
                _ax.grid(True, alpha=0.3)
            _axes[-1].set_xlabel("frame (downsampled)")
            _fig.suptitle(
                "LMA channels: file1 (tags_main) vs file2 (tags2) after assignment",
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()
            # ===== end LMA plot =====

        # ===== end lma_only =====

        min_len = min(rots.shape[0], rots2.shape[0])
        if not lma_only:
            rots[:min_len, 0, :] = rots2[:min_len, 0, :]
            og_rots[:min_len, 0, :] = og_rots2[:min_len, 0, :]

        # rots[:min_len,4] = rots2[:min_len,13] # head
        # rots[:min_len,8] = rots2[:min_len,17] # l hand
        # rots[:min_len,12] = rots2[:min_len,21]
        # rots[:min_len,15] = rots2[:min_len,4] # l foot
        # rots[:min_len,18] = rots2[:min_len,8]
        # pos[:min_len,0,:] = pos2[:min_len,0,:] * scalar
        # pos[:min_len,0,1] /= scalar
    ###############################################
    # initial_frame = None
    # if(foot_1_idx != -1):
    #     rots = quat.compute_incremental_quaternions_with_feet(rots,foot_1_idx,foot_2_idx)

    # else:
    # rots2 = quat.compute_incremental_quaternions(rots2)
    # for i in range(0,min_len):
    #     print(rots[i,0],rots2[i,0])

    # noise = np.random.randn(*rots.shape) * 0.1
    # rots = rots + noise

    # Create Models
    train_data = Train_Data(device, train_vq_vae.param)
    generator_model = Generator_Model(
        device, train_vq_vae.param, parents, train_data, is_vq_vae=True
    ).to(device)

    if args.eval_mode & IK != 0:
        ik_model = IK_Model(device, train_vq_vae.param, parents, train_data).to(device)

    # Load Models
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    ik_model_path = os.path.join(args.model_path, "ik.pt")

    means, stds = train_vq_vae.load_model(
        generator_model, generator_model_path, train_data, device
    )
    if args.eval_mode & IK != 0:
        means, stds = train_vq_vae.load_model(
            ik_model, ik_model_path, train_data, device
        )

    # codebook redundancy check!
    ######
    with torch.no_grad():
        import matplotlib.pyplot as plt

        W = generator_model.autoencoder.encoder.vq_codebooks[0].weight  # [K, D]

        # 1. Nearest-neighbor distances (cheap: O(K^2) but K=4096 is fine)
        # dists[i,j] = ||W[i] - W[j]||^2
        dists = torch.cdist(W, W)  # [K, K]
        # Zero the diagonal
        dists.fill_diagonal_(float("inf"))
        nn_dist, _ = dists.min(dim=1)  # [K] — dist to nearest neighbor

        print(
            f"NN dist: min={nn_dist.min():.4f}, median={nn_dist.median():.4f}, max={nn_dist.max():.4f}"
        )

        # 2. How many vectors are within epsilon of another?
        eps = nn_dist.median() * 0.1  # 10% of median as "near-duplicate" threshold
        near_dupes = (nn_dist < eps).sum().item()
        print(f"Near-duplicate entries (within {eps:.4f}): {near_dupes}/{W.shape[0]}")

        # 3. Histogram of NN distances
        plt.hist(nn_dist.cpu().numpy(), bins=50)
        plt.xlabel("Distance to nearest codebook neighbor")
        plt.title("Codebook redundancy check")
        plt.show()
    ######

    # TestMotionData
    eval_dataset.set_means_stds(means, stds)
    # pos_all_joints = translation_each_joint(rots, pos[:,0,:], parents, offsets)
    pos_all_joints = bvh.compute_global_pos()
    # pos_all_joints = to_root_space(rots, pos[:,0,:], parents, offsets)

    eval_dataset.add_motion(
        offsets,
        pos[:, 0, :],  # only global position
        rots,
        parents,
        bvh,
        filename,
        pos_all_joints,
        og_rots=og_rots,
        end_sites=bvh.data["end_sites"],
        end_sites_parents=bvh.data["end_sites_parents"],
        lma_features=lma_main,
    )

    if args.input_path2:
        eval_dataset.set_tags(tags2)
    eval_dataset.normalize()
    # phase = eval_dataset.get_phase()

    # if(use_second_file):
    # min_len = min(phase.shape[-1], phase2.shape[-1])
    # phase[:,:min_len] = phase2[:,:min_len]

    # train_data.set_phase(torch.tensor(phase).unsqueeze(0))
    # train_data.set_phase(torch.tensor(predicted_phase).unsqueeze(0))

    results = train_vq_vae.evaluate_generator(generator_model, train_data, eval_dataset)

    # results = results_modes["stochastic"]

    if args.eval_mode & IK != 0:
        results_ik = train_vq_vae.evaluate_ik(
            ik_model, results, train_data, eval_dataset
        )
        results = results_ik

    if use_second_file:
        bvh.data["positions"] = pos2[:min_len, :].copy()

    # initial_frame = rots[0,0,:] # for incremental rots

    # initial_frame = None
    # Save Result
    # plot_generated_motion_signals(results, means, stds)

    eval_path, eval_filename = train_vq_vae.result_to_bvh(
        results[0][0],
        means,
        stds,
        bvh,
        filename,
        save=True,
        initial_frame=initial_frame,
        feet_idx=None,
        copy_init_frame=False,
        initial_sin_cos=sin_cos,
        initial_ortho=initial_ortho,
    )

    if args.visualize_contacts:
        visualize_requested_contacts(
            args.visualize_contacts_source,
            dir,
            filename,
            eval_path,
            eval_filename,
        )

    # Evaluate Positional Error
    mpjpe, mpeepe = eval_metrics.eval_pos_error(
        train_vq_vae.get_bvh_from_disk(dir, filename),
        train_vq_vae.get_bvh_from_disk(eval_path, eval_filename),
        device,
    )

    print("Evaluate Loss: {}".format(mpjpe + mpeepe))
    print("Mean Per Joint Position Error: {}".format(mpjpe))
    print("Mean End Effector Position Error: {}".format(mpeepe))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Motion Upsampling Network")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to pytorch model folder",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="path to the input .bvh file",
    )
    if use_second_file:
        parser.add_argument(
            "input_path2",
            type=str,
            help="path to the input .bvh file",
        )
    parser.add_argument(
        "--input_path2",
        type=str,
        help="path to the input .bvh file 2",
    )
    parser.add_argument(
        "eval_mode",
        type=str.lower,
        choices=["generator", "ik"],
        help="evaluation mode",
    )
    parser.add_argument(
        "--visualize-contacts",
        action="store_true",
        help="launch the motion visualizer with foot-contact rows",
    )
    parser.add_argument(
        "--visualize-contacts-source",
        type=str.lower,
        choices=["generated", "original", "both"],
        default="generated",
        help="select whether contact visualization shows the generated result, the original input, or both",
    )
    args = parser.parse_args()
    if args.eval_mode == "generator":
        args.eval_mode = GENERATOR
    elif args.eval_mode == "ik":
        args.eval_mode = IK
    main(args)
