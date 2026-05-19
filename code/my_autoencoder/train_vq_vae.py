import copy
import random
import torch
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as ortho6d
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
import eval_metrics
from torch.utils.data.dataloader import DataLoader
from motion_data import (
    TestMotionData,
    TrainMotionData,
    ROOT_CHANNELS_ARE_GLOBAL_POSITIONS,
    USE_CANONICAL_XZ_POSITIONS,
    integrate_root_translation_np,
    split_motion_joints,
)
from pymotion.ops.skeleton import from_root_dual_quat
from pymotion.io.bvh import BVH
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model

scale = 1.0

# Train Modes
GENERATOR = 1
IK = 2

human_param = {
    "batch_size": 64,  # 1024, # 128 for vqvae, 64 for rnn
    "epochs": 1000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # left foot
        8,  # right foot
        # 11, # chest
        13,  # head
        17,  # left hand
        21,  # right hand
        # 22,  #dummy
    ],
    "window_size": 128,  # 256 for vqvae, 1024 for rnn
    "window_step": 16,  # 16 for vqvae, 256 for rnn
    "seed": 2222,
    "extra_joint": -11,
    "ema_updates": 74,  # 36,#205,
    "codebook_size": 2048,
    "ema_decay": 0.9,
    "skeleton_height": 0.96,
    "head_idx": 4,
    "head_height": 1.6,
    "feet_idxs": [42, 38],
    "feet_contact_threshold": 0.008,
    "not_dog": True,
    "root_branch_dim": 64,
    "gru_window": 8,
    "gru_layers": 2,
    "gru_hidden_dim": 512,
    "gru_vq_dim": 64,
    "training_stage": "ae",  # supported: ae, vq_vae, rnn
    "prior_gumbel_temperature": 1.0,
    "input_proj": -1,
    "root_invariant_encoder": True,
    "root_context_dim": 128,
    "p_drop_ctrl": 0.50,
    "p_drop_lma": 0.10,
    "p_drop_both": 0.05,
    "prior_diag_log_every": 100,
    "prior_inference_mode": "full",  # "full" or "lma_only" or "uncond"
    "enable_prior_root_override": True,
    "prior_root_loss_weight": 0.1,
    "autoencoder_module": "autoencoder_no_enc_9_no_vq",  # "autoencoder_no_enc_9_no_vq" or "autoencoder_no_enc_9_no_vq_multires"
    "use_vae": True,
    "vae_latent_dim": 504,
    "vae_hidden_dim": 504,
    "vae_posterior_dropout": 0.0,
    "vae_logvar_min": -6.0,
    "vae_logvar_max": 2.0,
    "vae_logvar_init_bias": -1.5,
    "vae_free_bits": 1e-4,
    "vae_kl_beta_start": 0.0,
    "vae_kl_beta": 1e-3,
    "vae_kl_warmup_steps": 10000,
    "vae_eval_sample": False,
    "synthetic_contact_joint_count": 1,
    "random_yaw_augmentation": True,
    "random_yaw_aug_max_degrees": 180.0,
    "root_velocity_loss_weight": 1.0,
    "root_position_xz_loss_weight": 1.0,
    "contact_aware_foot_sliding_loss_weight": 5.0,
    # ========== [AUG] Root branch augmentations toggles ==========
    "aug_extended_ctrl__": False,  # [AUG-A] include yaw_rate/yaw_accel in root ctrl
    "aug_split_heads": False,  # [AUG-B] split root rot/disp heads
    "aug_multires_vq": False,  # [AUG-C] inject VQ context at multiple resolutions
    "aug_vel_residual": False,  # [AUG-D] add velocity->disp residual shortcut
    "root_loss_epoch": 0,  # epoch at which root rot + disp loss is enabled
    "aug_extended_ctrl": True,
    "bvh_scale_factor": scale,
}


# dog params <-------
dog_param = {
    "batch_size": 128,
    "epochs": 150,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 1,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        4,  # head
        8,  # left hand
        12,  # right hand
        15,  # left foot
        18,  # right foot
        20,  # tail
    ],
    "window_size": 128,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 500,  # 51,
    "codebook_size": 512,
    "ema_decay": 0.9,
    "skeleton_height": 0.9,
    "head_idx": 2,
    "head_height": 1.0,
    "feet_idxs": [8, 12, 15, 18],
    "feet_contact_threshold": 0.07,
    "not_dog": False,
    "root_branch_dim": 32,
    "gru_window": 4,
    "gru_layers": 2,
    "gru_hidden_dim": 256,
    "gru_vq_dim": 64,
    "training_stage": "rnn",  # supported: vq_vae, rnn, rnn2
    "prior_gumbel_temperature": 1.0,
    "input_proj": -1,
    "root_invariant_encoder": True,
    "root_context_dim": 96,
    "p_drop_ctrl": 0.50,
    "p_drop_lma": 0.10,
    "p_drop_both": 0.05,
    "prior_diag_log_every": 100,
    "enable_prior_root_override": True,
    "prior_root_loss_weight": 1.0,
    "use_vae": False,
    "vae_latent_dim": 256,
    "vae_hidden_dim": 512,
    "vae_posterior_dropout": 0.0,
    "vae_logvar_min": -6.0,
    "vae_logvar_max": 2.0,
    "vae_logvar_init_bias": -1.5,
    "vae_free_bits": 1e-4,
    "vae_kl_beta_start": 0.0,
    "vae_kl_beta": 1e-3,
    "vae_kl_warmup_steps": 10000,
    "vae_eval_sample": False,
    "synthetic_contact_joint_count": 1,
    "random_yaw_augmentation": True,
    "random_yaw_aug_max_degrees": 180.0,
    # ========== [AUG] Root branch augmentations toggles ==========
    "aug_extended_ctrl": True,  # [AUG-A] include yaw_rate/yaw_accel in root ctrl
    "aug_split_heads": True,  # [AUG-B] split root rot/disp heads
    "aug_multires_vq": False,  # [AUG-C] inject VQ context at multiple resolutions
    "aug_vel_residual": False,  # [AUG-D] add velocity->disp residual shortcut
    "root_loss_epoch": 75,  # epoch at which root rot + disp loss is enabled
    "bvh_scale_factor": scale,
}

ostrich_param_ = {
    "batch_size": 32,
    "epochs": 4000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12,
        13,
        # 16,
        19,
        20,
        # 26,
        26,
        28,
        # 35,
        33,
        35,
        # 43,
        # 50,
        # 51,
        # 52,
        40,
        42,
    ],
    "window_size": 32,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 35,
    "codebook_size": 32,
    "ema_decay": 0.5,
    "root_invariant_encoder": True,
    "root_context_dim": 64,
    "p_drop_ctrl": 0.50,
    "p_drop_lma": 0.10,
    "p_drop_both": 0.05,
    "prior_diag_log_every": 100,
    "use_vae": False,
    "vae_latent_dim": 128,
    "vae_hidden_dim": 256,
    "vae_posterior_dropout": 0.0,
    "vae_logvar_min": -6.0,
    "vae_logvar_max": 2.0,
    "vae_logvar_init_bias": -1.5,
    "vae_free_bits": 1e-4,
    "vae_kl_beta_start": 0.0,
    "vae_kl_beta": 1e-3,
    "vae_kl_warmup_steps": 5000,
    "vae_eval_sample": False,
    # ========== [AUG] Root branch augmentations toggles ==========
    "aug_extended_ctrl": False,  # [AUG-A] include yaw_rate/yaw_accel in root ctrl
    "aug_split_heads": False,  # [AUG-B] split root rot/disp heads
    "aug_multires_vq": False,  # [AUG-C] inject VQ context at multiple resolutions
    "aug_vel_residual": False,  # [AUG-D] add velocity->disp residual shortcut
    "bvh_scale_factor": scale,
}

ostrich_param = {
    "batch_size": 32,
    "epochs": 300,
    "kernel_size_temporal_dim": 3,
    "neighbor_distance": 0,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-4,
    "lambda_root": 10,
    "lambda_ee": 10 / scale,
    "lambda_ee_reg": 1 / scale,
    "sparse_joints": [
        0,
        5,
        12,  #
        13,
        15,  #
        16,
        22,  #
        23,
        25,
        26,
        32,
        34,
        35,
        40,
        42,
        43,
        48,  #
        50,
        51,
        52,
        53,
    ],
    "window_size": 128,
    "window_step": 8,
    "seed": 2222,
    "extra_joint": -1,
    "ema_updates": 75,
    "codebook_size": 16,
    "ema_decay": 0.5,
    "skeleton_height": 0.69,
    "head_idx": 53,
    "head_height": 1.12,
    "feet_idxs": [20, 10, 20, 10],
    "feet_contact_threshold": 0.04,
    "not_dog": True,
    "root_branch_dim": 24,
    "gru_window": 1,
    "gru_layers": 1,
    "gru_hidden_dim": 16,
    "gru_vq_dim": 16,
    "input_proj": 32,
    "root_invariant_encoder": True,
    "root_context_dim": 64,
    "p_drop_ctrl": 0.50,
    "p_drop_lma": 0.10,
    "p_drop_both": 0.05,
    "prior_diag_log_every": 100,
    "use_vae": False,
    "vae_latent_dim": 128,
    "vae_hidden_dim": 256,
    "vae_posterior_dropout": 0.0,
    "vae_logvar_min": -6.0,
    "vae_logvar_max": 2.0,
    "vae_logvar_init_bias": -1.5,
    "vae_free_bits": 1e-4,
    "vae_kl_beta_start": 0.0,
    "vae_kl_beta": 1e-3,
    "vae_kl_warmup_steps": 5000,
    "vae_eval_sample": False,
    # ========== [AUG] Root branch augmentations toggles ==========
    "aug_extended_ctrl": False,  # [AUG-A] include yaw_rate/yaw_accel in root ctrl
    "aug_split_heads": False,  # [AUG-B] split root rot/disp heads
    "aug_multires_vq": False,  # [AUG-C] inject VQ context at multiple resolutions
    "aug_vel_residual": False,  # [AUG-D] add velocity->disp residual shortcut
    "bvh_scale_factor": scale,
}

param = human_param
rm_flag = False
frame_step = 1
param2 = human_param

assert param["kernel_size_temporal_dim"] % 2 == 1


def _validate_bvh_scale_factor(scale_factor):
    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError("bvh_scale_factor must be > 0")
    return scale_factor


def _apply_scale_to_param_dict(param_dict, scale_factor):
    if not isinstance(param_dict, dict):
        return
    scale_factor = _validate_bvh_scale_factor(scale_factor)
    param_dict["bvh_scale_factor"] = scale_factor
    param_dict["lambda_ee"] = 10 / scale_factor
    param_dict["lambda_ee_reg"] = 1 / scale_factor


def set_bvh_scale_factor(scale_factor):
    global scale
    scale = _validate_bvh_scale_factor(scale_factor)
    for param_dict in [
        human_param,
        dog_param,
        ostrich_param_,
        ostrich_param,
        param,
        param2,
    ]:
        _apply_scale_to_param_dict(param_dict, scale)
    return scale


def get_bvh_scale_factor(scale_factor=None):
    if scale_factor is None:
        return _validate_bvh_scale_factor(scale)
    return _validate_bvh_scale_factor(scale_factor)


def scale_bvh_data_arrays(positions, offsets, end_sites=None, scale_factor=None):
    scale_factor = get_bvh_scale_factor(scale_factor)
    positions = np.asarray(positions, dtype=np.float32).copy()
    offsets = np.asarray(offsets, dtype=np.float32).copy()
    positions *= scale_factor
    offsets *= scale_factor
    if end_sites is None:
        return positions, offsets, None
    end_sites = np.asarray(end_sites, dtype=np.float32).copy()
    end_sites *= scale_factor
    return positions, offsets, end_sites


def get_ctrl_loss_scalar(generator_model):
    ctrl_candidates = [getattr(generator_model, "ctrl_prediction_loss", None)]
    autoencoder = getattr(generator_model, "autoencoder", None)
    codebook_predictor = getattr(autoencoder, "codebook_predictor2", None)
    if codebook_predictor is not None:
        ctrl_candidates.append(
            getattr(codebook_predictor, "last_ctrl_prediction_loss", None)
        )

    for ctrl_val in ctrl_candidates:
        if isinstance(ctrl_val, torch.Tensor):
            return ctrl_val.detach().item()
        if isinstance(ctrl_val, (int, float)):
            return float(ctrl_val)

    return 0.0


def get_prior_diagnostics(generator_model):
    autoencoder = getattr(generator_model, "autoencoder", None)
    prior = getattr(autoencoder, "codebook_predictor2", None)
    if prior is None:
        return 0.0, 0.0, 0.0

    gate_param = getattr(prior, "lma_gate", None)
    if isinstance(gate_param, torch.Tensor):
        gate_sigmoid = torch.sigmoid(gate_param.detach()).mean().item()
    else:
        gate_sigmoid = 0.0

    ctrl_grad_norm = float(getattr(generator_model, "prior_ctrl_grad_norm", 0.0))
    lma_grad_norm = float(getattr(generator_model, "prior_lma_grad_norm", 0.0))
    return gate_sigmoid, ctrl_grad_norm, lma_grad_norm


def main(args):
    requested_scale_factor = getattr(args, "bvh_scale_factor", None)
    if requested_scale_factor is None:
        requested_scale_factor = param.get("bvh_scale_factor", scale)
    set_bvh_scale_factor(requested_scale_factor)

    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Prepare Data
    train_eval_dir = args.data_path

    # check if train and eval directories exist
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    train_files = os.listdir(train_dir)
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")
    eval_files = os.listdir(eval_dir)

    train_dataset = TrainMotionData(param, scale, device)
    eval_dataset = TestMotionData(param, scale, device)
    reference_parents = None  # used to make sure all bvh have the same structure

    # Train Files
    for filename in train_files:
        if filename[-4:] == ".bvh":

            bvh_from_disk = get_bvh_from_disk(train_dir, filename)
            rots, pos, parents, offsets, scaled_bvh, og_rots = get_info_from_bvh(
                bvh_from_disk, get_missing_frames=False
            )

            if reference_parents is None:
                reference_parents = parents.copy()
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            # Train Dataset
            pos_all_joints = scaled_bvh.compute_global_pos()

            # try to load LMA annotations (CSV) matching this BVH (first 6 numeric channels)
            lma_data = None
            try:
                from pathlib import Path
                import pandas as pd

                ann_path = (
                    Path(train_eval_dir)
                    / "annotations"
                    / "train"
                    / (Path(filename).stem + ".csv")
                )
                if not ann_path.exists():
                    # fall back to top-level annotations folder
                    ann_path = (
                        Path(train_eval_dir)
                        / "annotations"
                        / (Path(filename).stem + ".csv")
                    )
                if ann_path.exists():
                    df = pd.read_csv(ann_path)
                    # take numeric columns only and first 6 channels
                    numeric = df.select_dtypes(include=["number"]).to_numpy()
                    if numeric.size > 0:
                        if numeric.shape[1] >= 6:
                            lma_data = numeric[:, :6]
                            print(
                                f"[LMA] Loaded annotation for {filename}: {lma_data.shape}"
                            )
                        else:
                            print(
                                f"[LMA] No annotation found for {filename} - using zeros"
                            )  # <-- add this
                            lma_data = numeric
            except Exception:
                lma_data = None

            train_dataset.add_motion(
                offsets,
                pos[:, 0, :],  # only global position
                rots,
                parents,
                pos_all_joints,
                og_rots=og_rots,
                end_sites=scaled_bvh.data["end_sites"],
                end_sites_parents=scaled_bvh.data["end_sites_parents"],
                lma_features=lma_data,
            )
            chunk_size = 512
            num_frames = pos_all_joints.shape[0]

    # Once all train files are added, compute the means and stds and normalize
    train_dataset.normalize()
    eval_dataset.set_means_stds(train_dataset.means, train_dataset.stds)
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".bvh":
            rots, pos, parents, offsets, bvh, og_rots = get_info_from_bvh(
                get_bvh_from_disk(eval_dir, filename)
            )
            assert (
                reference_parents == parents
            )  # make sure all bvh have the same structure
            pos_all_joints = bvh.compute_global_pos()
            # Eval Dataset

            # try to load LMA annotation for eval file from annotations/eval
            lma_eval = None
            try:
                from pathlib import Path
                import pandas as pd

                ann_path = (
                    Path(train_eval_dir)
                    / "annotations"
                    / "eval"
                    / (Path(filename).stem + ".csv")
                )
                if not ann_path.exists():
                    ann_path = (
                        Path(train_eval_dir)
                        / "annotations"
                        / (Path(filename).stem + ".csv")
                    )
                if ann_path.exists():
                    df = pd.read_csv(ann_path)
                    numeric = df.select_dtypes(include=["number"]).to_numpy()
                    if numeric.size > 0:
                        lma_eval = numeric[:, :6] if numeric.shape[1] >= 6 else numeric
            except Exception:
                lma_eval = None

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
                lma_features=lma_eval,
            )
    # Once all eval files are added, normalize
    eval_dataset.normalize()
    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=True)
    param["ema_updates"] = (
        len(train_dataset) // param["batch_size"]
    )  # EMA update about once per epoch

    # Create Models
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(
        device, param, reference_parents, train_data, is_vq_vae=True
    ).to(device)

    if args.train_mode & IK != 0:
        ik_model = IK_Model(device, param, reference_parents, train_data).to(device)
    train_data.set_means(train_dataset.means["dqs"])
    train_data.set_stds(train_dataset.stds["dqs"])
    train_data.set_root_means_stds(
        train_dataset.means["rots"], train_dataset.stds["rots"]
    )
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
    train_data.set_sin_cos_means_stds(
        train_dataset.means["yaw_sin"],
        train_dataset.means["yaw_cos"],
        train_dataset.stds["yaw_sin"],
        train_dataset.stds["yaw_cos"],
    )

    # Load Models
    print(args, "\n", args.train_mode & IK)
    _, generator_path, ik_path = get_model_paths(args.name, train_eval_dir)

    # if args.train_mode & GENERATOR == 0 or (args.load and args.train_mode & IK != 0):
    if args.train_mode & GENERATOR == 0 or (args.load):
        print("loading pretrained model")
        # Generator is always needed with IK, load it if not training it
        load_model(generator_model, generator_path, train_data, device)

    if args.train_mode & IK != 0 and args.load:
        if os.path.exists(ik_path):
            load_model(ik_model, ik_path, train_data, device)
        else:
            print(
                "ik.pt not found under {}. Keeping the loaded generator checkpoint and initializing the IK stage from scratch.".format(
                    os.path.dirname(ik_path)
                )
            )

    if (args.train_mode & GENERATOR == 0 or args.train_mode & IK == 0) and args.load:
        # Check previous best evaluation loss
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik
        mpjpe, mpeepe = eval_save_result(
            results,
            train_dataset.means,
            train_dataset.stds,
            eval_dir,
            device,
            save=False,
        )
        best_evaluation = mpjpe + mpeepe
    else:
        best_evaluation = float("inf")

    # Training Loop
    start_time = time.time()
    global_step = 0
    prior_diag_log_every = max(1, int(param.get("prior_diag_log_every", 100)))

    for epoch in range(param["epochs"]):
        avg_train_loss = 0.0
        avg_kld_loss = 0.0
        avg_foot_sliding_loss = 0.0
        avg_yaw_aux_loss = 0.0
        avg_vel_loss = 0.0
        avg_acc_loss = 0.0
        avg_spectral_loss = 0.0
        avg_pos_xz_loss = 0.0
        avg_ctrl_loss = 0.0
        avg_prior_root_loss = 0.0

        if epoch == param.get("root_loss_epoch", param["epochs"]):
            generator_model.enable_root_loss = True
            print(f"Epoch {epoch}: root rotation + displacement loss enabled")

        for step, (denorm_motion, norm_motion) in enumerate(train_dataloader):
            global_step += 1
            # Forward
            train_data.set_offsets(norm_motion["offsets"], denorm_motion["offsets"])
            train_data.set_rot_order(bvh.data["rot_order"])
            train_data.set_end_sites(
                denorm_motion["end_sites"], denorm_motion["end_sites_parents"]
            )
            train_data.set_motions(
                norm_motion["dqs"],
                norm_motion["displacement"],
                # norm_motion["disp_8"],
                # norm_motion["tags"]["sin_diff"],
                # norm_motion["tags"]["cos_diff"],
                # norm_motion["loss_weights"],
            )
            # train_data.set_phase(denorm_motion["phase"])
            # train_data.set_phase_per_8_frames(denorm_motion["phase_per_8_frames"])
            # train_data.set_velocity_per_8_frames(denorm_motion["velocity_per_8_frames"])
            train_data.set_tags(norm_motion["tags"])
            train_data.set_rots(norm_motion["rots"])
            train_data.set_global_pos(denorm_motion["global_pos"])
            if "foot_positions" in denorm_motion:
                train_data.set_foot_positions(denorm_motion["foot_positions"])
            else:
                train_data.foot_positions = None
            train_data.apply_random_yaw_augmentation()
            # train_data.set_energy(denorm_motion["energy_feet"].clone().detach().to(device, dtype=torch.float32))

            if args.train_mode & GENERATOR != 0:
                generator_model.train()
            if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                res_decoder = generator_model.forward()
            if args.train_mode & IK != 0:
                ik_model.train()
                ik_model.forward(res_decoder)
            # Loss
            loss = 0.0
            kld_loss = 0.0
            kld_loss = 0.0
            foot_sliding_loss = 0.0
            yaw_aux_loss = 0.0
            vel_loss = 0.0
            acc_loss = 0.0
            spectral_loss = 0.0
            pos_xz_loss = 0.0
            ctrl_loss = 0.0
            prior_root_loss = 0.0
            if args.train_mode & GENERATOR != 0:

                (
                    loss_generator,
                    kld_loss,
                    foot_sliding_loss,
                    yaw_aux_loss,
                    vel_loss,
                    acc_loss,
                    spectral_loss,
                    pos_xz_loss,
                    prior_root_loss,
                ) = generator_model.optimize_parameters_vq_vae()
                loss = loss_generator.item()
                kld_loss = kld_loss.item()
                foot_sliding_loss = (
                    foot_sliding_loss.item() if foot_sliding_loss is not None else 0.0
                )
                yaw_aux_loss = yaw_aux_loss.item()
                vel_loss = vel_loss.item()
                acc_loss = acc_loss.item()
                spectral_loss = spectral_loss.item()
                pos_xz_loss = pos_xz_loss.item() if pos_xz_loss is not None else 0.0
                prior_root_loss = (
                    prior_root_loss.item() if prior_root_loss is not None else 0.0
                )
                # collect ctrl prediction loss (from prior / autoencoder)
                ctrl_loss = get_ctrl_loss_scalar(generator_model)
                if (
                    generator_model.training_stage == "rnn"
                    and global_step % prior_diag_log_every == 0
                ):
                    gate_sigmoid, ctrl_grad_norm, lma_grad_norm = get_prior_diagnostics(
                        generator_model
                    )
                    print(
                        "Step: {} - gate(sigmoid): {:.4f} - ctrl grad: {:.4f} - lma grad: {:.4f}".format(
                            global_step,
                            gate_sigmoid,
                            ctrl_grad_norm,
                            lma_grad_norm,
                        )
                    )

            if args.train_mode & IK != 0:
                loss_ik = ik_model.optimize_parameters(res_decoder)
                loss += loss_ik.item()

            avg_train_loss += loss
            avg_kld_loss += kld_loss
            avg_foot_sliding_loss += foot_sliding_loss
            avg_yaw_aux_loss += yaw_aux_loss
            avg_vel_loss += vel_loss
            avg_acc_loss += acc_loss
            avg_spectral_loss += spectral_loss
            avg_pos_xz_loss += pos_xz_loss
            avg_ctrl_loss += ctrl_loss
            avg_prior_root_loss += prior_root_loss
            # Evaluate & Print
            if step == len(train_dataloader) - 1:
                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    results = evaluate_generator(
                        generator_model, train_data, eval_dataset
                    )

                    if args.train_mode & IK != 0:
                        results_ik = evaluate_ik(
                            ik_model,
                            results,
                            train_data,
                            eval_dataset,
                        )
                        results = results_ik
                    if epoch % 50 == 0:
                        save = True
                    else:
                        save = False
                    mpjpe, mpeepe = eval_save_result(
                        results,
                        train_dataset.means,
                        train_dataset.stds,
                        eval_dir,
                        device,
                        save=save,
                    )
                    evaluation_loss = mpjpe + mpeepe
                # If best, save model
                was_best = False
                if evaluation_loss < best_evaluation:
                    save_model(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )
                    best_evaluation = evaluation_loss
                    was_best = True

                elif epoch % 10 == 0 and epoch != 0:
                    save_model_shared(
                        generator_model if args.train_mode & GENERATOR != 0 else None,
                        ik_model if args.train_mode & IK != 0 else None,
                        train_dataset,
                        args.name,
                        train_eval_dir,
                    )

                # Print
                avg_train_loss /= len(train_dataloader)
                avg_kld_loss /= len(train_dataloader)
                avg_foot_sliding_loss /= len(train_dataloader)
                avg_yaw_aux_loss /= len(train_dataloader)
                avg_vel_loss /= len(train_dataloader)
                avg_acc_loss /= len(train_dataloader)
                avg_spectral_loss /= len(train_dataloader)
                avg_pos_xz_loss /= len(train_dataloader)
                avg_ctrl_loss /= len(train_dataloader)
                avg_prior_root_loss /= len(train_dataloader)

                if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
                    prior = getattr(
                        generator_model.autoencoder, "codebook_predictor2", None
                    )
                    if prior is not None:
                        anneal_teacher_forcing = getattr(
                            prior, "anneal_teacher_forcing", None
                        )
                        if callable(anneal_teacher_forcing):
                            anneal_teacher_forcing(factor=0.996)
                        tf_prob = float(getattr(prior, "teacher_forcing_prob", 0.0))
                    else:
                        tf_prob = 0.0
                    print(
                        "Epoch: {} - Loss: {:.4f} - KLD: {:.14f} - foot_slide: {:.4f} - yaw_aux: {:.4f} - vel: {:.4f} - acc: {:.4f} - root_xz: {:.4f} - spec: {:.4f} - ctrl: {:.4f} - prior_root: {:.4f} - beta: {:.5f} - TF: {:.3f} - Eval: {:.4f} ".format(
                            epoch,
                            avg_train_loss,
                            avg_kld_loss,
                            avg_foot_sliding_loss,
                            avg_yaw_aux_loss,
                            avg_vel_loss,
                            avg_acc_loss,
                            avg_pos_xz_loss,
                            avg_spectral_loss,
                            avg_ctrl_loss,
                            avg_prior_root_loss,
                            generator_model.current_kl_beta,
                            tf_prob,
                            evaluation_loss,
                        )
                        + ("*" if was_best else "")
                    )

    end_time = time.time()
    print("Training Time:", end_time - start_time)

    # Load Best Model -> Save and/or Evaluate
    if args.train_mode & GENERATOR != 0 or args.train_mode & IK != 0:
        load_model(generator_model, generator_path, train_data, device)
        results = evaluate_generator(generator_model, train_data, eval_dataset)
        if args.train_mode & IK != 0:
            load_model(ik_model, ik_path, train_data, device)
            results_ik = evaluate_ik(ik_model, results, train_data, eval_dataset)
            results = results_ik

        mpjpe, mpeepe = eval_save_result(
            results, train_dataset.means, train_dataset.stds, eval_dir, device
        )
        evaluation_loss = mpjpe + mpeepe

    print("Evaluate Loss: {}".format(evaluation_loss))
    if args.train_mode & (GENERATOR | IK) != 0:
        print("Mean Per Joint Position Error: {}".format(mpjpe))
        print("Mean End Effector Position Error: {}".format(mpeepe))


def plot_generated_motion_signals(results, means, stds):
    """Denormalize generated results and plot velocity, acceleration, height and yaw rate."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    axes[0].set_title("Root XZ Velocity (m/frame)")
    axes[1].set_title("Root XZ Acceleration (m/frame²)")
    axes[2].set_title("Root Height Y (m)")
    axes[3].set_title("Yaw Rate (rad/frame)")

    mean_dqs = means["dqs"].cpu().numpy()
    std_dqs = stds["dqs"].cpu().numpy()

    for step, (res, bvh, filename) in enumerate(results):
        # Denormalize: [1, J*9, F] -> [F, J*9]
        r = res.permute(0, 2, 1).flatten(0, 1).cpu().detach().numpy()
        dqs_dn = r * std_dqs + mean_dqs
        dqs_dn = dqs_dn.reshape(dqs_dn.shape[0], dqs_dn.shape[1] // 9, 9)  # [F, J, 9]

        # Root displacement: channels 6-8 are (dx, y_height, dz)
        disp = dqs_dn[:, 0, 6:9]
        dx = disp[:, 0]
        height = disp[:, 1]
        dz = disp[:, 2]

        velocity = np.sqrt(dx**2 + dz**2)
        acceleration = np.diff(velocity, prepend=velocity[0])

        # Yaw rate from ortho6d root rotation (channels 0-5)
        o6d = dqs_dn[:, 0, :6]
        a1 = o6d[:, :3]
        a2 = o6d[:, 3:6]
        b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
        b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
        b3 = np.cross(b1, b2)  # forward direction
        yaw = np.arctan2(b3[:, 0], b3[:, 2])
        yaw = np.unwrap(yaw)
        yaw_rate = np.diff(yaw, prepend=yaw[0])

        label = os.path.splitext(filename)[0] if filename else f"clip_{step}"
        frames = np.arange(len(velocity))
        axes[0].plot(frames, velocity, alpha=0.8, label=label)
        axes[1].plot(frames, acceleration, alpha=0.8, label=label)
        axes[2].plot(frames, height, alpha=0.8, label=label)
        axes[3].plot(frames, yaw_rate, alpha=0.8, label=label)

    for ax in axes:
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def eval_save_result(
    results,
    train_means,
    train_stds,
    eval_dir,
    device,
    save=True,
):

    # Save Result
    array_mpjpe = np.empty((len(results),))
    array_mpeepe = np.empty((len(results),))
    for step, (res, bvh, filename) in enumerate(results):
        if save:
            eval_bvh = copy.deepcopy(bvh)
            eval_path, eval_filename = result_to_bvh(
                res, train_means, train_stds, eval_bvh, filename
            )
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                get_bvh_from_disk(eval_path, eval_filename),
                device,
            )
        else:
            eval_bvh = copy.deepcopy(bvh)
            result_to_bvh(res, train_means, train_stds, eval_bvh, None, save=False)
            # Evaluate Positional Error
            mpjpe, mpeepe = eval_metrics.eval_pos_error(
                get_bvh_from_disk(eval_dir, filename),
                eval_bvh,
                device,
            )

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe

    return np.mean(array_mpjpe), np.mean(array_mpeepe)


def load_model(
    model,
    model_path,
    train_data,
    device,
    ignore_transform_net=True,
    return_incompatible_keys=False,
):
    model_name = os.path.basename(model_path)[: -len(".pt")]
    assert model_name == "generator" or model_name == "ik"
    if model_name == "generator":
        data_path = model_path[: -len("generator.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # dont load transform net
        # if ignore_transform_net:
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.lma')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor2.rot')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.codebook_predictor2.yaw')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.root_branch')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.ema')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_embeddings')}
        #     state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.logits')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.encoder.num_quantizers')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.input_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.forward_dir_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_branch')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.root_upsample')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.ctrl_and_input_proj')}
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith('autoencoder.decoder.final_cnn_layer')}
        # state_dict = {k: v for k, v in state_dict.items() if (not 'timing_predictor' in k)}
        # state_dict = {k: v for k, v in state_dict.items() if (not 'timing_decoder' in k)}
        incompatible = model.load_state_dict(state_dict, strict=False)

    elif model_name == "ik":
        data_path = model_path[: -len("ik.pt")] + "data.pt"
        checkpoint = torch.load(model_path, map_location=device)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"])
    data = torch.load(data_path, map_location=device)
    means = data["means"]
    stds = data["stds"]
    saved_bvh_scale_factor = float(data.get("bvh_scale_factor", 1.0))
    set_bvh_scale_factor(saved_bvh_scale_factor)
    if hasattr(train_data, "bvh_scale_factor"):
        train_data.bvh_scale_factor = saved_bvh_scale_factor
    if hasattr(model, "param") and isinstance(model.param, dict):
        model.param["bvh_scale_factor"] = saved_bvh_scale_factor
    train_data.set_means(means["dqs"])
    train_data.set_stds(stds["dqs"])
    train_data.set_root_means_stds(means["rots"], stds["rots"])
    train_data.set_sin_cos_means_stds(
        means["yaw_sin"], means["yaw_cos"], stds["yaw_sin"], stds["yaw_cos"]
    )
    if return_incompatible_keys:
        return means, stds, incompatible.missing_keys, incompatible.unexpected_keys
    return means, stds


def get_model_paths(name, train_eval_dir):
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_path = os.path.join(model_dir, "data.pt")
    generator_path = os.path.join(model_dir, "generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path


def get_model_paths_shared(
    name, train_eval_dir
):  # possible delete, no more like rename
    model_name = (
        "model_" + name + "_" + os.path.basename(os.path.normpath(train_eval_dir))
    )
    model_dir = os.path.join("models", model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # ensure the `best_root` subdirectory exists so saving won't fail
    autosave_dir = os.path.join(model_dir, "autosave")
    if not os.path.exists(autosave_dir):
        os.makedirs(autosave_dir)

    data_path = os.path.join(autosave_dir, "data.pt")
    generator_path = os.path.join(autosave_dir, "generator.pt")
    ik_path = os.path.join(model_dir, "ik.pt")
    return data_path, generator_path, ik_path


def save_model(
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths(name, train_eval_dir)
    model_param = None
    if generator_model is not None and hasattr(generator_model, "param"):
        model_param = copy.deepcopy(generator_model.param)

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
                "bvh_scale_factor": float(getattr(train_dataset, "scale", scale)),
                "model_param": model_param,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
                "model_param": model_param,
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )


def save_model_shared(  # possible delete, no more like rename
    generator_model,
    ik_model,
    train_dataset,
    name,
    train_eval_dir,
):
    data_path, generator_path, ik_path = get_model_paths_shared(name, train_eval_dir)
    # generator_path = os.path.join(generator_path,"best_root")
    model_param = None
    if generator_model is not None and hasattr(generator_model, "param"):
        model_param = copy.deepcopy(generator_model.param)

    if train_dataset is not None:
        torch.save(
            {
                "means": train_dataset.means,
                "stds": train_dataset.stds,
                "bvh_scale_factor": float(getattr(train_dataset, "scale", scale)),
                "model_param": model_param,
            },
            data_path,
        )
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
                "model_param": model_param,
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )


def get_bvh_from_disk(path, filename):
    path = os.path.join(path, filename)
    bvh = BVH()
    bvh.load(path)
    return bvh


def get_info_from_bvh(
    bvh,
    incremental_rots=False,
    get_missing_frames=False,
    get_phase=False,
    bvh_scale_factor=None,
):
    bvh_scale_factor = get_bvh_scale_factor(bvh_scale_factor)
    bvh = copy.deepcopy(bvh)
    rot_roder = np.tile(
        bvh.data["rot_order"][0],
        (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1),
    )  # made a change here
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    og_rots = rots.copy()

    # rots = quat.normalize(rots)
    pos, offsets, end_sites = scale_bvh_data_arrays(
        bvh.data["positions"],
        bvh.data["offsets"],
        bvh.data.get("end_sites"),
        scale_factor=bvh_scale_factor,
    )
    bvh.data["positions"] = pos
    bvh.data["offsets"] = offsets
    if end_sites is not None:
        bvh.data["end_sites"] = end_sites

    parents = list(bvh.data["parents"])
    parents[0] = 0  # BVH sets root as None
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    bvh.data["offsets"][0] = np.zeros(3, dtype=np.float32)

    return rots, pos, parents, offsets, bvh, og_rots


def evaluate_generator(
    generator_model,
    train_data,
    dataset,
    sparse_motions=None,
):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    generator_model.eval()
    results = []
    yaw_diffs = []
    disps = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_end_sites(
                torch.tensor(norm_motion["end_sites"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0),
                torch.tensor(norm_motion["end_sites_parents"], dtype=torch.float32)
                .to("cuda")
                .unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
            )
            train_data.set_rots(
                norm_motion["rots"].unsqueeze(0),
            )

            tags_tensor_dict = {
                key: value.clone()
                .detach()
                .unsqueeze(
                    0
                )  # torch.tensor(value, dtype=torch.float32).to('cuda').unsqueeze(0)
                for key, value in norm_motion["tags"].items()
            }
            train_data.set_tags(tags_tensor_dict)

            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            bvh, filename = dataset.get_bvh(index)
            train_data.set_rot_order(bvh.data["rot_order"])
            train_data.set_global_pos(
                torch.tensor(bvh.data["positions"][:, 0]).to("cuda").unsqueeze(0)
            )
            res = generator_model.forward()
            results.append((res, bvh, filename))

    return results


def evaluate_ik(ik_model, results_decoder, train_data, dataset):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    ik_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
                norm_motion["denorm_offsets"].unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["displacement"].unsqueeze(0),
            )
            res = ik_model.forward(results_decoder[index][0])
            bvh, filename = dataset.get_bvh(index)
            results.append((res, bvh, filename))
    return results


def run_set_data(train_data, dataset):
    with torch.no_grad():
        norm_motion = dataset.get_item()
        train_data.set_offsets(
            norm_motion["offsets"].unsqueeze(0),
            norm_motion["denorm_offsets"].unsqueeze(0),
        )
        train_data.set_motions(
            norm_motion["dqs"].unsqueeze(0),
            norm_motion["displacement"].unsqueeze(0),
            norm_motion["disp_8"].unsqueeze(0),
            norm_motion["tags"]["sin_diff"].unsqueeze(0),
            norm_motion["tags"]["cos_diff"].unsqueeze(0),
            norm_motion["loss_weights"].unsqueeze(0),
        )


def run_generator(model):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res_decoder = model.forward()
    return res_decoder


def run_ik(model, res_decoder, frame=None):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    model.eval()
    with torch.no_grad():
        res = model.forward(res_decoder, frame)
    return res


def _build_bvh_positions_buffer(bvh, frame_count):
    source_positions = bvh.data["positions"]
    if source_positions.shape[0] == 0:
        raise ValueError("BVH positions buffer is empty")
    if source_positions.shape[0] >= frame_count:
        return source_positions[:frame_count].copy()
    return np.repeat(source_positions[:1], frame_count, axis=0)


def result_to_bvh(
    res,
    means,
    stds,
    bvh,
    filename,
    save=True,
    initial_frame=None,
    feet_idx=None,
    copy_init_frame=False,
    initial_sin_cos=None,
    initial_ortho=None,
    output_dir=None,
    filename_prefix="eval_",
    bvh_scale_factor=None,
):
    bvh_scale_factor = get_bvh_scale_factor(bvh_scale_factor)

    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    frames = res.shape[0]
    dqs = res
    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
    dqs = dqs.reshape(dqs.shape[0], dqs.shape[1] // 9, 9)  # frames, n_joints, 9
    skeletal_dqs, _ = split_motion_joints(
        dqs,
        synthetic_joint_count=(
            int(means.get("synthetic_contact_joint_count", 1))
            if isinstance(means, dict)
            else 1
        ),
    )
    pred_positions = integrate_root_translation_np(
        skeletal_dqs[:, 0, :],
        bvh.data["positions"][:, 0, :3],
    )
    dqs = ortho6d.to_dual_quat(skeletal_dqs).reshape(skeletal_dqs.shape[0], -1, 8)
    _, rots = from_root_dual_quat(dqs, np.array(bvh.data["parents"]))

    if copy_init_frame:
        min_len = min(rots.shape[0], initial_frame.shape[0])
        rots[:min_len, 0, :] = initial_frame[:min_len, :]
        # rots[:,0,0] = yaw + rots[0,0,0]
        # rots[0,0,:] = initial_frame[0, :]

    rot_roder = np.tile(bvh.data["rot_order"][0], (rots.shape[0], rots.shape[1], 1))
    rotations = bvh.to_degrees(quat.to_euler(rots, order=rot_roder))
    rot_roder = np.tile(
        bvh.data["rot_order"][0],
        (bvh.data["rotations"].shape[0], bvh.data["rotations"].shape[1], 1),
    )  # made a change here

    bvh.data["rotations"] = rotations

    # positions
    positions = _build_bvh_positions_buffer(bvh, rotations.shape[0])

    # Get y-rotation angles (assuming these represent the character's facing direction)

    bvh.data["positions"] = positions
    bvh.data["positions"][:, 0] = pred_positions[: positions.shape[0]]
    if bvh_scale_factor != 1.0:
        bvh.data["positions"] = bvh.data["positions"] / bvh_scale_factor
        bvh.data["offsets"] = bvh.data["offsets"] / bvh_scale_factor
        if "end_sites" in bvh.data and bvh.data["end_sites"] is not None:
            bvh.data["end_sites"] = bvh.data["end_sites"] / bvh_scale_factor
    bvh.data["parents"][0] = None  # BVH sets root as None
    path = None

    if save:
        path = output_dir if output_dir is not None else "data"
        filename = filename_prefix + filename
        if not os.path.exists(path):
            os.makedirs(path)
        bvh.save(os.path.join(path, filename))
    return path, filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )
    parser.add_argument(
        "name",
        type=str,
        help="name of the experiment, used to save the model and the logs",
    )
    parser.add_argument(
        "train_mode",
        type=str.lower,
        choices=["generator", "ik", "all"],
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="load the model(s) from a checkpoint",
    )
    parser.add_argument(
        "--bvh_scale_factor",
        type=float,
        default=None,
        help="scale factor applied to imported BVH positions, offsets, and end-sites",
    )
    args = parser.parse_args()
    if args.train_mode == "generator":
        args.train_mode = GENERATOR
    elif args.train_mode == "ik":
        args.train_mode = IK
    elif args.train_mode == "all":
        args.train_mode = GENERATOR | IK
    main(args)
