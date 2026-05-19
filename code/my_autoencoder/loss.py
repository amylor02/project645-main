import torch
import torch.nn as nn
import torch.nn.functional as F
import pymotion.rotations.ortho6d_torch as ortho6d
from pymotion.ops.skeleton import compute_global_pos_torch
from pymotion.rotations.dual_quat_torch import to_rotation_translation

from motion_data import integrate_root_translation_torch, split_motion_joints


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
    if weights is None:
        return F.mse_loss(prediction, target)

    weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    while weights.dim() < prediction.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(prediction)
    denom = weights.sum().clamp_min(1e-6)
    return ((prediction - target) ** 2 * weights).sum() / denom


class MSE_DQ(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.param = param
        self.parents = parents
        self.device = device

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.joint_distances = torch.norm(offsets, dim=1)

    def forward_generator_vq_vae(
        self,
        input,
        target,
        vq_dict,
        output_logits,
        target_indices,
        velo=None,
        target_velo=None,
        means=None,
        stds=None,
        yaw_target=None,
        yaw_pred=None,
        phi=None,
        enable_root_loss=False,
        pos_accum=None,
        global_pos_zeroed=None,
        pos_xz_weight=1.0,
        predicted_feet=None,
        target_foot_positions=None,
        foot_contact_binary=None,
        root_velocity_weight=0.0,
        foot_sliding_weight=0.0,
        prior_root_pred=None,
        prior_root_weight=1.0,
    ):

        # feet_start_idxs = [(8-1)*9,(12-1)*9,(15-1)*9,(18-1)*9]

        # # Create tensors for feet and rest indices
        # feet_indices = []
        # for start_idx in feet_start_idxs:
        #     feet_indices.extend(list(range(start_idx, start_idx + 9)))

        # # Convert to tensor for easier indexing
        # feet_indices = torch.tensor(feet_indices, device=input.device)

        # # Create mask for all indices
        # all_indices = torch.arange(input.shape[1], device=input.device)

        # # Find non-feet indices (everything except the feet)
        # non_feet_mask = torch.ones(input.shape[1], dtype=torch.bool, device=input.device)
        # non_feet_mask[feet_indices] = False
        # rest_indices = all_indices[non_feet_mask]

        # # Extract feet and rest components
        # input_feet = input[:, feet_indices]
        # target_feet = target[:, feet_indices]
        # input_rest = input[:, rest_indices]
        # target_rest = target[:, rest_indices]

        # feet_loss = self.mse(input_feet, target_feet)
        # rest_loss = self.mse(input_rest, target_rest)

        # recon_loss = feet_loss * 1.0 + rest_loss * 1.0
        if phi is not None:
            means = means.unsqueeze(-1)
            stds = stds.unsqueeze(-1)
            stds = stds.clamp_min(1e-8)

            input = (input * stds) + means
            target = (target * stds) + means

            new_root_in, new_root_tgt = self.rotate_root_channels(
                input[:, :9], target[:, :9], phi
            )
            # input = torch.cat([new_root_in, input[:, 9:, :].clone()], dim=1)
            target = torch.cat([new_root_tgt, target[:, 9:, :].clone()], dim=1)

            input = (input - means) / stds
            target = (target - means) / stds

        root_recon_weight = float(self.param.get("root_recon_loss_weight", 10.0))
        synthetic_joint_count = int(self.param.get("synthetic_contact_joint_count", 0))
        synthetic_joint_channels = synthetic_joint_count * 9
        contact_channel_count = int(self.param.get("synthetic_contact_channels", 4))
        contact_recon_weight = float(self.param.get("contact_recon_loss_weight", 5.0))

        rots_loss = self.mse(input[:, :6], target[:, :6])
        disp_loss = self.mse(input[:, 6:9], target[:, 6:9])
        root_loss = rots_loss + disp_loss
        body_end = (
            input.size(1) - synthetic_joint_channels
            if synthetic_joint_channels > 0
            else input.size(1)
        )
        if body_end > 9:
            joints_loss = self.mse(input[:, 9:body_end], target[:, 9:body_end])
        else:
            joints_loss = torch.tensor(0.0, device=input.device)

        if synthetic_joint_channels > 0:
            contact_start = input.size(1) - synthetic_joint_channels
            contact_end = contact_start + contact_channel_count
            contact_loss = self.mse(
                input[:, contact_start:contact_end],
                target[:, contact_start:contact_end],
            )
            synthetic_pad_loss = self.mse(
                input[:, contact_end:],
                target[:, contact_end:],
            )
        else:
            contact_loss = torch.tensor(0.0, device=input.device)
            synthetic_pad_loss = torch.tensor(0.0, device=input.device)

        recon_loss = (
            root_recon_weight * root_loss
            + joints_loss
            + contact_recon_weight * contact_loss
            + synthetic_pad_loss
        )

        root_vel_loss = torch.tensor(0.0, device=input.device)
        if (
            root_velocity_weight > 0.0
            and pos_accum is not None
            and global_pos_zeroed is not None
            and pos_accum.size(1) > 1
            and global_pos_zeroed.size(1) > 1
        ):
            root_t = min(pos_accum.size(1), global_pos_zeroed.size(1))
            pred_root_vel = (
                pos_accum[:, 1:root_t, [0, 2]] - pos_accum[:, : root_t - 1, [0, 2]]
            )
            target_root_vel = (
                global_pos_zeroed[:, 1:root_t, [0, 2]]
                - global_pos_zeroed[:, : root_t - 1, [0, 2]]
            )
            root_vel_loss = F.mse_loss(pred_root_vel, target_root_vel)
            recon_loss = recon_loss + root_velocity_weight * root_vel_loss

        # Whole-body velocity + acceleration supervision (1B: derivative losses)
        lambda_vel = 10.0
        lambda_acc = 5.0
        if input.size(-1) > 2:
            body_pred = input[:, 9:, :]
            body_tgt = target[:, 9:, :]
            vel_pred = body_pred[:, :, 1:] - body_pred[:, :, :-1]
            vel_tgt = body_tgt[:, :, 1:] - body_tgt[:, :, :-1]
            vel_loss = F.mse_loss(vel_pred, vel_tgt)
            acc_pred = vel_pred[:, :, 1:] - vel_pred[:, :, :-1]
            acc_tgt = vel_tgt[:, :, 1:] - vel_tgt[:, :, :-1]
            acc_loss = F.mse_loss(acc_pred, acc_tgt)
        else:
            vel_loss = torch.tensor(0.0, device=input.device)
            acc_loss = torch.tensor(0.0, device=input.device)
        recon_loss = recon_loss + lambda_vel * vel_loss + lambda_acc * acc_loss

        # Spectral loss: penalize missing frequency content in body joint trajectories
        # Computes per-channel FFT magnitude spectra and compares via log-scale L1
        lambda_spectral = 0
        if input.size(-1) >= 8:
            body_pred_s = input[:, 9:, :]  # [B, C_body, T]
            body_tgt_s = target[:, 9:, :]
            # Hann window to reduce spectral leakage
            T = body_pred_s.size(-1)
            window = torch.hann_window(T, device=input.device, dtype=input.dtype)
            pred_w = body_pred_s * window.unsqueeze(0).unsqueeze(0)
            tgt_w = body_tgt_s * window.unsqueeze(0).unsqueeze(0)
            # rfft → magnitude spectrum (only positive frequencies)
            spec_pred = torch.fft.rfft(pred_w, dim=-1).abs()  # [B, C_body, T//2+1]
            spec_tgt = torch.fft.rfft(tgt_w, dim=-1).abs()
            # Log-scale L1: emphasizes relative error at all frequency magnitudes
            eps = 1e-7
            spectral_loss = F.l1_loss(
                torch.log(spec_pred + eps), torch.log(spec_tgt + eps)
            )
        else:
            spectral_loss = torch.tensor(0.0, device=input.device)
        recon_loss = recon_loss + lambda_spectral * spectral_loss

        rots_incr_loss = (
            self.mse(yaw_pred[:, :, :], yaw_target[:, :, :]) * 10
        )  # .mean() * 10 #+ position_loss

        recon_loss = recon_loss + rots_incr_loss * 0
        vq_loss = vq_dict * 1

        if output_logits is None:
            logits_loss = torch.tensor(0.0, device=input.device)
        else:
            output_logits = output_logits.view(-1, output_logits.size(-1))
            target_indices = target_indices.view(-1)
            logits_loss = self.cross_entropy_loss(output_logits, target_indices)

        if (
            foot_sliding_weight > 0.0
            and predicted_feet is not None
            and target_foot_positions is not None
            and foot_contact_binary is not None
        ):
            if target_foot_positions.dim() == 3:
                target_foot_positions = target_foot_positions.unsqueeze(0)

            foot_t = min(predicted_feet.size(1), target_foot_positions.size(1))
            predicted_feet = predicted_feet[:, :foot_t]
            target_foot_positions = target_foot_positions[:, :foot_t].to(
                device=predicted_feet.device,
                dtype=predicted_feet.dtype,
            )
            contact_binary = foot_contact_binary[:, :foot_t].to(
                device=predicted_feet.device,
                dtype=predicted_feet.dtype,
            )

            if predicted_feet.size(1) > 1:
                predicted_velocity = predicted_feet[:, 1:] - predicted_feet[:, :-1]
                target_velocity = (
                    target_foot_positions[:, 1:] - target_foot_positions[:, :-1]
                )
                contact_velocity = 0.5 * (
                    contact_binary[:, 1:, :] + contact_binary[:, :-1, :]
                )
                velo_loss = weighted_mse(
                    predicted_velocity,
                    target_velocity,
                    contact_velocity,
                )
                recon_loss = recon_loss + foot_sliding_weight * velo_loss
            else:
                velo_loss = torch.tensor(0.0, device=input.device)
        else:
            velo_loss = torch.tensor(0.0, device=input.device)

        # XZ position loss: compare accumulated root XZ against global pos XZ
        if pos_accum is not None and global_pos_zeroed is not None:
            T_pred = pos_accum.shape[1]
            T_gt = global_pos_zeroed.shape[1]
            T_min = min(T_pred, T_gt)
            pred_xz = pos_accum[:, :T_min, :][:, :, [0, 2]]  # [B, T_min, 2]
            gt_xz = global_pos_zeroed[:, :T_min, :][:, :, [0, 2]]  # [B, T_min, 2]
            pos_xz_loss = F.mse_loss(pred_xz, gt_xz) * pos_xz_weight
            recon_loss = recon_loss + pos_xz_loss
        else:
            pos_xz_loss = torch.tensor(0.0, device=input.device)

        if prior_root_pred is not None and prior_root_weight > 0:
            if prior_root_pred.dim() == 3 and prior_root_pred.size(-1) == 9:
                prior_root = prior_root_pred.permute(0, 2, 1)
            else:
                prior_root = prior_root_pred

            root_t = min(prior_root.size(-1), target.size(-1))
            prior_root = prior_root[:, :, :root_t]
            target_root = target[:, :9, :root_t]

            prior_root_rot_loss = self.mse(prior_root[:, :6], target_root[:, :6]) * 10
            prior_root_disp_loss = (
                self.mse(prior_root[:, 6:9], target_root[:, 6:9]) * 10
            )

            if root_t > 1:
                prior_root_vel_loss = F.mse_loss(
                    prior_root[:, 6:9, 1:] - prior_root[:, 6:9, :-1],
                    target_root[:, 6:9, 1:] - target_root[:, 6:9, :-1],
                )
            else:
                prior_root_vel_loss = torch.tensor(0.0, device=input.device)

            prior_root_loss = prior_root_weight * (
                prior_root_rot_loss + prior_root_disp_loss + prior_root_vel_loss
            )
        else:
            prior_root_loss = torch.tensor(0.0, device=input.device)

        return (
            recon_loss,
            vq_loss,
            logits_loss,
            velo_loss,
            rots_incr_loss,
            vel_loss,
            acc_loss,
            spectral_loss,
            pos_xz_loss,
            prior_root_loss,
        )

    def rotate_root_channels(
        self, inp: torch.Tensor, tgt: torch.Tensor, phi: torch.Tensor
    ):
        """
        Rotate root channels (first 9 channels: 6D ortho6d + [vel_x, height, vel_z]) by yaw angle(s) phi.

        Args:
            inp: [B, C, T] input tensor
            tgt: [B, C, T] target tensor
            phi: scalar or torch.Tensor of shape [B] (radians). Broadcasts if needed.
        Returns:
            inp_rot, tgt_rot: rotated copies (do not modify inputs in-place)
        """
        if phi is None:
            return inp, tgt

        device = inp.device
        dtype = inp.dtype
        B = inp.shape[0]

        # normalize phi to tensor of shape [B]
        if isinstance(phi, (float, int)):
            phi_t = torch.full((B,), float(phi), device=device, dtype=dtype)
        else:
            phi_t = torch.as_tensor(phi, device=device, dtype=dtype)
            if phi_t.dim() == 0:
                phi_t = phi_t.view(1).expand(B)
            elif phi_t.numel() == 1:
                phi_t = phi_t.view(1).expand(B)
            elif phi_t.numel() == B and phi_t.dim() == 1:
                pass
            else:
                # try to squeeze/broadcast to (B,)
                try:
                    phi_t = phi_t.view(B)
                except Exception:
                    raise ValueError(
                        f"phi must be scalar or length batch_size ({B}), got {phi_t.shape}"
                    )

        inp_c = inp.clone()
        tgt_c = tgt.clone()

        # extract root channels: [B, T, 9]
        root_in = inp_c[:, :9, :].permute(0, 2, 1).contiguous()  # [B, T, 9]
        root_tgt = tgt_c[:, :9, :].permute(0, 2, 1).contiguous()  # [B, T, 9]
        B, T, _ = root_in.shape
        N = B * T

        flat_in = root_in.reshape(N, 9)
        flat_tgt = root_tgt.reshape(N, 9)

        rot6_in = flat_in[:, :6].reshape(N, 3, 2)
        pos_in = flat_in[:, 6:9]
        rot6_tgt = flat_tgt[:, :6].reshape(N, 3, 2)
        pos_tgt = flat_tgt[:, 6:9]

        # to matrices [N,3,3]
        mats_in = ortho6d.to_matrix(rot6_in)
        mats_tgt = ortho6d.to_matrix(rot6_tgt)

        # build per-batch yaw R [B,3,3]
        cos_phi = torch.cos(phi_t).view(B, 1)
        sin_phi = torch.sin(phi_t).view(B, 1)
        R_batch = torch.zeros((B, 3, 3), device=device, dtype=dtype)
        R_batch[:, 0, 0] = cos_phi.squeeze(-1)
        R_batch[:, 0, 2] = sin_phi.squeeze(-1)
        R_batch[:, 1, 1] = 1.0
        R_batch[:, 2, 0] = -sin_phi.squeeze(-1)
        R_batch[:, 2, 2] = cos_phi.squeeze(-1)

        # expand to per-frame [N,3,3]
        Rb = R_batch.view(B, 1, 3, 3).expand(-1, T, -1, -1).reshape(N, 3, 3)

        # apply rotation (left-multiply)
        new_mats_in = torch.matmul(Rb, mats_in)
        new_mats_tgt = torch.matmul(Rb, mats_tgt)

        # back to ortho6d 6-vector and rotate positions
        new_rot6_in = ortho6d.from_matrix(new_mats_in).reshape(B, T, 6)
        new_rot6_tgt = ortho6d.from_matrix(new_mats_tgt).reshape(B, T, 6)

        pos_in_rot = torch.matmul(Rb, pos_in.unsqueeze(-1)).squeeze(-1).reshape(B, T, 3)
        pos_tgt_rot = (
            torch.matmul(Rb, pos_tgt.unsqueeze(-1)).squeeze(-1).reshape(B, T, 3)
        )

        # reassemble back to [B,9,T]
        new_root_in = (
            torch.cat([new_rot6_in, pos_in_rot], dim=-1).permute(0, 2, 1).contiguous()
        )
        new_root_tgt = (
            torch.cat([new_rot6_tgt, pos_tgt_rot], dim=-1).permute(0, 2, 1).contiguous()
        )

        return new_root_in, new_root_tgt


class MSE_DQ_FK(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = [None if parent is None else int(parent) for parent in parents]
        self.device = device
        head_idx = int(self.param.get("head_idx", -1))
        self.target_sparse_indices = [
            int(joint_id)
            for joint_id in self.param.get("sparse_joints", [])[1:]
            if int(joint_id) != head_idx
        ]

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs

    def set_offsets(self, offsets):
        self.offsets = offsets  # denormalized
        self.joint_distances = torch.norm(offsets, dim=1)

    def set_target_sparse_indices(self, joint_ids):
        self.target_sparse_indices = [int(joint_id) for joint_id in joint_ids]

    def _normalize_fk_end_sites(self, end_sites, batch_size):
        if end_sites is None:
            return None

        if not torch.is_tensor(end_sites):
            end_sites = torch.as_tensor(
                end_sites,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            end_sites = end_sites.to(device=self.device, dtype=torch.float32)

        while end_sites.dim() > 3:
            if end_sites.size(0) == batch_size:
                end_sites = end_sites[:, 0]
            else:
                end_sites = end_sites[0]

        if end_sites.dim() == 3 and end_sites.size(0) not in {1, batch_size}:
            end_sites = end_sites[0]
        return end_sites

    def _normalize_fk_end_site_parents(self, end_sites_parents, batch_size):
        if end_sites_parents is None:
            return None

        if not torch.is_tensor(end_sites_parents):
            end_sites_parents = torch.as_tensor(
                end_sites_parents,
                dtype=torch.long,
                device=self.device,
            )
        else:
            end_sites_parents = end_sites_parents.to(
                device=self.device,
                dtype=torch.long,
            )

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

    def _motion_to_global_kinematics(self, motion, train_data):
        batch_size, _, frame_count = motion.shape
        safe_std = self.std_dqs.clamp_min(1e-8).view(1, 1, -1)
        mean_dqs = self.mean_dqs.view(1, 1, -1)

        denormalized = motion.permute(0, 2, 1).contiguous() * safe_std + mean_dqs
        denormalized = denormalized.view(batch_size, frame_count, -1, 9)
        skeletal_motion, _ = split_motion_joints(
            denormalized,
            synthetic_joint_count=int(
                self.param.get("synthetic_contact_joint_count", 0)
            ),
        )
        dual_quats = ortho6d.to_dual_quat(skeletal_motion)
        local_rotations, _ = to_rotation_translation(dual_quats)
        root_positions = integrate_root_translation_torch(
            skeletal_motion[:, :, 0, :],
            train_data.global_pos,
        )

        fk_kwargs = {}
        if hasattr(train_data, "end_sites") and hasattr(
            train_data, "end_sites_parents"
        ):
            fk_kwargs["end_sites"] = self._normalize_fk_end_sites(
                train_data.end_sites,
                batch_size=batch_size,
            )
            fk_kwargs["end_sites_parents"] = self._normalize_fk_end_site_parents(
                train_data.end_sites_parents,
                batch_size=batch_size,
            )

        joint_positions, joint_rotations = compute_global_pos_torch(
            local_rotations,
            root_positions,
            train_data.denorm_offsets,
            self.parents,
            **fk_kwargs,
        )
        return joint_positions, joint_rotations

    def forward_ik(self, input_ik, input_decoder, target, train_data):
        refined_joint_poses, refined_joint_rot_mat = self._motion_to_global_kinematics(
            input_ik,
            train_data,
        )
        target_joint_poses, target_joint_rot_mat = self._motion_to_global_kinematics(
            target,
            train_data,
        )
        decoder_joint_poses, decoder_rot_mat = self._motion_to_global_kinematics(
            input_decoder,
            train_data,
        )

        joint_count = refined_joint_poses.shape[2]
        sparse_indices = [
            joint_id
            for joint_id in self.target_sparse_indices
            if 0 <= int(joint_id) < joint_count
        ]
        sparse_tensor = torch.as_tensor(
            sparse_indices,
            dtype=torch.long,
            device=self.device,
        )
        non_sparse_mask = torch.ones(joint_count, dtype=torch.bool, device=self.device)
        if sparse_tensor.numel() > 0:
            non_sparse_mask[sparse_tensor] = False
        non_sparse_tensor = torch.arange(joint_count, device=self.device)[
            non_sparse_mask
        ]

        loss_ee = torch.tensor(0.0, device=self.device)
        if sparse_tensor.numel() > 0:
            loss_ee = self.mse(
                refined_joint_poses[:, :, sparse_tensor, :],
                target_joint_poses[:, :, sparse_tensor, :],
            )
            loss_ee = loss_ee + self.mse(
                refined_joint_rot_mat[:, :, sparse_tensor, :, :],
                target_joint_rot_mat[:, :, sparse_tensor, :, :],
            )

        loss_ee_reg = torch.tensor(0.0, device=self.device)
        if non_sparse_tensor.numel() > 0:
            loss_ee_reg = self.mse(
                decoder_joint_poses[:, :, non_sparse_tensor, :],
                refined_joint_poses[:, :, non_sparse_tensor, :],
            )
            loss_ee_reg = loss_ee_reg + self.mse(
                decoder_rot_mat[:, :, non_sparse_tensor, :, :],
                refined_joint_rot_mat[:, :, non_sparse_tensor, :, :],
            )

        return (
            loss_ee * self.param["lambda_ee"]
            + loss_ee_reg * self.param["lambda_ee_reg"]
        )
