import torch
import numpy as np
import pymotion.rotations.ortho6d_torch as ortho6d


class Train_Data:
    def __init__(self, device, param):
        super().__init__()
        self.device = device
        self.param = param
        self.losses = []

    def set_offsets(self, offsets, denorm_offsets):
        self.offsets = offsets
        self.denorm_offsets = denorm_offsets
        for loss in self.losses:
            loss.set_offsets(denorm_offsets[0])

    def set_rot_order(self, rot_order):
        self.rot_order = rot_order

    def set_motions(
        self,
        dqs,
        displacement,
        disp_8=None,
        sin_diff=None,
        cos_diff=None,
        loss_weights=None,
    ):

        DL = displacement[0, 1].shape[0]  # displacement length (3 or 4)
        downsampling_factor = self.param["stride_encoder_conv"] ** 3
        synthetic_joint_count = int(self.param.get("synthetic_contact_joint_count", 0))

        # concatenate the displacement to the dqs
        self.motion = torch.cat([dqs, displacement], dim=2)
        # swap second and third dimensions for convolutions (last row should be time)
        self.motion = self.motion.permute(0, 2, 1)

        if disp_8 is not None:
            self.disp_8 = disp_8

        if loss_weights is not None:
            self.loss_weights = loss_weights

        if sin_diff is not None:
            self.yaw_diff = torch.cat(
                [sin_diff.unsqueeze(-1), cos_diff.unsqueeze(-1)], dim=-1
            )

        # self.motion is tensor of shape (batch_size, n_joints*8 + 3, frames)
        # if time dimension is not multiple of 8... make it multiple of 8
        # so that convolutions always return an even number (3 time downsamplings)
        if self.motion.shape[2] % downsampling_factor != 0:
            self.motion = self.motion[
                :,
                :,
                : self.motion.shape[2] - self.motion.shape[2] % downsampling_factor,
            ]
        # input for sparse encoder
        # sparse_dqs (batch_size, n_sparse_joints, 8, frames)
        sparse_dqs = (
            self.motion[:, :-DL, :]
            .reshape(self.motion.shape[0], -1, 9, self.motion.shape[-1])
            .clone()
        )

        if synthetic_joint_count > 0:
            skeletal_joint_count = sparse_dqs.shape[1] - synthetic_joint_count
            skeletal_sparse_dqs = sparse_dqs[:, :skeletal_joint_count, ...]
            synthetic_sparse_dqs = sparse_dqs[:, skeletal_joint_count:, ...]
        else:
            skeletal_sparse_dqs = sparse_dqs
            synthetic_sparse_dqs = None

        sparse_dqs = skeletal_sparse_dqs[:, self.param["sparse_joints"], ...]
        if synthetic_sparse_dqs is not None:
            sparse_dqs = torch.cat([sparse_dqs, synthetic_sparse_dqs], dim=1)
        sparse_dqs = sparse_dqs.flatten(start_dim=1, end_dim=2)

        # import matplotlib.pyplot as plt
        # input_cpu = sparse_dqs.permute(0,2,1).cpu().numpy()
        # plt.plot(input_cpu[0,:,8:16-4])
        # plt.show()
        # plt.plot(input_cpu[0,:,16:24-4])
        # plt.show()
        # plt.plot(input_cpu[0,:,24:32-4])
        # plt.show()
        # plt.plot(input_cpu[0,:,32:40])
        # plt.show()
        # plt.plot(input_cpu[0,:,40:48-4])
        # plt.show()
        # plt.plot(input_cpu[0,:,48:56-4])
        # plt.show()
        # sparse_displacement (batch_size, 3, frames)
        sparse_displacement = self.motion[:, -DL:, :].clone()
        # self.sparse_motion (batch_size, (n_sparse_joints + synthetic joints) * 9 + displacement, frames)
        self.sparse_motion = torch.cat([sparse_dqs, sparse_displacement], dim=1)

        # remove displacement from self.motion (we use it only as input)
        # self.motion is tensor of shape (batch_size, n_joints*8, frames)
        self.displacement = self.motion[:, -DL:, :]
        self.motion = self.motion[:, :-DL, :]
        # adding it back for now actually
        # self.motion[:,4:7] = self.displacement[:,0:3]

    def set_sparse_motion(self, sparse_motion):
        self.sparse_motion = sparse_motion

    def set_means(self, mean_dqs):
        self.mean_dqs = mean_dqs
        for loss in self.losses:
            loss.set_mean(mean_dqs)

    def set_stds(self, std_dqs):
        self.std_dqs = std_dqs
        for loss in self.losses:
            loss.set_std(std_dqs)

    def set_displacement_means_stds(self, mean_displacement, std_displacement):
        self.mean_displacement = mean_displacement
        self.std_displacement = std_displacement

    def set_tag_root_means_stds(
        self,
        mean_smooth_root_pos,
        std_smooth_root_pos,
        mean_rough_root_traj=None,
        std_rough_root_traj=None,
    ):
        self.mean_smooth_root_pos = mean_smooth_root_pos
        self.std_smooth_root_pos = std_smooth_root_pos
        self.mean_rough_root_traj = mean_rough_root_traj
        self.std_rough_root_traj = std_rough_root_traj

    def set_root_means_stds(self, mean_root, std_root):
        self.mean_root = mean_root
        self.std_root = std_root

    def set_sin_cos_means_stds(self, mean_sin, mean_cos, std_sin, std_cos):
        self.mean_sin = mean_sin
        self.mean_cos = mean_cos
        self.std_sin = std_sin
        self.std_cos = std_cos

    def set_phase(self, phase):
        self.phase = phase

    def set_phase_per_8_frames(self, phase_per_8_frames):
        self.phase_per_8_frames = phase_per_8_frames

    def set_velocity_per_8_frames(self, velocity_per_8_frames):
        self.velocity_per_8_frames = velocity_per_8_frames

    def set_tags(self, tags):
        self.tags = tags

        device = self.device
        for key, value in self.tags.items():
            if key == "style_id":
                if not isinstance(value, torch.Tensor):
                    self.tags[key] = torch.tensor(
                        value, dtype=torch.long, device=device
                    )
                else:
                    self.tags[key] = value.to(device=device, dtype=torch.long)
                continue

            if not isinstance(value, torch.Tensor):
                self.tags[key] = torch.tensor(
                    value, dtype=torch.float32, requires_grad=True
                ).to(device)

            elif self.tags[key].device != device:
                self.tags[key] = value.to(device, dtype=torch.float32).requires_grad_(
                    True
                )

            current = self.tags[key]
            if key == "velo_foot":
                continue

            if current.dim() == 1:
                self.tags[key] = current.unsqueeze(-1)
                continue

            if current.dim() == 2 and current.shape[-1] != 1:
                continue

            if current.dim() == 2:
                self.tags[key] = current.unsqueeze(-1)

        # self.velocity = tags["velocity"]
        # self.acceleration = tags["acceleration"]
        # self.ang_velocity = tags["ang_velocity"]
        # self.height = tags["height"]

    def set_global_pos(self, global_pos):
        if not isinstance(global_pos, torch.Tensor):
            self.global_pos = torch.tensor(global_pos, requires_grad=True).to(
                self.device
            )
        else:
            self.global_pos = global_pos.to(self.device, dtype=torch.float32)

    def set_foot_positions(self, foot_positions):
        if not isinstance(foot_positions, torch.Tensor):
            self.foot_positions = torch.tensor(foot_positions).to(
                self.device, dtype=torch.float32
            )
        else:
            self.foot_positions = foot_positions.to(self.device, dtype=torch.float32)

    def _build_yaw_rotation_batch(self, phi, dtype):
        cos_phi = torch.cos(phi).view(-1, 1)
        sin_phi = torch.sin(phi).view(-1, 1)
        batch = phi.shape[0]
        rotation = torch.zeros((batch, 3, 3), device=self.device, dtype=dtype)
        rotation[:, 0, 0] = cos_phi.squeeze(-1)
        rotation[:, 0, 2] = sin_phi.squeeze(-1)
        rotation[:, 1, 1] = 1.0
        rotation[:, 2, 0] = -sin_phi.squeeze(-1)
        rotation[:, 2, 2] = cos_phi.squeeze(-1)
        return rotation

    def _rotate_root_joint_normalized(self, root_joint, mean_root, std_root, phi):
        safe_std = std_root.clamp_min(1e-8).view(1, -1, 1)
        mean_root = mean_root.view(1, -1, 1)
        denormalized = root_joint * safe_std + mean_root

        batch, _, frames = denormalized.shape
        frame_major = denormalized.permute(0, 2, 1).contiguous().reshape(-1, 9)
        rot6 = frame_major[:, :6].reshape(-1, 3, 2)
        motion = frame_major[:, 6:9]

        rotation_mats = ortho6d.to_matrix(rot6)
        yaw_batch = self._build_yaw_rotation_batch(phi, dtype=denormalized.dtype)
        yaw_per_frame = (
            yaw_batch.view(batch, 1, 3, 3).expand(-1, frames, -1, -1).reshape(-1, 3, 3)
        )

        rotated_rot = torch.matmul(yaw_per_frame, rotation_mats)
        rotated_motion = torch.matmul(yaw_per_frame, motion.unsqueeze(-1)).squeeze(-1)

        rotated = (
            torch.cat(
                [
                    ortho6d.from_matrix(rotated_rot).reshape(batch, frames, 6),
                    rotated_motion.reshape(batch, frames, 3),
                ],
                dim=-1,
            )
            .permute(0, 2, 1)
            .contiguous()
        )
        return (rotated - mean_root) / safe_std

    def _rotate_displacement_normalized(self, displacement, phi):
        safe_std = self.std_displacement.clamp_min(1e-8).view(1, -1, 1)
        mean_displacement = self.mean_displacement.view(1, -1, 1)
        denormalized = displacement * safe_std + mean_displacement

        batch, _, frames = denormalized.shape
        frame_major = denormalized.permute(0, 2, 1).contiguous().reshape(-1, 3)
        yaw_batch = self._build_yaw_rotation_batch(phi, dtype=denormalized.dtype)
        yaw_per_frame = (
            yaw_batch.view(batch, 1, 3, 3).expand(-1, frames, -1, -1).reshape(-1, 3, 3)
        )
        rotated = torch.matmul(yaw_per_frame, frame_major.unsqueeze(-1)).squeeze(-1)
        rotated = rotated.reshape(batch, frames, 3).permute(0, 2, 1).contiguous()
        return (rotated - mean_displacement) / safe_std

    def _rotate_world_vectors_normalized(
        self, sequence, mean_feature, std_feature, phi
    ):
        safe_std = std_feature.clamp_min(1e-8).view(1, 1, -1)
        mean_feature = mean_feature.view(1, 1, -1)
        denormalized = sequence * safe_std + mean_feature

        batch, frames, _ = denormalized.shape
        yaw_batch = self._build_yaw_rotation_batch(phi, dtype=denormalized.dtype)
        yaw_per_frame = yaw_batch.view(batch, 1, 3, 3).expand(-1, frames, -1, -1)
        rotated = torch.matmul(yaw_per_frame, denormalized.unsqueeze(-1)).squeeze(-1)
        return (rotated - mean_feature) / safe_std

    def _rotate_world_vectors_denormalized(self, sequence, phi):
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence, dtype=torch.float32, device=self.device)
        else:
            sequence = sequence.to(device=self.device, dtype=torch.float32)

        original_shape = sequence.shape
        if sequence.dim() == 3:
            batch, frames, _ = sequence.shape
            flat = sequence.reshape(batch, frames, 1, 3)
        elif sequence.dim() == 4:
            batch, frames, item_count, _ = sequence.shape
            flat = sequence
        else:
            raise ValueError(
                f"Expected denormalized world vectors shaped [B,T,3] or [B,T,N,3], got {tuple(sequence.shape)}"
            )

        yaw_batch = self._build_yaw_rotation_batch(phi, dtype=flat.dtype)
        yaw_per_frame = yaw_batch.view(batch, 1, 1, 3, 3).expand(
            -1, flat.shape[1], flat.shape[2], -1, -1
        )
        rotated = torch.matmul(yaw_per_frame, flat.unsqueeze(-1)).squeeze(-1)
        return rotated.reshape(original_shape)

    def apply_random_yaw_augmentation(self):
        if not bool(self.param.get("random_yaw_augmentation", False)):
            self.last_random_yaw = None
            return None

        max_degrees = float(self.param.get("random_yaw_aug_max_degrees", 180.0))
        if max_degrees <= 0.0:
            self.last_random_yaw = None
            return None

        phi = (
            torch.rand(self.motion.shape[0], device=self.device) * 2.0 - 1.0
        ) * np.deg2rad(max_degrees)
        self.motion[:, :9, :] = self._rotate_root_joint_normalized(
            self.motion[:, :9, :],
            self.mean_root,
            self.std_root,
            phi,
        )

        if hasattr(self, "rots"):
            self.rots = (
                self._rotate_root_joint_normalized(
                    self.rots.permute(0, 2, 1).contiguous(),
                    self.mean_root,
                    self.std_root,
                    phi,
                )
                .permute(0, 2, 1)
                .contiguous()
            )

        if hasattr(self, "displacement"):
            self.displacement = self._rotate_displacement_normalized(
                self.displacement,
                phi,
            )

        if hasattr(self, "sparse_motion"):
            self.sparse_motion[:, :9, :] = self._rotate_root_joint_normalized(
                self.sparse_motion[:, :9, :],
                self.mean_root,
                self.std_root,
                phi,
            )
            disp_channels = self.displacement.shape[1]
            self.sparse_motion[:, -disp_channels:, :] = self.displacement

        if hasattr(self, "tags") and isinstance(self.tags, dict):
            if (
                "smooth_root_pos" in self.tags
                and hasattr(self, "mean_smooth_root_pos")
                and hasattr(self, "std_smooth_root_pos")
            ):
                self.tags["smooth_root_pos"] = self._rotate_world_vectors_normalized(
                    self.tags["smooth_root_pos"],
                    self.mean_smooth_root_pos,
                    self.std_smooth_root_pos,
                    phi,
                )

            if (
                "rough_root_traj" in self.tags
                and getattr(self, "mean_rough_root_traj", None) is not None
                and getattr(self, "std_rough_root_traj", None) is not None
            ):
                self.tags["rough_root_traj"] = (
                    self._rotate_root_joint_normalized(
                        self.tags["rough_root_traj"].permute(0, 2, 1).contiguous(),
                        self.mean_rough_root_traj,
                        self.std_rough_root_traj,
                        phi,
                    )
                    .permute(0, 2, 1)
                    .contiguous()
                )

        if hasattr(self, "global_pos"):
            self.global_pos = self._rotate_world_vectors_denormalized(
                self.global_pos,
                phi,
            )

        if hasattr(self, "foot_positions"):
            self.foot_positions = self._rotate_world_vectors_denormalized(
                self.foot_positions,
                phi,
            )

        self.last_random_yaw = phi
        return phi

    def set_rots(self, rots):
        if not isinstance(rots, torch.Tensor):
            self.rots = torch.tensor(rots, requires_grad=True).to(self.device)
        else:
            self.rots = rots.to(self.device, dtype=torch.float32)

    def set_end_sites(self, end_sites, end_sites_parents):
        if not isinstance(end_sites, torch.Tensor):
            self.end_sites = torch.tensor(end_sites, dtype=torch.float32).to(
                self.device
            )
            self.end_sites_parents = torch.as_tensor(
                end_sites_parents, dtype=torch.long, device=self.device
            )
        else:
            self.end_sites = end_sites.to(self.device, dtype=torch.float32)
            self.end_sites_parents = end_sites_parents.to(self.device, dtype=torch.long)

    def set_energy(self, energy):
        self.energy = energy
