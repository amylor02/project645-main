import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pymotion.rotations.ortho6d_torch as ortho6d

from autoencoder_no_enc_9_groups_no_rootbranch import (
    ConvLayerNorm,
    Decoder as BaseDecoder,
    Encoder as BaseEncoder,
    StaticEncoder,
)
from latent_diffusion_prior import ContinuousLatentDiffusionPrior
from motion_data import LMA_KEYS
from skeleton import SkeletonConv, find_neighbor


training_stage = "ae"


class Encoder(BaseEncoder):
    @staticmethod
    def _build_structured_keep_indices(joint_num, channels_per_joint, target_dim):
        base_channels = target_dim // joint_num
        extra_channels = target_dim % joint_num
        keep_indices = []
        for joint_index in range(joint_num):
            joint_keep = base_channels + (1 if joint_index < extra_channels else 0)
            joint_offset = joint_index * channels_per_joint
            keep_indices.extend(range(joint_offset, joint_offset + joint_keep))
        return torch.tensor(keep_indices, dtype=torch.long)

    def __init__(self, param, parents, device, is_vae=False, is_vq_vae=False):
        super().__init__(param, parents, device, is_vae=is_vae, is_vq_vae=False)
        self.is_vq_vae = False
        self.latent_mean = None
        self.latent_logvar = None
        self.latent_sample = None
        self.latent_kl = None

        feature_dim = self.channel_list[-1]
        self.vae_latent_dim = int(param.get("vae_latent_dim", feature_dim))
        if self.is_vae:
            low_res_parents = self.parents[-1]
            posterior_neighbor_list, _ = find_neighbor(
                low_res_parents,
                param["neighbor_distance"],
                add_displacement=True,
            )
            self.posterior_feature_channels_per_joint = feature_dim // self.num_joints
            self.posterior_latent_channels_per_joint = max(
                1,
                math.ceil(self.vae_latent_dim / self.num_joints),
            )
            self.posterior_structured_latent_dim = (
                self.posterior_latent_channels_per_joint * self.num_joints
            )
            self.register_buffer(
                "posterior_keep_indices",
                self._build_structured_keep_indices(
                    self.num_joints,
                    self.posterior_latent_channels_per_joint,
                    self.vae_latent_dim,
                ),
                persistent=False,
            )

            posterior_conv_kwargs = dict(
                param=param,
                neighbor_list=posterior_neighbor_list,
                kernel_size=1,
                joint_num=self.num_joints,
                in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                padding=0,
                stride=1,
                device=device,
                add_offset=False,
            )

            self.posterior_input_norm = ConvLayerNorm(feature_dim)
            self.posterior_mean_head = SkeletonConv(
                in_channels_per_joint=self.posterior_feature_channels_per_joint,
                out_channels_per_joint=self.posterior_latent_channels_per_joint,
                **posterior_conv_kwargs,
            )
            self.posterior_logvar_head = SkeletonConv(
                in_channels_per_joint=self.posterior_feature_channels_per_joint,
                out_channels_per_joint=self.posterior_latent_channels_per_joint,
                **posterior_conv_kwargs,
            )
            if self.vae_latent_dim != feature_dim:
                self.posterior_to_decoder = nn.Sequential(
                    nn.Linear(self.vae_latent_dim, feature_dim),
                    nn.GELU(),
                )
            else:
                self.posterior_to_decoder = nn.Identity()

            nn.init.normal_(self.posterior_mean_head.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.posterior_mean_head.bias)
            nn.init.normal_(self.posterior_logvar_head.weight, mean=0.0, std=1e-2)
            nn.init.constant_(
                self.posterior_logvar_head.bias,
                float(param.get("vae_logvar_init_bias", -1.5)),
            )

    def _reparameterize(self, mean, logvar):
        if not self.training and not self.param.get("vae_eval_sample", False):
            return mean

        # print latent statistics

        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _compute_kl_loss(self, mean, logvar):
        kl_per_latent = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())
        free_bits = float(self.param.get("vae_free_bits", 0.0))
        if free_bits > 0.0:
            kl_per_latent = kl_per_latent.clamp_min(free_bits)

        return kl_per_latent.sum(dim=-1).mean()

    def get_posterior_stats(self):
        return self.latent_mean, self.latent_logvar, self.latent_kl

    def forward(self, input):
        encoded = super().forward(input)

        if not self.is_vae:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_sample = None
            self.latent_kl = encoded.new_zeros(())
            self.latent = encoded.permute(0, 2, 1).contiguous()
            return encoded

        posterior_features = encoded  # self.posterior_input_norm(encoded)

        logvar_min = float(self.param.get("vae_logvar_min", -6.0))
        logvar_max = float(self.param.get("vae_logvar_max", 2.0))
        structured_mean = (
            self.posterior_mean_head(posterior_features).permute(0, 2, 1).contiguous()
        )
        structured_logvar = (
            self.posterior_logvar_head(posterior_features).permute(0, 2, 1).contiguous()
        )
        self.latent_mean = structured_mean.index_select(
            dim=-1, index=self.posterior_keep_indices
        )
        self.latent_logvar = structured_logvar.index_select(
            dim=-1, index=self.posterior_keep_indices
        ).clamp(
            logvar_min,
            logvar_max,
        )

        self.latent_sample = self._reparameterize(self.latent_mean, self.latent_logvar)
        self.latent_kl = self._compute_kl_loss(self.latent_mean, self.latent_logvar)
        self.latent = self.latent_sample

        decoder_latent = self.posterior_to_decoder(self.latent_sample)
        return decoder_latent.permute(0, 2, 1).contiguous()


class Decoder(BaseDecoder):
    _ROOT_BRANCH_ATTRS = (
        "root_branch_dim",
        "forward_context_dim",
        "coarse_dim",
        "coarse_skel_conv1",
        "coarse_norm1",
        "coarse_skel_conv2",
        "coarse_norm2",
        "coarse_compress",
        "coarse_upsample",
        "fine_dim",
        "fine_skel_conv1",
        "fine_norm1",
        "fine_skel_conv2",
        "fine_norm2",
        "fine_compress",
        "root_input_fuse",
        "root_predictor",
        "root_rot_head",
        "root_disp_head",
    )

    def __init__(self, param, enc, device):
        super().__init__(param, enc, device)
        for attr in self._ROOT_BRANCH_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

        self.body_motion_dim = self.channel_size[0] * max(0, self.num_joints - 1)

    def forward(
        self,
        input,
        offset,
        mean_dqs,
        std_dqs,
        denorm_offsets,
        parents,
        smooth_root_pos=None,
        root_quats=None,
        ctrl=None,
        ctrl_embedding=None,
        vq_out=None,
    ):
        if root_quats is not None:
            input = torch.cat([input, root_quats.permute(0, 2, 1)], dim=2)

        decoded = self._run_decoder_layers(input, offset)
        return self._normalize_motion_channels(decoded, mean_dqs, std_dqs)


class Autoencoder(nn.Module):
    supports_continuous_latent = True

    def __init__(
        self, param, parents, device, transform_net=False, is_vae=False, is_vq_vae=False
    ):
        super(Autoencoder, self).__init__()
        self.is_vae = is_vae
        self.is_vq_vae = False
        self.encoder = Encoder(
            param, parents, device, is_vae=self.is_vae, is_vq_vae=False
        )
        self.decoder = Decoder(param, self.encoder, device)
        self.parents = parents
        self.param = param
        self.training_stage = param.get("training_stage", training_stage)
        self.prior_inference_mode = "disabled"
        self.enable_prior_root_override = False
        self.codebook_predictor2 = None
        self.diffusion_prior = None
        self.enc_output = None
        self.lstm_output = None
        self.latent = None
        self.latent_mean = None
        self.latent_logvar = None
        self.latent_kl = None
        self.decoder.training_stage = self.training_stage

        if bool(param.get("use_diffusion_prior", False)):
            self.diffusion_prior = ContinuousLatentDiffusionPrior(
                latent_dim=int(
                    param.get("vae_latent_dim", self.encoder.vae_latent_dim)
                ),
                hidden_dim=int(param.get("diffusion_hidden_dim", 512)),
                num_layers=int(param.get("diffusion_num_layers", 8)),
                num_heads=int(param.get("diffusion_num_heads", 8)),
                dropout=float(param.get("diffusion_dropout", 0.1)),
                lma_dim=len(LMA_KEYS),
                traj_dim=int(param.get("rough_root_traj_dim", 9)),
                num_train_timesteps=int(
                    param.get("diffusion_num_train_timesteps", 1000)
                ),
                use_per_lma_channel_encoders=bool(
                    param.get("diffusion_use_per_lma_channel_encoders", False)
                ),
                per_lma_channel_dropout=float(
                    param.get("diffusion_per_lma_channel_dropout", 0.1)
                ),
                latent_velocity_weight=float(
                    param.get("diffusion_latent_velocity_weight", 0.0)
                ),
                num_styles=int(param.get("diffusion_num_styles", 0)),
                style_drop=float(param.get("diffusion_style_drop", 0.0)),
                style_condition_scale=float(
                    param.get("diffusion_style_condition_scale", 1.0)
                ),
            )

    def set_training_stage(self, stage: str):
        self.training_stage = stage
        self.decoder.training_stage = stage

    def set_prior_inference_mode(self, mode: str):
        self.prior_inference_mode = mode

    def get_latents(self):
        return self.enc_output, self.lstm_output

    def get_variational_stats(self):
        return self.latent_mean, self.latent_logvar, self.latent_kl

    def encode_latent_sequence(self, input, use_mean=True):
        encoded = self.encoder(input.clone())
        self.latent_mean, self.latent_logvar, self.latent_kl = (
            self.encoder.get_posterior_stats()
        )

        if self.is_vae and self.latent_mean is not None:
            self.latent = (
                self.latent_mean
                if use_mean
                else getattr(self.encoder, "latent_sample", self.latent_mean)
            )
            self.enc_output = self.latent
            return self.latent

        self.latent = getattr(self.encoder, "latent", None)
        if self.latent is None:
            self.latent = encoded.permute(0, 2, 1).contiguous()
        self.enc_output = self.latent
        return self.latent

    def build_diffusion_condition_tensors(self, tags):
        if tags is None:
            return None, None, None

        lma_parts = []
        for key in LMA_KEYS:
            value = tags.get(key)
            if value is None:
                continue
            if value.dim() == 2:
                value = value.unsqueeze(-1)
            if value.dim() == 4 and value.size(-1) == 1:
                value = value.squeeze(-1)
            if value.dim() != 3:
                raise ValueError(f"Unsupported LMA tag shape for {key}: {value.shape}")
            lma_parts.append(value)

        lma_tensor = torch.cat(lma_parts, dim=-1) if lma_parts else None
        rough_root_traj = tags.get("rough_root_traj")
        if rough_root_traj is not None and rough_root_traj.dim() == 2:
            rough_root_traj = rough_root_traj.unsqueeze(0)
        style_id = tags.get("style_id")
        if style_id is not None:
            target_device = None
            if lma_tensor is not None:
                target_device = lma_tensor.device
            elif rough_root_traj is not None:
                target_device = rough_root_traj.device
            if not torch.is_tensor(style_id):
                style_id = torch.tensor(style_id, dtype=torch.long)
            if target_device is not None:
                style_id = style_id.to(dtype=torch.long, device=target_device)
            else:
                style_id = style_id.to(dtype=torch.long)
            if style_id.dim() == 0:
                style_id = style_id.unsqueeze(0)
            if style_id.dim() > 1:
                style_id = style_id.reshape(style_id.shape[0], -1)[:, 0]
        return lma_tensor, rough_root_traj, style_id

    def _project_latent_for_decoder(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 3:
            raise ValueError(f"Expected latent [B, T, C], got {latent.shape}")

        if not self.is_vae:
            return latent

        projection = getattr(self.encoder, "posterior_to_decoder", None)
        if projection is None:
            return latent

        decoder_dim = int(self.encoder.channel_list[-1])
        latent_dim = int(getattr(self.encoder, "vae_latent_dim", decoder_dim))

        if latent.size(-1) == decoder_dim:
            return latent

        if latent.size(-1) == latent_dim:
            return projection(latent)

        raise ValueError(
            "Unexpected latent width for decoder: "
            f"got {latent.size(-1)}, expected {latent_dim} or {decoder_dim}"
        )

    def decode_latent_sequence(
        self,
        latent,
        offset,
        mean_dqs,
        std_dqs,
        denorm_offsets,
        mean_root=None,
        std_root=None,
        tags=None,
        root_override=None,
        root_override_blend: float = 1.0,
    ):
        if latent.dim() != 3:
            raise ValueError(f"Expected latent [B, T, C], got {latent.shape}")

        latent = self._project_latent_for_decoder(latent)

        decoded = self.decoder(
            latent.permute(0, 2, 1).contiguous(),
            offset,
            mean_dqs,
            std_dqs,
            denorm_offsets,
            self.parents,
            smooth_root_pos=(
                tags["smooth_root_pos"]
                if tags is not None and "smooth_root_pos" in tags
                else None
            ),
        )

        if root_override is not None:
            decoded, root = self._apply_root_override(
                decoded,
                root_override,
                mean_root=mean_root,
                std_root=std_root,
                blend=root_override_blend,
            )
        else:
            root = self._normalize_root_channels(
                decoded[:, :9].clone().permute(0, 2, 1),
                mean_root=mean_root,
                std_root=std_root,
            )
        return decoded, root

    @torch.no_grad()
    def sample_latent_sequence(
        self,
        seq_len,
        lma_seq=None,
        rough_root_traj=None,
        style_id=None,
        mode=None,
        cfg_scale=None,
        lma_cfg_scale=None,
        traj_cfg_scale=None,
        style_cfg_scale=None,
        num_steps=None,
        chunk_len=None,
        overlap_len=None,
        halo_len=None,
        eta=None,
        temperature=None,
        batch_size=1,
    ):
        if self.diffusion_prior is None:
            raise RuntimeError(
                "No diffusion prior attached to this autoencoder. Enable use_diffusion_prior in param."
            )

        mode = mode or self.prior_inference_mode or "full"
        cfg_scale = float(
            cfg_scale
            if cfg_scale is not None
            else self.param.get("diffusion_cfg_scale", 2.5)
        )
        num_steps = int(
            num_steps
            if num_steps is not None
            else self.param.get("diffusion_sample_steps", 50)
        )
        eta = float(eta if eta is not None else self.param.get("diffusion_eta", 0.0))
        temperature = float(
            temperature
            if temperature is not None
            else self.param.get("diffusion_temperature", 1.0)
        )
        chunk_len = int(
            chunk_len
            if chunk_len is not None
            else self.param.get("diffusion_chunk_len", 128)
        )
        overlap_len = int(
            overlap_len
            if overlap_len is not None
            else self.param.get("diffusion_overlap_len", 32)
        )
        halo_len = int(
            halo_len
            if halo_len is not None
            else self.param.get("diffusion_halo_len", 8)
        )

        if seq_len > chunk_len:
            return self.diffusion_prior.sample_long(
                total_seq_len=seq_len,
                batch_size=batch_size,
                lma_seq=lma_seq,
                traj_seq=rough_root_traj,
                style_id=style_id,
                mode=mode,
                cfg_scale=cfg_scale,
                lma_cfg_scale=lma_cfg_scale,
                traj_cfg_scale=traj_cfg_scale,
                style_cfg_scale=style_cfg_scale,
                num_steps=num_steps,
                chunk_len=chunk_len,
                overlap_len=overlap_len,
                halo_len=halo_len,
                eta=eta,
                temperature=temperature,
            )

        return self.diffusion_prior.sample(
            seq_len=seq_len,
            batch_size=batch_size,
            lma_seq=lma_seq,
            traj_seq=rough_root_traj,
            mode=mode,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            eta=eta,
            temperature=temperature,
        )

    def _normalize_root_channels(self, root, mean_root=None, std_root=None):
        if root is None:
            return None

        if mean_root is None or std_root is None:
            return ortho6d.normalize(root)

        safe_std_root = std_root.clamp_min(1e-8)
        root = root * safe_std_root + mean_root
        root = ortho6d.normalize(root)
        return (root - mean_root) / safe_std_root

    def _resample_root_override(self, root_override, target_len: int):
        if root_override is None:
            return None

        if not torch.is_tensor(root_override):
            root_override = torch.tensor(root_override, dtype=torch.float32)
        if root_override.dim() == 2:
            root_override = root_override.unsqueeze(0)
        if root_override.dim() != 3:
            raise ValueError(
                f"Expected root override [B, T, 9], got {root_override.shape}"
            )
        if root_override.size(-1) != 9:
            raise ValueError(
                f"Expected 9 root override channels, got {root_override.shape}"
            )
        if root_override.size(1) == target_len:
            return root_override.float()

        return F.interpolate(
            root_override.float().transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _apply_root_override(
        self, decoded, root_override, mean_root=None, std_root=None, blend: float = 1.0
    ):
        target_len = decoded.size(-1)
        override_root = self._resample_root_override(
            root_override, target_len=target_len
        ).to(decoded.device)
        override_root = self._normalize_root_channels(
            override_root,
            mean_root=mean_root,
            std_root=std_root,
        )

        decoded_root = decoded[:, :9].clone().permute(0, 2, 1).contiguous()
        blend = float(blend)
        if blend < 1.0:
            override_root = blend * override_root + (1.0 - blend) * decoded_root
            override_root = self._normalize_root_channels(
                override_root,
                mean_root=mean_root,
                std_root=std_root,
            )

        decoded = decoded.clone()
        decoded[:, :9] = override_root.permute(0, 2, 1).contiguous()
        return decoded, override_root

    def forward(
        self,
        input,
        offset,
        mean_dqs,
        std_dqs,
        denorm_offsets,
        mean_root=None,
        std_root=None,
        mean_sin_cos=None,
        std_sin_cos=None,
        tags=None,
    ):
        self.decoder.training_stage = self.training_stage

        encoded = self.encoder(input.clone())
        self.latent_mean, self.latent_logvar, self.latent_kl = (
            self.encoder.get_posterior_stats()
        )

        self.latent = getattr(self.encoder, "latent_sample", None)
        if self.latent is None:
            self.latent = encoded.permute(0, 2, 1).contiguous()
        self.enc_output = self.latent
        self.lstm_output = None

        output = self.decoder(
            encoded,
            offset,
            mean_dqs,
            std_dqs,
            denorm_offsets,
            self.parents,
            smooth_root_pos=(
                tags["smooth_root_pos"]
                if tags is not None and "smooth_root_pos" in tags
                else None
            ),
        )

        root = self._normalize_root_channels(
            output[:, :9].clone().permute(0, 2, 1),
            mean_root=mean_root,
            std_root=std_root,
        )

        latent_regularizer = output.new_zeros(())
        if self.is_vae and self.latent_kl is not None:
            latent_regularizer = self.latent_kl

        return output, latent_regularizer, None, None, root, None, None, None
