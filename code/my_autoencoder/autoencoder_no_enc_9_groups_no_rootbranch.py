import torch
import torch.nn as nn
import pymotion.rotations.dual_quat_torch as dquat
from skeleton import (
    SkeletonPool,
    SkeletonUnpool,
    find_neighbor,
    SkeletonConv,
    SkeletonLinear,
    create_pooling_list,
)
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pymotion.rotations.quat_torch as quat
import pymotion.rotations.ortho6d_torch as ortho6d
import torch.nn.functional as F
from lma_transformer_prior_v5 import LMATransformerPrior

num_layers = 3
dummy_joint = False
# vae_latent_dim = 1584 # 576
# vae_latent_dim = 576
vae_latent_dim = 504
attn_dim = 448
plot = 0
training_stage = "rnn"
# training_stage = "vq_vae"
# training_stage = "debug"
# is_vq_vae = False
# In autoencoder.py


class Autoencoder(nn.Module):
    def __init__(
        self, param, parents, device, transform_net=False, is_vae=False, is_vq_vae=False
    ):
        super(Autoencoder, self).__init__()
        self.is_vae = is_vae
        self.is_vq_vae = is_vq_vae
        self.encoder = Encoder(
            param, parents, device, is_vae=self.is_vae, is_vq_vae=self.is_vq_vae
        )
        self.decoder = Decoder(param, self.encoder, device)
        self.parents = parents
        _lma_in_dim = 6  # BODY, EFFORT_WEIGHT_STRONG, EFFORT_TIME_SUDDEN, EFFORT_FLOW_BOUND, SHAPE, SPACE
        self.lma_condition_keys = [
            "BODY",
            "EFFORT_WEIGHT_STRONG",
            "EFFORT_TIME_SUDDEN",
            "EFFORT_FLOW_BOUND",
            "SHAPE",
            "SPACE",
        ]
        self.forward_context_dim = param.get(
            "root_context_dim", param.get("root_branch_dim", 64)
        )
        self.forward_ctrl_dim = 11 if param.get("aug_extended_ctrl", False) else 8
        self.forward_condition_encoder = ForwardConditionEncoder(
            in_dim=self.forward_ctrl_dim,
            out_dim=self.forward_context_dim,
            hidden_dim=max(self.forward_context_dim, self.forward_ctrl_dim * 8),
        )
        self.codebook_predictor2 = LMATransformerPrior(
            ctrl_dim=self.forward_context_dim,
            num_codebook_vectors=param["codebook_size"],
            num_levels=1,
            window_size=param["gru_window"],
            num_layers=param["gru_layers"],
            hidden_dim=param["gru_hidden_dim"],
            vq_dim=param["gru_vq_dim"],
            body_motion_dim=self.decoder.body_motion_dim,
            lma_dim=_lma_in_dim,
            p_drop_ctrl=param.get("p_drop_ctrl", 0.50),
            p_drop_lma=param.get("p_drop_lma", 0.10),
            p_drop_both=param.get("p_drop_both", 0.05),
        )
        self.enc_output = None
        self.lstm_output = None
        self.param = param
        self.training_stage = param.get("training_stage", training_stage)
        self.prior_inference_mode = param.get("prior_inference_mode", "full")
        self.enable_prior_root_override = param.get("enable_prior_root_override", True)
        self.decoder.training_stage = self.training_stage

    def set_training_stage(self, stage: str):
        self.training_stage = stage
        self.decoder.training_stage = stage

    def set_prior_inference_mode(self, mode: str):
        self.prior_inference_mode = mode

    def _normalize_root_channels(self, root, mean_root=None, std_root=None):
        if root is None:
            return None

        if mean_root is None or std_root is None:
            return ortho6d.normalize(root)

        safe_std_root = std_root.clamp_min(1e-8)
        root = root * safe_std_root + mean_root
        root = ortho6d.normalize(root)
        return (root - mean_root) / safe_std_root

    @torch.no_grad()
    def _decode_body_motion_from_prior_indices(
        self, predicted_indices, offset, mean_dqs, std_dqs
    ):
        if predicted_indices is None:
            return None

        predicted_quantized = self.encoder.indices_to_vector_(predicted_indices)
        decoder_latents = predicted_quantized.permute(0, 2, 1).contiguous()
        return self.decoder.decode_body_motion(
            decoder_latents,
            offset,
            mean_dqs,
            std_dqs,
        )

    def get_latents(self):
        return self.enc_output, self.lstm_output

    def _build_forward_controls(self, tags):
        forward_dim_parts = [
            tags["ctrl_forward_alignment"],
            tags["ctrl_lateral_alignment"],
            tags["ctrl_velocity"],
            tags["ctrl_acceleration"],
            tags["ctrl_height"],
            tags["ctrl_vertical_velocity"],
            tags["yaw_sin"],
            tags["yaw_cos"],
        ]

        if self.param.get("aug_extended_ctrl", False):
            forward_dim_parts.extend(
                [
                    tags["ctrl_yaw_rate"],
                    tags["ctrl_yaw_accel"],
                    tags["ctrl_head_height"],
                ]
            )

        return torch.cat(forward_dim_parts, dim=-1)

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
        stage = getattr(
            self, "training_stage", self.param.get("training_stage", training_stage)
        )
        self.decoder.training_stage = stage

        if all(k in tags for k in self.lma_condition_keys):
            lma_down = {key: tags[key] for key in self.lma_condition_keys}
        else:
            lma_down = None

        # zero out rots and displacements - codebook vectors must encode only motion itself,
        # not absolute pos/orientation etc.
        # not trimming the matrix because it messes with the Skeleton Aware Network
        # root is kept in encoder input for root-aware encoding
        input[:, :9] = input[:, :9] * 0
        # input[:,0,:] = 1
        input[:, -4:] = input[:, -4:] * 0

        quantized, encoding_indices, vq_loss, encoding_indices_all = self.encoder(
            input.clone()
        )
        self.latent = quantized
        if (
            self.training
            and self.encoder.is_vq_vae
            and hasattr(self.encoder, "ema_cluster_size")
            and stage == "vq_vae"
        ):

            try:
                idxs = encoding_indices_all
                if idxs.dim() == 3:
                    # Update each level separately
                    for lvl in range(1):  # range(idxs.size(2)):
                        idxs_lvl = idxs[..., lvl]  # [B,T]

                        # For level 0, use encoder.latent
                        # For level 1, encoder.residual will be used inside ema_update
                        if (
                            hasattr(self.encoder, "latent")
                            and int(self.encoder.ema_updates.item()) > 0
                        ):
                            self.encoder.ema_update(
                                idxs_lvl, self.encoder.latent, lvl=lvl
                            )

                    # Reset unused vectors (only needed for level 0 typically)
                    if (
                        int(self.encoder.ema_updates.item()) % self.param["ema_updates"]
                        == 0
                        and int(self.encoder.ema_updates.item()) > 0
                    ):

                        for lvl in range(1):  # range(idxs.size(2)):
                            self.encoder.reset_unused_codebook_vectors(
                                min_usage_frac=max(
                                    1e-4, 0.25 / float(self.encoder.num_embeddings)
                                ),
                                sample_z=(
                                    self.encoder.latent
                                    if lvl == 0
                                    else self.encoder.residual
                                ),
                                lvl=lvl,
                            )
                else:
                    # Backward compatibility for single level
                    self.encoder.ema_update(idxs, self.encoder.latent, lvl=0)
                    if (
                        int(self.encoder.ema_updates.item()) % self.param["ema_updates"]
                        == 0
                    ):
                        self.encoder.reset_unused_codebook_vectors(
                            min_usage_frac=max(
                                1e-4, 0.25 / float(self.encoder.num_embeddings)
                            ),
                            sample_z=self.encoder.latent,
                            lvl=0,
                        )
            except Exception as e:
                print(f"EMA update error: {e}")

        B = tags["yaw_sin"].shape[0]
        device = tags["yaw_sin"].device
        phi = torch.rand(B, device=device) * (4.0 * math.pi) - 2.0 * math.pi

        if (
            mean_sin_cos is not None
            and self.training
            and (stage == "vq_vae" or stage == "refiner")
        ):
            mean_sin = mean_sin_cos[0]
            mean_cos = mean_sin_cos[1]
            std_sin = std_sin_cos[0]
            std_cos = std_sin_cos[1]

            sin = (tags["yaw_sin"] * std_sin) + mean_sin
            cos = (tags["yaw_cos"] * std_cos) + mean_cos

            sin_phi = torch.sin(phi).view(B, 1, 1)
            cos_phi = torch.cos(phi).view(B, 1, 1)

            sin_rot = sin * cos_phi + cos * sin_phi
            cos_rot = cos * cos_phi - sin * sin_phi

            tags["yaw_sin"] = (sin_rot - mean_sin) / std_sin
            tags["yaw_cos"] = (cos_rot - mean_cos) / std_cos
        else:
            # phi *= 0
            phi = None

        forward_controls = self._build_forward_controls(tags)
        forward_context = self.forward_condition_encoder(forward_controls)

        batch, seq_len_times_8, _ = forward_controls.shape
        seq_len = seq_len_times_8 // 8 // self.encoder.group_size
        if self.training:
            encoding_indices = encoding_indices.view(batch, seq_len, -1)
        else:
            encoding_indices = encoding_indices.unsqueeze(0)

        predicted_indices = encoding_indices_all
        primary_logits = None
        forward_context_predicted = None
        prior_root_prediction = None
        codebooks = None  # self.encoder.vq_codebooks
        real_signal_context = torch.cat(
            [
                tags["yaw_sin"],
                tags["yaw_cos"],
                tags["ctrl_forward_alignment"],
                tags["ctrl_lateral_alignment"],
            ],
            dim=-1,
        )

        if stage == "rnn":
            prior_mode = (
                "full"
                if self.training
                else getattr(self, "prior_inference_mode", "full")
            )
            self.codebook_predictor2.set_body_motion_decoder(
                lambda indices: self._decode_body_motion_from_prior_indices(
                    indices,
                    offset=offset,
                    mean_dqs=mean_dqs,
                    std_dqs=std_dqs,
                )
            )
            prior_output = self.codebook_predictor2(
                forward_context,
                target_indices=encoding_indices_all,
                codebooks=codebooks,
                yaw_sin_cos=real_signal_context,
                lma_down=lma_down,
                mode=prior_mode,
            )

            # v5 returns indices/logits, while v6 also returns a predicted context.
            if isinstance(prior_output, tuple) and len(prior_output) == 4:
                (
                    predicted_indices,
                    primary_logits,
                    forward_context_predicted,
                    prior_root_prediction,
                ) = prior_output
            elif isinstance(prior_output, tuple) and len(prior_output) == 3:
                predicted_indices, primary_logits, forward_context_predicted = (
                    prior_output
                )
            else:
                predicted_indices, primary_logits = prior_output
                forward_context_predicted = forward_context

            forward_context = forward_context_predicted

        visualize = False
        if visualize and primary_logits is not None:
            temperature = 1.0
            B, T_ds, L, K = primary_logits.shape
            for t in range(T_ds):
                plt.figure(figsize=(8, 3))
                logits_level0 = (
                    (primary_logits[0, t, 0] / temperature).detach().cpu().numpy()
                )
                # Print top 3 logits and their indices
                top3_logits_idx = logits_level0.argsort()[-3:][::-1]
                top3_logits = logits_level0[top3_logits_idx]
                print(
                    f"Frame {t}: Top 3 logits indices: {top3_logits_idx}, Logits: {top3_logits}"
                )
                probs = F.softmax(primary_logits[0, t] / temperature, dim=-1)  # [L, K]
                # if single level:
                probs_level0 = probs[0].cpu().numpy()
                # Print top 3 indices and probabilities
                top3_idx = probs_level0.argsort()[-3:][::-1]
                top3_probs = probs_level0[top3_idx]
                print(
                    f"Frame {t}: Top 3 prob indices: {top3_idx}, Probabilities: {top3_probs}"
                )
                plt.bar(range(K), probs_level0, width=3.8)
                plt.xlabel("codebook index")
                plt.ylabel("probability")
                plt.title(f"Frame {t} (downsampled) - temperature={temperature}")
                plt.ylim(0, 1)
                plt.show()

        predicted_quantized = self.encoder.indices_to_vector_(predicted_indices)
        if self.encoder.ema_updates > 0 and (stage == "vq_vae" or stage == "refiner"):
            to_decoder = quantized
        else:
            to_decoder = self.encoder.enc_out
            vq_loss *= 0

        decoder_forward_context = forward_context
        if stage == "rnn":
            to_decoder = predicted_quantized.permute(0, 2, 1).clone()

        output = self.decoder(
            to_decoder,
            offset,
            mean_dqs,
            std_dqs,
            denorm_offsets,
            self.parents,
            smooth_root_pos=tags["smooth_root_pos"],
            ctrl_embedding=decoder_forward_context,
            vq_out=to_decoder,
        )

        normalized_prior_root = self._normalize_root_channels(
            prior_root_prediction,
            mean_root=mean_root,
            std_root=std_root,
        )
        if normalized_prior_root is not None and self.enable_prior_root_override:
            output = torch.cat(
                [
                    normalized_prior_root.permute(0, 2, 1),
                    output[:, 9:, :],
                ],
                dim=1,
            )

        root = self._normalize_root_channels(
            output[:, :9].clone().permute(0, 2, 1),
            mean_root=mean_root,
            std_root=std_root,
        )

        return (
            output,
            vq_loss,
            primary_logits,
            encoding_indices_all,
            root,
            phi,
            forward_context_predicted,
            normalized_prior_root,
        )


class ConvLayerNorm(nn.Module):
    """
    Apply LayerNorm to Conv1d outputs (input [B, C, T]).
    Behavior: normalize across channel dim per time step (permute -> LayerNorm -> permute back).
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, T, C] -> LayerNorm over channels -> [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        return x.permute(0, 2, 1)


class LMAEncoder(nn.Module):
    """
    Lightweight 1D-conv encoder for LMA annotation features.
    Applies a single stride-2 downsampling layer followed by a refinement
    conv at half resolution, expanding channels for richer representation.

    Input:  [B, T,     in_dim]
    Output: [B, T//2, out_dim]
    """

    def __init__(
        self, in_dim: int, out_dim: int, kernel_size: int = 7, dropout: float = 0.1
    ):
        super().__init__()
        mid_dim = max(out_dim // 2, in_dim)
        self.net = nn.Sequential(
            # ---- stride-2 downsample ----
            nn.Conv1d(
                in_dim,
                mid_dim,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
            ),
            ConvLayerNorm(mid_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            # ---- refine at half resolution ----
            nn.Conv1d(
                mid_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            ConvLayerNorm(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in_dim]  →  [B, T//2, out_dim]
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class ForwardConditionEncoder(nn.Module):
    """Encode full-rate root controls into a dense forward-context sequence."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = None,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(out_dim, in_dim * 8) if hidden_dim is None else hidden_dim
        padding = kernel_size // 2
        dilation_padding = padding * 2

        self.in_proj = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            ConvLayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.res_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            ConvLayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=2,
                padding=dilation_padding,
            ),
            ConvLayerNorm(hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=1),
            ConvLayerNorm(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.in_proj(x.permute(0, 2, 1))
        hidden = F.leaky_relu(hidden + self.res_block(hidden), negative_slope=0.2)
        return self.out_proj(hidden).permute(0, 2, 1)


class Encoder(nn.Module):
    def __init__(self, param, parents, device, is_vae=False, is_vq_vae=False):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.convs = []
        self.convs_2 = []
        self.parents = [parents]
        self.pooling_lists = []
        self.channel_list = []
        self.channel_list_2 = []
        self.is_vae = is_vae
        self.is_vq_vae = is_vq_vae
        self.device = device
        self.param = param

        kernel_size = 9  # param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        stride = param["stride_encoder_conv"]

        survivors_list = []

        # Compute pooled skeletons
        number_layers = num_layers
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(
                layer_parents, l == number_layers - 1
            )

            # original joint indices that survived (not pooled away)
            survivors = [grp[0] for grp in pooling_list]
            survivors_list.append(survivors)
            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)

        default_channels_per_joints = 9
        self.channel_size = [default_channels_per_joints] + [
            (default_channels_per_joints) * (2**i) for i in range(1, number_layers + 1)
        ]  # first each joint has 8 (dual quaternion) channels, then we multiply by 2 every layer to increase the number of conv. filters
        low_res_parents = self.parents[-1]
        neighbor_list, _ = find_neighbor(
            low_res_parents, param["neighbor_distance"], add_displacement=True  # <---
        )
        self.num_joints = len(neighbor_list)  # it includes the displacement fake joint
        if dummy_joint:
            self.num_joints += 1

        for l in range(number_layers):
            seq = []
            in_channels = self.channel_size[l] * self.num_joints
            out_channels = self.channel_size[l + 1] * self.num_joints
            if l == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            if l == number_layers - 1:
                last_layer_scale = 1
            else:
                last_layer_scale = 1

            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[l],
                    out_channels_per_joint=self.channel_size[l + 1] * last_layer_scale,
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=padding,
                    stride=stride * last_layer_scale,
                    device=device,
                    add_offset=False,
                )
            )

            self.convs.append(seq[-1])
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

        vq_dim = self.channel_list[-1] * last_layer_scale  # self.channel_list[0] * 8
        self.group_size = 1
        self.grouped_vq_dim = vq_dim * self.group_size

        base_ch = self.channel_list[0]  # vq_dim / input channels
        in_ch_per_joint = base_ch // max(1, self.num_joints)
        num_stages = 3
        kernel_size = 3
        # Encoder: per-stage SkeletonConv with stride=2
        enc_layers = []
        for i in range(num_stages):
            in_c = in_ch_per_joint * (2**i)
            out_c = in_ch_per_joint * (2 ** (i + 1))
            enc_layers.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=in_c,
                    out_channels_per_joint=out_c,
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=kernel_size // 2,
                    stride=2,
                    device=device,
                    add_offset=False,
                )
            )
            enc_layers.append(nn.LeakyReLU(0.2))
        # Projection layer for logits

        self.num_embeddings = self.param["codebook_size"]
        self.logits_proj = nn.Linear(vq_dim, self.num_embeddings)
        self.logits_proj_grouped = nn.Linear(self.grouped_vq_dim, self.num_embeddings)

        self.num_quantizers = 1
        self.vq_codebooks = nn.ModuleList(
            [
                nn.Embedding(self.num_embeddings, self.grouped_vq_dim)
                for _ in range(self.num_quantizers)
            ]
        )
        for codebook in self.vq_codebooks:
            codebook.weight.data.uniform_(
                -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
            )
            # codebook.weight.data.uniform_(-100.0 / self.num_embeddings, 100.0 / self.num_embeddings)

        self.commitment_cost = 0.75  # / 2

        self.use_codebook = False

        # ---------- EMA buffers for codebook maintenance ----------
        self.ema_decay = self.param["ema_decay"]
        self.ema_eps = 1e-5

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        # ema_w must match the codebook embedding dimensionality (grouped_vq_ddim)
        self.register_buffer(
            "ema_w", torch.zeros(self.num_embeddings, self.grouped_vq_dim)
        )
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

    # ================= EMA & Reset Utilities =================
    def incr_ema_updates(self):
        with torch.no_grad():
            self.ema_updates += 1

    def _get_level_ema_buffers(self, lvl: int = 0):
        if lvl == 0:
            return self.ema_cluster_size, self.ema_w
        if hasattr(self, f"ema_cluster_size_lvl{lvl}") and hasattr(
            self, f"ema_w_lvl{lvl}"
        ):
            return getattr(self, f"ema_cluster_size_lvl{lvl}"), getattr(
                self, f"ema_w_lvl{lvl}"
            )
        return None, None

    def _sample_codebook_replacements(
        self, sample_z: torch.Tensor, num_reset: int, dim: int, device
    ):
        if sample_z is not None and sample_z.numel() > 0:
            flat = sample_z.reshape(-1, dim)
            if flat.size(0) >= num_reset:
                rand_idx = torch.randperm(flat.size(0), device=flat.device)[:num_reset]
            else:
                rand_idx = torch.randint(
                    0, flat.size(0), (num_reset,), device=flat.device
                )
            new_vecs = flat[rand_idx].clone()
            noise_scale = flat.std(dim=0, unbiased=False).mean().clamp_min(1e-3) * 0.01
            return new_vecs + torch.randn_like(new_vecs) * noise_scale

        return torch.randn(num_reset, dim, device=device) * 0.02

    @torch.no_grad()
    def ema_update(
        self, indices: torch.Tensor, encoder_latent: torch.Tensor, lvl: int = 0
    ):
        """
        Update EMA statistics and refresh codebook weights for the specified level.
        Args:
            indices: [B,T] indices for the specified level
            encoder_latent: [B,T,D] latent before quantization (or residual for lvl>0)
            lvl: Level to update (default 0 - first level)
        """
        if indices is None or encoder_latent is None:
            return

        # # Use residual for level 1 if available
        if lvl == 1 and hasattr(self, "residual"):
            source_z = self.residual  # Use stored residual for level 1
        else:
            source_z = encoder_latent  # Use original latents for level 0
        # source_z = encoder_latent
        # Ensure we have appropriate EMA buffers for this level
        if not hasattr(self, f"ema_cluster_size_lvl{lvl}"):
            # Create level-specific EMA buffers if they don't exist
            if lvl == 0:
                # Level 0 uses the original buffers
                ema_cluster_size = self.ema_cluster_size
                ema_w = self.ema_w
            else:
                # For other levels, create new buffers
                self.register_buffer(
                    f"ema_cluster_size_lvl{lvl}",
                    torch.zeros_like(self.ema_cluster_size),
                )
                self.register_buffer(f"ema_w_lvl{lvl}", torch.zeros_like(self.ema_w))
                ema_cluster_size = getattr(self, f"ema_cluster_size_lvl{lvl}")
                ema_w = getattr(self, f"ema_w_lvl{lvl}")
        else:
            # Use existing level-specific buffers
            ema_cluster_size = getattr(self, f"ema_cluster_size_lvl{lvl}")
            ema_w = getattr(self, f"ema_w_lvl{lvl}")

        # Flatten indices and source vectors
        flat_idx = indices.reshape(-1).clamp(0, self.num_embeddings - 1)  # [N]
        flat_z = source_z.reshape(-1, source_z.size(-1))  # [N,D]

        # Compute counts and weighted sum
        # one_hot = F.one_hot(flat_idx, num_classes=self.num_embeddings).float()  # [N,K]
        # counts = one_hot.sum(0)                                                # [K]
        # dw = one_hot.t() @ flat_z                                              # [K,D]

        # Replacement (tiny memory):
        counts = torch.zeros(self.num_embeddings, device=flat_idx.device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float))
        dw = torch.zeros(self.num_embeddings, flat_z.size(-1), device=flat_idx.device)
        dw.scatter_add_(0, flat_idx.unsqueeze(1).expand(-1, flat_z.size(-1)), flat_z)

        # Update EMA statistics for this level
        ema_cluster_size.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Compute new weights
        n = ema_cluster_size.sum()
        if n.item() == 0:
            return
        cluster_size = (
            (ema_cluster_size + self.ema_eps)
            / (n + self.num_embeddings * self.ema_eps)
            * n
        )
        denom = cluster_size.unsqueeze(1).clamp_min(self.ema_eps)
        new_weight = ema_w / denom

        # Update the appropriate codebook weights
        self.vq_codebooks[lvl].weight.data.copy_(new_weight)

    @torch.no_grad()
    def reset_unused_codebook_vectors(
        self, min_usage_frac=1e-3, sample_z: torch.Tensor = None, lvl=0
    ):
        """
        Reinitialize rarely used codebook entries at specified level.
        Args:
            min_usage_frac: usage threshold
            sample_z: optional [B,T,D] tensor to sample replacements from
            lvl: level to reset (default 0)
        """
        # Get appropriate EMA buffers for this level
        ema_cluster_size, ema_w = self._get_level_ema_buffers(lvl)
        if ema_cluster_size is None or ema_w is None:
            print(f"No EMA cluster size for level {lvl}, skipping reset")
            return

        # Calculate usage statistics
        total = ema_cluster_size.sum().clamp_min(1.0)
        usage = ema_cluster_size / total
        mask = usage < min_usage_frac

        # Skip if no vectors need resetting
        if not mask.any():
            return

        # Determine how many vectors to reset
        num_reset = mask.sum().item()
        D = self.vq_codebooks[lvl].weight.size(1)

        new_vecs = self._sample_codebook_replacements(
            sample_z=sample_z,
            num_reset=num_reset,
            dim=D,
            device=self.vq_codebooks[lvl].weight.device,
        )

        # Reset vectors and synchronize EMA state so the next EMA refresh keeps
        # the respawned vectors instead of snapping back to stale centers.
        bootstrap_count = torch.full(
            (num_reset,),
            float(max(min_usage_frac * total.item(), self.ema_eps)),
            device=ema_cluster_size.device,
        )
        self.vq_codebooks[lvl].weight.data[mask] = new_vecs
        ema_cluster_size.data[mask] = bootstrap_count
        ema_w.data[mask] = new_vecs * bootstrap_count.unsqueeze(1)

    # =============== Group Stuff ================#
    def _group_latent(self, z: torch.Tensor):
        """
        Group consecutive timesteps (size=self.group_size) and concatenate features.
        Args:
            z: [B, T, D]
        Returns:
            z_group: [B, G, group_size*D]
            T_orig: original T
            pad: number of padded frames (0 or group_size-1)
        """
        B, T, D = z.shape
        g = self.group_size
        pad = (g - (T % g)) % g
        if pad > 0:
            z = torch.cat(
                [z, torch.zeros(B, pad, D, device=z.device, dtype=z.dtype)], dim=1
            )
        T_new = z.size(1)
        G = T_new // g
        z_group = z.view(B, G, g, D).reshape(B, G, g * D)
        return z_group, T, pad

    def split_quantized_groups(self, grouped: torch.Tensor, T_orig: int, pad: int):
        """
        Split grouped quantized embeddings back to per-frame embeddings.
        Args:
            grouped: [B, G, group_size*D]
            T_orig: original (unpadded) sequence length
            pad: number of padded frames added
        Returns:
            per_frame: [B, T_orig, D]
        """
        B, G, GD = grouped.shape
        g = self.group_size
        D = GD // g
        per_frame = grouped.view(B, G, g, D).reshape(B, G * g, D)
        if pad > 0:
            per_frame = per_frame[:, :T_orig]
        return per_frame

    def forward(self, input):
        expected_channels = self.channel_size[0] * self.num_joints
        if input.shape[1] < expected_channels:
            pad_channels = expected_channels - input.shape[1]
            input = torch.cat(
                [
                    input,
                    torch.zeros(
                        input.shape[0],
                        pad_channels,
                        input.shape[2],
                        dtype=input.dtype,
                        device=input.device,
                    ),
                ],
                dim=1,
            )
        elif input.shape[1] > expected_channels:
            raise ValueError(
                f"Encoder expected at most {expected_channels} sparse channels, got {input.shape[1]}"
            )

        for i, layer in enumerate(self.layers):
            input = layer(input)

        self.enc_out = input.clone()

        #####################
        if self.is_vq_vae:

            if self.training:
                self.incr_ema_updates()

            z = input.permute(0, 2, 1).clone()
            B, T, D = z.shape

            # Group consecutive timesteps and concatenate features
            z_grouped, T_orig, pad = self._group_latent(
                z
            )  # [B, G, group_size*D] where G = T//group_size
            self.latent = z_grouped.clone()
            B_g, G, GD = z_grouped.shape
            flat_inputs = z_grouped.reshape(-1, GD)  # [B*G, group_size*vq_dim]

            # Residual vector quantization on grouped features
            device = flat_inputs.device
            residual = flat_inputs.clone()
            quantized_sum = torch.zeros_like(residual, device=device)
            all_indices = []
            vq_loss_total = torch.tensor(0.0, device=device)

            for level, codebook in enumerate(self.vq_codebooks):
                # Compute distances between residual and codebook vectors
                # distances: [N, K]

                distances = (
                    torch.sum(residual * residual, dim=1, keepdim=True)  # [N,1]
                    + torch.sum(codebook.weight * codebook.weight, dim=1).unsqueeze(
                        0
                    )  # [1,K]
                    - 2.0 * (residual @ codebook.weight.t())  # [N,K]
                )

                #####
                topk = torch.topk(distances, k=3, dim=1, largest=False)
                topk_indices = topk.indices  # [N, 3]
                idx0 = topk_indices[:, 0]  # always the closest
                N = distances.size(0)
                # 7.5% chance to pick randomly among top 3 (excluding argmin)
                rand = torch.rand(N, device=distances.device)
                choose_random = rand < 0.075
                choose_random = rand < -1
                idx_final = idx0.clone()

                if choose_random.any():
                    # For those, randomly pick 1 or 2 (not 0)
                    alt_idx = torch.randint(
                        1, 3, (choose_random.sum(),), device=distances.device
                    )
                    idx_final[choose_random] = topk_indices[choose_random, alt_idx]
                encoding_indices = idx_final.unsqueeze(1)  # [N,1]

                #####

                all_indices.append(encoding_indices)

                # One-hot and lookup
                N = encoding_indices.size(0)
                encodings = torch.zeros(N, codebook.num_embeddings, device=device)
                encodings.scatter_(1, encoding_indices, 1.0)
                quantized = encodings @ codebook.weight  # [N, GD]

                # VQ losses (pull codebook to residual and encoder to codebook)
                codebook_loss = F.mse_loss(quantized, residual.detach())
                commitment_loss = self.commitment_cost * F.mse_loss(
                    quantized.detach(), residual
                )
                vq_loss = codebook_loss * 0 + commitment_loss
                vq_loss_total = vq_loss_total + vq_loss

                # Straight-through estimator for this level
                quantized_st = residual + (quantized - residual).detach()
                # Accumulate quantized vectors and update residual
                quantized_sum = quantized_sum + quantized_st
                residual = residual - quantized_st
                if level == 1:
                    self.residual = quantized.clone()

            # Reshape outputs - keep in grouped (downsampled) form
            encoding_indices = torch.cat(all_indices, dim=1)  # [N, L]
            encoding_indices = encoding_indices.view(B, G, -1)  # [B, G, L]
            quantized_grouped = quantized_sum.view(
                B, G, GD
            )  # [B, G, group_size*vq_dim]

            # Split back to per-frame for decoder (returns [B, T_orig, D_per_frame])
            quantized_per_frame = self.split_quantized_groups(
                quantized_grouped, T_orig, pad
            )

            return (
                quantized_per_frame.permute(0, 2, 1),
                encoding_indices[:, :, 0].clone(),
                vq_loss_total,
                encoding_indices.clone(),
            )

        return input

    def indices_to_vector_(self, indices):
        """
        Converts a sequence of indices to a sequence of codebook vectors.
        Args:
            indices: Tensor of shape [batch, seq_len, num_quantizers] containing indices.
        Returns:
            vectors: Tensor of shape [batch, seq_len, vq_dim] containing codebook vectors.
        """
        if indices.dim() == 2:  # Handle single-level indices for compatibility
            indices = indices.unsqueeze(-1)

        B, T, L = indices.shape
        vectors = (
            torch.zeros_like(self.vq_codebooks[0].weight[0])
            .expand(B, T, -1)
            .to(indices.device)
        )

        for l in range(L):
            vectors = vectors + self.vq_codebooks[l](
                indices[..., l]
            )  # Sum embeddings from all levels
        vectors = self.split_quantized_groups(vectors, T * self.group_size, pad=0)

        return vectors


class Decoder(nn.Module):
    def __init__(self, param, enc: Encoder, device):
        super(Decoder, self).__init__()

        self.param = param
        self.device = device
        self.training_stage = param.get("training_stage", training_stage)
        self.layers = nn.ModuleList()
        self.convs = []
        self.is_vae = enc.is_vae
        self.is_vq_vae = enc.is_vq_vae
        self.channels_last_layer = enc.channel_list[-1]
        self.stride = param["stride_encoder_conv"]

        latent_rot_dim = 0
        if self.is_vae or self.is_vq_vae:
            self.fc_dec = nn.Linear(
                vae_latent_dim + latent_rot_dim, enc.channel_list[-1]
            )

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        number_layers = num_layers
        default_channels_per_joints = 9
        self.channel_size = [default_channels_per_joints] + [
            default_channels_per_joints * (2**i) for i in range(1, number_layers + 1)
        ]  # first each joint has 8 channels, after collapse 16, then 32...
        for i in range(number_layers):
            seq = []
            neighbor_list, _ = find_neighbor(
                enc.parents[number_layers - i - 1], param["neighbor_distance"]
            )
            num_joints = len(neighbor_list)
            self.num_joints = num_joints
            if dummy_joint:
                num_joints += 1
            unpool = SkeletonUnpool(
                pooling_list=enc.pooling_lists[number_layers - i - 1],
                channels_per_edge=self.channel_size[number_layers - i],
                device=device,
            )
            seq.append(
                nn.Upsample(
                    scale_factor=self.stride, mode="linear", align_corners=False
                )
            )
            seq.append(unpool)
            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[number_layers - i],
                    out_channels_per_joint=self.channel_size[number_layers - i] // 2,
                    joint_num=num_joints,
                    in_offset_channel=3
                    * enc.channel_size[number_layers - i - 1]
                    // enc.channel_size[0],
                    padding=padding,
                    stride=1,
                    device=device,
                    add_offset=True,
                )
            )
            self.convs.append(seq[-1])

            if i != number_layers - 1:
                # seq.append(nn.Dropout(p=0.2))
                seq.append(nn.LeakyReLU(negative_slope=0.2))
                # seq.append(ConvLayerNorm(num_joints * self.channel_size[number_layers-i]//2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

        if param["input_proj"] == -1:
            self.project_input = False
            self.input_proj_dim = vae_latent_dim
        else:
            self.project_input = True
            self.input_proj_dim = param["input_proj"]

        self.input_proj = nn.Sequential(nn.Linear(vae_latent_dim, self.input_proj_dim))

        self.aug_multires_vq = param.get("aug_multires_vq", False)
        self.aug_vel_residual = param.get("aug_vel_residual", False)
        self.synthetic_contact_joint_count = int(
            param.get("synthetic_contact_joint_count", 0)
        )

        # =====================================================================
        # Dual-stream skeleton-aware root predictor
        # Stream 1 (coarse): quantized VQ output at T/8, skeleton-aware at low-res
        # Stream 2 (fine):   decoder output at T, skeleton-aware at full-res
        # =====================================================================

        self.root_branch_dim = param.get("root_branch_dim", 64)
        _rbd = self.root_branch_dim
        self.forward_context_dim = param.get("root_context_dim", _rbd)
        self.body_motion_dim = self.channel_size[0] * max(0, self.num_joints - 1)

        # -- Coarse stream: SkeletonConv on VQ output at low-res topology, stays at T/8 --
        # enc.parents[-1] is the low-res skeleton; enc.num_joints includes displacement
        _low_res_parents = enc.parents[-1]
        _low_res_neighbor_list, _ = find_neighbor(
            _low_res_parents, param["neighbor_distance"], add_displacement=True
        )
        _J_low = enc.num_joints  # includes displacement joint
        _enc_ch_top = enc.channel_size[-1]  # 72 channels/joint at encoder output
        self.coarse_dim = 32

        self.coarse_skel_conv1 = SkeletonConv(
            param=param,
            neighbor_list=_low_res_neighbor_list,
            kernel_size=5,
            in_channels_per_joint=_enc_ch_top,
            out_channels_per_joint=_enc_ch_top // 2,
            joint_num=_J_low,
            in_offset_channel=1,
            padding=2,
            stride=1,
            device=device,
            add_offset=False,
        )
        self.coarse_norm1 = ConvLayerNorm((_enc_ch_top // 2) * _J_low)
        self.coarse_skel_conv2 = SkeletonConv(
            param=param,
            neighbor_list=_low_res_neighbor_list,
            kernel_size=5,
            in_channels_per_joint=_enc_ch_top // 2,
            out_channels_per_joint=_enc_ch_top // 4,
            joint_num=_J_low,
            in_offset_channel=1,
            padding=2,
            stride=1,
            device=device,
            add_offset=False,
        )
        self.coarse_norm2 = ConvLayerNorm((_enc_ch_top // 4) * _J_low)
        # Compress across joints (still at T/8), then upsample only coarse_dim channels
        self.coarse_compress = nn.Sequential(
            nn.Conv1d((_enc_ch_top // 4) * _J_low, self.coarse_dim, kernel_size=1),
            ConvLayerNorm(self.coarse_dim),
            nn.LeakyReLU(0.2),
        )
        self.coarse_upsample = nn.Upsample(
            scale_factor=8, mode="linear", align_corners=False
        )

        # -- Fine stream: SkeletonConv on decoder output at full-res topology --
        _full_res_parents = enc.parents[0]
        _full_res_neighbor_list, _ = find_neighbor(
            _full_res_parents, param["neighbor_distance"], add_displacement=False
        )
        _J_full = len(_full_res_neighbor_list)
        self._J_full_no_disp = _J_full  # store for forward
        _fine_per_joint = 16
        self.fine_dim = 64

        self.fine_skel_conv1 = SkeletonConv(
            param=param,
            neighbor_list=_full_res_neighbor_list,
            kernel_size=7,
            in_channels_per_joint=self.channel_size[0],  # 9
            out_channels_per_joint=_fine_per_joint,
            joint_num=_J_full,
            in_offset_channel=1,
            padding=3,
            stride=1,
            device=device,
            add_offset=False,
        )
        self.fine_norm1 = ConvLayerNorm(_fine_per_joint * _J_full)
        self.fine_skel_conv2 = SkeletonConv(
            param=param,
            neighbor_list=_full_res_neighbor_list,
            kernel_size=7,
            in_channels_per_joint=_fine_per_joint,
            out_channels_per_joint=_fine_per_joint,
            joint_num=_J_full,
            in_offset_channel=1,
            padding=3,
            stride=1,
            device=device,
            add_offset=False,
        )
        self.fine_norm2 = ConvLayerNorm(_fine_per_joint * _J_full)
        self.fine_compress = nn.Sequential(
            nn.Conv1d(_fine_per_joint * _J_full, self.fine_dim, kernel_size=1),
            ConvLayerNorm(self.fine_dim),
            nn.LeakyReLU(0.2),
        )

        # -- Fusion: coarse + fine + forward context → root_branch_dim --
        self.root_input_fuse = nn.Sequential(
            nn.Conv1d(
                self.coarse_dim + self.fine_dim + self.forward_context_dim,
                _rbd,
                kernel_size=1,
            ),
            ConvLayerNorm(_rbd),
            nn.LeakyReLU(0.2),
        )

        # -- Dilated conv stack (temporal, stride=1 throughout) --
        _rk = 15
        self.root_predictor = nn.Sequential(
            nn.Conv1d(_rbd, _rbd, kernel_size=_rk, stride=1, padding=_rk // 2),
            ConvLayerNorm(_rbd),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv1d(
                _rbd,
                _rbd,
                kernel_size=_rk,
                stride=1,
                dilation=2,
                padding=(_rk // 2) * 2,
            ),
            ConvLayerNorm(_rbd),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Conv1d(
                _rbd,
                _rbd,
                kernel_size=_rk,
                stride=1,
                dilation=4,
                padding=(_rk // 2) * 4,
            ),
            ConvLayerNorm(_rbd),
            nn.LeakyReLU(0.2),
            nn.Conv1d(_rbd, _rbd // 2, kernel_size=_rk, stride=1, padding=_rk // 2),
            ConvLayerNorm(_rbd // 2),
            nn.LeakyReLU(0.2),
        )

        # -- Split heads: ortho6d rotation (6ch) + displacement (3ch) --
        self.root_rot_head = nn.Conv1d(_rbd // 2, 6, kernel_size=7, stride=1, padding=3)
        self.root_disp_head = nn.Conv1d(
            _rbd // 2, 3, kernel_size=7, stride=1, padding=3
        )

    def _run_decoder_layers(self, input: torch.Tensor, offset) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        return input

    def _normalize_motion_channels(
        self,
        motion: torch.Tensor,
        mean_dqs: torch.Tensor,
        std_dqs: torch.Tensor,
    ) -> torch.Tensor:
        safe_std = std_dqs.clamp_min(1e-8)
        motion = motion * safe_std.unsqueeze(-1) + mean_dqs.unsqueeze(-1)
        motion = motion.reshape(motion.shape[0], -1, 9, motion.shape[-1]).permute(
            0,
            3,
            1,
            2,
        )
        synthetic_count = max(int(getattr(self, "synthetic_contact_joint_count", 0)), 0)
        if synthetic_count > 0:
            skeletal_motion = motion[:, :, :-synthetic_count, :]
            synthetic_motion = motion[:, :, -synthetic_count:, :]
            skeletal_motion = ortho6d.normalize(skeletal_motion)
            motion = torch.cat([skeletal_motion, synthetic_motion], dim=2)
        else:
            motion = ortho6d.normalize(motion)
        motion = motion.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        return (motion - mean_dqs.unsqueeze(-1)) / safe_std.unsqueeze(-1)

    def decode_body_motion(
        self,
        input: torch.Tensor,
        offset,
        mean_dqs: torch.Tensor,
        std_dqs: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self._run_decoder_layers(input, offset)
        body_motion = decoded[:, 9:, :]
        return self._normalize_motion_channels(
            body_motion,
            mean_dqs[9:],
            std_dqs[9:],
        )

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
            input = torch.cat(
                [input, root_quats.permute(0, 2, 1)], dim=2
            )  # add root rotations to the input

        if self.project_input:
            input_proj = self.input_proj(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            input_proj = input

        if ctrl_embedding is None:
            ctrl_embedding = ctrl

        # ---- Coarse stream: skeleton-aware processing of VQ output at T/8 ----
        if vq_out is not None:
            # vq_out: [B, vq_dim, T/8] — quantized encoder output
            # Keep the old frozen-prior behavior in rnn, but allow recon gradients
            # through the coarse stream in rnn2 and vq_vae stages.
            stage = getattr(
                self, "training_stage", self.param.get("training_stage", training_stage)
            )
            coarse_in = vq_out.detach() if stage == "rnn" else vq_out
            coarse = F.leaky_relu(
                self.coarse_norm1(self.coarse_skel_conv1(coarse_in)), 0.2
            )
            coarse = F.leaky_relu(
                self.coarse_norm2(self.coarse_skel_conv2(coarse)), 0.2
            )
            coarse = self.coarse_compress(coarse)  # [B, 32, T/8]
            coarse = self.coarse_upsample(coarse)  # [B, 32, T]
        else:
            coarse = None

        # ---- Decoder conv layers: upsample + unpool skeleton ----
        input = self._run_decoder_layers(input, offset)

        # ---- Fine stream: skeleton-aware processing of decoder output at T ----
        # Decoder output is [B, 9*J_full_with_disp, T]; fine SkeletonConv expects
        # 9*J_full_no_disp channels (no displacement joint).
        _fine_ch = self.channel_size[0] * self._J_full_no_disp  # 9 * J_full
        fine_in = input[:, :_fine_ch, :].clone()  # .detach()
        fine = F.leaky_relu(self.fine_norm1(self.fine_skel_conv1(fine_in)), 0.2)
        fine = F.leaky_relu(self.fine_norm2(self.fine_skel_conv2(fine)), 0.2)
        fine = self.fine_compress(fine)  # [B, 64, T]

        # ---- Fusion & root prediction: coarse + fine + forward context → 9 root channels ----
        if ctrl_embedding is not None:
            ctrl_feat = ctrl_embedding.permute(0, 2, 1)

            if ctrl_feat.shape[-1] != fine.shape[-1]:
                ctrl_feat = F.interpolate(
                    ctrl_feat, size=fine.shape[-1], mode="linear", align_corners=False
                )

            # Align coarse temporal dim to T (handle minor rounding mismatches)
            if coarse is not None and coarse.shape[-1] != fine.shape[-1]:
                coarse = F.interpolate(
                    coarse, size=fine.shape[-1], mode="linear", align_corners=False
                )

            if coarse is not None:
                fuse_in = torch.cat([coarse, fine, ctrl_feat], dim=1)
            else:
                # Fallback: zeros for coarse if vq_out was not provided
                fuse_in = torch.cat(
                    [
                        torch.zeros(
                            fine.shape[0],
                            self.coarse_dim,
                            fine.shape[-1],
                            device=fine.device,
                        ),
                        fine,
                        ctrl_feat,
                    ],
                    dim=1,
                )

            fused = self.root_input_fuse(fuse_in)  # [B, rbd, T]
            feat = self.root_predictor(fused)  # [B, rbd//2, T]
            root_rot = self.root_rot_head(feat)  # [B, 6, T]
            root_disp = self.root_disp_head(feat)  # [B, 3, T]
            root = torch.cat([root_rot, root_disp], dim=1)  # [B, 9, T]
            input = torch.cat([root, input[:, 9:, :]], dim=1)

        return self._normalize_motion_channels(input, mean_dqs, std_dqs)


# encoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, param, parents, device):
        super(StaticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        channels = 3  # position

        number_layers = num_layers
        layer_parents = parents

        for i in range(number_layers):
            neighbor_list, _ = find_neighbor(layer_parents, param["neighbor_distance"])

            seq = []
            if not dummy_joint:
                seq.append(
                    SkeletonLinear(
                        param=param,
                        neighbor_list=neighbor_list,
                        in_channels=channels * len(neighbor_list),
                        out_channels=channels * 2 * len(neighbor_list),
                        device=device,
                    )
                )
            else:
                seq.append(
                    SkeletonLinear(
                        param=param,
                        neighbor_list=neighbor_list,
                        in_channels=channels * (len(neighbor_list) + 1),
                        out_channels=channels * 2 * (len(neighbor_list) + 1),
                        device=device,
                    )
                )

            if i < number_layers - 1:
                pool = SkeletonPool(
                    parents=layer_parents, channels_per_edge=channels * 2, device=device
                )
                layer_parents = pool.new_parents
                seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        output = [input]

        for layer in self.layers:
            input = layer(input)
            input = input.squeeze(-1)
            output.append(input)
        return output


class RecurrentPrior(nn.Module):
    """
    Simple autoregressive recurrent prior for VQ-VAE codebook indices.
    - Small window (4 timesteps) of recent control + embedding history
    - LSTM/GRU with windowed context only
    - Outputs categorical distribution over codebook indices
    - No long-term dependencies, focused on short-term prediction
    """

    def __init__(
        self,
        ctrl_dim=6,
        num_codebook_vectors=256,
        num_levels=1,
        vq_dim=256,
        hidden_dim=512,
        window_size=4,
        stride=8,
        kernel_size=7,
        num_layers=2,
        dropout=0.1,
        lma_dim=0,
    ):
        super().__init__()
        self.ctrl_dim = ctrl_dim
        self.lma_dim = lma_dim
        self.num_codebook_vectors = num_codebook_vectors
        self.num_levels = num_levels
        self.vq_dim = vq_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.stride = stride

        self.per_channel_ctrl_out = 8
        self.ctrl_out_dim = self.ctrl_dim * self.per_channel_ctrl_out

        self.ctrl_channel_dropout_prob = 0.2
        # If True, drop per-timestep instead of whole-channel; default False (whole-channel drop)
        self.ctrl_channel_dropout_per_timestep = True

        self.ctrl_downsamplers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        1,
                        self.per_channel_ctrl_out // 4,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    # CausalConv1d(1, self.per_channel_ctrl_out//4, kernel_size=kernel_size, stride=2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Conv1d(
                        self.per_channel_ctrl_out // 4,
                        self.per_channel_ctrl_out // 2,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    # CausalConv1d(self.per_channel_ctrl_out//4, self.per_channel_ctrl_out//2, kernel_size=kernel_size, stride=2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Conv1d(
                        self.per_channel_ctrl_out // 2,
                        self.per_channel_ctrl_out,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                    ),
                    # CausalConv1d(self.per_channel_ctrl_out//2, self.per_channel_ctrl_out, kernel_size=kernel_size, stride=2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    ConvLayerNorm(self.per_channel_ctrl_out),
                )
                for _ in range(self.ctrl_dim)
            ]
        )

        self.ctrl_downsampler = nn.Sequential(
            nn.Conv1d(
                self.ctrl_dim,
                self.ctrl_out_dim,
                kernel_size=15,
                stride=8,
                padding=15 // 2,
            ),
            nn.LeakyReLU(0.2),
            ConvLayerNorm(self.ctrl_out_dim),
        )

        # Embedding tables for each level (match VQ codebooks)
        self.level_embeddings = nn.ModuleList(
            [nn.Embedding(num_codebook_vectors, vq_dim) for _ in range(num_levels)]
        )
        for emb in self.level_embeddings:
            nn.init.normal_(emb.weight, 0.0, 0.02)

        # Project summed level embeddings to hidden space
        self.emb_proj = nn.Linear(vq_dim, hidden_dim)

        # Recurrent core with windowed context
        # Two stride-2 convs → 4× reduction.  Brings LMA from T//2 to T_ds=T//8.
        if lma_dim > 0:
            self.lma_align = nn.Sequential(
                nn.Conv1d(lma_dim, lma_dim, kernel_size=7, stride=2, padding=3),
                nn.LeakyReLU(0.2),
                nn.Conv1d(lma_dim, lma_dim, kernel_size=7, stride=2, padding=3),
                ConvLayerNorm(lma_dim),
                nn.LeakyReLU(0.2),
            )
            # Projects LMA → ctrl_out_dim so it fills the ctrl budget when lma_only=True
            self.lma_to_ctrl = nn.Sequential(
                nn.Linear(lma_dim, self.ctrl_out_dim),
                nn.LeakyReLU(0.2),
            )

        # rnn_input_size = hidden_dim * 2
        rnn_input_size = hidden_dim + self.ctrl_out_dim + lma_dim
        # self.rnn = nn.LSTM(
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.teacher_forcing_prob = 0.95
        self.lma_only = (
            True  # if True, zero ctrl signal — generation driven by LMA only
        )

        # Output heads for each level
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_codebook_vectors),
                )
                for _ in range(num_levels)
            ]
        )

        self.rot_proj = nn.Sequential(
            nn.Linear(hidden_dim + 16, 128),
            nn.LeakyReLU(0.2),
        )

        kernel_size = 5
        self.rot_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                128, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                64, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                32, 9, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            ),
            nn.LeakyReLU(0.2),  # <-------
        )

        self.rot_smoother = nn.Conv1d(
            9, 9, kernel_size=5, padding=5 // 2, groups=9, bias=False
        )
        # init as simple averaging kernel
        with torch.no_grad():
            avg_kernel = torch.ones(5, dtype=torch.float32) / 5.0
            k = avg_kernel.view(1, 1, -1)  # [out_ch_per_group, in_ch_per_group, K]
            # assign to each group's weights
            self.rot_smoother.weight.data.copy_(k.repeat(9, 1, 1))

        self.yaw_downsampler = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, stride=8, padding=7),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.rot_embed_proj = nn.Linear(vq_dim, 16)

        self.dropout = nn.Dropout(dropout)

    def anneal_teacher_forcing(self, factor: float):
        """Multiply teacher forcing prob by factor (call from training loop per epoch)."""
        # self.teacher_forcing_prob.mul_(factor)
        self.teacher_forcing_prob = float(self.teacher_forcing_prob) * float(factor)

    def _encode_controls(self, ctrl_seq: torch.Tensor) -> torch.Tensor:
        """
        Per-channel downsample and concat.
        ctrl_seq: [B, T_full, C]
        Returns: [B, T_ds, C * per_channel_ctrl_out]
        """

        B, T_full, C = ctrl_seq.shape
        device = ctrl_seq.device
        # Apply channel dropout corruption during training.
        # Default behavior: for each sample b and channel c, with probability p zero the entire channel time series.
        seq = ctrl_seq
        p = float(self.ctrl_channel_dropout_prob)

        if self.training and p > 0.0:
            if not self.ctrl_channel_dropout_per_timestep:
                # mask shape [B, C, 1] -> broadcast to [B, T_full, C]
                keep_mask = (torch.rand(B, C, device=device) >= p).float().unsqueeze(1)
                seq = seq * keep_mask  # zero-out dropped channels across all frames
            else:
                # per-timestep masking: [B, T_full, C]
                keep_mask = (torch.rand(B, T_full, C, device=device) >= p).float()
                seq = seq * keep_mask

        per_ch_outputs = []
        for i in range(C):
            sig = seq[:, :, i : i + 1]  # [B, T_full, 1]
            ds = self.ctrl_downsamplers[i](
                sig.permute(0, 2, 1)
            )  # [B, per_channel_ctrl_out, T_ds]
            per_ch_outputs.append(
                ds.permute(0, 2, 1)
            )  # [B, T_ds, per_channel_ctrl_out]
        # concat channels -> [B, T_ds, C * per_channel_ctrl_out]
        ctrl_concat = torch.cat(per_ch_outputs, dim=-1)
        return ctrl_concat

    def _encode_controls_(self, ctrl_seq: torch.Tensor):
        B, T_full, C = ctrl_seq.shape
        device = ctrl_seq.device
        seq = ctrl_seq
        p = float(self.ctrl_channel_dropout_prob)
        # channel dropout (optionally per-timestep). keep similar corruption semantics.
        if self.training and p > 0.0:
            if not self.ctrl_channel_dropout_per_timestep:
                # zero entire channels with probability p per sample
                mask = (
                    (torch.rand(B, C, device=device) > p).float().unsqueeze(1)
                )  # [B,1,C]
                seq = seq * mask  # broadcast over time
            else:
                # per-timestep channel dropout
                mask = (torch.rand(B, T_full, C, device=device) > p).float()
                seq = seq * mask
        # prepare for Conv1d: [B, C, T]
        x = seq.permute(0, 2, 1).contiguous()
        # single downsample pass -> [B, ctrl_out_dim, T_ds]
        ds = self.ctrl_downsampler(x)
        # return as [B, T_ds, ctrl_out_dim]
        return ds.permute(0, 2, 1).contiguous()

    def forward(
        self,
        ctrl_seq: torch.Tensor,
        target_indices: torch.Tensor = None,
        temperature: float = 1.0,
        max_length: int = None,
        codebooks: torch.Tensor = None,
        yaw_sin_cos: torch.Tensor = None,
        lma_down: torch.Tensor = None,
    ):
        """
        Args:
            ctrl_seq: [B, T_full, ctrl_dim]  control signal
            target_indices: [B, T_ds, num_levels]  ground truth (training)
            temperature: sampling temperature (inference)
            max_length: max sequence length to generate (inference)
            lma_down: [B, T_full//2, lma_dim]  pre-downsampled LMA (optional)

        Returns:
            indices: [B, T_out, num_levels]
            logits:  [B, T_out, num_levels, num_codebook_vectors]
        """
        # Downsample control signal
        ctrl_down = self._encode_controls(ctrl_seq)
        B, T_ds, H = ctrl_down.shape
        device = ctrl_down.device

        # Align LMA from T//2 → T_ds (T//8), or fill with zeros when unavailable
        if self.lma_dim > 0:
            if lma_down is not None:
                lma_aligned = self.lma_align(lma_down.permute(0, 2, 1)).permute(0, 2, 1)
                if lma_aligned.size(1) != T_ds:  # safety: exact-length match
                    lma_aligned = F.interpolate(
                        lma_aligned.permute(0, 2, 1),
                        size=T_ds,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
            else:
                lma_aligned = torch.zeros(B, T_ds, self.lma_dim, device=device)
        else:
            lma_aligned = None

        # When lma_only, replace ctrl with a learned projection of LMA (same budget, no dead zeros)
        if self.lma_only:
            if lma_aligned is not None and self.lma_dim > 0:
                ctrl_down = self.lma_to_ctrl(lma_aligned)  # [B, T_ds, ctrl_out_dim]
            else:
                ctrl_down = torch.zeros_like(ctrl_down)

        if self.training and target_indices is not None:
            return self._forward_train(
                ctrl_down,
                target_indices,
                codebooks=codebooks,
                yaw_sin_cos=yaw_sin_cos,
                lma_aligned=lma_aligned,
            )
        else:
            seq_len = max_length if max_length is not None else T_ds
            return self._forward_inference(
                ctrl_down,
                seq_len,
                temperature,
                codebooks=codebooks,
                yaw_sin_cos=yaw_sin_cos,
                lma_aligned=lma_aligned,
            )

    def _forward_train(
        self,
        ctrl_down: torch.Tensor,
        target_indices: torch.Tensor,
        codebooks=None,
        yaw_sin_cos: torch.Tensor = None,
        lma_aligned: torch.Tensor = None,
    ):
        """Training with teacher forcing 90% of the time"""
        B, T_ds, H = ctrl_down.shape
        seq_len = min(T_ds, target_indices.size(1))
        device = ctrl_down.device

        all_logits = []
        all_indices = []
        rnn_outs = []
        all_embs = []

        # Initialize context
        emb_history = torch.zeros(B, self.window_size, self.vq_dim, device=device)
        hidden_state = None

        # Track predictions vs ground truth usage
        predicted_embs = torch.zeros(B, seq_len, self.vq_dim, device=device)

        # First pass: collect GT embeddings for fallback
        gt_level_embs = [
            self.level_embeddings[l](target_indices[..., l])
            for l in range(self.num_levels)
        ]
        gt_sum_emb = torch.stack(gt_level_embs, dim=2).sum(dim=2)  # [B, T_ds, vq_dim]

        for t in range(seq_len):

            w = min(self.window_size, t + 1)
            emb_window = emb_history[:, -w:, :]
            emb_proj_seq = self.emb_proj(emb_window)

            start = max(0, t - w + 1)
            ctrl_window = ctrl_down[:, start : t + 1]
            if self.lma_dim > 0 and lma_aligned is not None:
                lma_window = lma_aligned[
                    :, max(0, t - w + 1) : t + 1, :
                ]  # [B, w, lma_dim]
                rnn_input = torch.cat([ctrl_window, emb_proj_seq, lma_window], dim=-1)
            else:
                rnn_input = torch.cat([ctrl_window, emb_proj_seq], dim=-1)

            rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)
            rnn_out = rnn_out[:, -1:]
            rnn_outs.append(rnn_out)
            feat = self.dropout(rnn_out.squeeze(1))  # [B, H]

            # Predict each level
            level_logits = []
            level_indices = []
            level_embs = []

            for l in range(self.num_levels):
                logits_l = self.output_heads[l](feat)  # [B, K]
                level_logits.append(logits_l)

                # Teacher forcing decision (90% of time)
                use_teacher_forcing = (
                    torch.rand(B, device=device) < self.teacher_forcing_prob
                )

                if self.training:
                    tau = 1.0  # 0.5
                    soft_sample = F.gumbel_softmax(
                        logits_l, tau=tau, hard=True
                    )  # [B, K]
                    # Teacher forcing decision
                    use_teacher_forcing = (
                        torch.rand(B, device=device) < self.teacher_forcing_prob
                    )
                    # Convert GT index to one-hot
                    gt_onehot = F.one_hot(
                        target_indices[:, t, l], num_classes=logits_l.size(-1)
                    ).float()
                    # Mix teacher forcing and soft sample
                    mixed_emb_input = torch.where(
                        use_teacher_forcing.unsqueeze(-1),  # [B,1] broadcast
                        gt_onehot,
                        soft_sample,
                    )  # [B, K]
                    final_idx = torch.argmax(logits_l, dim=-1)
                else:
                    # Inference: always use predictions
                    final_idx = torch.argmax(logits_l, dim=-1)

                level_indices.append(final_idx)

                # Get embedding (use final decision)
                if codebooks == None:
                    # emb_l = self.level_embeddings[l](final_idx)  # [B, vq_dim]
                    emb_l = mixed_emb_input @ self.level_embeddings[l].weight
                else:
                    emb_l = codebooks[l](final_idx)
                level_embs.append(emb_l)
                all_embs.append(emb_l.unsqueeze(1))

            # Update embedding history (sliding window)
            current_emb = torch.stack(level_embs, dim=1).sum(dim=1)  # [B, vq_dim]
            emb_history = torch.cat(
                [emb_history[:, 1:, :], current_emb.unsqueeze(1)], dim=1
            )

            all_logits.append(torch.stack(level_logits, dim=1))  # [B, L, K]
            all_indices.append(torch.stack(level_indices, dim=1))  # [B, L]

        rnn_outs = torch.cat(rnn_outs, dim=1)  # [B, T, hidden_dim]
        indices = torch.stack(all_indices, dim=1)  # [B, T, L]
        logits = torch.stack(all_logits, dim=1)  # [B, T, L, K]

        # rots = self.predict_rots(rnn_outs, yaw_sin_cos, embeddings)
        return indices, logits  # , rots

    def _forward_inference(
        self,
        ctrl_down: torch.Tensor,
        seq_len: int,
        temperature: float,
        codebooks=None,
        yaw_sin_cos: torch.Tensor = None,
        lma_aligned: torch.Tensor = None,
    ):
        """Autoregressive inference with windowed context"""
        B, T_ds, H = ctrl_down.shape
        seq_len = min(seq_len, T_ds)
        device = ctrl_down.device

        all_logits = []
        all_indices = []
        rnn_outs = []
        all_embs = []
        # Initialize context
        emb_history = torch.zeros(B, self.window_size, self.vq_dim, device=device)
        hidden_state = None

        for t in range(seq_len):

            w = min(self.window_size, t + 1)
            emb_window = emb_history[:, -w:, :]
            emb_proj_seq = self.emb_proj(emb_window)

            start = max(0, t - w + 1)
            ctrl_window = ctrl_down[:, start : t + 1]
            if self.lma_dim > 0 and lma_aligned is not None:
                lma_window = lma_aligned[
                    :, max(0, t - w + 1) : t + 1, :
                ]  # [B, w, lma_dim]
                rnn_input = torch.cat([ctrl_window, emb_proj_seq, lma_window], dim=-1)
            else:
                rnn_input = torch.cat([ctrl_window, emb_proj_seq], dim=-1)

            rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)
            rnn_out = rnn_out[:, -1:]
            rnn_outs.append(rnn_out)
            feat = rnn_out.squeeze(1)

            # Sample each level
            level_logits = []
            level_indices = []
            level_embs = []

            for l in range(self.num_levels):
                logits_l = self.output_heads[l](feat)
                level_logits.append(logits_l)

                # Sample or argmax
                if temperature > 0:
                    probs = F.softmax(logits_l / temperature, dim=-1)
                    idx_l = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    idx_l = torch.argmax(logits_l, dim=-1)

                level_indices.append(idx_l)
                if codebooks == None:
                    emb_l = self.level_embeddings[l](idx_l)
                else:
                    emb_l = codebooks[l](idx_l)

                level_embs.append(emb_l)
                all_embs.append(emb_l.unsqueeze(1))

            # Update history
            current_emb = torch.stack(level_embs, dim=1).sum(dim=1)
            emb_history = torch.cat(
                [emb_history[:, 1:, :], current_emb.unsqueeze(1)], dim=1
            )

            all_logits.append(torch.stack(level_logits, dim=1))
            all_indices.append(torch.stack(level_indices, dim=1))

        indices = torch.stack(all_indices, dim=1)
        logits = torch.stack(all_logits, dim=1)
        rnn_outs = torch.cat(rnn_outs, dim=1)

        return indices, logits  # , rots

    @torch.no_grad()
    def sample(
        self,
        ctrl_seq: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = None,
        codebooks: torch.Tensor = None,
        yaw_sin_cos: torch.Tensor = None,
        lma_down: torch.Tensor = None,
    ):
        """Convenience sampling method"""
        self.eval()
        indices, logits = self.forward(
            ctrl_seq,
            temperature=temperature,
            codebooks=codebooks,
            yaw_sin_cos=yaw_sin_cos,
            lma_down=lma_down,
        )

        if top_k is not None:
            # Apply top-k filtering to logits
            B, T, L, K = logits.shape
            for b in range(B):
                for t in range(T):
                    for l in range(L):
                        vals, _ = torch.topk(logits[b, t, l], min(top_k, K))
                        threshold = vals[-1]
                        logits[b, t, l] = torch.where(
                            logits[b, t, l] >= threshold,
                            logits[b, t, l],
                            torch.full_like(logits[b, t, l], float("-inf")),
                        )

        return indices, logits
