import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lma_transformer_prior_v5 import (
    BidirectionalBlock,
    ConvNorm,
    LMA_CHANNELS,
    ResidualTemporalBlock,
)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def extract_schedule(
    buffer: torch.Tensor, timesteps: torch.Tensor, target_shape
) -> torch.Tensor:
    values = buffer.gather(0, timesteps.clamp(0, buffer.numel() - 1))
    while values.dim() < len(target_shape):
        values = values.unsqueeze(-1)
    return values


class DiffusionTimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        half_dim = self.hidden_dim // 2
        if half_dim == 0:
            raise ValueError("hidden_dim must be >= 2 for timestep embedding")

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.hidden_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class TemporalConditionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        downsample_stride: int = 1,
        kernel_size: int = 7,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        mid_dim = max(hidden_dim // 2, input_dim * 4)
        self.proj = nn.Sequential(
            nn.Conv1d(
                input_dim,
                mid_dim,
                kernel_size,
                stride=downsample_stride,
                padding=padding,
            ),
            ConvNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=3, padding=1),
            ConvNorm(hidden_dim),
            nn.GELU(),
        )
        blocks = []
        for block_index in range(num_res_blocks):
            blocks.append(
                ResidualTemporalBlock(
                    hidden_dim,
                    kernel_size=5,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
        self.refine = nn.Sequential(*blocks)

    def forward(
        self, sequence: torch.Tensor, target_len: Optional[int] = None
    ) -> torch.Tensor:
        if sequence.dim() != 3:
            raise ValueError(f"Expected [B, T, C] sequence, got {sequence.shape}")

        encoded = self.proj(sequence.transpose(1, 2))
        encoded = self.refine(encoded).transpose(1, 2)
        if target_len is not None and encoded.size(1) != target_len:
            encoded = F.interpolate(
                encoded.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return encoded


class TemporalRootHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        traj_dim: int,
        dropout: float = 0.1,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    hidden_dim,
                    kernel_size=5,
                    dilation=2**block_index,
                    dropout=dropout,
                )
                for block_index in range(num_blocks)
            ]
        )
        self.output_norm = ConvNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, traj_dim, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.input_proj(self.input_norm(hidden)))
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_proj(self.output_norm(hidden))
        return hidden.transpose(1, 2)


class ContinuousLatentDiffusionPrior(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        lma_dim: int = len(LMA_CHANNELS),
        traj_dim: int = 9,
        num_train_timesteps: int = 1000,
        condition_mode_probs: Optional[Dict[str, float]] = None,
        use_explicit_condition_mode_probs: bool = False,
        use_per_lma_channel_encoders: bool = False,
        per_lma_channel_dropout: float = 0.1,
        latent_velocity_weight: float = 0.0,
        root_traj_loss_weight: float = 0.0,
        root_traj_velocity_weight: float = 0.0,
        root_contact_loss_weight: float = 0.0,
        root_contact_velocity_weight: float = 0.0,
        root_contact_acceleration_weight: float = 0.0,
        root_rot_loss_scale: float = 1.0,
        root_pos_loss_scale: float = 3.0,
        p_drop_lma: float = 0.15,
        p_drop_traj: float = 0.45,
        p_drop_both: float = 0.05,
        final_p_drop_lma: Optional[float] = None,
        final_p_drop_traj: Optional[float] = None,
        final_p_drop_both: Optional[float] = None,
        lma_condition_scale: float = 1.15,
        traj_condition_scale: float = 1.0,
        num_styles: int = 0,
        style_drop: float = 0.0,
        final_style_drop: Optional[float] = None,
        style_condition_scale: float = 1.0,
        condition_focus_start: float = 0.4,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lma_dim = lma_dim
        self.traj_dim = traj_dim
        self.num_train_timesteps = int(num_train_timesteps)
        self.latent_velocity_weight = float(latent_velocity_weight)
        self.root_traj_loss_weight = float(root_traj_loss_weight)
        self.root_traj_velocity_weight = float(root_traj_velocity_weight)
        self.root_contact_loss_weight = float(root_contact_loss_weight)
        self.root_contact_velocity_weight = float(root_contact_velocity_weight)
        self.root_contact_acceleration_weight = float(root_contact_acceleration_weight)
        self.root_rot_loss_scale = float(root_rot_loss_scale)
        self.root_pos_loss_scale = float(root_pos_loss_scale)
        self.p_drop_lma = float(p_drop_lma)
        self.p_drop_traj = float(p_drop_traj)
        self.p_drop_both = float(p_drop_both)
        self.final_p_drop_lma = float(
            p_drop_lma if final_p_drop_lma is None else final_p_drop_lma
        )
        self.final_p_drop_traj = float(
            p_drop_traj if final_p_drop_traj is None else final_p_drop_traj
        )
        self.final_p_drop_both = float(
            p_drop_both if final_p_drop_both is None else final_p_drop_both
        )
        self.lma_condition_scale = float(lma_condition_scale)
        self.traj_condition_scale = float(traj_condition_scale)
        self.num_styles = int(num_styles)
        self.style_drop = float(style_drop)
        self.final_style_drop = float(
            style_drop if final_style_drop is None else final_style_drop
        )
        self.style_condition_scale = float(style_condition_scale)
        self.condition_focus_start = float(condition_focus_start)
        self.training_progress = 0.0
        self.use_explicit_condition_mode_probs = bool(use_explicit_condition_mode_probs)
        self.use_per_lma_channel_encoders = bool(use_per_lma_channel_encoders)
        self.per_lma_channel_dropout = float(per_lma_channel_dropout)

        if self.per_lma_channel_dropout < 0.0 or self.per_lma_channel_dropout > 1.0:
            raise ValueError(
                "per_lma_channel_dropout must be in [0, 1], "
                f"got {self.per_lma_channel_dropout}"
            )

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.condition_mode_probs = condition_mode_probs or {
            "full": 0.55,
            "lma_only": 0.40,
            "traj_only": 0.00,
            "uncond": 0.05,
        }
        self._validate_condition_mode_probs()
        self.mode_names = tuple(self.condition_mode_probs.keys())
        self.mode_to_index = {name: index for index, name in enumerate(self.mode_names)}

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.timestep_embed = DiffusionTimestepEmbedding(hidden_dim)

        if self.use_per_lma_channel_encoders:
            self.lma_encoder = None
            self.lma_channel_encoders = nn.ModuleList(
                [
                    TemporalConditionEncoder(
                        input_dim=1,
                        hidden_dim=hidden_dim,
                        downsample_stride=2,
                        kernel_size=9,
                        num_res_blocks=2,
                        dropout=dropout,
                    )
                    for _ in range(lma_dim)
                ]
            )
        else:
            self.lma_encoder = TemporalConditionEncoder(
                input_dim=lma_dim,
                hidden_dim=hidden_dim,
                downsample_stride=2,
                kernel_size=9,
                num_res_blocks=2,
                dropout=dropout,
            )
            self.lma_channel_encoders = None
        self.traj_encoder = TemporalConditionEncoder(
            input_dim=traj_dim,
            hidden_dim=hidden_dim,
            downsample_stride=1,
            kernel_size=7,
            num_res_blocks=2,
            dropout=dropout,
        )

        self.null_lma = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.null_traj = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.lma_condition_norm = nn.LayerNorm(hidden_dim)
        self.traj_condition_norm = nn.LayerNorm(hidden_dim)
        if self.num_styles > 0:
            self.style_embedding = nn.Embedding(self.num_styles, hidden_dim)
            self.null_style = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.style_condition_norm = nn.LayerNorm(hidden_dim)
        else:
            self.style_embedding = None
            self.null_style = None
            self.style_condition_norm = None
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                BidirectionalBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, latent_dim)
        self.root_traj_head = TemporalRootHead(
            hidden_dim=hidden_dim,
            traj_dim=traj_dim,
            dropout=dropout,
        )

        betas = cosine_beta_schedule(self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev, persistent=False
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

    def _validate_condition_mode_probs(self):
        expected = {"full", "lma_only", "traj_only", "uncond"}
        actual = set(self.condition_mode_probs.keys())
        if actual != expected:
            raise ValueError(
                f"condition_mode_probs keys must be {expected}, got {actual}"
            )
        total = sum(float(value) for value in self.condition_mode_probs.values())
        if total <= 0.0:
            raise ValueError("condition_mode_probs must sum to a positive value")

    def set_training_progress(self, progress: float):
        self.training_progress = float(min(max(progress, 0.0), 1.0))

    def _scheduled_probability(self, start_value: float, end_value: float) -> float:
        focus_start = min(max(self.condition_focus_start, 0.0), 0.999)
        if self.training_progress <= focus_start:
            return float(start_value)
        alpha = (self.training_progress - focus_start) / max(1.0 - focus_start, 1e-6)
        alpha = min(max(alpha, 0.0), 1.0)
        return float(start_value + alpha * (end_value - start_value))

    def get_current_dropout_state(self) -> Dict[str, float]:
        return {
            "p_drop_lma": self._scheduled_probability(
                self.p_drop_lma, self.final_p_drop_lma
            ),
            "p_drop_traj": self._scheduled_probability(
                self.p_drop_traj, self.final_p_drop_traj
            ),
            "p_drop_both": self._scheduled_probability(
                self.p_drop_both, self.final_p_drop_both
            ),
            "style_drop": self._scheduled_probability(
                self.style_drop, self.final_style_drop
            ),
        }

    @staticmethod
    def _ensure_sequence_tensor(
        sequence: torch.Tensor, feature_dim: Optional[int] = None
    ) -> torch.Tensor:
        if sequence is None:
            return None
        if not torch.is_tensor(sequence):
            sequence = torch.tensor(sequence, dtype=torch.float32)
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        if sequence.dim() == 4 and sequence.size(-1) == 1:
            sequence = sequence.squeeze(-1)
        if sequence.dim() != 3:
            raise ValueError(f"Expected [B, T, C] sequence, got {sequence.shape}")
        if feature_dim is not None and sequence.size(-1) != feature_dim:
            raise ValueError(
                f"Expected feature dim {feature_dim}, got {sequence.size(-1)} for sequence {sequence.shape}"
            )
        return sequence.float()

    def _stack_lma_dict(self, lma_seq: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(lma_seq, dict):
            raise TypeError("LMA dict input must be a dict of tensors")

        example = None
        parts = []
        for key in LMA_CHANNELS:
            value = lma_seq.get(key)
            if value is not None:
                value = self._ensure_sequence_tensor(value)
                if value.size(-1) != 1:
                    raise ValueError(
                        f"Expected scalar LMA channel for {key}, got {value.shape}"
                    )
                example = value if example is None else example
                parts.append(value)
                continue

            if example is None:
                continue
            parts.append(torch.zeros_like(example))

        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def _fuse_per_lma_channel_conditions(
        self, encoded_channels: torch.Tensor
    ) -> torch.Tensor:
        if encoded_channels.dim() != 4:
            raise ValueError(
                "Expected per-channel LMA encodings with shape [B, C, T, H], "
                f"got {encoded_channels.shape}"
            )

        if (
            not self.training
            or self.per_lma_channel_dropout <= 0.0
            or encoded_channels.size(1) <= 1
        ):
            return encoded_channels.mean(dim=1)

        batch_size, num_channels = encoded_channels.shape[:2]
        keep_mask = (
            torch.rand(
                batch_size,
                num_channels,
                device=encoded_channels.device,
            )
            >= self.per_lma_channel_dropout
        )
        dropped_all = ~keep_mask.any(dim=1)
        if dropped_all.any():
            rescued_rows = dropped_all.nonzero(as_tuple=False).squeeze(-1)
            rescued_channels = torch.randint(
                low=0,
                high=num_channels,
                size=(rescued_rows.numel(),),
                device=encoded_channels.device,
            )
            keep_mask[rescued_rows, rescued_channels] = True

        channel_weights = keep_mask.to(dtype=encoded_channels.dtype)
        channel_weights = channel_weights / channel_weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        return (encoded_channels * channel_weights[:, :, None, None]).sum(dim=1)

    def _encode_lma_condition(self, lma_seq, target_len: int) -> Optional[torch.Tensor]:
        if lma_seq is None:
            return None
        if isinstance(lma_seq, dict):
            lma_tensor = self._stack_lma_dict(lma_seq)
        else:
            lma_tensor = self._ensure_sequence_tensor(lma_seq)
        if lma_tensor is None:
            return None
        if lma_tensor.size(-1) != self.lma_dim:
            if lma_tensor.size(-1) < self.lma_dim:
                pad = torch.zeros(
                    lma_tensor.size(0),
                    lma_tensor.size(1),
                    self.lma_dim - lma_tensor.size(-1),
                    device=lma_tensor.device,
                    dtype=lma_tensor.dtype,
                )
                lma_tensor = torch.cat([lma_tensor, pad], dim=-1)
            else:
                lma_tensor = lma_tensor[..., : self.lma_dim]
        if self.use_per_lma_channel_encoders:
            encoded_channels = []
            for channel_index, channel_encoder in enumerate(self.lma_channel_encoders):
                channel_tensor = lma_tensor[..., channel_index : channel_index + 1]
                encoded_channels.append(
                    channel_encoder(channel_tensor, target_len=target_len)
                )
            return self._fuse_per_lma_channel_conditions(
                torch.stack(encoded_channels, dim=1)
            )
        return self.lma_encoder(lma_tensor, target_len=target_len)

    def _encode_traj_condition(
        self, traj_seq, target_len: int
    ) -> Optional[torch.Tensor]:
        if traj_seq is None:
            return None
        traj_tensor = self._ensure_sequence_tensor(traj_seq)
        if traj_tensor.size(-1) != self.traj_dim:
            raise ValueError(
                f"Expected trajectory feature dim {self.traj_dim}, got {traj_tensor.size(-1)}"
            )
        return self.traj_encoder(traj_tensor, target_len=target_len)

    def _sample_condition_mask(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        dropout_state = self.get_current_dropout_state()
        drop_lma = torch.rand(batch_size, device=device) < dropout_state["p_drop_lma"]
        drop_traj = torch.rand(batch_size, device=device) < dropout_state["p_drop_traj"]
        drop_style = torch.rand(batch_size, device=device) < dropout_state["style_drop"]
        drop_both = torch.rand(batch_size, device=device) < dropout_state["p_drop_both"]
        drop_lma |= drop_both
        drop_traj |= drop_both
        keep_lma = ~drop_lma
        keep_traj = ~drop_traj
        keep_style = ~drop_style
        return {
            "keep_lma": keep_lma,
            "keep_traj": keep_traj,
            "keep_style": keep_style,
            "drop_lma": drop_lma,
            "drop_traj": drop_traj,
            "drop_style": drop_style,
        }

    def _sample_condition_mask_from_mode_probs(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        probabilities = torch.tensor(
            [float(self.condition_mode_probs[name]) for name in self.mode_names],
            dtype=torch.float32,
            device=device,
        )
        probabilities = probabilities / probabilities.sum().clamp_min(1e-8)
        sampled_mode_indices = torch.multinomial(
            probabilities,
            num_samples=batch_size,
            replacement=True,
        )

        keep_lma = torch.logical_or(
            sampled_mode_indices == self.mode_to_index["full"],
            sampled_mode_indices == self.mode_to_index["lma_only"],
        )
        keep_traj = torch.logical_or(
            sampled_mode_indices == self.mode_to_index["full"],
            sampled_mode_indices == self.mode_to_index["traj_only"],
        )
        keep_style = sampled_mode_indices != self.mode_to_index["uncond"]
        return {
            "keep_lma": keep_lma,
            "keep_traj": keep_traj,
            "keep_style": keep_style,
            "mode_indices": sampled_mode_indices,
        }

    def _sample_training_condition_mask(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        if self.use_explicit_condition_mode_probs:
            return self._sample_condition_mask_from_mode_probs(batch_size, device)
        return self._sample_condition_mask(batch_size, device)

    def _mask_from_mode(
        self, mode: str, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        mode = (mode or "full").lower()
        if mode not in self.mode_names:
            raise ValueError(f"Unsupported conditioning mode: {mode}")
        keep_lma = torch.zeros(batch_size, dtype=torch.bool, device=device)
        keep_traj = torch.zeros(batch_size, dtype=torch.bool, device=device)
        keep_style = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if mode in {"full", "lma_only"}:
            keep_lma[:] = True
        if mode in {"full", "traj_only"}:
            keep_traj[:] = True
        if mode != "uncond":
            keep_style[:] = True
        return {"keep_lma": keep_lma, "keep_traj": keep_traj, "keep_style": keep_style}

    def _normalize_style_ids(
        self, style_id, batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if style_id is None:
            return None
        if not torch.is_tensor(style_id):
            style_id = torch.tensor(style_id, dtype=torch.long, device=device)
        else:
            style_id = style_id.to(device=device, dtype=torch.long)

        if style_id.dim() == 0:
            style_id = style_id.unsqueeze(0)
        if style_id.dim() > 1:
            style_id = style_id.reshape(style_id.shape[0], -1)[:, 0]
        if style_id.numel() == 1 and batch_size > 1:
            style_id = style_id.expand(batch_size)
        if style_id.shape[0] != batch_size:
            raise ValueError(
                f"style_id batch dimension {style_id.shape[0]} does not match batch size {batch_size}"
            )
        return style_id

    def _encode_style_condition(
        self,
        style_id,
        batch_size: int,
        target_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.style_embedding is None:
            return None

        style_id = self._normalize_style_ids(style_id, batch_size, device)
        if style_id is None:
            return None

        valid_mask = (style_id >= 0) & (style_id < self.num_styles)
        safe_style_id = style_id.clamp(0, max(self.num_styles - 1, 0))
        style_cond = self.style_embedding(safe_style_id).unsqueeze(1)
        null_style = self.null_style.expand(batch_size, 1, -1)
        style_cond = torch.where(valid_mask[:, None, None], style_cond, null_style)
        return style_cond.expand(batch_size, target_len, -1)

    @staticmethod
    def _condition_available(condition) -> bool:
        if condition is None:
            return False
        if torch.is_tensor(condition):
            return condition.numel() > 0
        return True

    def _apply_condition_mask(
        self,
        lma_cond: Optional[torch.Tensor],
        traj_cond: Optional[torch.Tensor],
        style_cond: Optional[torch.Tensor],
        condition_mask: Dict[str, torch.Tensor],
        target_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = None
        if lma_cond is not None:
            batch_size = lma_cond.size(0)
        if traj_cond is not None:
            batch_size = traj_cond.size(0) if batch_size is None else batch_size
        if style_cond is not None:
            batch_size = style_cond.size(0) if batch_size is None else batch_size
        if batch_size is None:
            batch_size = condition_mask["keep_lma"].numel()

        if lma_cond is None:
            lma_cond = self.null_lma.expand(batch_size, target_len, -1)
        else:
            null_lma = self.null_lma.expand(batch_size, target_len, -1)
            lma_cond = torch.where(
                condition_mask["keep_lma"][:, None, None], lma_cond, null_lma
            )

        if traj_cond is None:
            traj_cond = self.null_traj.expand(batch_size, target_len, -1)
        else:
            null_traj = self.null_traj.expand(batch_size, target_len, -1)
            traj_cond = torch.where(
                condition_mask["keep_traj"][:, None, None], traj_cond, null_traj
            )

        lma_cond = self.lma_condition_norm(lma_cond) * self.lma_condition_scale
        traj_cond = self.traj_condition_norm(traj_cond) * self.traj_condition_scale

        style_sum = 0.0
        if self.style_embedding is not None:
            if style_cond is None:
                style_cond = self.null_style.expand(batch_size, target_len, -1)
            else:
                null_style = self.null_style.expand(batch_size, target_len, -1)
                style_cond = torch.where(
                    condition_mask["keep_style"][:, None, None],
                    style_cond,
                    null_style,
                )
            style_sum = (
                self.style_condition_norm(style_cond) * self.style_condition_scale
            )

        return lma_cond + traj_cond + style_sum

    def _decode_hidden(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        mode: str = "full",
        condition_mask: Optional[Dict[str, torch.Tensor]] = None,
    ):
        noisy_latent = self._ensure_sequence_tensor(
            noisy_latent, feature_dim=self.latent_dim
        )
        batch_size, seq_len, _ = noisy_latent.shape
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(batch_size)

        lma_cond = self._encode_lma_condition(lma_seq, target_len=seq_len)
        traj_cond = self._encode_traj_condition(traj_seq, target_len=seq_len)
        style_cond = self._encode_style_condition(
            style_id,
            batch_size=batch_size,
            target_len=seq_len,
            device=noisy_latent.device,
        )

        if condition_mask is None:
            condition_mask = self._mask_from_mode(mode, batch_size, noisy_latent.device)

        cond = self._apply_condition_mask(
            lma_cond,
            traj_cond,
            style_cond,
            condition_mask=condition_mask,
            target_len=seq_len,
            device=noisy_latent.device,
        )

        hidden = self.latent_proj(noisy_latent)
        hidden = hidden + cond + self.timestep_embed(timesteps).unsqueeze(1)
        hidden = self.input_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        mode: str = "full",
        condition_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        hidden = self._decode_hidden(
            noisy_latent,
            timesteps,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
            mode=mode,
            condition_mask=condition_mask,
        )
        x0_pred = self.output_head(hidden)
        root_traj_pred = self.root_traj_head(hidden)
        if return_aux:
            return x0_pred, root_traj_pred
        return x0_pred

    def q_sample(
        self,
        clean_latent: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_latent)
        sqrt_alpha = extract_schedule(
            self.sqrt_alphas_cumprod, timesteps, clean_latent.shape
        )
        sqrt_one_minus = extract_schedule(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            clean_latent.shape,
        )
        return sqrt_alpha * clean_latent + sqrt_one_minus * noise

    @staticmethod
    def _split_root_channels(root_traj: torch.Tensor):
        if root_traj is None:
            return None, None
        if root_traj.size(-1) < 9:
            raise ValueError(
                f"Expected root trajectory with 9 channels, got {root_traj.shape}"
            )
        return root_traj[..., :6], root_traj[..., 6:9]

    @staticmethod
    def _align_temporal_condition(
        sequence: Optional[torch.Tensor], target_len: int
    ) -> Optional[torch.Tensor]:
        if sequence is None:
            return None
        sequence = ContinuousLatentDiffusionPrior._ensure_sequence_tensor(sequence)
        if sequence.size(1) == target_len:
            return sequence
        return F.interpolate(
            sequence.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    @staticmethod
    def _weighted_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if weights is None:
            return F.mse_loss(prediction, target)

        weights = weights.to(device=prediction.device, dtype=prediction.dtype)
        while weights.dim() < prediction.dim():
            weights = weights.unsqueeze(-1)
        weights = weights.expand_as(prediction)
        denom = weights.sum().clamp_min(1e-6)
        return ((prediction - target) ** 2 * weights).sum() / denom

    def _resolve_support_contact(
        self,
        support_contact_latent,
        foot_contact_latent,
        target_len: int,
    ) -> Optional[torch.Tensor]:
        support_contact = self._align_temporal_condition(
            support_contact_latent,
            target_len=target_len,
        )
        if support_contact is None and foot_contact_latent is not None:
            foot_contact = self._align_temporal_condition(
                foot_contact_latent,
                target_len=target_len,
            )
            if foot_contact is not None:
                support_contact = foot_contact.max(dim=-1, keepdim=True).values

        if support_contact is None:
            return None
        return support_contact[..., :1].clamp(0.0, 1.0)

    def compute_loss(
        self,
        clean_latent: torch.Tensor,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        foot_contact_latent=None,
        support_contact_latent=None,
    ) -> Dict[str, torch.Tensor]:
        clean_latent = self._ensure_sequence_tensor(
            clean_latent, feature_dim=self.latent_dim
        )
        traj_target = (
            self._ensure_sequence_tensor(traj_seq, feature_dim=self.traj_dim)
            if traj_seq is not None
            else None
        )
        batch_size = clean_latent.size(0)
        device = clean_latent.device
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        )
        noise = torch.randn_like(clean_latent)
        noisy_latent = self.q_sample(clean_latent, timesteps, noise=noise)
        condition_mask = self._sample_training_condition_mask(
            batch_size,
            device=device,
        )
        x0_pred, root_traj_pred = self.forward(
            noisy_latent,
            timesteps,
            lma_seq=lma_seq,
            traj_seq=traj_seq,
            style_id=style_id,
            condition_mask=condition_mask,
            return_aux=True,
        )
        x0_loss = F.mse_loss(x0_pred, clean_latent)
        loss = x0_loss

        latent_velocity_loss = clean_latent.new_zeros(())
        if self.latent_velocity_weight > 0.0 and clean_latent.size(1) > 1:
            pred_delta = x0_pred[:, 1:] - x0_pred[:, :-1]
            tgt_delta = clean_latent[:, 1:] - clean_latent[:, :-1]
            latent_velocity_loss = F.mse_loss(pred_delta, tgt_delta)
            loss = loss + self.latent_velocity_weight * latent_velocity_loss

        root_traj_loss = clean_latent.new_zeros(())
        root_traj_velocity_loss = clean_latent.new_zeros(())
        root_rotation_loss = clean_latent.new_zeros(())
        root_translation_loss = clean_latent.new_zeros(())
        root_translation_velocity_loss = clean_latent.new_zeros(())
        root_contact_loss = clean_latent.new_zeros(())
        root_contact_velocity_loss = clean_latent.new_zeros(())
        root_contact_acceleration_loss = clean_latent.new_zeros(())
        if traj_target is not None:
            pred_root_rot, pred_root_pos = self._split_root_channels(root_traj_pred)
            tgt_root_rot, tgt_root_pos = self._split_root_channels(traj_target)
            root_rotation_loss = F.mse_loss(pred_root_rot, tgt_root_rot)
            root_translation_loss = F.mse_loss(pred_root_pos, tgt_root_pos)
            root_traj_loss = (
                self.root_rot_loss_scale * root_rotation_loss
                + self.root_pos_loss_scale * root_translation_loss
            )
            if self.root_traj_loss_weight > 0.0:
                loss = loss + self.root_traj_loss_weight * root_traj_loss

            if self.root_traj_velocity_weight > 0.0 and traj_target.size(1) > 1:
                pred_root_delta = pred_root_pos[:, 1:] - pred_root_pos[:, :-1]
                tgt_root_delta = tgt_root_pos[:, 1:] - tgt_root_pos[:, :-1]
                root_translation_velocity_loss = F.mse_loss(
                    pred_root_delta, tgt_root_delta
                )
                root_traj_velocity_loss = root_translation_velocity_loss
                loss = loss + self.root_traj_velocity_weight * root_traj_velocity_loss

        return {
            "loss": loss,
            "x0_loss": x0_loss,
            "latent_velocity_loss": latent_velocity_loss,
            "root_traj_loss": root_traj_loss,
            "root_traj_velocity_loss": root_traj_velocity_loss,
            "root_rotation_loss": root_rotation_loss,
            "root_translation_loss": root_translation_loss,
            "root_translation_velocity_loss": root_translation_velocity_loss,
            "root_contact_loss": root_contact_loss,
            "root_contact_velocity_loss": root_contact_velocity_loss,
            "root_contact_acceleration_loss": root_contact_acceleration_loss,
            "x0_pred": x0_pred,
            "root_traj_pred": root_traj_pred,
            "noisy_latent": noisy_latent,
            "timesteps": timesteps,
            "condition_mask": condition_mask,
        }

    def _guided_prediction(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        mode: str = "full",
        cfg_scale: float = 2.5,
        lma_cfg_scale: Optional[float] = None,
        traj_cfg_scale: Optional[float] = None,
        style_cfg_scale: Optional[float] = None,
    ):
        mode = (mode or "full").lower()
        lma_scale = float(cfg_scale if lma_cfg_scale is None else lma_cfg_scale)
        traj_scale = float(cfg_scale if traj_cfg_scale is None else traj_cfg_scale)
        style_scale = float(cfg_scale if style_cfg_scale is None else style_cfg_scale)
        batch_size = x_t.size(0)
        device = x_t.device
        normalized_style_id = self._normalize_style_ids(style_id, batch_size, device)
        has_style = bool(
            self.style_embedding is not None
            and normalized_style_id is not None
            and torch.any(
                (normalized_style_id >= 0) & (normalized_style_id < self.num_styles)
            ).item()
        )

        def predict_mask(keep_lma: bool, keep_traj: bool, keep_style: bool):
            condition_mask = {
                "keep_lma": torch.full(
                    (batch_size,), keep_lma, dtype=torch.bool, device=device
                ),
                "keep_traj": torch.full(
                    (batch_size,), keep_traj, dtype=torch.bool, device=device
                ),
                "keep_style": torch.full(
                    (batch_size,), keep_style, dtype=torch.bool, device=device
                ),
            }
            return self.forward(
                x_t,
                timesteps,
                lma_seq=lma_seq,
                traj_seq=traj_seq,
                style_id=normalized_style_id,
                mode="full",
                condition_mask=condition_mask,
                return_aux=True,
            )

        x0_uncond, root_uncond = predict_mask(False, False, False)
        if mode == "uncond":
            return x0_uncond, root_uncond

        x0_pred = x0_uncond
        root_pred = root_uncond
        base_x0 = x0_uncond
        base_root = root_uncond

        if has_style:
            x0_style, root_style = predict_mask(False, False, True)
            x0_pred = x0_pred + style_scale * (x0_style - x0_uncond)
            root_pred = root_pred + style_scale * (root_style - root_uncond)
            base_x0 = x0_style
            base_root = root_style

        if mode == "lma_only":
            if self._condition_available(lma_seq):
                x0_lma, root_lma = predict_mask(True, False, has_style)
                x0_pred = x0_pred + lma_scale * (x0_lma - base_x0)
                root_pred = root_pred + lma_scale * (root_lma - base_root)
            return x0_pred, root_pred

        if mode == "traj_only":
            if self._condition_available(traj_seq):
                x0_traj, root_traj = predict_mask(False, True, has_style)
                x0_pred = x0_pred + traj_scale * (x0_traj - base_x0)
                root_pred = root_pred + traj_scale * (root_traj - base_root)
            return x0_pred, root_pred

        if mode != "full":
            raise ValueError(f"Unsupported conditioning mode: {mode}")

        if self._condition_available(traj_seq):
            x0_traj, root_traj = predict_mask(False, True, has_style)
            x0_pred = x0_pred + traj_scale * (x0_traj - base_x0)
            root_pred = root_pred + traj_scale * (root_traj - base_root)
            base_x0 = x0_traj
            base_root = root_traj

        if self._condition_available(lma_seq):
            x0_full, root_full = predict_mask(
                True, self._condition_available(traj_seq), has_style
            )
            x0_pred = x0_pred + lma_scale * (x0_full - base_x0)
            root_pred = root_pred + lma_scale * (root_full - base_root)

        return x0_pred, root_pred

    def _predict_eps(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, x0_pred: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = extract_schedule(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus = extract_schedule(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )
        return (x_t - sqrt_alpha * x0_pred) / sqrt_one_minus.clamp_min(1e-8)

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        eta: float,
    ) -> torch.Tensor:
        batch_timesteps = torch.full(
            (x_t.size(0),), timestep, device=x_t.device, dtype=torch.long
        )
        eps = self._predict_eps(x_t, batch_timesteps, x0_pred)
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = (
            x_t.new_tensor(1.0)
            if prev_timestep < 0
            else self.alphas_cumprod[prev_timestep]
        )

        sigma = 0.0
        if prev_timestep >= 0 and eta > 0.0:
            sigma = eta * torch.sqrt(
                ((1 - alpha_prev) / (1 - alpha_t)).clamp_min(0.0)
                * (1 - alpha_t / alpha_prev).clamp_min(0.0)
            )

        direction = torch.sqrt((1 - alpha_prev - sigma**2).clamp_min(0.0)) * eps
        x_prev = torch.sqrt(alpha_prev).clamp_min(1e-8) * x0_pred + direction
        if prev_timestep >= 0 and float(sigma) > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev

    @staticmethod
    def _match_sequence_length(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
        if sequence.dim() != 3:
            raise ValueError(f"Expected [B, T, C] source latent, got {sequence.shape}")

        target_len = max(int(target_len), 1)
        if sequence.size(1) == target_len:
            return sequence.contiguous()
        if sequence.size(1) <= 0:
            raise ValueError("source_latent must have at least one timestep")
        if sequence.size(1) == 1:
            return sequence.expand(sequence.size(0), target_len, sequence.size(2))

        return (
            F.interpolate(
                sequence.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _prepare_source_latent(
        self,
        source_latent,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        dtype = self.alphas_cumprod.dtype
        if not torch.is_tensor(source_latent):
            source_latent = torch.as_tensor(
                source_latent,
                dtype=dtype,
                device=device,
            )
        else:
            source_latent = source_latent.to(device=device, dtype=dtype)

        if source_latent.dim() == 2:
            source_latent = source_latent.unsqueeze(0)
        if source_latent.dim() != 3:
            raise ValueError(
                f"Expected [B, T, C] source latent, got {source_latent.shape}"
            )
        if source_latent.size(-1) != self.latent_dim:
            raise ValueError(
                "source_latent width does not match the prior latent width: "
                f"got {source_latent.size(-1)}, expected {self.latent_dim}"
            )
        if source_latent.size(0) == 1 and batch_size > 1:
            source_latent = source_latent.expand(batch_size, -1, -1)
        elif source_latent.size(0) != batch_size:
            raise ValueError(
                f"source_latent batch size must be 1 or {batch_size}, got {source_latent.size(0)}"
            )

        return self._match_sequence_length(source_latent, seq_len)

    @staticmethod
    def _build_sampling_schedule(
        start_timestep: int,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        start_timestep = max(int(start_timestep), 0)
        step_count = max(int(num_steps), 1)
        step_count = min(step_count, start_timestep + 1)
        timesteps = (
            torch.linspace(start_timestep, 0, steps=step_count, device=device)
            .round()
            .long()
        )
        return torch.unique_consecutive(timesteps)

    @staticmethod
    def _resolve_sampling_mode(
        mode: str,
        step_index: int,
        total_steps: int,
        use_full_mode_lma_prefix: bool,
        full_mode_lma_prefix_fraction: float,
        use_full_mode_lma_suffix: bool,
        full_mode_lma_suffix_fraction: float,
        has_lma_condition: bool,
    ) -> str:
        if mode != "full" or not has_lma_condition:
            return mode

        if use_full_mode_lma_prefix and use_full_mode_lma_suffix:
            raise ValueError(
                "full-mode LMA prefix and suffix schedules are mutually exclusive"
            )

        if use_full_mode_lma_prefix:
            fraction = float(full_mode_lma_prefix_fraction)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(
                    "full_mode_lma_prefix_fraction must be in [0, 1], "
                    f"got {fraction}"
                )
            prefix_steps = int(math.ceil(max(total_steps, 0) * fraction))
            if prefix_steps > 0 and step_index < prefix_steps:
                return "lma_only"

        if use_full_mode_lma_suffix:
            fraction = float(full_mode_lma_suffix_fraction)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(
                    "full_mode_lma_suffix_fraction must be in [0, 1], "
                    f"got {fraction}"
                )
            suffix_steps = int(math.ceil(max(total_steps, 0) * fraction))
            if suffix_steps > 0 and step_index >= max(total_steps - suffix_steps, 0):
                return "lma_only"

        return mode

    @torch.no_grad()
    def sample(
        self,
        seq_len: int,
        batch_size: int = 1,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        source_latent=None,
        source_noise_timestep: Optional[int] = None,
        mode: str = "full",
        cfg_scale: float = 2.5,
        lma_cfg_scale: Optional[float] = None,
        traj_cfg_scale: Optional[float] = None,
        style_cfg_scale: Optional[float] = None,
        use_full_mode_lma_prefix: bool = False,
        full_mode_lma_prefix_fraction: float = 0.5,
        use_full_mode_lma_suffix: bool = False,
        full_mode_lma_suffix_fraction: float = 0.5,
        num_steps: int = 50,
        eta: float = 0.0,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        if device is None:
            if torch.is_tensor(traj_seq):
                device = traj_seq.device
            elif torch.is_tensor(lma_seq):
                device = lma_seq.device
            elif torch.is_tensor(style_id):
                device = style_id.device
            else:
                device = next(self.parameters()).device

        if source_latent is not None:
            if source_noise_timestep is None:
                raise ValueError(
                    "source_noise_timestep is required when source_latent is provided"
                )
            max_timestep = max(self.num_train_timesteps - 1, 0)
            start_timestep = int(source_noise_timestep)
            if start_timestep < 0 or start_timestep > max_timestep:
                raise ValueError(
                    f"source_noise_timestep must be in [0, {max_timestep}], got {start_timestep}"
                )
            source_latent = self._prepare_source_latent(
                source_latent,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            if start_timestep <= 0:
                x_t = source_latent.clone()
            else:
                timestep_batch = torch.full(
                    (batch_size,),
                    start_timestep,
                    device=device,
                    dtype=torch.long,
                )
                source_noise = torch.randn_like(source_latent) * float(temperature)
                x_t = self.q_sample(
                    source_latent,
                    timestep_batch,
                    noise=source_noise,
                )
        else:
            start_timestep = self.num_train_timesteps - 1
            x_t = torch.randn(
                batch_size,
                seq_len,
                self.latent_dim,
                device=device,
            ) * float(temperature)
        timesteps = self._build_sampling_schedule(start_timestep, num_steps, device)

        for index, timestep in enumerate(timesteps.tolist()):
            t_batch = torch.full(
                (batch_size,), timestep, device=device, dtype=torch.long
            )
            effective_mode = self._resolve_sampling_mode(
                mode=mode,
                step_index=index,
                total_steps=len(timesteps),
                use_full_mode_lma_prefix=use_full_mode_lma_prefix,
                full_mode_lma_prefix_fraction=full_mode_lma_prefix_fraction,
                use_full_mode_lma_suffix=use_full_mode_lma_suffix,
                full_mode_lma_suffix_fraction=full_mode_lma_suffix_fraction,
                has_lma_condition=self._condition_available(lma_seq),
            )
            x0_pred, root_traj_pred = self._guided_prediction(
                x_t,
                t_batch,
                lma_seq=lma_seq,
                traj_seq=traj_seq,
                style_id=style_id,
                mode=effective_mode,
                cfg_scale=cfg_scale,
                lma_cfg_scale=lma_cfg_scale,
                traj_cfg_scale=traj_cfg_scale,
                style_cfg_scale=style_cfg_scale,
            )
            prev_timestep = (
                timesteps[index + 1].item() if index + 1 < len(timesteps) else -1
            )
            x_t = self._ddim_step(x_t, x0_pred, timestep, prev_timestep, eta=eta)
        if return_aux:
            return x0_pred, root_traj_pred
        return x0_pred

    @staticmethod
    def _chunk_starts(total_len: int, chunk_len: int, overlap_len: int):
        if chunk_len <= 0:
            raise ValueError("chunk_len must be positive")
        if overlap_len >= chunk_len:
            raise ValueError("overlap_len must be smaller than chunk_len")

        starts = [0]
        step = chunk_len - overlap_len
        while starts[-1] + chunk_len < total_len:
            next_start = starts[-1] + step
            if next_start + chunk_len >= total_len:
                next_start = max(total_len - chunk_len, 0)
            if next_start == starts[-1]:
                break
            starts.append(next_start)
        return starts

    @staticmethod
    def _slice_condition(condition, start: int, end: int, total_target_len: int):
        if condition is None:
            return None
        if isinstance(condition, dict):
            return {
                key: ContinuousLatentDiffusionPrior._slice_condition(
                    value, start, end, total_target_len
                )
                for key, value in condition.items()
            }

        tensor = condition
        if not torch.is_tensor(tensor):
            dtype = torch.long if isinstance(condition, int) else torch.float32
            tensor = torch.tensor(tensor, dtype=dtype)
        if tensor.dim() <= 1:
            return tensor
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 4 and tensor.size(-1) == 1:
            tensor = tensor.squeeze(-1)

        total_seq_len = tensor.size(1)
        scale = float(total_seq_len) / float(max(total_target_len, 1))
        start_idx = max(int(round(start * scale)), 0)
        end_idx = min(int(round(end * scale)), total_seq_len)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, total_seq_len)
        return tensor[:, start_idx:end_idx]

    @staticmethod
    def _blend_weights(
        length: int, overlap_left: int, overlap_right: int, device: torch.device
    ) -> torch.Tensor:
        weights = torch.ones(length, device=device)
        if overlap_left > 0:
            left = torch.linspace(0.0, 1.0, overlap_left + 1, device=device)[1:]
            weights[:overlap_left] = 0.5 - 0.5 * torch.cos(left * math.pi)
        if overlap_right > 0:
            right = torch.linspace(1.0, 0.0, overlap_right + 1, device=device)[:-1]
            weights[-overlap_right:] = torch.minimum(
                weights[-overlap_right:],
                0.5 - 0.5 * torch.cos(right * math.pi),
            )
        return weights

    @torch.no_grad()
    def sample_long(
        self,
        total_seq_len: int,
        batch_size: int = 1,
        lma_seq=None,
        traj_seq=None,
        style_id=None,
        source_latent=None,
        source_noise_timestep: Optional[int] = None,
        mode: str = "full",
        cfg_scale: float = 2.5,
        lma_cfg_scale: Optional[float] = None,
        traj_cfg_scale: Optional[float] = None,
        style_cfg_scale: Optional[float] = None,
        use_full_mode_lma_prefix: bool = False,
        full_mode_lma_prefix_fraction: float = 0.5,
        use_full_mode_lma_suffix: bool = False,
        full_mode_lma_suffix_fraction: float = 0.5,
        num_steps: int = 50,
        chunk_len: int = 128,
        overlap_len: int = 32,
        halo_len: int = 8,
        eta: float = 0.0,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        if total_seq_len <= chunk_len:
            return self.sample(
                seq_len=total_seq_len,
                batch_size=batch_size,
                lma_seq=lma_seq,
                traj_seq=traj_seq,
                style_id=style_id,
                source_latent=source_latent,
                source_noise_timestep=source_noise_timestep,
                mode=mode,
                cfg_scale=cfg_scale,
                lma_cfg_scale=lma_cfg_scale,
                traj_cfg_scale=traj_cfg_scale,
                style_cfg_scale=style_cfg_scale,
                use_full_mode_lma_prefix=use_full_mode_lma_prefix,
                full_mode_lma_prefix_fraction=full_mode_lma_prefix_fraction,
                use_full_mode_lma_suffix=use_full_mode_lma_suffix,
                full_mode_lma_suffix_fraction=full_mode_lma_suffix_fraction,
                num_steps=num_steps,
                eta=eta,
                temperature=temperature,
                device=device,
                return_aux=return_aux,
            )

        if device is None:
            device = next(self.parameters()).device

        combined = torch.zeros(
            batch_size, total_seq_len, self.latent_dim, device=device
        )
        combined_root = torch.zeros(
            batch_size, total_seq_len, self.traj_dim, device=device
        )
        weights = torch.zeros(batch_size, total_seq_len, 1, device=device)
        starts = self._chunk_starts(total_seq_len, chunk_len, overlap_len)

        for chunk_start in starts:
            chunk_end = min(chunk_start + chunk_len, total_seq_len)
            context_start = max(chunk_start - halo_len, 0)
            context_end = min(chunk_end + halo_len, total_seq_len)
            context_len = context_end - context_start

            chunk_lma = self._slice_condition(
                lma_seq, context_start, context_end, total_seq_len
            )
            chunk_traj = self._slice_condition(
                traj_seq, context_start, context_end, total_seq_len
            )
            chunk_style = self._slice_condition(
                style_id, context_start, context_end, total_seq_len
            )
            chunk_source_latent = self._slice_condition(
                source_latent, context_start, context_end, total_seq_len
            )
            sampled_latent, sampled_root = self.sample(
                seq_len=context_len,
                batch_size=batch_size,
                lma_seq=chunk_lma,
                traj_seq=chunk_traj,
                style_id=chunk_style,
                source_latent=chunk_source_latent,
                source_noise_timestep=source_noise_timestep,
                mode=mode,
                cfg_scale=cfg_scale,
                lma_cfg_scale=lma_cfg_scale,
                traj_cfg_scale=traj_cfg_scale,
                style_cfg_scale=style_cfg_scale,
                use_full_mode_lma_prefix=use_full_mode_lma_prefix,
                full_mode_lma_prefix_fraction=full_mode_lma_prefix_fraction,
                use_full_mode_lma_suffix=use_full_mode_lma_suffix,
                full_mode_lma_suffix_fraction=full_mode_lma_suffix_fraction,
                num_steps=num_steps,
                eta=eta,
                temperature=temperature,
                device=device,
                return_aux=True,
            )

            core_start = chunk_start - context_start
            core_end = core_start + (chunk_end - chunk_start)
            sampled_core = sampled_latent[:, core_start:core_end]
            sampled_root_core = sampled_root[:, core_start:core_end]

            overlap_left = (
                0 if chunk_start == 0 else min(overlap_len, chunk_end - chunk_start)
            )
            overlap_right = (
                0
                if chunk_end == total_seq_len
                else min(overlap_len, chunk_end - chunk_start)
            )
            chunk_weights = self._blend_weights(
                sampled_core.size(1),
                overlap_left=overlap_left,
                overlap_right=overlap_right,
                device=device,
            ).view(1, -1, 1)

            combined[:, chunk_start:chunk_end] += sampled_core * chunk_weights
            combined_root[:, chunk_start:chunk_end] += sampled_root_core * chunk_weights
            weights[:, chunk_start:chunk_end] += chunk_weights

        combined = combined / weights.clamp_min(1e-8)
        combined_root = combined_root / weights.clamp_min(1e-8)
        if return_aux:
            return combined, combined_root
        return combined
