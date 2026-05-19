"""LMA-conditioned MaskGIT Transformer prior for VQ-VAE motion generation (v5).

MaskGIT-style masked token prediction with iterative parallel decoding.

Key architectural differences from v1–v4 (all autoregressive GPT):
──────────────────────────────────────────────────────────────────
• Bidirectional self-attention — every token attends to ALL others, not
  just predecessors.  This gives dramatically better temporal coherence:
  the model can "see" both past and future context when predicting.

• Masked token prediction — during training, a random subset of tokens is
  masked and the model predicts them from the visible tokens + conditioning.
  This is the BERT / MaskGIT paradigm applied to motion code indices.

• Iterative parallel decoding — at inference, all tokens start masked and
  are progressively unmasked over S steps (default 12).  Each step predicts
  all masked tokens in parallel; the most confident predictions are kept,
  the rest are re-masked for the next iteration.  Yields higher quality
  than single-shot AR generation through iterative refinement.

• Classifier-free guidance (CFG) — ctrl and LMA are dropped
    independently during training via learned null embeddings.  At inference,
    the model runs conditioned and nulled forwards, and the final logits are:
      logits = uncond + cfg_scale × (cond − uncond)
  This amplifies the effect of ctrl / LMA conditioning.

• RoPE positional encoding — Rotary Position Embeddings for bidirectional
  attention.  Better suited than ALiBi (designed for causal) in the
  bidirectional setting.  Generalises to variable sequence lengths.

• Mask ratio embedding — a sinusoidal + MLP encoding of the current masking
  ratio, added globally.  Tells the transformer "how much is still unknown",
  analogous to the timestep embedding in diffusion models.

Compatibility contract (unchanged from v1–v4 where possible):
  • ``forward()`` returns ``(indices, logits)`` → ``[B, T, L]``, ``[B, T, L, K]``.
  • ``sample()`` returns the same.  Extra kwargs: ``num_steps``, ``cfg_scale``.
  • ``anneal_teacher_forcing(factor)`` supported (no-op — MaskGIT uses
    adaptive cosine masking, no manual annealing required).
  • Accepts ``lma_down`` as pre-encoded tensor or raw-channel dict.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── LMA channel registry ──────────────────────────────────────────

LMA_CHANNELS = (
    "BODY",
    "EFFORT_WEIGHT_STRONG",
    "EFFORT_TIME_SUDDEN",
    "EFFORT_FLOW_BOUND",
    "SHAPE",
    "SPACE",
)
NUM_LMA_RAW = len(LMA_CHANNELS)  # 6


# ─── Building blocks ───────────────────────────────────────────────


class ConvNorm(nn.Module):
    """LayerNorm applied over the channel dim of (B, C, T) tensors."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ResidualTemporalBlock(nn.Module):
    """Dilated residual temporal block used by the full-rate root refiner."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            ConvNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            ConvNorm(channels),
        )
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x + self.net(x))


# ─── RoPE Bidirectional Self-Attention ──────────────────────────────


class RoPESelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with Rotary Position Embeddings.

    RoPE encodes relative position information by rotating Q/K vectors,
    enabling the model to generalise across different sequence lengths
    without learned positional embeddings.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Precompute inverse frequencies for RoPE
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _build_rope(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)           # [T, Dh/2]
        emb = torch.cat([freqs, freqs], dim=-1)          # [T, Dh]
        cos = emb.cos().unsqueeze(0).unsqueeze(0)         # [1, 1, T, Dh]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        qkv = self.qkv_proj(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each [B, H, T, Dh]

        cos, sin = self._build_rope(T, x.device)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ─── Transformer block ─────────────────────────────────────────────


class BidirectionalBlock(nn.Module):
    """Pre-norm bidirectional transformer block with RoPE self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RoPESelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ─── Mask ratio embedding ──────────────────────────────────────────


class MaskRatioEmbedding(nn.Module):
    """Sinusoidal + MLP embedding for the current mask ratio ∈ [0, 1].

    Analogous to the timestep embedding in diffusion models — tells the
    transformer how much of the sequence is still unknown so it can trade
    off between bold first-pass guesses and fine-grained refinements.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, ratio: torch.Tensor) -> torch.Tensor:
        """ratio: [B] tensor in [0, 1] → [B, hidden_dim]."""
        if ratio.dim() == 0:
            ratio = ratio.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=ratio.device).float()
            / half
        )
        args = ratio[:, None].float() * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class PriorRootPredictor(nn.Module):
    """Predict full-rate root motion from latent state, conditioning, and decoded body motion."""

    def __init__(self, hidden_dim: int, body_motion_dim: int, dropout: float = 0.1):
        super().__init__()
        fusion_dim = max(hidden_dim, 256)
        head_dim = max(hidden_dim // 2, 128)

        self.hidden_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ConvNorm(hidden_dim),
            nn.GELU(),
        )
        self.cond_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ConvNorm(hidden_dim),
            nn.GELU(),
        )
        self.body_proj = nn.Sequential(
            nn.Conv1d(body_motion_dim, hidden_dim, kernel_size=7, padding=3),
            ConvNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ConvNorm(hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, fusion_dim, kernel_size=1),
            ConvNorm(fusion_dim),
            nn.GELU(),
        )
        self.body_gate = nn.Sequential(
            nn.Conv1d(hidden_dim, fusion_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ResidualTemporalBlock(fusion_dim, kernel_size=5, dilation=1, dropout=dropout),
            ResidualTemporalBlock(fusion_dim, kernel_size=5, dilation=2, dropout=dropout),
            ResidualTemporalBlock(fusion_dim, kernel_size=5, dilation=4, dropout=dropout),
            ResidualTemporalBlock(fusion_dim, kernel_size=5, dilation=8, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Conv1d(fusion_dim, head_dim, kernel_size=5, padding=2),
            ConvNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.root_rot_head = nn.Conv1d(head_dim, 6, kernel_size=7, padding=3)
        self.root_disp_head = nn.Conv1d(head_dim, 3, kernel_size=7, padding=3)

    @staticmethod
    def _upsample_stream(x: torch.Tensor, full_seq_len: int) -> torch.Tensor:
        if x.size(-1) == full_seq_len:
            return x
        return F.interpolate(
            x,
            size=full_seq_len,
            mode="linear",
            align_corners=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cond: torch.Tensor,
        full_seq_len: int,
        decoded_body: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_full = self._upsample_stream(
            self.hidden_proj(hidden.transpose(1, 2)),
            full_seq_len,
        )
        cond_full = self._upsample_stream(
            self.cond_proj(cond.transpose(1, 2)),
            full_seq_len,
        )

        if decoded_body is not None:
            body_feat = self._upsample_stream(self.body_proj(decoded_body), full_seq_len)
        else:
            body_feat = torch.zeros_like(hidden_full)

        fused = self.fuse(torch.cat([hidden_full, cond_full, body_feat], dim=1))
        fused = fused * (1.0 + self.body_gate(body_feat))
        fused = self.refine(fused)
        root_feat = self.head(fused)
        root_rot = self.root_rot_head(root_feat)
        root_disp = self.root_disp_head(root_feat)
        return torch.cat([root_rot, root_disp], dim=1).transpose(1, 2)


# ─── Main module ────────────────────────────────────────────────────


class LMATransformerPrior(nn.Module):
    """MaskGIT-style masked-prediction prior for VQ-VAE code indices (v5).

    Core recipe:
    1. Ctrl + LMA → per-position condition vectors (Conv1d stacks).
    2. Randomly mask a fraction of target tokens (cosine schedule).
    3. Bidirectional transformer with RoPE predicts all token positions.
    4. Gradient flows only through masked positions (clean learning signal).
    5. At inference, iteratively unmask from fully-masked using confidence.
    6. Classifier-free guidance amplifies conditioning at inference.
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
        body_motion_dim=0,
        p_drop_ctrl=0.50,
        p_drop_lma=0.10,
        p_drop_both=0.05,
    ):
        super().__init__()

        # Store config
        self.ctrl_dim = ctrl_dim
        self.num_codebook_vectors = num_codebook_vectors
        self.num_levels = num_levels
        self.vq_dim = vq_dim
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.lma_dim = lma_dim
        self.body_motion_dim = body_motion_dim

        # Compat knobs (kept for interface compatibility with v1–v4)
        self.teacher_forcing_prob = 0.75
        self.corruption_rate = 0.5

        # Asymmetric per-signal conditioning dropout.
        self.p_drop_ctrl = p_drop_ctrl
        self.p_drop_lma = p_drop_lma
        self.p_drop_both = p_drop_both

        # MaskGIT inference defaults
        self.default_num_steps = 12
        self.default_cfg_scale = 2.5
        self.body_motion_decoder = None

        num_heads = self._choose_num_heads(hidden_dim)

        # ── Ctrl encoder (3 × stride-2 = 8× downsample → hidden_dim) ──
        ch1 = max(hidden_dim // 4, ctrl_dim * 4)
        ch2 = max(hidden_dim // 2, ch1)
        self.ctrl_encoder = nn.Sequential(
            nn.Conv1d(ctrl_dim, ch1, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(ch1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch1, ch2, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(ch2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch2, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # ── LMA encoder (pre-encoded tensor path: 2 × stride-2 = 4×) ──
        if lma_dim > 0:
            lma_mid = max(hidden_dim // 4, lma_dim * 2)
            self.lma_encoded_proj = nn.Sequential(
                nn.Conv1d(lma_dim, lma_mid, kernel_size=5, stride=2, padding=2),
                ConvNorm(lma_mid), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(lma_mid, hidden_dim, kernel_size=3, stride=2, padding=1),
                ConvNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            )

        # ── LMA encoder (raw dict path: 3 × stride-2 = 8×) ──
        raw_mid1 = max(hidden_dim // 8, NUM_LMA_RAW * 4)
        raw_mid2 = max(hidden_dim // 4, raw_mid1)
        self.lma_raw_encoder = nn.Sequential(
            nn.Conv1d(NUM_LMA_RAW, raw_mid1, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(raw_mid1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(raw_mid1, raw_mid2, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(raw_mid2), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(raw_mid2, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2),
            ConvNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

        # ── LMA gate (starts low so LMA encoders can warm up) ──
        self.lma_gate = nn.Parameter(torch.tensor(-2.0))
        self.null_ctrl = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.null_lma = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # ── Token embeddings ───────────────────────────────────────
        self.level_embeddings = nn.ModuleList(
            [nn.Embedding(num_codebook_vectors, vq_dim) for _ in range(num_levels)]
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, vq_dim))
        self.token_proj = nn.Linear(vq_dim, hidden_dim)

        # ── Mask ratio embedding ───────────────────────────────────
        self.mask_ratio_emb = MaskRatioEmbedding(hidden_dim)

        # ── Transformer (bidirectional) ────────────────────────────
        self.blocks = nn.ModuleList(
            [BidirectionalBlock(hidden_dim, num_heads, dropout)
             for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_codebook_vectors) for _ in range(num_levels)]
        )
        self.prior_root_predictor = PriorRootPredictor(
            hidden_dim,
            body_motion_dim=max(1, body_motion_dim),
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

        # ── Initialisation ─────────────────────────────────────────
        for emb in self.level_embeddings:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    # ─── Public API ─────────────────────────────────────────────────

    def anneal_teacher_forcing(self, factor: float):
        """Compat no-op.  MaskGIT uses cosine masking — no annealing needed."""
        pass

    def set_body_motion_decoder(self, decoder_fn):
        self.body_motion_decoder = decoder_fn

    def forward(
        self,
        ctrl_seq: torch.Tensor,
        target_indices: torch.Tensor = None,
        temperature: float = 1.0,
        max_length: int = None,
        codebooks: torch.Tensor = None,
        yaw_sin_cos: torch.Tensor = None,
        lma_down=None,
        mode: str = "full",
    ):

        del codebooks, yaw_sin_cos  # kept for interface compat

        target_len = None
        full_seq_len = ctrl_seq.size(1)
        if target_indices is not None:
            target_indices = self._normalize_indices(target_indices)
            target_len = target_indices.size(1)

        ctrl_enc, lma_enc = self._encode_condition_streams(ctrl_seq, lma_down, target_len)

        if self.training and target_indices is not None:
            ctrl_enc, lma_enc = self._apply_condition_dropout(ctrl_enc, lma_enc)
            cond = self._combine_conditions(ctrl_enc, lma_enc)
        else:
            cond, cond_uncond = self._build_inference_conditions(ctrl_enc, lma_enc, mode)

        seq_len = cond.size(1)

        if self.training and target_indices is not None:
            target_indices = target_indices[:, :seq_len]
            indices, logits, root_prediction = self._train_forward(
                cond,
                target_indices,
                full_seq_len=full_seq_len,
            )
            return indices, logits, ctrl_seq[:, :full_seq_len], root_prediction

        gen_len = seq_len if max_length is None else min(max_length, seq_len)
        full_gen_len = min(full_seq_len, gen_len * self.stride)
        indices, logits, root_prediction = self._sample_maskgit(
            cond[:, :gen_len],
            gen_len,
            temperature,
            cond_uncond=cond_uncond[:, :gen_len],
            full_seq_len=full_gen_len,
        )
        return indices, logits, ctrl_seq[:, :full_gen_len], root_prediction

    @torch.no_grad()
    def sample(
        self,
        ctrl_seq: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = None,
        codebooks: torch.Tensor = None,
        yaw_sin_cos: torch.Tensor = None,
        lma_down=None,
        num_steps: int = None,
        cfg_scale: float = None,
        mode: str = "full",
    ):
        del codebooks, yaw_sin_cos
        ctrl_enc, lma_enc = self._encode_condition_streams(ctrl_seq, lma_down, target_len=None)
        cond, cond_uncond = self._build_inference_conditions(ctrl_enc, lma_enc, mode)
        seq_len = cond.size(1)
        full_seq_len = ctrl_seq.size(1)
        steps = num_steps if num_steps is not None else self.default_num_steps
        scale = cfg_scale if cfg_scale is not None else self.default_cfg_scale
        indices, logits, root_prediction = self._sample_maskgit(
            cond,
            seq_len,
            temperature,
            top_k,
            steps,
            scale,
            cond_uncond=cond_uncond,
            full_seq_len=full_seq_len,
        )
        return indices, logits, ctrl_seq[:, :full_seq_len], root_prediction

    # ─── Condition Encoding ─────────────────────────────────────────

    def _encode_condition_streams(self, ctrl_seq, lma_down, target_len):
        """Encode ctrl and LMA streams separately at the latent rate."""
        ctrl_enc = self.ctrl_encoder(ctrl_seq.transpose(1, 2)).transpose(1, 2)
        if target_len is not None:
            ctrl_enc = self._align_length(ctrl_enc, target_len)

        T_ds = ctrl_enc.size(1)
        lma_enc = self._encode_lma(lma_down, T_ds)
        return ctrl_enc, lma_enc

    def _apply_condition_dropout(self, ctrl_enc, lma_enc):
        """Drop ctrl and LMA independently using learned null embeddings."""
        B = ctrl_enc.size(0)
        device = ctrl_enc.device

        drop_ctrl = torch.rand(B, device=device) < self.p_drop_ctrl
        drop_lma = torch.rand(B, device=device) < self.p_drop_lma
        drop_both = torch.rand(B, device=device) < self.p_drop_both

        ctrl_enc = torch.where(
            (drop_ctrl | drop_both)[:, None, None],
            self._expand_null_ctrl(ctrl_enc),
            ctrl_enc,
        )

        if lma_enc is not None:
            lma_enc = torch.where(
                (drop_lma | drop_both)[:, None, None],
                self._expand_null_lma(lma_enc),
                lma_enc,
            )

        return ctrl_enc, lma_enc

    def _build_inference_conditions(self, ctrl_enc, lma_enc, mode):
        """Build conditioned and unconditioned inputs for inference CFG."""
        mode = "full" if mode is None else mode.lower()

        null_ctrl = self._expand_null_ctrl(ctrl_enc)
        null_lma = self._expand_null_lma(lma_enc) if lma_enc is not None else None
        cond_uncond = self._combine_conditions(null_ctrl, null_lma)

        if mode == "full":
            cond = self._combine_conditions(ctrl_enc, lma_enc)
        elif mode == "lma_only":
            cond = self._combine_conditions(null_ctrl, lma_enc)
        elif mode == "uncond":
            cond = cond_uncond
        else:
            raise ValueError(f"Unsupported conditioning mode: {mode}")

        return cond, cond_uncond

    def _combine_conditions(self, ctrl_enc, lma_enc):
        cond = ctrl_enc
        if lma_enc is not None:
            # no gating
            # cond = cond + self._gate_value() * lma_enc
            cond = cond + lma_enc
        return cond

    def _expand_null_ctrl(self, like: torch.Tensor) -> torch.Tensor:
        return self.null_ctrl.expand(like.size(0), like.size(1), -1)

    def _expand_null_lma(self, like: torch.Tensor) -> torch.Tensor:
        return self.null_lma.expand(like.size(0), like.size(1), -1)

    def _gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.lma_gate) * 0.9 + 0.1

    def _encode_lma(self, lma_down, target_len):
        """Dispatch LMA encoding based on the input type."""
        if lma_down is None:
            return None
        if isinstance(lma_down, dict):
            return self._encode_lma_dict(lma_down, target_len)
        if torch.is_tensor(lma_down):
            return self._encode_lma_tensor(lma_down, target_len)
        return None

    def _encode_lma_dict(self, lma_dict, target_len):
        """Encode a dict of raw per-channel signals → [B, target_len, H]."""
        parts = []
        for ch_name in LMA_CHANNELS:
            if ch_name in lma_dict and torch.is_tensor(lma_dict[ch_name]):
                t = lma_dict[ch_name]
                if t.dim() == 1:
                    t = t.unsqueeze(0).unsqueeze(-1)
                elif t.dim() == 2:
                    t = t.unsqueeze(-1)
                parts.append(t)
            else:
                if parts:
                    parts.append(torch.zeros_like(parts[0]))
                else:
                    return None
        lma_raw = torch.cat(parts, dim=-1)
        lma_enc = self.lma_raw_encoder(
            lma_raw.transpose(1, 2),
        ).transpose(1, 2)
        return self._align_length(lma_enc, target_len)

    def _encode_lma_tensor(self, lma_tensor, target_len):
        """Encode a pre-encoded or raw LMA tensor → [B, target_len, H]."""
        if lma_tensor.dim() == 2:
            lma_tensor = lma_tensor.unsqueeze(0)

        last_dim = lma_tensor.size(-1)
        if self.lma_dim > 0 and last_dim == self.lma_dim:
            lma_enc = self.lma_encoded_proj(
                lma_tensor.transpose(1, 2),
            ).transpose(1, 2)
        elif last_dim <= NUM_LMA_RAW:
            if last_dim < NUM_LMA_RAW:
                pad = torch.zeros(
                    *lma_tensor.shape[:-1], NUM_LMA_RAW - last_dim,
                    device=lma_tensor.device, dtype=lma_tensor.dtype,
                )
                lma_tensor = torch.cat([lma_tensor, pad], dim=-1)
            lma_enc = self.lma_raw_encoder(
                lma_tensor.transpose(1, 2),
            ).transpose(1, 2)
        elif self.lma_dim > 0:
            lma_enc = self.lma_encoded_proj(
                lma_tensor[..., : self.lma_dim].transpose(1, 2),
            ).transpose(1, 2)
        else:
            return None

        return self._align_length(lma_enc, target_len)

    # ─── Training forward ──────────────────────────────────────────

    def _train_forward(self, cond, target_indices, full_seq_len):
        """Masked token prediction with cosine masking and learned-null dropout.

        1. Sample a per-sample masking ratio r ~ cos(U[0,1] · π/2).
        2. Mask each token independently with probability r  (Bernoulli).
        3. Condition dropout is applied upstream, before ctrl/LMA are combined.
        4. Run the bidirectional transformer.
        5. Detach gradient at unmasked logit positions so loss only trains
           the model through predictions at masked tokens.
        """
        B, T, L = target_indices.shape
        device = cond.device

        # ── 1. Cosine masking schedule ──
        # Biased toward high masking: avg ≈ 64% masked.
        u = torch.rand(B, device=device)
        mask_ratio = torch.cos(u * math.pi * 0.5)

        # ── 2. Per-token Bernoulli mask ──
        is_masked = torch.rand(B, T, device=device) < mask_ratio[:, None]
        # Guarantee at least one masked token per sample
        no_mask = ~is_masked.any(dim=1)
        if no_mask.any():
            force = torch.randint(T, (int(no_mask.sum()),), device=device)
            is_masked[no_mask, force] = True

        # ── 3. Build token embeddings ──
        gt_emb = self._embed_indices(target_indices)           # [B, T, vq_dim]
        mask_emb = self.mask_token.expand(B, T, -1)
        token_emb = torch.where(
            is_masked.unsqueeze(-1).expand_as(gt_emb),
            mask_emb, gt_emb,
        )

        # ── 4. Mask ratio embedding ──
        ratio_emb = self.mask_ratio_emb(
            is_masked.float().mean(dim=1),
        )                                                      # [B, hidden_dim]

        # ── 5. Bidirectional transformer ──
        hidden = self.drop(
            self.token_proj(token_emb) + cond + ratio_emb.unsqueeze(1),
        )
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self._project_logits(hidden)                  # [B, T, L, K]

        # ── 6. Detach gradient at unmasked positions ──
        # The external CE loss is computed on all positions.  We ensure
        # gradient flows ONLY through masked positions by detaching the
        # unmasked ones.  The visible tokens still help predict masked tokens
        # via the bidirectional attention (that gradient path is intact).
        mask_4d = is_masked[:, :, None, None].expand_as(logits).float()
        logits = logits * mask_4d + logits.detach() * (1.0 - mask_4d)

        indices = torch.argmax(logits, dim=-1)
        root_prediction = self._predict_root_from_indices(indices, cond, full_seq_len)
        return indices, logits, root_prediction

    # ─── MaskGIT iterative sampling ────────────────────────────────

    def _sample_maskgit(
        self,
        cond: torch.Tensor,
        seq_len: int,
        temperature: float = 1.0,
        top_k: int = None,
        num_steps: int = None,
        cfg_scale: float = None,
        cond_uncond: torch.Tensor = None,
        full_seq_len: int = None,
    ):
        """Iterative parallel decoding: fully masked → fully unmasked.

        At each step:
        1. Predict logits for all positions (bidirectional).
        2. Sample tokens at masked positions.
        3. Compute per-position confidence (prob of sampled token).
        4. Keep the ``num_to_unmask`` most confident predictions;
           re-mask the rest for the next iteration.

        Cosine schedule determines how many tokens to unmask per step,
        biased toward unmasking more early (when context is sparse).
        """
        if num_steps is None:
            num_steps = self.default_num_steps
        if cfg_scale is None:
            cfg_scale = self.default_cfg_scale
        if cond_uncond is None:
            cond_uncond = torch.zeros_like(cond)

        B = cond.size(0)
        device = cond.device
        L = self.num_levels
        K = self.num_codebook_vectors

        tokens = torch.zeros(B, seq_len, L, dtype=torch.long, device=device)
        is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        final_logits = torch.zeros(B, seq_len, L, K, device=device)

        for step in range(num_steps):
            # ── How many tokens remain masked after this step ──
            r_after = math.cos(math.pi * 0.5 * (step + 1) / num_steps)
            num_masked_after = max(0, round(seq_len * r_after))
            if step == num_steps - 1:
                num_masked_after = 0

            # ── Build input token embeddings ──
            tok_emb = self._embed_indices(tokens.clamp(min=0))
            mask_emb = self.mask_token.expand(B, seq_len, -1)
            tok_emb = torch.where(
                is_masked.unsqueeze(-1).expand_as(tok_emb),
                mask_emb, tok_emb,
            )

            cur_ratio = is_masked.float().mean(dim=1)          # [B]
            ratio_emb = self.mask_ratio_emb(cur_ratio)          # [B, H]

            # ── Conditioned forward pass ──
            hidden_c = self.token_proj(tok_emb) + cond + ratio_emb.unsqueeze(1)
            for block in self.blocks:
                hidden_c = block(hidden_c)
            logits_cond = self._project_logits(self.final_norm(hidden_c))

            # ── Classifier-free guidance ──
            if cfg_scale != 1.0:
                hidden_u = self.token_proj(tok_emb) + cond_uncond + ratio_emb.unsqueeze(1)
                for block in self.blocks:
                    hidden_u = block(hidden_u)
                logits_uncond = self._project_logits(self.final_norm(hidden_u))
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            final_logits = logits

            # ── Sample all positions ──
            all_sampled = torch.zeros_like(tokens)
            confidence = torch.ones(B, seq_len, device=device)

            for level in range(L):
                lv_logits = logits[:, :, level]              # [B, T, K]

                if temperature > 0:
                    scaled = lv_logits / max(temperature, 1e-5)
                else:
                    scaled = lv_logits * 1e5

                if top_k is not None and 0 < top_k < K:
                    cutoff = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
                    scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))

                probs = F.softmax(scaled, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, K), 1,
                ).view(B, seq_len)
                all_sampled[:, :, level] = sampled

                conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                confidence = confidence * conf

            # Update tokens at masked positions only
            update_mask = is_masked.unsqueeze(-1).expand_as(tokens)
            tokens = torch.where(update_mask, all_sampled, tokens)

            # ── Re-mask least confident predictions ──
            if num_masked_after > 0 and is_masked.any():
                conf = confidence.clone()
                conf[~is_masked] = float("inf")              # protect unmasked

                _, sorted_idx = conf.sort(dim=1)              # ascending
                threshold = (
                    torch.arange(seq_len, device=device).unsqueeze(0)
                    .expand(B, -1) < num_masked_after
                )
                new_mask = torch.zeros_like(is_masked)
                new_mask.scatter_(1, sorted_idx, threshold)
                is_masked = new_mask
            else:
                is_masked.fill_(False)

        if full_seq_len is None:
            full_seq_len = seq_len * self.stride
        root_prediction = self._predict_root_from_indices(tokens, cond, full_seq_len)
        return tokens, final_logits, root_prediction

    # ─── Logits projection ─────────────────────────────────────────

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        per_level = [head(hidden) for head in self.output_heads]
        return torch.stack(per_level, dim=2)                   # [B, T, L, K]

    # ─── Index / embedding helpers ──────────────────────────────────

    def _embed_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = self._normalize_indices(indices)
        emb = 0.0
        for lvl in range(min(indices.size(-1), self.num_levels)):
            emb = emb + self.level_embeddings[lvl](indices[..., lvl])
        return emb

    def _predict_root_from_indices(
        self,
        indices: torch.Tensor,
        cond: torch.Tensor,
        full_seq_len: int,
    ) -> torch.Tensor:
        hidden = self._refine_hidden_from_indices(indices, cond)
        decoded_body = self._decode_body_motion(indices)
        return self.prior_root_predictor(
            hidden,
            cond,
            full_seq_len,
            decoded_body=decoded_body,
        )

    def _decode_body_motion(self, indices: torch.Tensor):
        if self.body_motion_decoder is None:
            return None
        return self.body_motion_decoder(indices)

    def _refine_hidden_from_indices(
        self,
        indices: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        token_emb = self._embed_indices(indices)
        zero_ratio = torch.zeros(indices.size(0), device=cond.device)
        ratio_emb = self.mask_ratio_emb(zero_ratio)
        hidden = self.drop(
            self.token_proj(token_emb) + cond + ratio_emb.unsqueeze(1),
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def _normalize_indices(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)
        return indices.long().clamp(0, self.num_codebook_vectors - 1)

    def _align_length(
        self, tokens: torch.Tensor, target_len: int,
    ) -> torch.Tensor:
        if tokens.size(1) == target_len:
            return tokens
        return F.interpolate(
            tokens.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    @staticmethod
    def _choose_num_heads(hidden_dim: int) -> int:
        candidate = max(1, min(8, hidden_dim // 64))
        while candidate > 1 and hidden_dim % candidate != 0:
            candidate -= 1
        return candidate
