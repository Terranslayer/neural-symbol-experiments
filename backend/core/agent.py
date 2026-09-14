# -*- coding: utf-8 -*-
"""
GRU-based agent for Stage 1 multi-comparison task.

Architecture:
    input_proj (Linear 3 -> d_model)
        |
        v
    GRUCell (d_model -> d_model)   # shared across all phases
        |
        +--> write_head (Linear d_model -> 1)  : outputs scratch pad marks
        +--> compare_head (Linear d_model -> 3): outputs >/</= logits

The agent runs a single continuous forward pass through the full episode
sequence. During the write window it collects write_head outputs into a
scratch pad tensor. During read windows, the scratch pad is fed back
through the signal channel (channel 0 of the input), replacing the
placeholder zeros that the scene generator leaves there.

Gradient flows from compare_head (at compare timesteps) all the way back
through the GRU, through the read-phase signal inputs (which were the
write-phase outputs), and into write_head. This closed loop is what
lets the agent learn what to write.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.scene import SceneConfig


@dataclass
class AgentConfig:
    d_model: int = 16
    input_dim: int = 3          # [signal, is_input_valid, post_gap_flag]
    signal_output_dim: int = 1  # write head outputs a scalar (same-medium)
    compare_classes: int = 3    # >, <, ==
    quantize_levels: Optional[int] = None  # None = no quantization
    quantize_range: str = "symmetric"      # "symmetric" -> [-1, 1] via tanh; "unit" -> [0, 1] via sigmoid
    # Scratch-pad attention at compare timesteps: gives the policy content-based
    # addressing into the W notebook slots in parallel, instead of relying on
    # GRU integration of one-slot-per-timestep readback.
    scratch_attn: bool = False
    scratch_attn_heads: int = 2
    gaussian_attn: bool = False           # True: replace MHA with PFC-style Gaussian receptive-field attn
    compare_mlp_hidden: Optional[int] = None  # if set, compare_head becomes 2-layer MLP
    # PFC-inspired attractor mechanisms — try to push hidden geometry from
    # cumulative-sigmoid thermometers toward Gaussian-peak population codes.
    inner_iter: int = 1                 # >1: hidden does K fixed-point steps per timestep (attractor convergence)
    hidden_noise_std: float = 0.0       # >0: inject Gaussian noise into hidden at training (attractor basin selection)
    lateral_inhibition: bool = False    # True: hidden = h * softmax(alpha * h) — competitive specialization
    lateral_alpha: float = 2.0          # softmax temperature for lateral inhibition (lower=softer)
    cell_type: str = "gru"              # "gru" | "lif" | "none" | "rcnn" | "lstm"
    lif_decay: float = 0.9              # leak rate (smaller = faster decay)
    lif_threshold: float = 1.0          # firing threshold
    rcnn_iter: int = 4                  # RCNN inner-iteration steps per timestep
    # Stage-3 perception+PFC architecture
    read_cnn_kernel: int = 0            # >0: causal 1D conv front-end with this kernel (V1/V2 windowed perception)
    read_cnn_layers: int = 1            # Number of stacked Conv1d layers (1=V1 only, 2=V1+V2, 3=V1+V2+V4).
                                        #   Layer 0: input_dim -> d_model. Layers 1+: d_model -> d_model.
                                        #   GELU between layers. Each has its own causal buffer of size kernel-1.
    pfc_layers: int = 0                 # >0: transformer encoder on [h, scratch_pad] at compare time (PFC executive)
    pfc_heads: int = 2                  # transformer heads
    pfc_iter: int = 1                   # >1: apply the pfc encoder K times in a loop (Universal-Transformer style
                                        # multi-round reasoning). Same module reused at write AND compare so both
                                        # benefit. K=1 is current behavior. Increasing K trades compute for depth.
    reset_h_at_compare_block: bool = False  # If True, zero h at start of each compare block — WM wipe forcing
                                            # alpha info to flow only through scratch_pad (emergence pressure)
    pfc_split_ab: bool = False              # v3: snapshot h at beta_start (h_alpha) and reset; another snapshot at
                                            # compare_step (h_beta). PFC takes [h_alpha, h_beta, scratch_slots] at
                                            # compare time. Requires reset_h_at_compare_block + pfc_layers>0.
    pfc_at_write: bool = False              # v3-w: PFC also fires at every write timestep with [h, prior_scratch].
                                            # Shared weights with compare-time PFC (single self.pfc module).
    pfc_compare_drop_raw_scratch: bool = False  # If True (and pfc_split_ab), PFC at compare reads only
                                                 # [h_alpha, h_beta] — drops the direct path-2 access to raw
                                                 # scratch_pad slots. Forces alpha info to flow only via the
                                                 # CNN-perceived path (scratch -> CNN -> GRU -> h_alpha).
    compare_via_raw_scratch: bool = False  # If True, at compare time skip [h_alpha, h_beta] and feed
                                            # compare_head(concat(scratch_alpha, scratch_beta)) where
                                            # scratch_beta is generated by W internal GRU+write_head steps
                                            # on h_beta with zero inputs. Symmetrizes alpha/beta encoding —
                                            # both pass through the same quantize-STE bottleneck before compare.
    write_noise_std: float = 0.0  # If >0, add Gaussian noise to write_head logit before quantize-STE
                                   # during training only. Mechanism-level sharpening — forces write
                                   # outputs far from quantize boundaries, reducing off-by-one ambiguity.
    use_gap_head: bool = False    # If True, replace 3-class compare_head with a scalar gap_head
                                   # (Linear -> 1). Used by gap_mse4 supervised loss and rl_top1
                                   # categorical-selection loss; both treat compare output as a
                                   # per-block scalar (gap prediction or preference score).


class _SurrogateThreshold(torch.autograd.Function):
    """Hard step in forward, sigmoid surrogate gradient in backward.
    Used by LIFCell so that backprop has nonzero gradient through the
    discontinuous spike."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(ctx.alpha * x)
        return grad_output * ctx.alpha * sig * (1 - sig), None


class LIFCell(nn.Module):
    """Leaky integrate-and-fire cell with surrogate gradient.

    Drop-in for GRUCell (same forward signature `(x, h) -> h_new`), where h
    is interpreted as the post-reset membrane potential v_t. Each step:
        v_pre = decay * v_prev + W_in(x) + W_rec(v_prev)
        spike = (v_pre > threshold)   # binary in fwd, sigmoid surrogate in bwd
        v_new = v_pre * (1 - spike)   # reset to 0 on spike

    The output h_new = v_new, so the rest of the agent reads continuous
    membrane potential. The native sparsity comes from frequent resets:
    most dims at most timesteps will be at low v after a recent spike,
    while a few high-firing dims dominate. This is the population-code
    geometry PFC numerosity neurons exhibit.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 decay: float = 0.9, threshold: float = 1.0,
                 surrogate_alpha: float = 4.0):
        super().__init__()
        self.W_in = nn.Linear(input_size, hidden_size)
        # Per-dim learnable decay rate (Lambda-style multi-rate prior, like
        # RG-LRU's per-dim Lambda). Without this, all dims share decay=0.9
        # and 5 write timesteps see near-identical hidden -> 5-pos full
        # redundancy collapse (verified in CL1 v0: per_pos_R2 = [0.11]*5).
        # Parameterize as logit so each dim's decay stays in (0, 1).
        init_logit = float(torch.logit(torch.tensor(decay)).item())
        self.decay_logit = nn.Parameter(torch.full((hidden_size,), init_logit))
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Pure leaky integrate, no W_rec (W_rec causes blowup unless tiny init).
        # Per-dim decay lets each hidden dim specialize to a different time-
        # constant — the prior that lets PFC numerosity neurons tune to
        # different preferred-N ranges via differential temporal accumulation.
        decay = torch.sigmoid(self.decay_logit)                          # (hidden,)
        v_pre = decay * h + self.W_in(x)
        spike = _SurrogateThreshold.apply(v_pre - self.threshold, self.surrogate_alpha)
        v_new = v_pre * (1 - spike)
        return v_new


def _quantize_ste_with_int(
    x: torch.Tensor, levels: int, output_range: str = "symmetric"
):
    """V28: like _quantize_ste but also returns integer level indices for
    one-hot expansion (needed by V28-Sin predict_head MLP+multi-hot path).

    output_range options:
    - "symmetric": tanh(x), output in [-1, 1] (NOT same medium as signal [0,1])
    - "unit":      sigmoid(x), output in [0, 1] (same medium ✓)
    - "linear_unit": x clamped to [0, 1] directly, NO nonlinearity
                     (for V28 modules whose raw output is already bounded
                      [-1, +1] and pre-mapped via (raw+1)/2 to [0, 1] —
                      avoids re-applying sigmoid which compresses to (0.27, 0.73))
    """
    if output_range == "symmetric":
        soft = torch.tanh(x)
        lo, hi = -1.0, 1.0
    elif output_range == "unit":
        soft = torch.sigmoid(x)
        lo, hi = 0.0, 1.0
    elif output_range == "linear_unit":
        soft = x.clamp(0.0, 1.0)
        lo, hi = 0.0, 1.0
    else:
        raise ValueError(f"Unknown output_range: {output_range!r}")
    if levels <= 1:
        return soft, torch.zeros_like(soft, dtype=torch.long)
    span = hi - lo
    scaled = (soft - lo) / span * (levels - 1)
    rounded = torch.round(scaled)
    hard = lo + rounded / (levels - 1) * span
    out = soft + (hard - soft).detach()  # STE
    int_levels = rounded.long().clamp(0, levels - 1)
    return out, int_levels


def _quantize_ste(
    x: torch.Tensor, levels: int, output_range: str = "symmetric"
) -> torch.Tensor:
    """
    Straight-through-estimator quantization: forward uses hard levels,
    backward acts as identity on the squashed value.

    output_range="symmetric": values in [-1, 1] via tanh.
        e.g. levels=3 -> {-1, 0, +1};  levels=9 -> {-1, -0.75, ..., +0.75, +1}.
    output_range="unit": values in [0, 1] via sigmoid.
        e.g. levels=6 -> {0, 0.2, 0.4, 0.6, 0.8, 1.0}.
    """
    if output_range == "symmetric":
        soft = torch.tanh(x)
        lo, hi = -1.0, 1.0
    elif output_range == "unit":
        soft = torch.sigmoid(x)
        lo, hi = 0.0, 1.0
    else:
        raise ValueError(f"Unknown output_range: {output_range!r}")
    if levels <= 1:
        return soft
    span = hi - lo
    scaled = (soft - lo) / span * (levels - 1)
    rounded = torch.round(scaled)
    hard = lo + rounded / (levels - 1) * span
    return soft + (hard - soft).detach()


class RCNNCell(nn.Module):
    """Recurrent CNN-style cell (Liang & Hu 2015, untied for time axis).

    For 1D vector hidden state (no spatial dim), this is a vanilla RNN with
    K-step inner iteration and shared self-feedback. Each timestep:
        u   = W_x(x_t)
        h_0 = h_prev
        h_k = tanh(W_h(h_{k-1}) + u)   for k = 1..K
        h_t = h_K

    Shared W_h across iterations is the "recurrent" property — depth without
    extra parameters. Compared to GRUCell: no gates, but K-step deeper update
    per timestep.
    """
    def __init__(self, input_size: int, hidden_size: int, n_iter: int = 4):
        super().__init__()
        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.n_iter = n_iter

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        u = self.W_x(x)
        for _ in range(self.n_iter):
            h = torch.tanh(self.W_h(h) + u)
        return h


class GaussianAttn(nn.Module):
    """PFC-style attention: each head is a Gaussian receptive field over the
    1-D scratch-pad value axis (each scratch_pad slot stores a scalar in [0,1]
    when q=3 unit; can be interpreted as a "preferred N" estimate).

    Each head h has a learnable preferred-N center mu_h (read from hidden via
    a Linear projection -> sigmoid into [0,1]) and a learnable width sigma_h.
    Attention weight over slot i is `exp(-(scratch_value[i] - mu_h)^2 / sigma_h^2)`,
    not the usual content-based dot product. Retrieval is the weighted average
    of value-projected scratch values, then output-projected.

    This embeds population-code geometry as inductive bias *without* fixing the
    centers (mu_h, sigma_h are free-learned), so emergence is preserved.
    """
    def __init__(self, d_model: int, n_heads: int, signal_dim: int = 1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, n_heads)              # hidden -> H preferred-N centers
        self.log_widths = nn.Parameter(torch.zeros(n_heads) - 1.5)  # per-head sigma in log-space
        self.W_v = nn.Linear(signal_dim, d_model)           # scratch -> per-head value embedding
        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, scratch_values: torch.Tensor) -> torch.Tensor:
        # h: (B, d), scratch_values: (B, W, signal_dim=1)
        B, W, _ = scratch_values.shape
        H = self.n_heads
        # Each head's preferred-N query in [0, 1]
        q = torch.sigmoid(self.W_q(h))                     # (B, H)
        sigma = self.log_widths.exp().clamp(min=0.05, max=2.0).view(1, H, 1)  # (1, H, 1)
        # Gaussian distance weights
        diff = scratch_values.squeeze(-1).unsqueeze(1) - q.unsqueeze(2)  # (B, H, W)
        weights = torch.exp(-(diff / sigma) ** 2)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)       # normalize over W
        # Per-head value embeddings
        v = self.W_v(scratch_values).view(B, W, H, self.d_head).permute(0, 2, 1, 3)  # (B, H, W, d_head)
        # Retrieve: weighted sum over W, per head
        retrieved = (weights.unsqueeze(-1) * v).sum(2).reshape(B, self.d_model)       # (B, d)
        return self.W_out(retrieved)


class GRUAgent(nn.Module):
    def __init__(self, agent_cfg: AgentConfig, scene_cfg: SceneConfig):
        super().__init__()
        self.agent_cfg = agent_cfg
        self.scene_cfg = scene_cfg

        # Perceptual front-end (Stage 3): causal 1D conv with kernel K, used as
        # the unified "eye" that processes both world signal and scratch reads.
        # When kernel=0 (default), fallback to plain Linear projection (legacy).
        if agent_cfg.read_cnn_kernel > 0:
            self.read_cnn = nn.Conv1d(
                agent_cfg.input_dim, agent_cfg.d_model,
                kernel_size=agent_cfg.read_cnn_kernel, bias=True,
            )
            # Optional extra layers (V1+V2[+V4] hierarchy). Each consumes (B, d_model, K)
            # buffers. Empty ModuleList for single-layer baseline; checkpoints from
            # 1-layer runs still load cleanly because read_cnn keeps its name.
            n_extra = max(0, agent_cfg.read_cnn_layers - 1)
            self.read_cnn_extra = nn.ModuleList([
                nn.Conv1d(
                    agent_cfg.d_model, agent_cfg.d_model,
                    kernel_size=agent_cfg.read_cnn_kernel, bias=True,
                )
                for _ in range(n_extra)
            ])
            self.input_proj = None
        else:
            self.read_cnn = None
            self.read_cnn_extra = nn.ModuleList()
            self.input_proj = nn.Linear(agent_cfg.input_dim, agent_cfg.d_model)
        if agent_cfg.cell_type == "lif":
            self.processor = LIFCell(
                agent_cfg.d_model, agent_cfg.d_model,
                decay=agent_cfg.lif_decay,
                threshold=agent_cfg.lif_threshold,
            )
        elif agent_cfg.cell_type == "rcnn":
            self.processor = RCNNCell(
                agent_cfg.d_model, agent_cfg.d_model,
                n_iter=agent_cfg.rcnn_iter,
            )
        elif agent_cfg.cell_type == "lstm":
            # Standard LSTMCell. The (h, c) tuple is managed in the forward
            # loop; only h is exposed downstream (write_head/scratch_attn read h).
            self.processor = nn.LSTMCell(agent_cfg.d_model, agent_cfg.d_model)
        elif agent_cfg.cell_type == "none":
            # Phase D control: no recurrent integration; hidden = current u_t
            self.processor = None
        else:
            self.processor = nn.GRUCell(agent_cfg.d_model, agent_cfg.d_model)
        self.write_head = nn.Linear(agent_cfg.d_model, agent_cfg.signal_output_dim)
        if agent_cfg.scratch_attn:
            if agent_cfg.gaussian_attn:
                # PFC-style Gaussian receptive-field attn over 1-D scratch axis
                self.scratch_gauss = GaussianAttn(
                    agent_cfg.d_model,
                    agent_cfg.scratch_attn_heads,
                    signal_dim=agent_cfg.signal_output_dim,
                )
            else:
                # Default: standard MHA with learned scratch_proj + pos_emb
                self.scratch_proj = nn.Linear(agent_cfg.signal_output_dim, agent_cfg.d_model)
                self.scratch_pos_emb = nn.Parameter(
                    torch.randn(scene_cfg.W, agent_cfg.d_model) * 0.02
                )
                self.scratch_mha = nn.MultiheadAttention(
                    agent_cfg.d_model, agent_cfg.scratch_attn_heads, batch_first=True
                )
            compare_in_dim = 2 * agent_cfg.d_model
        else:
            compare_in_dim = agent_cfg.d_model
        # No lateral_norm — magnitude-preserving lateral_v2 doesn't need it.
        # PFC transformer (Stage 3): operates at compare-time on the sequence
        # [h, scratch_pad slots with pos_emb]. Uses query-position output as
        # h_pfc into compare_head. When pfc_layers=0, this is bypassed.
        if agent_cfg.pfc_layers > 0:
            # Build minimal scratch projection if not already created by scratch_attn
            if not agent_cfg.scratch_attn:
                self.scratch_proj = nn.Linear(agent_cfg.signal_output_dim, agent_cfg.d_model)
                self.scratch_pos_emb = nn.Parameter(
                    torch.randn(scene_cfg.W, agent_cfg.d_model) * 0.02
                )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=agent_cfg.d_model,
                nhead=agent_cfg.pfc_heads,
                dim_feedforward=agent_cfg.d_model * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.pfc = nn.TransformerEncoder(enc_layer, num_layers=agent_cfg.pfc_layers)
            # In v3 (pfc_split_ab): compare_head reads concat[h_alpha_pfc, h_beta_pfc] = 2d.
            # Otherwise (PhaseB style): compare_head reads h_pfc (=position 0 of refined seq) = d.
            compare_in_dim = (2 * agent_cfg.d_model) if agent_cfg.pfc_split_ab else agent_cfg.d_model
        else:
            self.pfc = None
        if agent_cfg.compare_mlp_hidden is not None:
            self.compare_head = nn.Sequential(
                nn.Linear(compare_in_dim, agent_cfg.compare_mlp_hidden),
                nn.ReLU(),
                nn.Linear(agent_cfg.compare_mlp_hidden, agent_cfg.compare_classes),
            )
        else:
            self.compare_head = nn.Linear(compare_in_dim, agent_cfg.compare_classes)

        # Optional gap_head: scalar per compare block. Used by gap_mse4 (predicts
        # signed gap N_α-N_β) and rl_top1 (preference score for categorical sample
        # over K options). Lives alongside compare_head; forward picks one based on
        # use_gap_head. Same input dim, so identical PFC/scratch-attn upstream paths.
        if agent_cfg.use_gap_head:
            if agent_cfg.compare_mlp_hidden is not None:
                self.gap_head = nn.Sequential(
                    nn.Linear(compare_in_dim, agent_cfg.compare_mlp_hidden),
                    nn.ReLU(),
                    nn.Linear(agent_cfg.compare_mlp_hidden, 1),
                )
            else:
                self.gap_head = nn.Linear(compare_in_dim, 1)
        else:
            self.gap_head = None

        # E_F: separate compare head that reads concat(scratch_alpha, scratch_beta) = 2*W trits
        if agent_cfg.compare_via_raw_scratch:
            raw_in_dim = 2 * scene_cfg.W * agent_cfg.signal_output_dim  # = 10 for W=5, sig=1
            if agent_cfg.compare_mlp_hidden is not None:
                self.compare_head_raw = nn.Sequential(
                    nn.Linear(raw_in_dim, agent_cfg.compare_mlp_hidden),
                    nn.ReLU(),
                    nn.Linear(agent_cfg.compare_mlp_hidden, agent_cfg.compare_classes),
                )
            else:
                self.compare_head_raw = nn.Linear(raw_in_dim, agent_cfg.compare_classes)
        else:
            self.compare_head_raw = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _emit_compare(self, x: torch.Tensor) -> torch.Tensor:
        """Pick gap_head (scalar per block) over compare_head (3-class) when
        use_gap_head is set. Centralizes the branch so every compare-time
        path (PFC-split, PhaseB, scratch_attn, vanilla) honours the flag."""
        if self.gap_head is not None:
            return self.gap_head(x)
        return self.compare_head(x)

    # ---- Forward-pass helpers ----------------------------------------------

    def _write_timesteps(self) -> List[int]:
        """Indices (absolute t) where write_head output goes to scratch pad."""
        cfg = self.scene_cfg
        return list(range(cfg.L, cfg.write_end))

    def _read_timesteps(self) -> List[Tuple[int, int]]:
        """For each comparison block, list of (absolute_t, scratch_idx) pairs."""
        cfg = self.scene_cfg
        blocks = []
        for i in range(cfg.K):
            read_start = cfg.compare_block_start(i)
            for j in range(cfg.W):
                blocks.append((read_start + j, j))
        return blocks

    def _compare_timesteps(self, compare_indices: torch.Tensor) -> List[int]:
        """compare_indices has shape (K,) or (B, K); we assume shared across batch."""
        if compare_indices.dim() == 2:
            return compare_indices[0].tolist()
        return compare_indices.tolist()

    # ---- Main forward pass --------------------------------------------------

    def forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        initial_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a full episode forward pass.

        Args:
            inputs: (B, T, 3) — phase markers set by scene generator;
                scratch pad slots have signal=0 (placeholder to be filled).
            compare_indices: (B, K) or (K,) — absolute timestep indices where
                compare_head should emit logits. Assumed shared across batch.
            initial_hidden: (B, d_model) — optional, defaults to zeros.

        Returns:
            compare_logits: (B, K, 3)
            scratch_pad: (B, W) — the floats agent wrote during write phase
            hidden_at_compares: (B, K, d_model) — useful for analysis
        """
        B, T, _ = inputs.shape
        d = self.agent_cfg.d_model
        W = self.scene_cfg.W
        cfg = self.scene_cfg

        if initial_hidden is None:
            h = inputs.new_zeros(B, d)
        else:
            h = initial_hidden
        # LSTM cell additionally needs cell-state c. Tracked separately.
        lstm_c = inputs.new_zeros(B, d) if self.agent_cfg.cell_type == "lstm" else None

        write_set = set(self._write_timesteps())
        # For reads: map absolute_t -> scratch_idx
        read_map = dict(self._read_timesteps())
        compare_set = set(self._compare_timesteps(compare_indices))

        # Pre-allocate scratch pad and output containers. We fill them
        # in the loop; autograd will track the dependencies.
        scratch_pad_list: List[Optional[torch.Tensor]] = [None] * W
        compare_logits_list: List[torch.Tensor] = []
        hidden_at_compares_list: List[torch.Tensor] = []

        # Causal CNN buffers, one per layer (for the V1/V2[+V4] front-end).
        # Layer 0 buffer: (B, input_dim, K-1) — past raw inputs.
        # Layer i>0 buffer: (B, d_model, K-1) — past activations from layer i-1.
        K_cnn = self.agent_cfg.read_cnn_kernel
        if K_cnn > 0:
            n_cnn_layers = 1 + len(self.read_cnn_extra)
            cnn_bufs: List[Optional[torch.Tensor]] = [
                inputs.new_zeros(
                    B,
                    self.agent_cfg.input_dim if li == 0 else d,
                    K_cnn - 1,
                )
                for li in range(n_cnn_layers)
            ]
        else:
            n_cnn_layers = 0
            cnn_bufs = []

        # WM-reset boundaries:
        # v2 reset (reset_h_at_compare_block): zero h at every read_start.
        # v3 second reset (pfc_split_ab): also zero h at every beta_start, AND
        #   snapshot h_alpha at beta_start before the reset (h_alpha = h after
        #   reading notes). h_beta = h at compare_step.
        if self.agent_cfg.reset_h_at_compare_block:
            wm_reset_set = {cfg.compare_block_start(i) for i in range(cfg.K)}
        else:
            wm_reset_set = set()
        if self.agent_cfg.pfc_split_ab:
            beta_start_to_idx = {cfg.compare_block_start(i) + cfg.W: i for i in range(cfg.K)}
            compare_step_to_idx = {cfg.compare_block_start(i) + cfg.W + cfg.L: i for i in range(cfg.K)}
        else:
            beta_start_to_idx = {}
            compare_step_to_idx = {}
        h_alpha_per_block: List[Optional[torch.Tensor]] = [None] * cfg.K

        for t in range(T):
            # v3: snapshot h_alpha at beta_start, then reset h second time
            if t in beta_start_to_idx:
                bidx = beta_start_to_idx[t]
                h_alpha_per_block[bidx] = h
                h = inputs.new_zeros(B, d)
                if lstm_c is not None:
                    lstm_c = inputs.new_zeros(B, d)
                if cnn_bufs:
                    cnn_bufs = [
                        inputs.new_zeros(
                            B,
                            self.agent_cfg.input_dim if li == 0 else d,
                            K_cnn - 1,
                        )
                        for li in range(n_cnn_layers)
                    ]
            if t in wm_reset_set:
                h = inputs.new_zeros(B, d)
                if lstm_c is not None:
                    lstm_c = inputs.new_zeros(B, d)
                if cnn_bufs:
                    # Also clear perceptual buffers — fresh "look" at compare phase
                    cnn_bufs = [
                        inputs.new_zeros(
                            B,
                            self.agent_cfg.input_dim if li == 0 else d,
                            K_cnn - 1,
                        )
                        for li in range(n_cnn_layers)
                    ]
            raw_t = inputs[:, t, :]  # (B, 3) view; don't modify in place

            # If this is a read-notes timestep, build x_t by concatenating
            # the scratch-pad float (requires_grad=True via write_head) with
            # the phase-marker channels (non-grad). Using torch.cat keeps
            # the computation graph clean (in-place assignment on a clone
            # can silently drop the grad dependency).
            if t in read_map:
                scratch_idx = read_map[t]
                stored = scratch_pad_list[scratch_idx]
                if stored is not None:
                    phase_channels = raw_t[:, 1:]  # (B, 2)
                    x_t = torch.cat([stored, phase_channels], dim=-1)  # (B, 3)
                else:
                    x_t = raw_t
            else:
                x_t = raw_t

            # Perceptual front-end: causal 1D conv (V1/V2 windowed scanner)
            # processes both world signal and scratch reads through the same
            # module — recycling-style unified perception. Falls back to plain
            # Linear projection when read_cnn_kernel == 0.
            if self.read_cnn is not None:
                # Stacked causal CNN: layer 0 reads raw input, deeper layers read
                # the previous layer's activations. Each layer maintains its own
                # K-1 past-frames buffer so total receptive field grows linearly
                # with depth (V1 sees ~K cells, V2 sees ~2K-1, V4 sees ~3K-2).
                feat = x_t  # (B, input_dim) at layer 0
                new_bufs: List[torch.Tensor] = []
                cnn_in_0 = torch.cat([cnn_bufs[0], feat.unsqueeze(-1)], dim=-1)  # (B, input_dim, K)
                feat = self.read_cnn(cnn_in_0).squeeze(-1)                       # (B, d_model)
                new_bufs.append(cnn_in_0[..., 1:])
                for li, conv in enumerate(self.read_cnn_extra):
                    feat = F.gelu(feat)
                    cnn_in_l = torch.cat([cnn_bufs[li + 1], feat.unsqueeze(-1)], dim=-1)  # (B, d_model, K)
                    feat = conv(cnn_in_l).squeeze(-1)                                     # (B, d_model)
                    new_bufs.append(cnn_in_l[..., 1:])
                cnn_bufs = new_bufs
                u_t = feat
            else:
                u_t = self.input_proj(x_t)  # (B, d_model)
            if self.processor is None:
                # Phase D control: no recurrent state, h := u_t (CNN output).
                h = u_t
            elif self.agent_cfg.cell_type == "lstm":
                # LSTM: returns (h, c) tuple; we keep c internal, expose h.
                h, lstm_c = self.processor(u_t, (h, lstm_c))
            elif self.agent_cfg.cell_type in ("rcnn", "lif"):
                # RCNN/LIF have their own internal iteration; ignore inner_iter.
                h = self.processor(u_t, h)
            else:
                # GRU: leak-integrator inner iteration (K=1 = standard GRU).
                K = max(self.agent_cfg.inner_iter, 1)
                dt = 1.0 / K
                for _ in range(K):
                    h_cand = self.processor(u_t, h)
                    h = (1.0 - dt) * h + dt * h_cand
            # Hidden noise: train-time Gaussian injection picks attractor basins
            # over saddles. Disabled at eval so inspection sees deterministic codes.
            if self.training and self.agent_cfg.hidden_noise_std > 0.0:
                h = h + self.agent_cfg.hidden_noise_std * torch.randn_like(h)
            # Lateral inhibition v2: magnitude-preserving soft winner-take-all.
            # Old version used LayerNorm which destroys the magnitude trace that
            # the GRU uses to accumulate N across timesteps — every cell that
            # included it collapsed (R^2 ~ -0.05 across all stages).
            # New approach: redistribute mass between dims (sharper peak on the
            # winner) but rescale to preserve the ORIGINAL hidden norm. This
            # changes shape, not amplitude — preserving the cumulative trace.
            if self.agent_cfg.lateral_inhibition:
                weights = torch.softmax(self.agent_cfg.lateral_alpha * h.abs(), dim=-1) * h.shape[-1]
                h_competed = h * weights
                orig_norm = h.norm(dim=-1, keepdim=True) + 1e-6
                comp_norm = h_competed.norm(dim=-1, keepdim=True) + 1e-6
                h = h_competed * orig_norm / comp_norm

            # Write head fires only during write window.
            if t in write_set:
                scratch_idx = t - cfg.L
                if self.agent_cfg.pfc_at_write and self.pfc is not None:
                    # v3-w: run PFC over [h, prior_scratch_pad]; write_head reads
                    # the PFC-symbolized representation. PFC weights shared with
                    # compare-time PFC. Unwritten slots are zeros (placeholder).
                    Bw = h.shape[0]
                    sp_slots_w: List[torch.Tensor] = []
                    for s in scratch_pad_list:
                        if s is None:
                            sp_slots_w.append(h.new_zeros(Bw, self.agent_cfg.signal_output_dim))
                        else:
                            sp_slots_w.append(s)
                    sp_w = torch.stack(sp_slots_w, dim=1)
                    kv_w = self.scratch_proj(sp_w) + self.scratch_pos_emb.unsqueeze(0)
                    seq_w = torch.cat([h.unsqueeze(1), kv_w], dim=1)  # (B, 1+W, d)
                    seq_w_refined = seq_w
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_w_refined = self.pfc(seq_w_refined)
                    h_for_write = seq_w_refined[:, 0, :]
                else:
                    h_for_write = h
                raw = self.write_head(h_for_write)  # (B, 1)
                if self.training and self.agent_cfg.write_noise_std > 0.0:
                    raw = raw + torch.randn_like(raw) * self.agent_cfg.write_noise_std
                if self.agent_cfg.quantize_levels is not None:
                    raw = _quantize_ste(
                        raw,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
                scratch_pad_list[scratch_idx] = raw

            # Compare head fires at compare timesteps.
            if t in compare_set:
                # Build scratch tensor (used by PFC and/or scratch_attn).
                B = h.shape[0]
                sp_slots: List[torch.Tensor] = []
                for s in scratch_pad_list:
                    if s is None:
                        sp_slots.append(h.new_zeros(B, self.agent_cfg.signal_output_dim))
                    else:
                        sp_slots.append(s)
                sp = torch.stack(sp_slots, dim=1)  # (B, W, signal_dim)

                if self.agent_cfg.compare_via_raw_scratch:
                    # E_F: symmetrize alpha/beta encoding. Both pass through write_head + quantize.
                    # alpha already in scratch_pad. For beta, run W internal GRU+write_head steps
                    # on h with zero perceptual input (mirrors the alpha write phase shape).
                    Be = h.shape[0]
                    zero_x = h.new_zeros(Be, self.agent_cfg.input_dim)
                    if self.read_cnn is not None:
                        K_cnn = self.agent_cfg.read_cnn_kernel
                        cnn_in_z = h.new_zeros(Be, self.agent_cfg.input_dim, K_cnn)
                        u_z = self.read_cnn(cnn_in_z).squeeze(-1)
                    elif self.input_proj is not None:
                        u_z = self.input_proj(zero_x)
                    else:
                        u_z = zero_x  # cell_type=none would already use raw u
                    h_internal = h
                    scratch_beta_list: List[torch.Tensor] = []
                    for _ in range(cfg.W):
                        if self.processor is not None:
                            if self.agent_cfg.cell_type == "lstm":
                                # use last lstm_c if available, else zeros
                                c_local = lstm_c if lstm_c is not None else h.new_zeros(h.shape)
                                h_internal, _ = self.processor(u_z, (h_internal, c_local))
                            else:
                                h_internal = self.processor(u_z, h_internal)
                        raw_b = self.write_head(h_internal)
                        if self.training and self.agent_cfg.write_noise_std > 0.0:
                            raw_b = raw_b + torch.randn_like(raw_b) * self.agent_cfg.write_noise_std
                        if self.agent_cfg.quantize_levels is not None:
                            raw_b = _quantize_ste(
                                raw_b,
                                self.agent_cfg.quantize_levels,
                                self.agent_cfg.quantize_range,
                            )
                        scratch_beta_list.append(raw_b)
                    scratch_beta = torch.stack(scratch_beta_list, dim=1)  # (B, W, sig)
                    compare_in = torch.cat(
                        [sp.flatten(1), scratch_beta.flatten(1)], dim=-1
                    )  # (B, 2*W*sig)
                    compare_logits_list.append(self.compare_head_raw(compare_in))
                elif self.pfc is not None and self.agent_cfg.pfc_split_ab:
                    # v3 / v3-w: PFC reads [h_alpha, h_beta, scratch_slots].
                    # h_alpha was snapshotted at beta_start (after notes-readback)
                    # h_beta is the current h (after beta scan, post second reset)
                    bidx = compare_step_to_idx[t]
                    h_alpha = h_alpha_per_block[bidx]
                    if h_alpha is None:  # safety; should never happen if config consistent
                        h_alpha = h.new_zeros(h.shape)
                    if self.agent_cfg.pfc_compare_drop_raw_scratch:
                        # Drop path-2: PFC sees only [h_alpha, h_beta]. Forces alpha
                        # info to flow strictly via CNN -> GRU -> h_alpha during the
                        # notes-readback phase. Keeps PFC's self-attention sequence
                        # length at 2 (still nontrivial when iterated K times).
                        seq = torch.cat([h_alpha.unsqueeze(1), h.unsqueeze(1)], dim=1)  # (B, 2, d)
                    else:
                        kv_seq = self.scratch_proj(sp) + self.scratch_pos_emb.unsqueeze(0)
                        seq = torch.cat([h_alpha.unsqueeze(1), h.unsqueeze(1), kv_seq], dim=1)  # (B, 2+W, d)
                    seq_refined = seq
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_refined = self.pfc(seq_refined)
                    h_alpha_pfc = seq_refined[:, 0, :]
                    h_beta_pfc = seq_refined[:, 1, :]
                    compare_in = torch.cat([h_alpha_pfc, h_beta_pfc], dim=-1)
                    compare_logits_list.append(self._emit_compare(compare_in))
                elif self.pfc is not None:
                    # PhaseB: PFC executive on [h, scratch_pad].
                    kv_seq = self.scratch_proj(sp) + self.scratch_pos_emb.unsqueeze(0)
                    seq = torch.cat([h.unsqueeze(1), kv_seq], dim=1)
                    seq_refined = seq
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_refined = self.pfc(seq_refined)
                    h_pfc = seq_refined[:, 0, :]
                    compare_logits_list.append(self._emit_compare(h_pfc))
                elif self.agent_cfg.scratch_attn:
                    if self.agent_cfg.gaussian_attn:
                        retrieved = self.scratch_gauss(h, sp)
                    else:
                        kv = self.scratch_proj(sp) + self.scratch_pos_emb.unsqueeze(0)
                        retrieved_mha, _ = self.scratch_mha(
                            h.unsqueeze(1), kv, kv, need_weights=False
                        )
                        retrieved = retrieved_mha.squeeze(1)
                    compare_in = torch.cat([h, retrieved], dim=-1)
                    compare_logits_list.append(self._emit_compare(compare_in))
                else:
                    compare_logits_list.append(self._emit_compare(h))
                hidden_at_compares_list.append(h)

        # Stack outputs.
        compare_logits = torch.stack(compare_logits_list, dim=1)  # (B, K, 3)
        hidden_at_compares = torch.stack(hidden_at_compares_list, dim=1)  # (B, K, d)
        scratch_pad = torch.stack(scratch_pad_list, dim=1).squeeze(-1)  # (B, W)

        return compare_logits, scratch_pad, hidden_at_compares
