"""V29: R&F SNN encoder + event-driven attention PFC + R&F SNN write head.

Keeps V27 scratch interface (5 cells x q=3 levels) but replaces all internal
substrate with spike-based dynamics. R&F neurons at encoder AND write head
provide complex-eigenvalue oscillation. Event attention PFC operates on
emerged spike events as discrete tokens.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RFSpikeFunction(torch.autograd.Function):
    """Hard threshold forward, fast-sigmoid surrogate backward.

    Surrogate: dspike/dv ~ alpha / (1 + alpha*|v - theta|)^2  (Zenke 2018 fast sigmoid).
    """
    @staticmethod
    def forward(ctx, v, threshold, alpha):
        ctx.save_for_backward(v - threshold)
        ctx.alpha = alpha
        return (v > threshold).to(v.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors
        grad = ctx.alpha / (1.0 + ctx.alpha * diff.abs()).pow(2)
        return grad_output * grad, None, None


def rf_spike(v: torch.Tensor, threshold: float, alpha: float = 4.0) -> torch.Tensor:
    return _RFSpikeFunction.apply(v, threshold, alpha)


class RFEncoder(nn.Module):
    """16 R&F neurons receiving signal channel as driving current.

    Per-neuron learnable (b_i, omega_i, w_i, bias_i):
      I_i(t) = w_i * signal(t) + bias_i
      z_i(t+1) = exp(b_i + i*omega_i) * z_i(t) + I_i(t)  (z complex)
      spike_i(t) = (Re(z_i(t)) > threshold)
      Re(z_i(t)) -= spike_i(t) * threshold     (soft reset)

    Returns spike tensor (B, L, n_neurons) in {0, 1}.
    """
    def __init__(
        self,
        n_neurons: int = 16,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
        omega_init_min: float = 0.5,
        omega_init_max: float = 2.0,
        b_init_min: float = -0.3,
        b_init_max: float = -0.05,
        soft_reset: bool = True,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha
        self.soft_reset = soft_reset
        # Learnable per-neuron eigenvalue (b, omega)
        omega_init = torch.linspace(omega_init_min, omega_init_max, n_neurons)
        b_init = torch.empty(n_neurons).uniform_(b_init_min, b_init_max)
        self.omega = nn.Parameter(omega_init)
        self.b = nn.Parameter(b_init)
        # Learnable per-neuron input projection (signal scalar -> driving current)
        # Heterogeneous w_in init: scale by ω so high-ω neurons get larger gain
        # (compensates for shorter sub-threshold integration time at higher frequencies).
        # Scale = (omega / omega_mean), so neurons with above-mean ω get up-scaled w_in.
        omega_scale = omega_init / omega_init.mean()  # (N,) ∈ ~[0.4, 1.6]
        self.w_in = nn.Parameter(torch.randn(n_neurons) * 0.5 * omega_scale)
        self.bias_in = nn.Parameter(torch.full((n_neurons,), 0.3))  # nonzero bias bumps initial membrane up
        # Stash init for clamp regularization
        self.register_buffer("omega_init", omega_init.clone())
        self.register_buffer("b_init", b_init.clone())

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """signal: (B, L, 1) or (B, L) - first channel of inputs.
        Returns:
          spikes: (B, L, n_neurons) in {0,1}, soft via surrogate gradient
          stash: dict with 'z_re_final', 'z_im_final', 'I_drive' for diagnostics
        """
        if signal.dim() == 3:
            signal = signal[..., 0]  # (B, L)
        B, L = signal.shape
        N = self.n_neurons
        device = signal.device

        # Driving current per neuron: I_i(t) = w_i * signal(t) + bias_i
        # signal: (B, L), w_in/bias_in: (N,)
        I_drive = signal.unsqueeze(-1) * self.w_in + self.bias_in  # (B, L, N)

        # Complex eigenvalue: lambda = exp(b + i*omega) = exp(b)*(cos(omega) + i*sin(omega))
        # b is encouraged < 0 for stable decay
        decay = torch.exp(self.b)  # (N,) - magnitude per step
        cos_w = torch.cos(self.omega)
        sin_w = torch.sin(self.omega)

        z_re = torch.zeros(B, N, device=device, dtype=signal.dtype)
        z_im = torch.zeros(B, N, device=device, dtype=signal.dtype)
        spikes = []
        for t in range(L):
            # z_new = decay*R(omega)*z + I (I treated as +0j)
            new_re = decay * (z_re * cos_w - z_im * sin_w) + I_drive[:, t, :]
            new_im = decay * (z_re * sin_w + z_im * cos_w)
            spike_t = rf_spike(new_re, self.threshold, self.surrogate_alpha)  # (B, N)
            if self.soft_reset:
                # Re part minus theta on spike (soft reset preserves im)
                z_re = new_re - spike_t * self.threshold
                z_im = new_im
            else:
                # Hard reset: zero both Re and Im on spike
                z_re = new_re * (1.0 - spike_t)
                z_im = new_im * (1.0 - spike_t)
            spikes.append(spike_t)
        spikes_tensor = torch.stack(spikes, dim=1)  # (B, L, N)
        stash = {
            "z_re_final": z_re,
            "z_im_final": z_im,
            "I_drive": I_drive,
            "spike_rate": spikes_tensor.sum(dim=1),  # (B, N) total spike count per α episode per neuron
        }
        return spikes_tensor, stash

    def regularize_loss(self) -> torch.Tensor:
        """L2 regularization on |omega - omega_init|. Soft clamp via training pressure.

        Caller should weight this by --v29-omega-clamp-decay (default 1e-4).
        """
        return ((self.omega - self.omega_init) ** 2).mean() + ((self.b - self.b_init) ** 2).mean()


class Time2VecPE(nn.Module):
    """Learnable Time2Vec positional encoding.

    p(t) = [sin(omega_k * t + phi_k)]_{k=1..d_pe}, omega_k log-spaced init, phi_k uniform init.
    """
    def __init__(self, d_pe: int = 16, t_max_init: float = 100.0):
        super().__init__()
        self.d_pe = d_pe
        # Log-spaced initial periods covering 1 to t_max_init
        log_omega_init = torch.linspace(
            math.log(2 * math.pi / t_max_init),  # slowest
            math.log(2 * math.pi / 1.0),         # fastest (1-step period)
            d_pe,
        )
        self.omega = nn.Parameter(log_omega_init.exp())
        self.phi = nn.Parameter(torch.empty(d_pe).uniform_(0, 2 * math.pi))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...) tensor of float timestamps. Returns (..., d_pe)."""
        t_expand = t.unsqueeze(-1)  # (..., 1)
        return torch.sin(self.omega * t_expand + self.phi)


def extract_events_padded(
    spikes: torch.Tensor,
    n_id_embed: nn.Embedding,
    time_pe: Time2VecPE,
    max_events: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert spike tensor to padded event token sequences.

    Args:
      spikes: (B, L, N) in {0, 1}
      n_id_embed: nn.Embedding(N, d_id)
      time_pe: Time2VecPE producing (..., d_pe)
      max_events: int or None (use batch max)

    Returns:
      tokens: (B, max_events, d_id + d_pe) - padded with zeros
      mask: (B, max_events) bool - True where token is valid (not padded)
    """
    B, L, N = spikes.shape
    device = spikes.device
    # Find spike indices per batch
    # spikes_bool: (B, L, N)
    spikes_bool = spikes > 0.5
    # Per-batch event count
    counts = spikes_bool.view(B, -1).sum(dim=1)  # (B,)
    if max_events is None:
        max_events = int(counts.max().item()) if counts.max().item() > 0 else 1
    # Build padded tensors
    d_id = n_id_embed.embedding_dim
    d_pe = time_pe.d_pe
    d = d_id + d_pe
    tokens = torch.zeros(B, max_events, d, device=device, dtype=spikes.dtype)
    mask = torch.zeros(B, max_events, device=device, dtype=torch.bool)
    # Loop per batch (variable-length, naturally per-row)
    for b in range(B):
        # Find (t, neuron_id) where spike occurred
        idxs = spikes_bool[b].nonzero(as_tuple=False)  # (n_events_b, 2) [(t, neuron)]
        n_b = min(idxs.shape[0], max_events)
        if n_b == 0:
            continue
        ts = idxs[:n_b, 0].to(spikes.dtype)  # (n_b,)
        nids = idxs[:n_b, 1]                  # (n_b,)
        id_emb = n_id_embed(nids)             # (n_b, d_id)
        time_emb = time_pe(ts)                # (n_b, d_pe)
        # Multiply spike value to keep grad flow through surrogate
        spike_vals = spikes[b][ts.long(), nids].unsqueeze(-1)  # (n_b, 1)
        # Token has soft surrogate-grad-bearing scaling
        tok = torch.cat([id_emb, time_emb], dim=-1) * spike_vals
        tokens[b, :n_b] = tok
        mask[b, :n_b] = True
    return tokens, mask


class EventAttentionPFC(nn.Module):
    """Pre-norm transformer encoder, L=2 weight-shared, run n_passes times.

    Universal-transformer style: same parameter set applied iteratively.
    Multi-head full attention with key padding mask for variable-length events.
    """
    def __init__(
        self,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        n_passes: int = 2,
        ffn_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_passes = n_passes
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Canonical pre-norm transformer requires final LN post-stack (GPT-2 onwards).
        # Without this, residual accumulation across layers + passes makes output
        # magnitude unbounded → cascade through cross-attn → write_head saturation.
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d). mask: (B, N) bool, True for VALID (not padded).

        TransformerEncoder expects src_key_padding_mask (True=PAD), so we negate.
        Returns (B, N, d).
        """
        # NaN guard: if any batch row has zero valid tokens, force the first
        # token valid. Its embedding is zero (extract_events_padded fills with
        # zeros), so contributes only the bias path, but prevents softmax(-inf)
        # NaN propagation downstream.
        has_any = mask.any(dim=1, keepdim=True)  # (B, 1)
        mask = mask | (~has_any)                  # if all-False, make first True
        pad_mask = ~mask  # (B, N) True = pad
        out = x
        for _ in range(self.n_passes):
            out = self.encoder(out, src_key_padding_mask=pad_mask)
        return out


class CrossAttentionExtractor(nn.Module):
    """V29.4: sigmoid-attention cross-attn (count-sensitive readout).

    Replaces softmax (convex combination → count-blind) with sigmoid (independent
    per-event scores → output magnitude grows with event count). Each of the
    n_queries learnable queries attends to all event tokens; weights are NOT
    normalized, so output is sum-pooled with [0, 1] independent per-event gates.
    """
    def __init__(self, d_model: int = 32, n_queries: int = 5, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_heads = n_heads  # kept in signature for compat; not used internally
        # Learnable queries (used for orthogonality_loss)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.5)
        # Manual K/V projections (per-token) + output projection (post-aggregation)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d), mask: (B, N) bool valid. Returns (B, n_queries, d).

        Sigmoid-attention output is NOT bounded by convexity:
        out_w = Σ_j sigmoid(q_w·K_j/√d) · V_j  for each query w.
        With more valid events, more high-weight contributions accumulate →
        output magnitude scales with event count (count-sensitive).
        """
        B = x.size(0)
        # NaN guard: if a row has zero valid tokens, force first valid (empty token)
        has_any = mask.any(dim=1, keepdim=True)  # (B, 1)
        mask = mask | (~has_any)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_queries, d)
        k = self.k_proj(x)                                # (B, N, d)
        v = self.v_proj(x)                                # (B, N, d)
        # Scores: (B, n_queries, N) — un-normalized
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        # Independent per-event gates ∈ [0, 1], zero out padding positions
        weights = torch.sigmoid(scores) * mask.unsqueeze(1).float()  # (B, n_queries, N)
        # Sum-pool: (B, n_queries, d) — magnitude scales with N
        out = torch.matmul(weights, v)  # (B, n_queries, d)
        out = self.out_proj(out)
        return out

    def orthogonality_loss(self) -> torch.Tensor:
        """Sum_{w != w'} (q_w · q_w' / |q_w||q_w'|)^2."""
        q = self.queries
        q_norm = F.normalize(q, dim=-1)
        cos_sim = q_norm @ q_norm.t()  # (n_queries, n_queries)
        n = q.size(0)
        # Off-diagonal sum of squares
        mask = ~torch.eye(n, dtype=torch.bool, device=q.device)
        return (cos_sim[mask] ** 2).sum()


class _SurrogateSpikeCount(torch.autograd.Function):
    """Forward: hard count of spikes over time window.
    Backward: surrogate count = sum of fast_sigmoid(alpha*(v-theta)) over time.
    """
    @staticmethod
    def forward(ctx, v_seq, threshold, alpha):
        # v_seq: (B, T, N). Returns (B, N) hard spike counts.
        ctx.save_for_backward(v_seq - threshold)
        ctx.alpha = alpha
        spikes = (v_seq > threshold).to(v_seq.dtype)
        return spikes.sum(dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, N). Distribute over T uniformly via surrogate.
        diff, = ctx.saved_tensors  # (B, T, N)
        surrogate = ctx.alpha / (1.0 + ctx.alpha * diff.abs()).pow(2)  # (B, T, N)
        # d_count/d_v_t = surrogate(v_t - theta)
        grad_v = grad_output.unsqueeze(1) * surrogate  # (B, T, N)
        return grad_v, None, None


def surrogate_spike_count(v_seq: torch.Tensor, threshold: float, alpha: float = 4.0) -> torch.Tensor:
    return _SurrogateSpikeCount.apply(v_seq, threshold, alpha)


class RFWriteHead(nn.Module):
    """5 R&F neurons with constant input drive over T_write window.

    Drive: I_w = drive_proj(u_w) in scalar, constant for T_write timesteps.
    Each neuron simulates with own (b_w, omega_w), produces spike train, count is
    quantized to {0, 0.5, 1.0} via STE on bin {0, 1, >=2}.

    T_write fixed at construction = round(2π / median(omega_init)).
    """
    def __init__(
        self,
        d_model: int = 32,
        n_neurons: int = 5,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
        omega_init_min: float = 0.7,
        omega_init_max: float = 1.8,
        b_init_min: float = -0.3,
        b_init_max: float = -0.05,
        soft_reset: bool = True,
        t_write: Optional[int] = None,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.surrogate_alpha = surrogate_alpha
        self.soft_reset = soft_reset
        omega_init = torch.linspace(omega_init_min, omega_init_max, n_neurons)
        b_init = torch.empty(n_neurons).uniform_(b_init_min, b_init_max)
        self.omega = nn.Parameter(omega_init)
        self.b = nn.Parameter(b_init)
        self.register_buffer("omega_init", omega_init.clone())
        self.register_buffer("b_init", b_init.clone())
        # V29.5 (D2a): drive_proj followed by LayerNorm(2) + per-cell scale.
        # LayerNorm bounds (Re, Im) to mean=0, var=1 per-sample → |I| ~ O(1),
        # giving R&F a stable threshold-relative regime. Per-cell learnable
        # scale lets different cells learn different magnitude ranges.
        # NaN-safe: small init for drive_proj weight + LN handles any growth.
        self.drive_proj = nn.Linear(d_model, 2)
        nn.init.normal_(self.drive_proj.weight, std=0.1)  # slightly larger; LN bounds output anyway
        nn.init.zeros_(self.drive_proj.bias)
        # LayerNorm over (Re, Im) — elementwise_affine=False to keep pure
        # mean-zero unit-variance (no learnable γ, β shift).
        self.drive_ln = nn.LayerNorm(2, elementwise_affine=False)
        # Per-cell learnable magnitude scale (init 1.0 → |I| ≈ threshold initially).
        self.drive_scale = nn.Parameter(torch.ones(n_neurons))
        # T_write: 1 cycle of median omega
        if t_write is None:
            median_omega = float(omega_init.median().item())
            t_write = max(2, int(round(2 * math.pi / median_omega)))
        self.t_write = t_write

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u: (B, n_queries=n_neurons, d_model) - output of cross-attention.

        We pair query w to neuron w (n_queries == n_neurons): take diagonal of drive_proj output.

        Returns:
          scratch_q: (B, n_neurons) in {0.0, 0.5, 1.0} (STE quantized)
          scratch_int: (B, n_neurons) in {0, 1, 2} long (for read K/V value embed)
        """
        B, Nq, d = u.shape
        N = self.n_neurons
        assert Nq == N, f"n_queries ({Nq}) must equal n_neurons ({N})"
        # V29.5: drive_proj → LayerNorm(2) → per-cell scale.
        # Step 1: Linear projection (B*N, d) → (B*N, 2)
        I_raw = self.drive_proj(u.reshape(B * N, d))
        # Step 2: LayerNorm over (Re, Im) → mean=0, var=1 per sample
        I_normalized = self.drive_ln(I_raw)  # (B*N, 2)
        I_normalized = I_normalized.reshape(B, N, 2)
        # Step 3: Per-cell scale → magnitude tunable per write neuron
        # drive_scale shape (N,) broadcasts to (B, N) via outer dim
        I_re = I_normalized[..., 0] * self.drive_scale  # (B, N)
        I_im = I_normalized[..., 1] * self.drive_scale

        # Simulate R&F over T_write with constant complex drive (I_re, I_im)
        decay = torch.exp(self.b)  # (N,)
        cos_w = torch.cos(self.omega)
        sin_w = torch.sin(self.omega)
        device = u.device
        z_re = torch.zeros(B, N, device=device, dtype=u.dtype)
        z_im = torch.zeros(B, N, device=device, dtype=u.dtype)
        v_seq = []
        for t in range(self.t_write):
            new_re = decay * (z_re * cos_w - z_im * sin_w) + I_re
            new_im = decay * (z_re * sin_w + z_im * cos_w) + I_im
            v_seq.append(new_re)
            # Reset state for next iteration based on hard threshold (no surrogate path here;
            # surrogate path is in surrogate_spike_count below for the count gradient).
            spike_t = (new_re > self.threshold).to(u.dtype)
            if self.soft_reset:
                z_re = new_re - spike_t * self.threshold
                z_im = new_im
            else:
                z_re = new_re * (1.0 - spike_t)
                z_im = new_im * (1.0 - spike_t)
        v_stack = torch.stack(v_seq, dim=1)  # (B, T_write, N)
        spike_count = surrogate_spike_count(v_stack, self.threshold, self.surrogate_alpha)  # (B, N)

        # V29.5 (D1): max_count = T_write (not T_write·ω/(2π)).
        # With soft reset and |I| in O(threshold) regime, R&F can fire at every
        # timestep, so actual max = T_write timesteps. The old formula
        # T_write·ω/(2π) (1 spike per cycle) was too conservative — it caused
        # 100% clamp saturation when |I|>>threshold (V29.4 issue).
        max_count_per_neuron = float(self.t_write)
        normalized = (spike_count / max_count_per_neuron).clamp(0, 1)  # (B, N)
        # Quantize to 3 levels {0, 0.5, 1.0} via round to nearest of {0, 0.5, 1.0}.
        # scratch_int: 0 -> 0, [1/3, 2/3] -> 1, [2/3, 1] -> 2
        scratch_int = (normalized * 2).round().long().clamp(0, 2)  # (B, N) ∈ {0, 1, 2}
        scratch_hard = scratch_int.float() * 0.5  # {0.0, 0.5, 1.0}
        # STE: forward hard, backward identity on normalized
        scratch_soft = normalized
        scratch_q = scratch_soft + (scratch_hard - scratch_soft).detach()
        return scratch_q, scratch_int

    def regularize_loss(self) -> torch.Tensor:
        return ((self.omega - self.omega_init) ** 2).mean() + ((self.b - self.b_init) ** 2).mean()


class ScratchKVBuilder(nn.Module):
    """Build 5 K/V tokens from scratch_int (cell_id_embed + value_embed)."""
    def __init__(self, n_cells: int = 5, n_levels: int = 3, d_id: int = 16, d_val: int = 16):
        super().__init__()
        self.cell_id_embed = nn.Embedding(n_cells, d_id)
        self.value_embed = nn.Embedding(n_levels, d_val)
        self.n_cells = n_cells

    def forward(self, scratch_int: torch.Tensor) -> torch.Tensor:
        """scratch_int: (B, n_cells) in {0,1,2}. Returns (B, n_cells, d_id+d_val)."""
        B = scratch_int.size(0)
        cell_ids = torch.arange(self.n_cells, device=scratch_int.device).unsqueeze(0).expand(B, -1)
        id_emb = self.cell_id_embed(cell_ids)        # (B, n_cells, d_id)
        val_emb = self.value_embed(scratch_int)      # (B, n_cells, d_val)
        return torch.cat([id_emb, val_emb], dim=-1)


class V29Pipeline(nn.Module):
    """End-to-end V29 pipeline: signal -> spikes -> events -> PFC -> write head -> scratch.

    Plus beta-side: signal -> spikes -> events -> cross-attention over scratch K/V -> predict.
    """
    def __init__(
        self,
        n_encoder_neurons: int = 16,
        n_write_neurons: int = 5,
        d_id: int = 16,
        d_pe: int = 16,
        n_pfc_layers: int = 2,
        n_pfc_heads: int = 4,
        n_pfc_passes: int = 2,
        threshold: float = 1.0,
        surrogate_alpha: float = 4.0,
        omega_enc_min: float = 0.5,
        omega_enc_max: float = 2.0,
        omega_write_min: float = 0.7,
        omega_write_max: float = 1.8,
        b_init_min: float = -0.3,
        b_init_max: float = -0.05,
        time_max_init: float = 100.0,
        firing_rate_lambda: float = 0.01,
        target_firing_rate: float = 1.5,
        soft_reset: bool = True,
        t_write: Optional[int] = None,
        max_events: int = 256,
        n_levels: int = 3,
        k_max: int = 5,
        bidirectional: bool = True,
    ):
        super().__init__()
        d_model = d_id + d_pe
        self.d_model = d_model
        self.max_events = max_events
        self.firing_rate_lambda = firing_rate_lambda
        self.target_firing_rate = target_firing_rate
        # Encoder (shared alpha/beta)
        self.encoder = RFEncoder(
            n_neurons=n_encoder_neurons,
            threshold=threshold, surrogate_alpha=surrogate_alpha,
            omega_init_min=omega_enc_min, omega_init_max=omega_enc_max,
            b_init_min=b_init_min, b_init_max=b_init_max,
            soft_reset=soft_reset,
        )
        self.neuron_id_embed = nn.Embedding(n_encoder_neurons, d_id)
        self.time_pe = Time2VecPE(d_pe=d_pe, t_max_init=time_max_init)
        # PFC (shared alpha/beta for self-attn over events)
        self.pfc = EventAttentionPFC(
            d_model=d_model, n_layers=n_pfc_layers, n_heads=n_pfc_heads,
            n_passes=n_pfc_passes,
        )
        # alpha-side: 5 query -> write head
        self.alpha_query_extract = CrossAttentionExtractor(
            d_model=d_model, n_queries=n_write_neurons, n_heads=n_pfc_heads,
        )
        self.write_head = RFWriteHead(
            d_model=d_model, n_neurons=n_write_neurons,
            threshold=threshold, surrogate_alpha=surrogate_alpha,
            omega_init_min=omega_write_min, omega_init_max=omega_write_max,
            b_init_min=b_init_min, b_init_max=b_init_max,
            soft_reset=soft_reset, t_write=t_write,
        )
        # Scratch K/V builder
        self.scratch_kv = ScratchKVBuilder(
            n_cells=n_write_neurons, n_levels=n_levels,
            d_id=d_id, d_val=d_pe,  # match d_model = d_id + d_val so K/V tokens are d_model
        )
        # STE soft path for scratch -> K/V: provides gradient channel from K/V
        # back to scratch_q (and thus to write_head). Forward uses discrete
        # lookup (hard_kv); backward flows through this Linear projection.
        self.scratch_kv_soft_proj = nn.Linear(1, d_model)
        # beta-side: cross-attention over scratch K/V
        self.beta_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_pfc_heads, batch_first=True,
        )
        # Predict head: bidirectional successor -> 2*k_max classes
        n_classes = (2 * k_max) if bidirectional else k_max
        self.predict_head = nn.Linear(d_model, n_classes)
        self.bidirectional = bidirectional
        self.k_max = k_max

    def encode_to_events(self, signal: torch.Tensor):
        """signal: (B, L, 1) or (B, L). Returns (tokens, mask, spikes, stash)."""
        spikes, stash = self.encoder(signal)
        tokens, mask = extract_events_padded(spikes, self.neuron_id_embed, self.time_pe,
                                             max_events=self.max_events)
        return tokens, mask, spikes, stash

    def forward(self, alpha_signal: torch.Tensor, beta_signal: torch.Tensor):
        """alpha_signal: (B, L_a, 1).  beta_signal: (B, L_b, 1).

        Returns dict with:
          'logits': (B, n_classes)
          'scratch_q': (B, n_cells)  in {0.0, 0.5, 1.0}
          'scratch_int': (B, n_cells)
          'alpha_spikes', 'beta_spikes', 'alpha_tokens', 'beta_tokens', 'alpha_mask', 'beta_mask'
          'orth_loss': scalar
          'reg_loss': scalar
        """
        # alpha path
        a_tokens, a_mask, a_spikes, _ = self.encode_to_events(alpha_signal)
        a_pfc = self.pfc(a_tokens, a_mask)
        u = self.alpha_query_extract(a_pfc, a_mask)  # (B, n_neurons, d)
        scratch_q, scratch_int = self.write_head(u)
        # K/V tokens with STE: forward = discrete lookup; backward = soft projection
        # so CE loss can reach write_head (and through it, encoder + PFC).
        hard_kv = self.scratch_kv(scratch_int)                              # (B, n_cells, d_model)
        soft_kv = self.scratch_kv_soft_proj(scratch_q.unsqueeze(-1))        # (B, n_cells, d_model)
        kv = soft_kv + (hard_kv - soft_kv).detach()                         # forward = hard_kv, backward = grad through soft_kv
        # beta path
        b_tokens, b_mask, b_spikes, _ = self.encode_to_events(beta_signal)
        b_pfc = self.pfc(b_tokens, b_mask)
        # Cross-attention: beta tokens query, scratch K/V
        # nn.MultiheadAttention does NOT need K/V mask if all 5 cells valid
        attn_out, _ = self.beta_cross_attn(query=b_pfc, key=kv, value=kv)
        # Aggregate beta output across (variable-length) tokens -> mean over valid positions
        mask_f = b_mask.float().unsqueeze(-1)  # (B, N, 1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        b_summary = (attn_out * mask_f).sum(dim=1) / denom  # (B, d)
        logits = self.predict_head(b_summary)  # (B, n_classes)
        # Aux losses
        orth_loss = self.alpha_query_extract.orthogonality_loss()
        reg_loss = self.encoder.regularize_loss() + self.write_head.regularize_loss()
        # Firing rate reg: each encoder neuron should fire at least target_firing_rate
        # spikes per α episode on average. Penalize neurons that fire less.
        # a_spikes: (B, L, N_enc). Mean rate per neuron = sum over L, avg over batch.
        rate_per_neuron = a_spikes.sum(dim=1).mean(dim=0)  # (N_enc,)
        firing_reg = torch.clamp(self.target_firing_rate - rate_per_neuron, min=0.0).sum()
        return {
            "logits": logits,
            "scratch_q": scratch_q,
            "scratch_int": scratch_int,
            "alpha_spikes": a_spikes,
            "beta_spikes": b_spikes,
            "alpha_tokens": a_tokens,
            "beta_tokens": b_tokens,
            "alpha_mask": a_mask,
            "beta_mask": b_mask,
            "orth_loss": orth_loss,
            "reg_loss": reg_loss,
            "firing_reg": firing_reg,
        }
