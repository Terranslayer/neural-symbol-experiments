"""V32 multi-gate (A=2) primitive blocks.

Multi-gate factorization: y = sigmoid(W_role · tau) ⊙ sigmoid(W_cont · x) ⊙ (W_v · x)

- W_role: role gate, conditional on type/position embedding `tau`
- W_cont: content gate, conditional on input `x` (Mamba-native selective gate)
- W_v: value projection

References:
- Rumelhart-Hinton-Williams 1986 (PDP vol 1): Sigma-Pi units
- Sutskever-Martens-Hinton 2011: multiplicative RNN
- Krause-Lu-Murray-Renals 2016: multiplicative LSTM (mLSTM)
- Multi-gate channel-wise gating: input × gate × value 3-way interaction

Design choice: A=2 (two multiplicative gates × value = 3 multiplicands). A=3
(Sigma-Pi proper) was considered but Sigma-Pi training is empirically brittle.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGateLinear(nn.Module):
    """A=2 multi-gate linear block.

    Args:
        d_in: input feature dim
        d_role: role embedding dim
        d_out: output dim
    """
    def __init__(self, d_in: int, d_role: int, d_out: int, gate_bias_init: float = 2.0):
        super().__init__()
        self.W_role = nn.Linear(d_role, d_out)
        self.W_cont = nn.Linear(d_in, d_out)
        self.W_v = nn.Linear(d_in, d_out)
        # F5 fix P1 (2026-05-19): sigmoid gates die at init (output ~0.5 each → cumulative
        # 0.25× gain per multi-gate layer, ~0.004× after 4 layers). Bias-init gates to
        # +2.0 so sigmoid(2.0) ≈ 0.88 → cumulative ~0.6× over 4 layers, ~150× recovery
        # of inter-N signal at write_head raw.
        nn.init.constant_(self.W_role.bias, gate_bias_init)
        nn.init.constant_(self.W_cont.bias, gate_bias_init)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        x: (B, ..., d_in)
        tau: (B, ..., d_role)
        returns: (B, ..., d_out)
        """
        g_role = torch.sigmoid(self.W_role(tau))
        g_cont = torch.sigmoid(self.W_cont(x))
        v = self.W_v(x)
        return g_role * g_cont * v


class _STEQuantize(torch.autograd.Function):
    """Forward: hard quantize to {0, 1/(q-1), ..., 1.0}. Backward: identity.

    F3c-1 (2026-05-20): replaced sigmoid mapping with linear clamp. With sigmoid,
    raw values outside ~[-2, +2] produce near-constant q (sigmoid saturation),
    creating a huge dead zone where forward q doesn't respond to raw changes —
    while STE backward still reports grad=1. The training run reaches a saturated
    state (raw≈-125 → sigmoid≈0 → q=0 for all N) that gradient signal cannot
    escape. Linear clamp has a bounded active region [-1, +1] with exact unit
    derivative inside — gradient flow matches forward sensitivity.
    """

    @staticmethod
    def forward(ctx, raw, q_levels):
        # F3c-1: linear clamp maps raw ∈ [-1, +1] → [0, 1] then bin to q_levels.
        x = ((raw + 1.0) / 2.0).clamp(0.0, 1.0)
        bin_idx = torch.floor((q_levels - 1) * x + 0.5).clamp(0, q_levels - 1)
        return bin_idx / (q_levels - 1)

    @staticmethod
    def backward(ctx, grad_out):
        # Straight-through: pass gradient as if quantize were identity.
        return grad_out, None


def ste_quantize(raw: torch.Tensor, q_levels: int) -> torch.Tensor:
    return _STEQuantize.apply(raw, q_levels)


class MultiGateSequentialWriteHead(nn.Module):
    """V32 multi-gate sequential write head.

    Generalizes V27's SequentialWriteHead (backend/core/mamba_agent.py:617).
    V27 used shared MLP looking at [h_pfc, written_so_far]. Empirically cells
    showed cell-cell Pearson +0.97 (all cells were near-identical 1D functions
    of h_pfc with different offsets) — pure thermometer.

    Multi-gate version adds learnable per-cell role embeddings into a 3-way
    multiplicative factorization:
        v_w = MultiGateLinear(input=[h_pfc, written_so_far], tau=role_embeds[w])

    Role gate is conditional on `w` (cell position), so 5 cells receive
    different multiplicative bias and CANNOT degenerate to identical 1D
    functions even when h_pfc is 1D-dominant.

    Quantize is STE: forward hard bin to {0, 0.5, 1.0} (q=3 standard project
    same-medium setting in the design notes). Backward identity.
    """
    def __init__(self, d_pfc: int, W: int = 5, hidden: int = 32, d_role: int = 8,
                 output_scale_init: float = 8.0):
        super().__init__()
        self.W = W
        self.d_role = d_role
        # Per-cell role embedding (W=5 distinct cells)
        self.role_embeds = nn.Parameter(torch.randn(W, d_role) * 0.5)
        # Multi-gate block: input = [h_pfc, written_so_far] dim = d_pfc + W
        # Output: 1 scalar per cell (squeezed)
        self.gate_block = MultiGateLinear(
            d_in=d_pfc + W, d_role=d_role, d_out=hidden,
        )
        self.out_proj = nn.Linear(hidden, 1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden)
        # F5 fix P2 (2026-05-19): learnable gain that scales raw before sigmoid+quantize.
        # Without this, multi-gate attenuation keeps raw range in [-0.15, 0.05], which
        # sigmoid maps to [0.46, 0.51], all quantized to middle bin → 1 distinct code.
        # Init at 8.0 → raw range × 8 → sigmoid sees [-1.2, 0.4] → spans 2-3 quantize bins.
        self.output_scale = nn.Parameter(torch.tensor(output_scale_init))

    def forward(self, h_pfc: torch.Tensor, quantize_levels: int = 3):
        """h_pfc: (B, d_pfc) → returns (raw_all, q_all) of shape (B, W) each.
        q is STE-quantized to {0, 1/(q-1), ..., 1.0}.
        """
        B = h_pfc.shape[0]
        device = h_pfc.device
        written = torch.zeros(B, self.W, device=device, dtype=h_pfc.dtype)
        cells_raw = []
        cells_q = []
        for w in range(self.W):
            input_w = torch.cat([h_pfc, written], dim=-1)  # (B, d_pfc + W)
            tau_w = self.role_embeds[w].unsqueeze(0).expand(B, -1)  # (B, d_role)
            hidden = self.norm(self.act(self.gate_block(input_w, tau_w)))  # (B, hidden)
            raw_w = self.out_proj(hidden).squeeze(-1) * self.output_scale  # (B,)  P2 gain
            q_w = ste_quantize(raw_w, quantize_levels)
            cells_raw.append(raw_w)
            cells_q.append(q_w)
            # Append to written buffer (clone to avoid in-place on grad tensor)
            written = written.clone()
            written[:, w] = q_w
        return torch.stack(cells_raw, dim=-1), torch.stack(cells_q, dim=-1)


class MultiGateLRUBlock(nn.Module):
    """Linear Recurrent Unit (Orvieto 2023) with A=2 multi-gate input projection.

    Standard LRU dynamics:
        h_{t+1} = λ ⊙ h_t + B · x_t      (complex h, λ inside unit disk)
        y_t = Re(C_real h_t - C_imag h_t) + D · x_t

    Multi-gate addition: input projection B · x_t is replaced with:
        B_eff(x_t, tau_t) = sigmoid(W_role tau_t) ⊙ sigmoid(W_cont x_t) ⊙ (W_v x_t)

    where tau_t is the per-token role embedding (e.g., SIGNAL vs SCRATCH_OWN
    vs SCRATCH_PARTNER token type). This lets LRU process different token
    types with conditional behavior without redesigning architecture.

    init_h_r/init_h_i support chunk-continuity (V27 invariant): calling on
    chunks sequentially with carried h equals calling on concatenation.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_role: int = 4,
                 r_min: float = 0.0, r_max: float = 1.0,
                 phase_max: float = math.pi * 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Eigenvalue parameterization (Orvieto 2023):
        # |λ| = exp(-exp(nu_log)) ∈ (0, 1) guaranteed stable
        # θ = exp(theta_log) ∈ (0, ∞) clamped via phase_max via init
        u1 = torch.rand(d_state)
        u2 = torch.rand(d_state)
        nu_log = torch.log(
            -0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2 + 1e-9)
        )
        theta_log = torch.log(u2 * phase_max + 1e-6)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        # Multi-gate input projection (B matrix): (d_model, d_role) → d_state
        # Real and imaginary parts: project to complex state.
        self.B_real = MultiGateLinear(d_in=d_model, d_role=d_role, d_out=d_state)
        self.B_imag = MultiGateLinear(d_in=d_model, d_role=d_role, d_out=d_state)
        # Output projection (C matrix): real + imaginary parts of state → d_model
        self.C_real = nn.Linear(d_state, d_model, bias=False)
        self.C_imag = nn.Linear(d_state, d_model, bias=False)
        # Skip connection D (Orvieto recommends)
        self.D = nn.Linear(d_model, d_model)

    def get_lambda(self):
        """λ = exp(-exp(nu_log))·(cos θ + i sin θ), θ = exp(theta_log)."""
        decay = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)
        return decay * torch.cos(phase), decay * torch.sin(phase)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                init_h_r: torch.Tensor = None,
                init_h_i: torch.Tensor = None,
                return_state: bool = False):
        """
        x: (B, T, d_model)
        tau: (B, T, d_role)
        init_h_r, init_h_i: (B, d_state) — carry-in state. Defaults to zero.
        Returns: out (B, T, d_model), and if return_state, (h_r_final, h_i_final).
        """
        B, T, _ = x.shape
        lam_r, lam_i = self.get_lambda()  # (d_state,) each
        # Per-step input projections (B matrix, multi-gate)
        Bx_r = self.B_real(x, tau)  # (B, T, d_state)
        Bx_i = self.B_imag(x, tau)  # (B, T, d_state)
        # Run scan
        h_r = init_h_r if init_h_r is not None else torch.zeros(
            B, self.d_state, device=x.device, dtype=x.dtype
        )
        h_i = init_h_i if init_h_i is not None else torch.zeros(
            B, self.d_state, device=x.device, dtype=x.dtype
        )
        outs = []
        for t in range(T):
            # Complex multiply: (h_r + i h_i)(lam_r + i lam_i)
            #   = (h_r lam_r - h_i lam_i) + i (h_r lam_i + h_i lam_r)
            new_r = h_r * lam_r - h_i * lam_i + Bx_r[:, t]
            new_i = h_r * lam_i + h_i * lam_r + Bx_i[:, t]
            h_r, h_i = new_r, new_i
            y_t = self.C_real(h_r) - self.C_imag(h_i) + self.D(x[:, t])
            outs.append(y_t)
        out = torch.stack(outs, dim=1)  # (B, T, d_model)
        if return_state:
            return out, h_r, h_i
        return out


class _MultiGatePFCStep(nn.Module):
    """One PFC step: self-attn over 3 tokens [pfc_state, cnn_token, lru_token]
    with multi-gate projections."""
    def __init__(self, d_model: int, d_role: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Multi-gate Q/K/V projections — each token's role gate fires on its
        # role identity (state vs cnn vs lru). role_embeds shape (3, d_role).
        self.role_embeds = nn.Parameter(torch.randn(3, d_role) * 0.5)
        self.Q = MultiGateLinear(d_in=d_model, d_role=d_role, d_out=d_model)
        self.K = MultiGateLinear(d_in=d_model, d_role=d_role, d_out=d_model)
        self.V = MultiGateLinear(d_in=d_model, d_role=d_role, d_out=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        # FFN after attn
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 3, d_model) where dim-1 ordering = [state, cnn, lru].
        Returns: (B, 3, d_model)
        """
        B = tokens.shape[0]
        # Build per-token role embedding (B, 3, d_role)
        tau = self.role_embeds.unsqueeze(0).expand(B, -1, -1)
        q = self.Q(tokens, tau).reshape(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        k = self.K(tokens, tau).reshape(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        v = self.V(tokens, tau).reshape(B, 3, self.n_heads, self.d_head).transpose(1, 2)
        # (B, h, 3, d_head)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, h, 3, d_head)
        out = out.transpose(1, 2).reshape(B, 3, self.d_model)
        out = self.norm(self.out_proj(out) + tokens)
        out = self.norm2(self.ffn(out) + out)
        return out


class MultiGateCoTPFC(nn.Module):
    """V32 multi-gate CoTPFC. Inherits V27 4-chunk chain design unchanged,
    upgrades internal PFC self-attn projections to A=2 multi-gate.

    Per chunk c:
        chunk_signal = signal[:, c*chunk_size:(c+1)*chunk_size, :]
        cnn_token = strided_conv(chunk)  # (B, d) — 1 token per chunk
        lru_state, h = LRU(chunk, init_h=prev_h)  # continuous across chunks
        lru_token = lru_state[:, -1, :]
        tokens = [pfc_state, cnn_token, lru_token]  # (B, 3, d)
        pfc_state = MultiGatePFCStep(tokens)[:, 0, :]

    Returns final pfc_state after n_chunks iterations.

    Multiple CNN heads (one per token_type_id) registered via `kernels_per_type`
    so a single shared instance can serve Phase 1 (SIGNAL, type=0) and Phase 3
    (SCRATCH, type=1) with different CNN kernel sizes. PFC + LRU + projections
    are shared across phases — only CNN front-end differs per phase.
    """
    def __init__(self, d_model: int, kernels_per_type: dict,
                 cnn_channels_per_scale: int, lru_block: nn.Module,
                 input_dim: int = 3, d_role: int = 4, n_pfc_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_role = d_role
        self.kernels_per_type = kernels_per_type
        # Per-token-type CNN heads + projection
        self.convs_per_type = nn.ModuleDict()
        self.cnn_proj_per_type = nn.ModuleDict()
        for type_id, kernels in kernels_per_type.items():
            self.convs_per_type[str(type_id)] = nn.ModuleList([
                nn.Conv1d(input_dim, cnn_channels_per_scale, k, padding=(k - 1) // 2)
                for k in kernels
            ])
            self.cnn_proj_per_type[str(type_id)] = nn.Linear(
                len(kernels) * cnn_channels_per_scale, d_model
            )
        # LRU input projection (raw signal dim → d_model)
        self.lru_input_proj = nn.Linear(input_dim, d_model)
        # LRU role embedding (per token-type-id). Default: 2 token types
        # (SIGNAL=0, SCRATCH=1). Caller passes token_type_id to indicate which.
        self.lru_type_embed = nn.Embedding(2, d_role)
        self.lru_block = lru_block
        # PFC step (multi-gate self-attn)
        self.pfc_step = _MultiGatePFCStep(d_model, d_role, n_heads=n_pfc_heads)
        # Initial PFC state
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)

    def cnn_chunked(self, signal: torch.Tensor, n_chunks: int,
                    token_type_id: int) -> torch.Tensor:
        """signal: (B, L, input_dim) → (B, n_chunks, d_model).
        Uses CNN convs registered for the given token_type_id.
        """
        B, L, _ = signal.shape
        chunk_size = max(1, L // n_chunks)
        xt = signal.transpose(1, 2)
        type_key = str(token_type_id)
        convs = self.convs_per_type[type_key]
        cnn_proj = self.cnn_proj_per_type[type_key]
        outs = []
        for conv in convs:
            o = F.relu(conv(xt))  # (B, ch, L)
            idx = torch.arange(n_chunks, device=o.device) * chunk_size + chunk_size // 2
            idx = idx.clamp(max=o.shape[-1] - 1)
            o = o[:, :, idx]  # (B, ch, n_chunks)
            outs.append(o)
        combined = torch.cat(outs, dim=1).transpose(1, 2)  # (B, n_chunks, total_ch)
        return cnn_proj(combined)

    def forward(self, signal: torch.Tensor, n_chunks: int, token_type_id: int = 0,
                stash: dict = None) -> torch.Tensor:
        """signal: (B, L, input_dim) → (B, d_model) final pfc_state.

        n_chunks: number of PFC chain iterations. Phase 1 typically 4; Phase 3 typically 2.
        token_type_id: 0=SIGNAL, 1=SCRATCH; selects which CNN set to use + LRU role embed.
        """
        B, L, _ = signal.shape
        chunk_size = max(1, L // n_chunks)

        cnn_chunks = self.cnn_chunked(signal, n_chunks, token_type_id)  # (B, n_chunks, d_model)
        pfc_state = self.init_state.unsqueeze(0).expand(B, -1)
        h_lru_r, h_lru_i = None, None

        if stash is not None:
            stash["cnn_chunks"] = cnn_chunks.detach()
            stash["pfc_states_per_chunk"] = []
            stash["lru_h_per_chunk"] = []

        # LRU role embedding for this signal's token type (constant per call)
        type_id_t = torch.tensor([token_type_id], device=signal.device, dtype=torch.long)
        tau_token = self.lru_type_embed(type_id_t)  # (1, d_role)

        for c in range(n_chunks):
            cs = c * chunk_size
            ce = (c + 1) * chunk_size if c < n_chunks - 1 else L
            chunk = signal[:, cs:ce, :]  # (B, chunk_size, input_dim)
            chunk_proj = self.lru_input_proj(chunk)  # (B, chunk_size, d_model)
            tau_chunk = tau_token.unsqueeze(0).expand(B, chunk.shape[1], -1)
            lru_out, h_lru_r, h_lru_i = self.lru_block(
                chunk_proj, tau_chunk,
                init_h_r=h_lru_r, init_h_i=h_lru_i, return_state=True,
            )
            lru_token = lru_out[:, -1, :]  # (B, d_model)
            cnn_token = cnn_chunks[:, c, :]
            tokens = torch.stack([pfc_state, cnn_token, lru_token], dim=1)  # (B, 3, d)
            refined = self.pfc_step(tokens)
            pfc_state = refined[:, 0, :]
            if stash is not None:
                stash["pfc_states_per_chunk"].append(pfc_state.detach().clone())
                stash["lru_h_per_chunk"].append(lru_token.detach().clone())

        return pfc_state
