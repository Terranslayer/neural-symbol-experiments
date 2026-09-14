# backend/core/v33.py
"""V33: HH-SSM substrate — multi-channel selective SSM with closed-loop gate dynamics.

Replaces V27 (LRU + CoT-PFC) recurrence core with 4-channel HH-inspired dynamics:
  Na-like (fast act + inact), K-like (slow act, inhibition polarity), NMDA-like
  (slow act, depolarizing), leak (no gate). Channels share d-dim state via
  non-orthogonal projection directions v_c, enabling multi-channel competition.
  Gates have own first-order relaxation dynamics with learnable time constants.

Forward path is sequential O(L) — no parallel scan (non-orthogonal v_c breaks
diagonal Jacobian). Semi-implicit Euler ensures stability across stiff τ ranges.

Discreteness is NOT in the substrate. Codes are read by the existing V25
sequential write_head with quantize bottleneck applied on h_t output.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNEL_NAMES = ("Na", "K", "NMDA", "leak")  # order matters for indexing
GATE_SPECS = (
    # (gate_name, channel_name, polarity_sign, log_tau_init)
    ("m_Na",   "Na",   +1.0, math.log(2.0)),
    ("h_Na",   "Na",   -1.0, math.log(2.0)),
    ("n_K",    "K",    +1.0, math.log(10.0)),
    ("s_NMDA", "NMDA", +1.0, math.log(40.0)),
)


class HHChannelBank(nn.Module):
    """Manages 4 channels (Na, K, NMDA, leak) — directions v_c, reversal E_c, conductance g_c.

    Geometric design: non-orthogonal v_c (Approach B). Each channel acts on its
    own 1-D projection v̂_c^T · h, pulls it toward E_c. Channels couple through
    shared subspace overlap (cos(v_c, v_c') ≠ 0 by init).

    E_c init is polarity-biased: Na/NMDA positive (depolarizing), K negative
    (hyperpolarizing/inhibition), leak ~0 (baseline). E_c is `nn.Parameter`,
    trained by SGD, fixed at inference.
    """

    def __init__(self, d_model: int = 16, perturbation_scale: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.n_channels = len(CHANNEL_NAMES)

        base = torch.randn(d_model) / math.sqrt(d_model)
        perturbations = torch.randn(self.n_channels, d_model) / math.sqrt(d_model)
        v_init = base.unsqueeze(0) + perturbation_scale * perturbations
        v_init = v_init / v_init.norm(dim=-1, keepdim=True)
        self.v = nn.Parameter(v_init)

        E_init = torch.tensor([+1.0, -1.0, +0.5, 0.0])  # Na, K, NMDA, leak
        E_init = E_init + 0.05 * torch.randn(self.n_channels)
        self.E = nn.Parameter(E_init)

        self.log_g = nn.Parameter(torch.zeros(self.n_channels))

    def unit_directions(self) -> torch.Tensor:
        return self.v / (self.v.norm(dim=-1, keepdim=True) + 1e-8)

    def g(self) -> torch.Tensor:
        return self.log_g.exp()

    def channel_overlap(self) -> torch.Tensor:
        """Pairwise cos similarity between v_c — for diagnostic only.
        Returns (n_channels, n_channels) symmetric matrix with 1s on diagonal.
        """
        v_hat = self.unit_directions()
        return v_hat @ v_hat.T


class HHGateBank(nn.Module):
    """4 gating variables (m_Na, h_Na, n_K, s_NMDA) with first-order relaxation.

    Per-gate parameters:
      w_i, b_i  - learnable scalar (read from V_c projection)
      s_i       - FROZEN polarity sign in {+1, -1}
      log_tau_i - learnable log time constant

    Equilibrium:  g_i_inf(V) = sigmoid( s_i * (w_i * V + b_i) )
    Dynamics:     g_i,t = g_i,t-1 + dt * (g_i_inf - g_i,t-1) / tau_i
    """

    def __init__(self):
        super().__init__()
        self.n_gates = len(GATE_SPECS)
        self.w = nn.Parameter(torch.randn(self.n_gates) * 0.5)
        self.b = nn.Parameter(torch.zeros(self.n_gates))
        log_tau_init = torch.tensor([s[3] for s in GATE_SPECS])
        self.log_tau = nn.Parameter(log_tau_init.clone())
        polarity_init = torch.tensor([s[2] for s in GATE_SPECS])
        self.register_buffer("polarity", polarity_init)
        ch_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}
        gate_ch = torch.tensor([ch_idx[s[1]] for s in GATE_SPECS], dtype=torch.long)
        self.register_buffer("gate_channel_idx", gate_ch)

    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def equilibrium(self, V_per_gate: torch.Tensor) -> torch.Tensor:
        """V_per_gate: (B, n_gates). Returns g_inf shape (B, n_gates) in [0, 1]."""
        z = self.polarity * (self.w * V_per_gate + self.b)
        return torch.sigmoid(z)

    def step(
        self, g_prev: torch.Tensor, V_per_gate: torch.Tensor, dt: torch.Tensor
    ) -> torch.Tensor:
        """One-step gate update.
        g_prev:      (B, n_gates) current gate state
        V_per_gate:  (B, n_gates) projection at each gate's channel
        dt:          scalar tensor
        Returns:     (B, n_gates) new gate state
        """
        g_inf = self.equilibrium(V_per_gate)
        return g_prev + dt * (g_inf - g_prev) / self.tau()


class HHSSMLayer(nn.Module):
    """One HH-SSM layer: gate dynamics + channel mediated update + leak input bypass.

    Sequential O(L) recurrence (no parallel scan). Semi-implicit Euler:
        h_t = (I + dt * A_eff)^{-1} . (h_{t-1} + dt * (b_eff + B_leak . x_t))
    where A_eff = sum_c g_c * P_c * v_hat_c v_hat_c^T (rank <= 4),
    b_eff = sum_c g_c P_c E_c v_hat_c.

    Inversion uses Sherman-Morrison-Woodbury for K=4 channels:
        (I + dt * V_hat D V_hat^T)^{-1} = I - dt * V_hat (D^{-1} + dt V_hat^T V_hat)^{-1} V_hat^T
    where V_hat in R^{d x K}, D = diag(g * P) in R^{K x K}.
    """

    def __init__(self, d_model: int = 16, d_input: int = 16, perturbation_scale: float = 0.3,
                 input_to_gate: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        # V33-D5 (5/20): when input_to_gate=True, input bypasses the direct-injection
        # u_seq path and instead modulates gate equilibria via W_input_to_gate. Forces
        # all input information to flow through channel machinery (gate dynamics →
        # P_c → b_eff → h), eliminating the SGD shortcut that left channels unused
        # in baseline. See layer_compare report (5/20) for diagnosis.
        self.input_to_gate = input_to_gate
        self.channels = HHChannelBank(d_model=d_model, perturbation_scale=perturbation_scale)
        self.gates = HHGateBank()
        self.B_leak = nn.Linear(d_input, d_model, bias=False)
        if input_to_gate:
            self.W_input_to_gate = nn.Linear(d_input, len(GATE_SPECS), bias=False)
        self.log_dt = nn.Parameter(torch.tensor(math.log(0.5)))

    def dt(self) -> torch.Tensor:
        return self.log_dt.clamp(min=math.log(1e-2), max=math.log(10.0)).exp()

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        """Initial state: h_0 = 0; gate init = (m_Na=0, h_Na=1, n_K=0, s_NMDA=0)."""
        h_0 = torch.zeros(batch_size, self.d_model, device=device)
        gate_init_per_gate = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
        g_0 = gate_init_per_gate.unsqueeze(0).expand(batch_size, -1).contiguous()
        return {"h": h_0, "g": g_0}

    def channel_open_prob(self, gate_state: torch.Tensor) -> torch.Tensor:
        """Compute P_c per channel from gate state.
        gate_state: (B, 4) = (m_Na, h_Na, n_K, s_NMDA)
        Returns: P (B, 4) = (P_Na, P_K, P_NMDA, P_leak=1)
        """
        m_Na, h_Na, n_K, s_NMDA = gate_state.unbind(dim=-1)
        P_Na = m_Na * h_Na
        P_K = n_K
        P_NMDA = s_NMDA
        P_leak = torch.ones_like(P_Na)
        return torch.stack([P_Na, P_K, P_NMDA, P_leak], dim=-1)

    def project_V_per_gate(self, h: torch.Tensor) -> torch.Tensor:
        """V_c per gate (each gate reads its channel's V_c projection). h: (B, d)."""
        v_hat = self.channels.unit_directions()  # (4, d)
        V_per_channel = h @ v_hat.T  # (B, 4)
        V_per_gate = V_per_channel[:, self.gates.gate_channel_idx]  # (B, n_gates)
        return V_per_gate

    def forward(self, x_seq: torch.Tensor, state: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        """Sequential forward over x_seq.
        x_seq: (B, L, d_input)
        state: optional initial state dict {h: (B, d), g: (B, n_gates)}
        Returns:
          h_seq: (B, L, d) — h trajectory
          final_state: state dict at end
        """
        B, L, _ = x_seq.shape
        device = x_seq.device
        if state is None:
            state = self.initial_state(B, device)
        h = state["h"]
        g_state = state["g"]

        dt = self.dt()
        v_hat = self.channels.unit_directions()
        E = self.channels.E
        g_c = self.channels.g()

        if self.input_to_gate:
            # D5: input drives gate equilibrium directly; no additive bypass to h.
            u_seq = torch.zeros(B, L, self.d_model, device=x_seq.device, dtype=x_seq.dtype)
            input_to_gate_drive = self.W_input_to_gate(x_seq)  # (B, L, n_gates)
        else:
            u_seq = self.B_leak(x_seq)
            input_to_gate_drive = None
        VtV = v_hat @ v_hat.T

        h_out = []
        for t in range(L):
            V_per_channel = h @ v_hat.T  # (B, 4)
            V_per_gate = V_per_channel[:, self.gates.gate_channel_idx]  # (B, n_gates)
            if input_to_gate_drive is not None:
                V_per_gate = V_per_gate + input_to_gate_drive[:, t]
            g_state = self.gates.step(g_state, V_per_gate, dt)

            P = self.channel_open_prob(g_state)
            gP = g_c.unsqueeze(0) * P

            b_eff = (gP * E.unsqueeze(0)) @ v_hat

            rhs = h + dt * (b_eff + u_seq[:, t])

            D_inv = 1.0 / gP.clamp(min=1e-3)
            M = torch.diag_embed(D_inv) + dt * VtV.unsqueeze(0)
            M_inv = torch.linalg.inv(M)

            VtRhs = rhs @ v_hat.T
            inner = (M_inv @ VtRhs.unsqueeze(-1)).squeeze(-1)
            correction = inner @ v_hat
            h = rhs - dt * correction

            h_out.append(h)

        h_seq = torch.stack(h_out, dim=1)
        final_state = {"h": h, "g": g_state}
        return h_seq, final_state


class V33Substrate(nn.Module):
    """Composes HHSSMLayer with optional multi-layer stacking.

    Input contract:
      x_seq: (B, L, d_input) — encoder features (CNN output, same as V27)
    Output contract:
      h_seq: (B, L, d_model) — hidden state trajectory (same role as LRU+PFC output in V27)

    The write_head is applied OUTSIDE this module by the caller (MambaAgent._v33_forward)
    at the specific write timestep, matching V25/V27 sequential write_head convention.
    """

    def __init__(
        self,
        d_model: int = 16,
        d_input: int = 16,
        n_layers: int = 1,
        perturbation_scale: float = 0.3,
        input_to_gate: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.n_layers = n_layers
        self.input_to_gate = input_to_gate
        self.layers = nn.ModuleList([
            HHSSMLayer(d_model=d_model, d_input=d_input if i == 0 else d_model,
                       perturbation_scale=perturbation_scale,
                       input_to_gate=input_to_gate)
            for i in range(n_layers)
        ])

    def forward(
        self, x_seq: torch.Tensor, state: Optional[List[dict]] = None
    ) -> Tuple[torch.Tensor, List[dict]]:
        """x_seq: (B, L, d_input). Returns h_seq: (B, L, d_model), per-layer states."""
        if state is None:
            state = [None] * self.n_layers
        h = x_seq
        new_states = []
        for layer, layer_state in zip(self.layers, state):
            h, new_state = layer(h, layer_state)
            new_states.append(new_state)
        return h, new_states

    def channel_ortho_loss(self) -> torch.Tensor:
        """Mean squared off-diagonal cos overlap across channels and layers.

        Pushes pairs (v̂_c, v̂_{c'}) toward orthogonality (cos = 0). Without this,
        SGD collapses the 4 channel directions toward a common axis (5/20 baseline
        diag: pairwise cos > 0.91), reducing the substrate to effective rank-1.

        Returns a scalar averaged over the K*(K-1)/2 unordered channel pairs and
        across all layers, so the same λ semantic ("target mean cos²") holds
        regardless of n_layers or K.
        """
        losses = []
        for layer in self.layers:
            v_hat = layer.channels.unit_directions()
            gram = v_hat @ v_hat.T
            K = gram.shape[0]
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            losses.append((gram[mask] ** 2).mean())
        return torch.stack(losses).mean()

    def max_pairwise_overlap(self) -> float:
        """Diagnostic: max |cos(v̂_c, v̂_{c'})| over all layers and off-diag pairs."""
        max_o = 0.0
        for layer in self.layers:
            v_hat = layer.channels.unit_directions().detach()
            gram = (v_hat @ v_hat.T).abs()
            K = gram.shape[0]
            mask = torch.triu(torch.ones_like(gram), diagonal=1).bool()
            max_o = max(max_o, float(gram[mask].max().item()))
        return max_o
