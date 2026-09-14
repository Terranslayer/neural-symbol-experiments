"""V35: d=1 HH-SSM substrate with fire-and-reset for base-theta cascade.

d_model=1 collapses V33's 4-channel R^16 substrate to a single scalar h.
Each channel (Na, K, NMDA, leak) keeps its own E_c, log_g_c, and a fixed
channel-sign in {+1, -1} (replacing the unit direction v_c that V33 had
in R^d). Each gate (m_Na, h_Na, n_K, s_NMDA) keeps its learnable tau,
polarity sign, and (w, b).

Fire-and-reset is appended: at each timestep after h update, if h >= theta,
emit binary spike y_t = 1 and subtract theta from h. Gradient flows through
the hard threshold via straight-through estimator (STE).
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNEL_NAMES = ("Na", "K", "NMDA", "leak")
GATE_SPECS = (
    # (gate_name, channel_name, polarity_sign, log_tau_init)
    ("m_Na",   "Na",   +1.0, math.log(2.0)),
    ("h_Na",   "Na",   -1.0, math.log(2.0)),
    ("n_K",    "K",    +1.0, math.log(10.0)),
    ("s_NMDA", "NMDA", +1.0, math.log(40.0)),
)


def _ste_threshold(h: torch.Tensor, theta: torch.Tensor,
                   temp: float = 0.5) -> torch.Tensor:
    """Hard spike y = 1[h >= theta] with straight-through gradient via
    sigmoid((h - theta) / temp).
    """
    hard = (h >= theta).float()
    soft = torch.sigmoid((h - theta) / temp)
    return soft + (hard - soft).detach()


def _ste_state_quantize(h: torch.Tensor, n_levels: int,
                        theta: torch.Tensor) -> torch.Tensor:
    """Straight-through quantization of the post-fire-reset membrane state h to
    n_levels evenly-spaced levels across [0, theta].

    Motivation (INSIGHTS sec.11-13): function-selecting radix needs a FULLY
    discrete substrate -- a discrete carry AND a discrete level/state. V35's
    carry (the fire-and-reset spike train) is already discrete, but the membrane
    state between resets is continuous, which the abstract program's 2x2 predicts
    is the failure-risk regime. This snaps the inter-spike state to a small
    alphabet so the level is discrete too. Forward: round-to-grid; backward:
    identity (STE) so the substrate stays trainable. n_levels<=1 returns h
    unchanged (callers gate on >1; guarded anyway).
    """
    if n_levels <= 1:
        return h
    step = theta / (n_levels - 1)
    # Clamp into [0, theta] for the grid, snap to nearest level, STE passthrough.
    h_clamped = h.clamp(min=torch.zeros_like(theta), max=theta)
    q = torch.round(h_clamped / step) * step
    return h + (q - h).detach()


def food_pattern_spike_gate(
    sig: torch.Tensor,
    center_thresh: float = 0.9,
    neighbor_thresh: float = 0.8,
) -> torch.Tensor:
    """Deterministic object detector replicating scene.count_foods cell-wise.

    A center cell c fires iff sig[c] >= center_thresh AND both neighbors
    sig[c-1], sig[c+1] >= neighbor_thresh. Boundary cells (c=0, c=L-1) never fire
    (no valid neighbor on one side; scene food centers live in [1, L-2]). Because
    the scene resamples until count_foods(signal) == N, the per-row spike sum
    equals N exactly, with spikes at the N food centers. Distractors
    (isolated_spike / weak_neighbor) have a food-level center but fail the neighbor
    test, so they are correctly excluded.

    Args:
      sig: (B, L) raw signal channel.
      center_thresh / neighbor_thresh: mirror scene.FOOD_CENTER_THRESH / FOOD_NEIGHBOR_THRESH.
    Returns:
      (B, L) float binary spike train. No parameters, no gradient.
    """
    center = sig >= center_thresh                          # (B, L) bool
    left_ok = torch.zeros_like(center)
    left_ok[:, 1:] = sig[:, :-1] >= neighbor_thresh        # neighbor at c-1
    right_ok = torch.zeros_like(center)
    right_ok[:, :-1] = sig[:, 1:] >= neighbor_thresh       # neighbor at c+1
    return (center & left_ok & right_ok).to(sig.dtype)


class LearnedLayerADetector(nn.Module):
    """LEARNED object counter (direction-1: de-scaffold counting; replaces the hand-coded
    food_pattern_spike_gate). A tied window Conv1d over the raw signal -> per-position logit;
    a LOCAL-MAX (lateral-inhibition / WTA) gate fires at most once per object peak. The spike
    train is binary in the forward (so the downstream cascade counts integer spikes) but carries
    a straight-through gradient via the sigmoid soft path, so SGD trains the detector from the
    task loss. WHAT a peak is = LEARNED; local-max is a generic perceptual readout (admissible).
    Mirrors the CPU count+carry recipe (window-MLP + local-max) that emerged exact, position-
    invariant counting under ES (RESEARCH_LOG 2026-06-16).

    Args:
      kernel: detector window width (odd). hidden: conv hidden width. temp: STE soft sharpness.
      local_max: if True, gate spikes to strict local maxima (one per peak); else plain threshold.
    """

    def __init__(self, kernel: int = 5, hidden: int = 8, temp: float = 4.0, local_max: bool = True,
                 nms_window: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(hidden, 1, kernel_size=1)
        self.temp = temp
        self.local_max = local_max
        self.nms_window = nms_window if nms_window % 2 == 1 else nms_window + 1  # odd

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        # sig: (B, L) raw signal channel -> (B, L) binary spike train (STE).
        x = sig.unsqueeze(1)                                   # (B, 1, L)
        logit = self.conv2(torch.relu(self.conv1(x))).squeeze(1)  # (B, L)
        soft = torch.sigmoid(self.temp * logit)                # (B, L) in (0,1), grad path
        fire = (soft > 0.5)
        if self.local_max:
            # WINDOWED NMS (anti-flood): a position fires only if it is the MAX logit in a window of
            # nms_window cells (so >=~nms_window/2 cells between spikes -> at most one spike per object),
            # + strict-left to break plateau ties. Prevents the over-fire -> flood -> collapse failure.
            wm = torch.nn.functional.max_pool1d(
                logit.unsqueeze(1), kernel_size=self.nms_window, stride=1, padding=self.nms_window // 2
            ).squeeze(1)
            lo = torch.full_like(logit, float("-inf"))
            lo[:, 1:] = logit[:, :-1]
            peak = fire & (logit >= wm - 1e-6) & (logit > lo)
        else:
            peak = fire
        hard = peak.to(sig.dtype)
        return hard + (soft - soft.detach())                   # forward = hard spikes; backward = sigmoid grad


class HHSSMLayer1D(nn.Module):
    """One d=1 HH-SSM layer with fire-and-reset.

    Args:
      d_input: Input feature width.
      theta_param: Optional shared learnable theta nn.Parameter. If None, the
                   layer owns its own theta.
      fire_temp: STE temperature for spike soft path.
    """

    def __init__(
        self,
        d_input: int,
        theta_param: Optional[nn.Parameter] = None,
        fire_temp: float = 0.5,
        biological_init: bool = False,
        theta_fixed: Optional[float] = None,
        log_g_leak_init: Optional[float] = None,
        b_leak_warm_scale: float = 1.0,
        b_leak_init: Optional[float] = None,
        state_discretize: int = 0,
    ):
        super().__init__()
        self.d_input = d_input
        self.n_channels = len(CHANNEL_NAMES)
        self.n_gates = len(GATE_SPECS)
        self.fire_temp = fire_temp
        # STE state-discretization of the post-fire-reset membrane (INSIGHTS sec.13).
        # 0 (or 1) = OFF -> forward is byte-for-byte identical to legacy V35.
        # K>1 = snap inter-spike state to K levels across [0, theta] (STE).
        self.state_discretize = int(state_discretize)

        # Channel reversal potentials E_c (learnable, polarity-biased init)
        E_init = torch.tensor([+1.0, -1.0, +0.5, 0.0]) + 0.05 * torch.randn(4)
        self.E = nn.Parameter(E_init)
        # 5/28 fix: log_g init = -4 (g≈0.018) instead of 0 (g=1). At g=1 the
        # leak channel (P_leak always = 1) makes a_eff_init = 1, and the
        # semi-implicit Euler divisor (1 + dt*a_eff) ≈ 1.5 caps h accumulation
        # per CNN bump at ~0.17. With N=30 bumps spaced ~6.7 timesteps apart,
        # max h ≈ 0.18 ≪ θ=3 → layer A never fires, cascade is dead-on-arrival.
        # log_g=-4 → a_eff_init ≈ 0.07 → divisor ≈ 1.04 → ~LIF accumulation.
        # Channels can still learn to activate (log_g is learnable).
        # 6/02: log_g_leak_init optionally overrides ONLY the leak channel (index 3) so
        # Layer A can be given a membrane leak (a_eff knob) without touching Na/K/NMDA.
        log_g_init = torch.full((self.n_channels,), -4.0)
        if log_g_leak_init is not None:
            log_g_init[3] = log_g_leak_init  # index 3 = "leak" (CHANNEL_NAMES)
        self.log_g = nn.Parameter(log_g_init)

        # Channel sign (frozen, replaces V33's v_hat in R^1)
        ch_signs = torch.tensor([+1.0, -1.0, +1.0, +1.0])  # Na+, K-, NMDA+, leak+
        self.register_buffer("ch_sign", ch_signs)

        # Per-gate: w, b (learnable scalars); polarity (frozen); log_tau (learnable)
        if biological_init:
            # HH-inspired init for V35 local-PC mode. gate_w sign + magnitude
            # gives Rao-Ballard fast(m_Na, τ=2) vs slow(s_NMDA, τ=40) prediction-error
            # pair when paired with fixed polarity. gate_b shifts equilibria off 0.5
            # so resting V≈0 leaves gates partly closed.
            gate_w_init = torch.tensor([2.0, 2.0, 1.0, 0.5])      # m_Na, h_Na, n_K, s_NMDA
            gate_b_init = torch.tensor([-0.5, +0.5, -0.5, -0.5])
        else:
            gate_w_init = torch.randn(self.n_gates) * 0.5
            gate_b_init = torch.zeros(self.n_gates)
        self.gate_w = nn.Parameter(gate_w_init.clone())
        self.gate_b = nn.Parameter(gate_b_init.clone())
        polarity = torch.tensor([s[2] for s in GATE_SPECS])
        self.register_buffer("gate_polarity", polarity)
        log_tau_init = torch.tensor([s[3] for s in GATE_SPECS])
        self.log_tau = nn.Parameter(log_tau_init.clone())
        ch_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}
        gate_ch = torch.tensor(
            [ch_idx[s[1]] for s in GATE_SPECS], dtype=torch.long
        )
        self.register_buffer("gate_channel_idx", gate_ch)

        self.B_leak = nn.Linear(d_input, 1, bias=False)
        # 6/02: b_leak_init fills B_leak with a constant (hand-set frozen counter, e.g. Layer B).
        # None = no-op (keeps default init). Applied BEFORE warm scale (not used together).
        if b_leak_init is not None:
            self.B_leak.weight.data.fill_(b_leak_init)
        # 6/02: warm B_leak init (cold-start firing guarantee replacing IP). No-op at 1.0.
        if b_leak_warm_scale != 1.0:
            self.B_leak.weight.data.mul_(b_leak_warm_scale)

        # Homeostatic intrinsic-plasticity tonic bias (5/28). Scalar standing drive
        # added every timestep; regulated by V35Substrate.pc_step IP rule toward a
        # target average firing rate. Init 0 (no bias). Substrate param -> excluded
        # from SGD. Spec: docs/design.md
        self.ip_bias = nn.Parameter(torch.zeros(1))

        self.log_dt = nn.Parameter(torch.tensor(math.log(0.5)))

        # theta: fixed frozen buffer (per-layer mode), shared param, or own param.
        if theta_fixed is not None:
            # Per-layer fixed theta: register as frozen buffer, no learnable raw.
            self.register_buffer("_theta_fixed", torch.tensor(theta_fixed))
            self._shared_theta = None
            self.theta_raw = None
        elif theta_param is None:
            self.theta_raw = nn.Parameter(
                torch.tensor(_inverse_softplus(2.0))  # softplus(_) + 1.0 = 3.0
            )
            self._shared_theta = None
        else:
            self._shared_theta = theta_param
            self.theta_raw = None

    def _theta(self) -> torch.Tensor:
        if hasattr(self, "_theta_fixed"):
            return self._theta_fixed
        raw = self._shared_theta if self._shared_theta is not None else self.theta_raw
        return 1.0 + F.softplus(raw)  # anchor in [1, inf), init at 3.0

    def _dt(self) -> torch.Tensor:
        return self.log_dt.clamp(min=math.log(1e-2), max=math.log(10.0)).exp()

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        h = torch.zeros(batch_size, 1, device=device)
        gate_init = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
        g = gate_init.unsqueeze(0).expand(batch_size, -1).contiguous()
        return {"h": h, "g": g}

    def _channel_open_prob(self, g: torch.Tensor) -> torch.Tensor:
        m_Na, h_Na, n_K, s_NMDA = g.unbind(dim=-1)
        P_Na = m_Na * h_Na
        P_K = n_K
        P_NMDA = s_NMDA
        P_leak = torch.ones_like(P_Na)
        return torch.stack([P_Na, P_K, P_NMDA, P_leak], dim=-1)

    def forward(
        self,
        x_seq: torch.Tensor,
        state: Optional[dict] = None,
        capture_for_pc: bool = False,
    ):
        """
        x_seq: (B, L, d_input)
        Returns:
          h_seq:    (B, L, 1) trajectory of scalar state after fire-and-reset
          spike_seq: (B, L) binary spike train (with STE grad)
          residual: (B,) final h(L) after L timesteps (i.e., h_seq[:,-1,0])
          final_state: dict {h, g}
        """
        B, L, _ = x_seq.shape
        device = x_seq.device
        if state is None:
            state = self.initial_state(B, device)
        h = state["h"]
        g = state["g"]

        dt = self._dt()
        theta = self._theta()
        u_seq = self.B_leak(x_seq)  # (B, L, 1)
        g_c = self.log_g.exp()
        E = self.E
        ch_sign = self.ch_sign
        gate_ch_idx = self.gate_channel_idx
        gate_polarity = self.gate_polarity
        tau = self.log_tau.exp()

        h_out: List[torch.Tensor] = []
        spike_out: List[torch.Tensor] = []
        gate_history: List[torch.Tensor] = []  # only populated when capture_for_pc=True
        for t in range(L):
            # V per gate: project h via channel sign at each gate's channel
            V_per_channel = h * ch_sign.unsqueeze(0)  # (B, 4)
            V_per_gate = V_per_channel[:, gate_ch_idx]  # (B, n_gates)
            g_inf = torch.sigmoid(
                gate_polarity * (self.gate_w * V_per_gate + self.gate_b)
            )
            g = g + dt * (g_inf - g) / tau
            if capture_for_pc:
                gate_history.append(g.clone())  # (B, n_gates) snapshot after gate update

            P = self._channel_open_prob(g)  # (B, 4)
            gP = g_c.unsqueeze(0) * P
            # Drive: sum_c gP_c * E_c * ch_sign_c (scalar per sample)
            drive = (gP * E.unsqueeze(0) * ch_sign.unsqueeze(0)).sum(
                dim=-1, keepdim=True
            ) + u_seq[:, t] + self.ip_bias
            a_eff = (gP * ch_sign.unsqueeze(0).abs()).sum(dim=-1, keepdim=True)
            # Semi-implicit Euler (scalar form): h_new = (h + dt*drive) / (1+dt*a_eff)
            h = (h + dt * drive) / (1.0 + dt * a_eff)

            # Fire-and-reset
            y = _ste_threshold(h, theta, temp=self.fire_temp)
            h = h - theta * y

            # Optional STE state discretization of the post-reset membrane
            # (INSIGHTS sec.13). OFF by default (state_discretize<=1 -> no-op),
            # so default V35 forward is unchanged byte-for-byte.
            if self.state_discretize > 1:
                h = _ste_state_quantize(h, self.state_discretize, theta)

            h_out.append(h)
            spike_out.append(y.squeeze(-1))

        h_seq = torch.stack(h_out, dim=1)         # (B, L, 1)
        spike_seq = torch.stack(spike_out, dim=1) # (B, L)
        residual = h_seq[:, -1, 0]                # (B,)
        final_state = {"h": h, "g": g}
        if capture_for_pc:
            gate_seq = torch.stack(gate_history, dim=1)            # (B, L, n_gates)
            spike_in = (x_seq.abs().sum(dim=-1) > 1e-6).float()    # (B, L)
            capture = {
                "gate_seq": gate_seq,
                "u_seq": u_seq,                                     # (B, L, 1), pre-computed above
                "input_seq": x_seq,                                 # (B, L, d_input)
                "spike_in": spike_in,
                "h_seq": h_seq,                                     # (B, L, 1), substrate state after fire-reset
                "spike_seq": spike_seq,                             # (B, L), this layer's OUTPUT spikes (for IP r_obs)
            }
            return h_seq, spike_seq, residual, final_state, capture
        return h_seq, spike_seq, residual, final_state


def _inverse_softplus(y: float) -> float:
    """Solve y = softplus(x) for x: x = log(exp(y) - 1)."""
    return math.log(math.expm1(y))


class FireResetCascade(nn.Module):
    """Layer A (translator) untied + Layer B (carry) tied across n_recursions.

    Layer A reads d_input features (CNN output during signal pass, or
    same-medium scratch readback as a 5-step single-channel input).
    Layer B is invoked n_recursions times sequentially; pass k's input
    is pass (k-1)'s spike train; weights are shared. Theta is a single
    shared scalar across all 6 sub-layer invocations.
    """

    def __init__(
        self,
        d_input: int,
        n_recursions: int = 5,
        fire_temp: float = 0.5,
        biological_init: bool = False,
        theta_a: Optional[float] = None,
        theta_b: Optional[float] = None,
        log_g_leak_a: Optional[float] = None,
        b_leak_warm_a: float = 1.0,
        b_leak_b_init: Optional[float] = None,
        state_discretize: int = 0,
        theta_shared_init: float = 3.0,
    ):
        super().__init__()
        self.n_recursions = n_recursions

        if theta_a is not None and theta_b is not None:
            # Per-layer fixed theta mode: frozen buffers, no shared theta_raw.
            self.theta_raw = None
            self.layer_A = HHSSMLayer1D(
                d_input=d_input,
                theta_param=None,
                theta_fixed=theta_a,
                fire_temp=fire_temp,
                biological_init=biological_init,
                log_g_leak_init=log_g_leak_a,
                b_leak_warm_scale=b_leak_warm_a,
                state_discretize=state_discretize,
            )
            # Layer B: log_g_leak_a / b_leak_warm_a deliberately NOT passed — the carry
            # fold requires the no-leak integrator property (log_g all -4); leaking layer_B
            # destroys the carry. Correctness lock (same rationale as the legacy branch below).
            self.layer_B = HHSSMLayer1D(
                d_input=1,
                theta_param=None,
                theta_fixed=theta_b,
                fire_temp=fire_temp,
                biological_init=biological_init,
                b_leak_init=b_leak_b_init,
                state_discretize=state_discretize,
            )
        else:
            # Legacy: shared theta scalar across all sub-layers. theta_shared_init sets the
            # initial effective theta (= base); default 3.0 reproduces the prior behavior.
            # Rung B2 (de-scaffold "invent" test) sets this AWAY from 3 to ask whether the
            # learnable base DISCOVERS 3 from a non-3 start (vs only MAINTAINS it, Rung B1).
            self.theta_raw = nn.Parameter(
                torch.tensor(_inverse_softplus(float(theta_shared_init) - 1.0))
            )
            self.layer_A = HHSSMLayer1D(
                d_input=d_input,
                theta_param=self.theta_raw,
                fire_temp=fire_temp,
                biological_init=biological_init,
                log_g_leak_init=log_g_leak_a,
                b_leak_warm_scale=b_leak_warm_a,
                state_discretize=state_discretize,
            )
            # Layer B sees binary spike train (single scalar per timestep).
            # log_g_leak_a / b_leak_warm_a are NOT passed to layer_B: layer_B is a
            # carry counter whose fold requires the no-leak integrator property
            # (log_g all -4). Giving it a leak destroys the carry. Correctness lock.
            self.layer_B = HHSSMLayer1D(
                d_input=1,
                theta_param=self.theta_raw,
                fire_temp=fire_temp,
                biological_init=biological_init,
                b_leak_init=b_leak_b_init,
                state_discretize=state_discretize,
            )

    def theta(self) -> torch.Tensor:
        if self.theta_raw is not None:
            return 1.0 + F.softplus(self.theta_raw)
        # Per-layer mode: return layer_B's theta (representative scalar for callers)
        return self.layer_B._theta()

    def forward(self, x_seq: torch.Tensor, capture_for_pc: bool = False):
        """
        x_seq: (B, L, d_input)
        capture_for_pc: if True, also return capture_data dict with per-call gate
                        trajectories for layer A and each Layer B recursion.

        Returns:
          residuals (B, n_recursions): h_k(L) for k in 1..n_recursions
          spike_trains_B (list of n_recursions tensors of shape (B, L))
          spike_train_A (B, L): Layer A's output spike train
          h_seqs_B (list of n_recursions tensors of shape (B, L, 1))
          [capture_data (dict)]   — only when capture_for_pc=True
        """
        if capture_for_pc:
            a_out = self.layer_A(x_seq, capture_for_pc=True)
            _h_a_seq, spike_train_A, _residual_a, _state_a, capture_A = a_out
        else:
            _h_a_seq, spike_train_A, _residual_a, _state_a = self.layer_A(x_seq)
            capture_A = None

        cur = spike_train_A.unsqueeze(-1)             # (B, L, 1)
        residuals_list: List[torch.Tensor] = []
        spike_trains_B: List[torch.Tensor] = []
        h_seqs_B: List[torch.Tensor] = []
        capture_B_list: List[dict] = []
        for _k in range(self.n_recursions):
            if capture_for_pc:
                b_out = self.layer_B(cur, capture_for_pc=True)
                h_seq, spike_seq, residual, _state, capture_B = b_out
                capture_B_list.append(capture_B)
            else:
                h_seq, spike_seq, residual, _state = self.layer_B(cur)
            residuals_list.append(residual)
            spike_trains_B.append(spike_seq)
            h_seqs_B.append(h_seq)
            cur = spike_seq.unsqueeze(-1)
        residuals = torch.stack(residuals_list, dim=-1)  # (B, n_recursions)

        if capture_for_pc:
            capture_data = {"layer_A": capture_A, "layer_B_recursions": capture_B_list}
            return residuals, spike_trains_B, spike_train_A, h_seqs_B, capture_data
        return residuals, spike_trains_B, spike_train_A, h_seqs_B


class V35Substrate(nn.Module):
    """Top-level V35 substrate: FireResetCascade with a forward API
    matching `(x_seq) -> (residuals, aux_dict)` for agent integration.
    """

    def __init__(
        self,
        d_input: int,
        n_recursions: int = 5,
        fire_temp: float = 0.5,
        biological_init: bool = False,
        theta_a: Optional[float] = None,
        theta_b: Optional[float] = None,
        log_g_leak_a: Optional[float] = None,
        b_leak_warm_a: float = 1.0,
        b_leak_b_init: Optional[float] = None,
        state_discretize: int = 0,
        theta_shared_init: float = 3.0,
    ):
        super().__init__()
        self.cascade = FireResetCascade(
            d_input=d_input,
            n_recursions=n_recursions,
            fire_temp=fire_temp,
            biological_init=biological_init,
            theta_a=theta_a,
            theta_b=theta_b,
            log_g_leak_a=log_g_leak_a,
            b_leak_warm_a=b_leak_warm_a,
            b_leak_b_init=b_leak_b_init,
            state_discretize=state_discretize,
            theta_shared_init=theta_shared_init,
        )
        # Populated by forward(..., capture_for_pc=True). Cleared by clear_pc_history().
        # Each element is one cascade-forward's capture_data dict.
        self._pc_data_history: List[dict] = []

    def clear_pc_history(self):
        self._pc_data_history = []

    @torch.no_grad()
    def pc_step(
        self,
        lr: float,
        use_h_upstream: bool = False,
        ip_enable: bool = False,
        ip_lr: float = 0.02,
        r_target: Optional[float] = None,
        ip_bias_clamp: float = 0.0,
        weight_decay: float = 0.0,
        ip_leak: float = 0.0,
        modulator: Optional[torch.Tensor] = None,
        weight_clip: float = 0.0,
        freeze_layer_b: bool = False,
        freeze_layer_a: bool = False,
    ):
        """Local PC update for layer_A.B_leak and layer_B.B_leak using captured
        gate / drive / input data from _pc_data_history.

        Prediction error per timestep: e_t = (m_Na · h_Na) - s_NMDA, i.e.
        P_Na (Na channel open probability) minus s_NMDA (slow NMDA gate). P_Na
        is transient spike-shaped; e is bipolar across episode (positive at
        spike onset, negative during sustained/refractory), enabling fire-rate
        self-calibration equilibrium under fixed gate weights.

        Reduction convention (per design spec 2026-05-28):
          - Within each capture: sum over time, divide by B (batch-mean of time-sum).
          - Across captures: sum (over alpha+readback+beta passes for layer A; over
            5 recursions × passes for layer B).

        Args:
          lr: learning rate for the weight update.
          weight_decay: Option-1 anchor (spec 2026-05-29). Plain L2 decay applied to
            layer A B_leak on the PRE-Hebbian weight: w_A <- w_A*(1-weight_decay) + lr*ΔW_A.
            Gives the Hebbian common-mode integrator a restoring force so B_leak cannot
            drift unbounded (the ±91 lockstep). 0.0 = off (no-op).
          ip_leak: Option-2 anchor (spec 2026-05-29). Soft leak on ip_bias (anti-windup):
            Δb = ip_lr*(r*-r_obs) - ip_leak*b. Bounds the controller integrator to the
            fixed point b* = ip_lr*(r*-r_obs)/ip_leak instead of winding up. 0.0 = off.
          weight_clip: NaN-divergence fix (spec 2026-05-29). Max L2 norm of each
            substrate B_leak (layer A vector, layer B scalar): after the local update,
            if ||w|| > weight_clip, renormalize w *= weight_clip/||w||. Bounds u_seq
            (hence h) without weight_decay's proportional shrinkage of small weights —
            preserves the learned read DIRECTION, caps only SCALE. 0.0 = off (no-op).
          modulator: Fix 3 three-factor signal (spec 2026-05-29), shape (B,) per-sample
            signed advantage M = p_target - baseline from the compare head. When provided,
            the Hebbian contrib is weighted per-sample by M before the batch sum
            (ΔW ~ e*input*M) on BOTH layer A and the (now unfrozen) layer B; layer B uses
            the event-gated mode regardless of use_h_upstream. None = unmodulated (current
            two-factor behavior; layer B stays frozen under ip_enable).
          use_h_upstream: if True, layer B PC update uses the preceding layer's
            continuous h_seq (shape B, L, 1) instead of the binary spike train
            (input_seq) as the multiplier signal. Layer A update is unchanged.
            For recursion 0, preceding = layer A (h_seq). For recursion k>=1,
            preceding = layer_B_recursions[k-1] (h_seq). Falls back to B's own
            input_seq when layer_A capture is None (threshold-A ablation mode).
          freeze_layer_b: if True, skip the Layer B B_leak update (hand-set frozen
            base counter, spec 2026-06-02).
          freeze_layer_a: if True, skip BOTH the Layer A B_leak update AND the IP
            (ip_bias) update — Layer A is fully frozen (e.g. a forward detector loaded
            via --init-from-ckpt). Full frozen-cascade mode pairs it with freeze_layer_b
            (spec 2026-06-02). False = off (no-op).

        Idempotent within a step. Run under @torch.no_grad — does NOT use any .grad.
        Empty history is a no-op (safe to call when forward used capture_for_pc=False).
        """
        if not self._pc_data_history:
            return
        _diag = os.environ.get("V35_PC_DIAG")  # env-gated divergence instrumentation
        _max_up = 0.0   # max |upstream_signal| feeding layer B update
        _max_hA = 0.0   # max |layer A h_seq|
        _max_e = 0.0    # max |error| feeding layer B
        _hA_min = 0.0   # signed min of layer A h_seq
        _hA_max = 0.0   # signed max of layer A h_seq
        _hB_min = 0.0   # signed min of layer B h_seq (any recursion)
        _hB_max = 0.0   # signed max of layer B h_seq
        _spk_A = 0.0    # mean layer A spike count per sample (positive-fire events)
        delta_W_A = None  # shape (d_input_A,) when populated
        delta_W_B = None  # shape (1,) — layer B has d_input=1
        for pass_data in self._pc_data_history:
            A = pass_data.get("layer_A")
            if A is not None:
                m_Na, h_Na, n_K, s_NMDA = A["gate_seq"].unbind(dim=-1)  # each (B, L)
                P_Na = m_Na * h_Na                                       # Na channel open prob (B, L)
                e = (P_Na - s_NMDA) * A["spike_in"]                     # (B, L)
                if modulator is not None:
                    e = e * modulator[:, None]                          # fix3: per-sample three-factor weight
                # (1/B) · Σ_{b, t} e[b, t] · input_seq[b, t, d]   for each d
                contrib = (e.unsqueeze(-1) * A["input_seq"]).sum(dim=(0, 1)) / e.shape[0]
                delta_W_A = contrib if delta_W_A is None else delta_W_A + contrib
                if _diag and A.get("h_seq") is not None:
                    _hs = A["h_seq"]
                    _max_hA = max(_max_hA, float(_hs.abs().max()))
                    _hA_min = min(_hA_min, float(_hs.min()))
                    _hA_max = max(_hA_max, float(_hs.max()))
                    # layer A "fires" where h >= theta_a (one-sided). Count per-sample mean.
                    _th_a = float(self.cascade.layer_A._theta())
                    _spk_A = max(_spk_A, float((_hs >= _th_a).float().sum(dim=1).mean()))
            layer_B_caps = pass_data.get("layer_B_recursions", [])
            for k, B_cap in enumerate(layer_B_caps):
                m_Na, h_Na, n_K, s_NMDA = B_cap["gate_seq"].unbind(dim=-1)
                P_Na = m_Na * h_Na
                raw_e = P_Na - s_NMDA                                       # (B, L)
                if modulator is not None:
                    # fix3 three-factor: event-gated (avoid h_upstream divergence),
                    # per-sample modulated. Overrides use_h_upstream.
                    e = raw_e * B_cap["spike_in"] * modulator[:, None]
                    upstream_signal = B_cap["input_seq"]
                elif use_h_upstream:
                    # h_upstream mode: continuous signal, no binary event gating.
                    # spike_in mask (derived from binary input spike train) zeroes
                    # the update during cold-start (cascade not yet firing), which
                    # masked the h_upstream change in v1 smoke. Drop the mask so
                    # PC accumulates Hebbian-style across all L timesteps using
                    # continuous h_upstream as multiplier. Layer A's update path
                    # (above) still uses spike_in because its input (CNN feat) is
                    # essentially always non-zero, making the mask a no-op there.
                    e = raw_e
                    if k == 0:
                        # Preceding layer is layer A; fall back to spike-train if unavailable
                        upstream_signal = A["h_seq"] if A is not None else B_cap["input_seq"]
                    else:
                        upstream_signal = layer_B_caps[k - 1]["h_seq"]
                else:
                    # Legacy event-gated mode: spike_in mask multiplies error
                    # so update only fires at binary spike timesteps. Correct for
                    # STDP-style "presynaptic event triggers plasticity" semantics.
                    e = raw_e * B_cap["spike_in"]
                    upstream_signal = B_cap["input_seq"]
                # upstream_signal: (B, L, 1) in both branches
                contrib = (e.unsqueeze(-1) * upstream_signal).sum(dim=(0, 1)) / e.shape[0]
                delta_W_B = contrib if delta_W_B is None else delta_W_B + contrib
                if _diag:
                    _max_up = max(_max_up, float(upstream_signal.abs().max()))
                    _max_e = max(_max_e, float(e.abs().max()))
                    if B_cap.get("h_seq") is not None:
                        _hB_min = min(_hB_min, float(B_cap["h_seq"].min()))
                        _hB_max = max(_hB_max, float(B_cap["h_seq"].max()))
        if _diag:
            type(self)._pc_diag_step = getattr(type(self), "_pc_diag_step", 0) + 1
            wA = self.cascade.layer_A.B_leak.weight
            wB = self.cascade.layer_B.B_leak.weight
            _mstr = (f"M=[{float(modulator.mean()):.3e},{float(modulator.std()):.3e}]"
                     if modulator is not None else "M=None")
            print(
                f"[v35-pc-diag] step={type(self)._pc_diag_step} "
                f"wA_norm={float(wA.norm()):.4e} wB={float(wB.reshape(-1)[0]):.6e} "
                f"dWA={float(delta_W_A.norm()) if delta_W_A is not None else 0.0:.4e} "
                f"dWB={float(delta_W_B.reshape(-1)[0]) if delta_W_B is not None else 0.0:.4e} "
                f"max|up|={_max_up:.4e} max|e|={_max_e:.4e} "
                f"hA=[{_hA_min:.3e},{_hA_max:.3e}] spkA={_spk_A:.3f} "
                f"hB=[{_hB_min:.3e},{_hB_max:.3e}] "
                f"ip_bias_A={float(self.cascade.layer_A.ip_bias):.4e} wd={weight_decay:.3e} {_mstr} "
                f"wA_nan={bool(torch.isnan(wA).any())} wB_nan={bool(torch.isnan(wB).any())}",
                flush=True,
            )
        if delta_W_A is not None and not freeze_layer_a:
            # Option-1 anchor: decay the (pre-Hebbian) weight, then the retained
            # Hebbian step. Order matters — decay must hit w_0, not the post-Hebbian
            # weight, to give w_A <- w_A*(1-weight_decay) + lr*ΔW_A (spec 2026-05-29).
            w_A = self.cascade.layer_A.B_leak.weight.data
            if weight_decay > 0.0:
                w_A.mul_(1.0 - weight_decay)
            w_A.add_(lr * delta_W_A.unsqueeze(0))
            if weight_clip > 0.0:
                _nA = w_A.norm()
                if float(_nA) > weight_clip:
                    w_A.mul_(weight_clip / _nA)  # renorm to cap, preserve direction
        # Layer B update. Frozen under ip_enable (pilot, spec 2026-05-28) UNLESS fix-3
        # three-factor is active (modulator provided) — then unfreeze and apply the
        # modulated update with the same weight-decay anchor as layer A (spec 2026-05-29).
        if delta_W_B is not None and (modulator is not None or not ip_enable) and not freeze_layer_b:
            w_B = self.cascade.layer_B.B_leak.weight.data
            if weight_decay > 0.0:
                w_B.mul_(1.0 - weight_decay)
            w_B.add_(lr * delta_W_B.unsqueeze(0))
            if weight_clip > 0.0:
                _nB = w_B.norm()
                if float(_nB) > weight_clip:
                    w_B.mul_(weight_clip / _nB)  # renorm to cap, preserve sign

        # Homeostatic intrinsic-plasticity update on Layer A ip_bias (negative
        # feedback toward target avg firing rate r_target). r_obs = Layer A batch-mean
        # spike count on the alpha pass (history[0]); falls back to 0 if unavailable.
        if ip_enable and r_target is not None and not freeze_layer_a:
            # history is non-empty here (pc_step early-returns on empty history).
            # history[0] is the alpha pass; fall back to r_obs=0 if layer_A capture
            # is absent (threshold-layer-A ablation path stores layer_A=None).
            first_A = self._pc_data_history[0].get("layer_A")
            r_obs = (
                float(first_A["spike_seq"].sum(dim=1).mean())
                if first_A is not None and first_A.get("spike_seq") is not None
                else 0.0
            )
            # Option-2 anchor: soft leak (anti-windup). delta = ip_lr*(r*-r_obs) - ip_leak*b.
            b_cur = float(self.cascade.layer_A.ip_bias)
            delta_ip = ip_lr * (r_target - r_obs) - ip_leak * b_cur
            self.cascade.layer_A.ip_bias.data.add_(delta_ip)
            if ip_bias_clamp > 0.0:
                self.cascade.layer_A.ip_bias.data.clamp_(-ip_bias_clamp, ip_bias_clamp)
            if _diag:
                # b_star = analytic leak fixed point (inf when leak off); shown to confirm
                # the controller integrator is anchored rather than winding up.
                b_star = (ip_lr * (r_target - r_obs) / ip_leak) if ip_leak > 0.0 else float("inf")
                print(
                    f"[v35-pc-diag-ip] r_obs={r_obs:.3f} r*={r_target:.3f} "
                    f"d_ip={delta_ip:.4e} ip_leak={ip_leak:.3e} b*={b_star:.4e} "
                    f"ip_bias_A={float(self.cascade.layer_A.ip_bias):.4e}",
                    flush=True,
                )

    def forward(self, x_seq: torch.Tensor, capture_for_pc: bool = False):
        if capture_for_pc:
            out = self.cascade(x_seq, capture_for_pc=True)
            residuals, spike_trains_B, spike_train_A, h_seqs_B, capture_data = out
            self._pc_data_history.append(capture_data)
        else:
            residuals, spike_trains_B, spike_train_A, h_seqs_B = self.cascade(x_seq)
        aux = {
            "spike_trains_B": spike_trains_B,
            "spike_train_A": spike_train_A,
            "h_seqs_B": h_seqs_B,
            "theta": self.cascade.theta(),
        }
        return residuals, aux

    def forward_with_external_layer_a_spike(
        self, spike_train_A: torch.Tensor, capture_for_pc: bool = False
    ):
        """Ablation path: bypass HH-based layer A; consume an externally-supplied
        binary spike train (e.g. from raw-signal threshold detector) and run only
        layer B's n_recursions cascade. For testing cascade carry math in isolation.

        spike_train_A: (B, L) — binary {0,1} (caller's responsibility to threshold)
        capture_for_pc: if True, append capture_data to self._pc_data_history.
                        layer_A capture is None (no HH-A here); only layer_B_recursions.

        Returns: (residuals (B, n_recursions), aux dict) matching forward().
        """
        cur = spike_train_A.unsqueeze(-1)  # (B, L, 1)
        residuals_list = []
        spike_trains_B = []
        h_seqs_B = []
        capture_B_list: List[dict] = []
        for _k in range(self.cascade.n_recursions):
            if capture_for_pc:
                out = self.cascade.layer_B(cur, capture_for_pc=True)
                h_seq, spike_seq, residual, _state, capture_B = out
                capture_B_list.append(capture_B)
            else:
                h_seq, spike_seq, residual, _state = self.cascade.layer_B(cur)
            residuals_list.append(residual)
            spike_trains_B.append(spike_seq)
            h_seqs_B.append(h_seq)
            cur = spike_seq.unsqueeze(-1)
        residuals = torch.stack(residuals_list, dim=-1)
        if capture_for_pc:
            self._pc_data_history.append({
                "layer_A": None,
                "layer_B_recursions": capture_B_list,
            })
        aux = {
            "spike_trains_B": spike_trains_B,
            "spike_train_A": spike_train_A,
            "h_seqs_B": h_seqs_B,
            "theta": self.cascade.theta(),
        }
        return residuals, aux

    def theta(self) -> torch.Tensor:
        return self.cascade.theta()
