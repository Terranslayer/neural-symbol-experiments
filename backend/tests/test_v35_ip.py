"""V35-PC homeostatic intrinsic-plasticity (IP) bias tests.

All tests run on the pod (project rule: feedback_tests_on_pod).
Spec: docs/design.md
"""
import numpy as np
import torch
from backend.core.v35 import HHSSMLayer1D, V35Substrate


def _bump_input(L=60, n_bumps=8, d_input=1):
    """Fixed input with n_bumps unit pulses spaced across L timesteps."""
    x = torch.zeros(1, L, d_input)
    positions = np.linspace(0, L - 1, n_bumps, dtype=int)
    x[0, positions.tolist(), 0] = 1.0
    return x


def test_ip_bias_raises_firing_monotonically():
    """Sweeping ip_bias up must not decrease Layer A spike count (ip_bias is the
    knob that pulls h toward the firing regime).

    Mean h is intentionally NOT asserted: in a fire-and-reset system it is
    non-monotone in ip_bias (high bias fires+resets every step, keeping mean
    post-reset h near 0; low bias lets h accumulate between sparse fires).
    Only spike-count monotonicity is checked.
    """
    torch.manual_seed(0)
    layer = HHSSMLayer1D(d_input=1, biological_init=True, theta_fixed=1.0)
    # Make B_leak a known small positive so input alone doesn't dominate the sweep.
    layer.B_leak.weight.data.fill_(0.2)
    x = _bump_input()

    counts = []
    for b in [-2.0, -0.5, 0.5, 2.0, 4.0]:
        layer.ip_bias.data.fill_(b)
        h_seq, spike_seq, _residual, _state = layer(x)
        counts.append(float(spike_seq.sum()))

    assert all(counts[i] <= counts[i + 1] + 1e-6 for i in range(len(counts) - 1)), (
        f"Layer A spike count not monotone non-decreasing in ip_bias: {counts}"
    )
    assert counts[-1] > counts[0], f"raising ip_bias produced no extra firing: {counts}"


def test_capture_carries_layer_a_spike_seq():
    """pc_step needs Layer A's OUTPUT spike train to compute r_obs.
    The per-layer capture dict must include 'spike_seq' (B, L)."""
    layer = HHSSMLayer1D(d_input=1, biological_init=True, theta_fixed=1.0)
    x = _bump_input()
    out = layer(x, capture_for_pc=True)
    assert len(out) == 5, "capture_for_pc forward must return 5-tuple"
    _h_seq, spike_seq, _residual, _state, capture = out
    assert "spike_seq" in capture, "capture dict missing 'spike_seq'"
    assert capture["spike_seq"].shape == spike_seq.shape == (1, 60)
    assert torch.allclose(capture["spike_seq"], spike_seq)


def _alpha_count(sub):
    """Layer A batch-mean spike count on the most recent single forward."""
    cap = sub._pc_data_history[0]["layer_A"]
    return float(cap["spike_seq"].sum(dim=1).mean())


def test_ip_negative_feedback_direction():
    """Delta ip_bias < 0 when r_obs > r*, and > 0 when r_obs < r*."""
    torch.manual_seed(1)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.B_leak.weight.data.fill_(0.2)
    # Drive Layer A to fire a fair amount by setting a positive ip_bias.
    sub.cascade.layer_A.ip_bias.data.fill_(2.0)
    x = _bump_input()

    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    r_obs = _alpha_count(sub)
    assert r_obs > 0, f"need Layer A firing to test direction; r_obs={r_obs}"

    # r* far BELOW r_obs -> bias should decrease.
    b0 = sub.cascade.layer_A.ip_bias.item()
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.1, r_target=r_obs - 5.0)
    assert sub.cascade.layer_A.ip_bias.item() < b0, "r_obs>r* should LOWER ip_bias"

    # Re-capture, then r* far ABOVE r_obs -> bias should increase.
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    r_obs2 = _alpha_count(sub)
    b1 = sub.cascade.layer_A.ip_bias.item()
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.1, r_target=r_obs2 + 5.0)
    assert sub.cascade.layer_A.ip_bias.item() > b1, "r_obs<r* should RAISE ip_bias"


def test_ip_breaks_no_fire_and_converges():
    """From a deeply negative ip_bias (no firing), IP drives firing up toward r*
    and ip_bias stabilizes (controller converges, does not diverge)."""
    torch.manual_seed(2)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.B_leak.weight.data.fill_(0.2)
    sub.cascade.layer_A.ip_bias.data.fill_(-3.0)  # start: no firing
    x = _bump_input()
    r_star = 4.0

    bias_hist, r_hist = [], []
    for _ in range(150):
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        r_hist.append(_alpha_count(sub))
        sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.05, r_target=r_star)
        bias_hist.append(sub.cascade.layer_A.ip_bias.item())

    assert r_hist[0] == 0.0, f"expected no firing at start, got {r_hist[0]}"
    assert max(r_hist) > 0.0, "IP never produced any firing (deadlock not broken)"
    # ip_bias finite + stabilized (no divergence)
    assert np.isfinite(bias_hist[-1]), f"ip_bias diverged: {bias_hist[-1]}"
    # steady-state bias oscillation is bounded by ip_lr*|r_obs-r*|; with ip_lr=0.05
    # this is small, so std<0.5 over the last 20 steps confirms convergence (recalibrate
    # this threshold if ip_lr or r_star change).
    assert float(np.std(bias_hist[-20:])) < 0.5, (
        f"ip_bias not stabilizing (last-20 std={np.std(bias_hist[-20:]):.3f})"
    )
    # r_obs ends closer to r* than the (zero) start
    assert abs(r_hist[-1] - r_star) < abs(r_hist[0] - r_star), (
        f"r_obs did not move toward r* (start {r_hist[0]}, end {r_hist[-1]}, r*={r_star})"
    )


def test_pilot_freezes_layer_b():
    """With ip_enable=True (pilot), Layer B B_leak and ip_bias are NOT updated."""
    torch.manual_seed(3)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.ip_bias.data.fill_(1.0)
    b_leak_before = sub.cascade.layer_B.B_leak.weight.detach().clone()
    b_ipbias_before = sub.cascade.layer_B.ip_bias.detach().clone()
    x = _bump_input()
    for _ in range(5):
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        sub.pc_step(lr=0.1, ip_enable=True, ip_lr=0.05, r_target=4.0)
    assert torch.allclose(sub.cascade.layer_B.B_leak.weight, b_leak_before), (
        "pilot must not update Layer B B_leak"
    )
    assert torch.allclose(sub.cascade.layer_B.ip_bias, b_ipbias_before), (
        "pilot must not update Layer B ip_bias"
    )


def test_ip_bias_is_substrate_param():
    """ip_bias must be a registered substrate parameter so the SGD-exclusion
    logic (which uses v35_substrate.parameters()) auto-excludes it."""
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    names = [n for n, _ in sub.named_parameters()]
    assert any("ip_bias" in n for n in names), f"ip_bias not a substrate param: {names}"


# ---------------------------------------------------------------------------
# Option 1 (B_leak plain weight decay) + Option 2 (ip_bias soft leak)
# Spec: docs/design.md
# ---------------------------------------------------------------------------


def test_weight_decay_shrinks_b_leak_with_zero_lr():
    """With lr=0 the Hebbian term vanishes, so weight decay alone applies:
    w_A *= (1 - weight_decay). Anchors the B_leak common mode toward 0."""
    torch.manual_seed(4)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.B_leak.weight.data.fill_(0.5)
    x = _bump_input()
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    w0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    wd = 0.1
    sub.pc_step(lr=0.0, weight_decay=wd)
    w1 = sub.cascade.layer_A.B_leak.weight
    assert torch.allclose(w1, w0 * (1.0 - wd), atol=1e-6), (
        f"weight decay alone should give w*(1-wd): w0={w0.item()}, w1={w1.item()}"
    )


def test_weight_decay_composes_with_hebbian():
    """w_A update == lr*Hebbian - weight_decay*w0. Two runs from identical state
    and input share the same Hebbian dW; their difference is exactly -wd*w0."""

    def run(wd):
        torch.manual_seed(7)  # before construction -> identical E init -> identical dW
        sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
        sub.cascade.layer_A.B_leak.weight.data.fill_(0.5)
        sub.cascade.layer_A.ip_bias.data.fill_(1.5)  # some firing -> nonzero dW
        x = _bump_input()
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        w0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
        sub.pc_step(lr=0.3, weight_decay=wd)
        return w0, sub.cascade.layer_A.B_leak.weight.detach().clone()

    w0a, wa = run(0.0)        # wa = w0 + lr*dW
    w0b, wb = run(0.2)        # wb = w0 + lr*dW - wd*w0
    assert torch.allclose(w0a, w0b), "runs must start from identical weights"
    assert torch.allclose(wb, wa - 0.2 * w0b, atol=1e-6), (
        f"weight decay must compose additively: wa={wa.item()}, wb={wb.item()}"
    )


def test_ip_leak_decays_bias_at_zero_control_error():
    """When r_target == r_obs the control term is 0, so the leak alone applies:
    ip_bias *= (1 - ip_leak)."""
    torch.manual_seed(5)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.B_leak.weight.data.fill_(0.2)
    sub.cascade.layer_A.ip_bias.data.fill_(3.0)
    x = _bump_input()
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    r_obs = _alpha_count(sub)
    b0 = sub.cascade.layer_A.ip_bias.item()
    lk = 0.1
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.05, r_target=r_obs, ip_leak=lk)
    b1 = sub.cascade.layer_A.ip_bias.item()
    assert abs(b1 - b0 * (1.0 - lk)) < 1e-5, (
        f"ip leak at zero error should give b*(1-leak): b0={b0}, b1={b1}"
    )


def test_ip_leak_bounds_windup():
    """Persistent unreachable target: WITHOUT leak ip_bias winds up unbounded;
    WITH leak it converges to the fixed point b* = ip_lr*(r*-r_obs)/ip_leak.

    theta is set huge so Layer A never fires -> r_obs is pinned at 0, decoupling
    the controller from the firing feedback (pure constant-error controller test).
    A single capture is reused across pc_step calls (r_obs stays 0)."""
    ip_lr, r_star, lk = 0.05, 10.0, 0.01
    x = _bump_input()

    def controller(leak):
        sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True,
                           theta_a=1000.0, theta_b=1000.0)
        sub.cascade.layer_A.B_leak.weight.data.fill_(0.2)
        sub.cascade.layer_A.ip_bias.data.fill_(0.0)
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        assert _alpha_count(sub) == 0.0, "theta=1000 should suppress all firing"
        for _ in range(800):
            sub.pc_step(lr=0.0, ip_enable=True, ip_lr=ip_lr, r_target=r_star, ip_leak=leak)
        return sub.cascade.layer_A.ip_bias.item()

    b_leaky = controller(lk)
    b_noleak = controller(0.0)
    b_star = ip_lr * r_star / lk  # = 50.0
    assert np.isfinite(b_leaky), f"leaky controller diverged: {b_leaky}"
    assert abs(b_leaky - b_star) < 1.5, (
        f"leaky ip_bias should converge to b*={b_star}, got {b_leaky}"
    )
    assert b_noleak > 100.0, (
        f"no-leak controller should wind up unbounded (~400), got {b_noleak}"
    )


def test_new_terms_default_to_noop():
    """Passing weight_decay=0 and ip_leak=0 explicitly must equal the default
    (no-kwargs) call — guards that the new terms are no-ops at default."""

    def run(explicit):
        torch.manual_seed(11)
        sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
        sub.cascade.layer_A.B_leak.weight.data.fill_(0.4)
        sub.cascade.layer_A.ip_bias.data.fill_(1.2)
        x = _bump_input()
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        r_obs = _alpha_count(sub)
        if explicit:
            sub.pc_step(lr=0.2, ip_enable=True, ip_lr=0.05, r_target=r_obs + 3.0,
                        weight_decay=0.0, ip_leak=0.0)
        else:
            sub.pc_step(lr=0.2, ip_enable=True, ip_lr=0.05, r_target=r_obs + 3.0)
        return (sub.cascade.layer_A.B_leak.weight.detach().clone(),
                sub.cascade.layer_A.ip_bias.item())

    w_default, b_default = run(False)
    w_explicit, b_explicit = run(True)
    assert torch.allclose(w_default, w_explicit, atol=1e-7), "weight_decay=0 not a no-op"
    assert abs(b_default - b_explicit) < 1e-7, "ip_leak=0 not a no-op"


def test_weight_decay_does_not_touch_layer_b():
    """In the pilot (ip_enable) Layer B is frozen; weight decay (applied only on
    the Layer A Hebbian path) must not modify Layer B B_leak either."""
    torch.manual_seed(6)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.ip_bias.data.fill_(1.0)
    b_leak_before = sub.cascade.layer_B.B_leak.weight.detach().clone()
    x = _bump_input()
    for _ in range(5):
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        sub.pc_step(lr=0.1, ip_enable=True, ip_lr=0.05, r_target=4.0,
                    weight_decay=1e-2, ip_leak=0.01)
    assert torch.allclose(sub.cascade.layer_B.B_leak.weight, b_leak_before), (
        "weight decay must not touch Layer B B_leak in the pilot"
    )
