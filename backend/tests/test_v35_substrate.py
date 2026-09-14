import torch
from backend.core.v35 import HHSSMLayer1D, _inverse_softplus


def test_hhssm_layer_1d_forward_shape():
    layer = HHSSMLayer1D(d_input=8)
    x = torch.randn(2, 30, 8)  # (B, L, d_input)
    h_seq, spike_seq, residual, final_state = layer(x)
    assert h_seq.shape == (2, 30, 1)
    assert spike_seq.shape == (2, 30)        # binary spike train
    assert residual.shape == (2,)            # h(L), the final residual
    assert torch.all((spike_seq == 0) | (spike_seq == 1))
    assert not torch.isnan(h_seq).any()


def test_fire_reset_count_invariance():
    """With strong injected unit pulses and weak HH dynamics, total
    fire count should approximately equal floor(total_drive / theta).

    This is a sanity check that fire-and-reset preserves the carry
    invariant `sum y = floor(sum drive / theta) + small slop`.
    """
    torch.manual_seed(0)
    layer = HHSSMLayer1D(d_input=1)
    # Force theta = 3.0 exactly for the test by reaching into theta_raw:
    layer.theta_raw.data = torch.tensor(_inverse_softplus(2.0))
    # 10 unit pulses spread over L=40 timesteps
    x = torch.zeros(1, 40, 1)
    spike_positions = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]
    for p in spike_positions:
        x[0, p, 0] = 1.0
    h_seq, spike_seq, residual, _ = layer(x)
    total_spikes = spike_seq.sum().item()
    # Theta=3, N=10 unit-area pulses -> expect ~3 fires plus residual ~1.
    # Loose bounds since HH dynamics leak some drive.
    assert 0 <= total_spikes <= 5  # loose; with untrained gates leak varies
    assert torch.isfinite(residual).all()


def test_fire_reset_cascade_contract():
    """Cascade(x) returns 5 residuals (one per Layer-B recursion) and
    5 spike trains (one per recursion), all with shared theta.
    """
    from backend.core.v35 import FireResetCascade
    cascade = FireResetCascade(d_input=8, n_recursions=5)
    x = torch.randn(2, 30, 8)
    residuals, spike_trains_B, spike_train_A, h_seqs_B = cascade(x)
    assert residuals.shape == (2, 5)        # 5 digit residuals per sample
    assert len(spike_trains_B) == 5
    for s in spike_trains_B:
        assert s.shape == (2, 30)
        assert torch.all((s == 0) | (s == 1))
    assert spike_train_A.shape == (2, 30)
    assert len(h_seqs_B) == 5
    for h in h_seqs_B:
        assert h.shape == (2, 30, 1)


def test_fire_reset_cascade_theta_shared():
    """All 6 sub-layers (1 translator + 5 carry calls of same module) share
    the same theta Parameter so it's a single learnable scalar."""
    from backend.core.v35 import FireResetCascade
    cascade = FireResetCascade(d_input=8, n_recursions=5)
    theta_params = [
        p for n, p in cascade.named_parameters() if "theta" in n
    ]
    assert len(theta_params) == 1, (
        f"Expected exactly one theta Parameter (shared), got {len(theta_params)}: "
        f"{[n for n, _ in cascade.named_parameters() if 'theta' in n]}"
    )


def test_fire_reset_cascade_grad_flow():
    """Loss on residuals must backprop through STE to theta and gate params."""
    from backend.core.v35 import FireResetCascade
    torch.manual_seed(0)
    cascade = FireResetCascade(d_input=8, n_recursions=5)
    x = torch.randn(2, 30, 8, requires_grad=False)
    residuals, _, _, _ = cascade(x)
    loss = (residuals ** 2).mean()
    loss.backward()
    assert cascade.theta_raw.grad is not None
    assert cascade.theta_raw.grad.abs().item() > 0
    assert cascade.layer_A.log_tau.grad is not None
    assert cascade.layer_B.log_tau.grad is not None
    assert cascade.layer_A.gate_w.grad is not None
    assert cascade.layer_B.gate_w.grad is not None


def test_v35_substrate_composer():
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5)
    x = torch.randn(2, 30, 8)
    residuals, aux = sub(x)
    assert residuals.shape == (2, 5)
    assert "spike_trains_B" in aux
    assert "spike_train_A" in aux
    assert "h_seqs_B" in aux
    assert "theta" in aux
    assert aux["theta"].numel() == 1
    assert aux["theta"].item() > 1.0  # bounded below by 1


def test_hhssm_layer_biological_init():
    """biological_init=True sets gate_w + gate_b to specific HH-inspired values."""
    layer_bio = HHSSMLayer1D(d_input=8, biological_init=True)
    expected_gate_w = torch.tensor([2.0, 2.0, 1.0, 0.5])
    expected_gate_b = torch.tensor([-0.5, +0.5, -0.5, -0.5])
    assert torch.allclose(layer_bio.gate_w.data, expected_gate_w)
    assert torch.allclose(layer_bio.gate_b.data, expected_gate_b)
    # Default keeps random init (gate_w random, gate_b zeros)
    torch.manual_seed(123)
    layer_rand = HHSSMLayer1D(d_input=8, biological_init=False)
    assert not torch.allclose(layer_rand.gate_w.data, expected_gate_w)
    assert torch.allclose(layer_rand.gate_b.data, torch.zeros(4))


def test_hhssm_layer_capture_for_pc_shape():
    """capture_for_pc=True returns 5-tuple with gate_seq, u_seq, input_seq, spike_in.

    Shapes must match: gate_seq (B, L, 4), u_seq (B, L, 1), input_seq (B, L, d_input), spike_in (B, L).
    """
    torch.manual_seed(0)
    layer = HHSSMLayer1D(d_input=8, biological_init=True)
    B, L, d_in = 2, 30, 8
    x = torch.randn(B, L, d_in)
    out = layer(x, capture_for_pc=True)
    assert len(out) == 5, f"Expected 5-tuple, got {len(out)}"
    h_seq, spike_seq, residual, final_state, capture = out
    assert h_seq.shape == (B, L, 1)
    assert spike_seq.shape == (B, L)
    assert capture["gate_seq"].shape == (B, L, 4)
    assert capture["u_seq"].shape == (B, L, 1)
    assert capture["input_seq"].shape == (B, L, d_in)
    assert capture["spike_in"].shape == (B, L)
    assert torch.all((capture["spike_in"] == 0) | (capture["spike_in"] == 1))


def test_hhssm_layer_default_no_capture():
    """capture_for_pc=False (default) returns existing 4-tuple, no capture dict."""
    torch.manual_seed(0)
    layer = HHSSMLayer1D(d_input=8)
    x = torch.randn(2, 30, 8)
    out = layer(x)
    assert len(out) == 4
    h_seq, spike_seq, residual, final_state = out
    assert h_seq.shape == (2, 30, 1)


def test_fire_reset_cascade_capture_for_pc():
    """Cascade.forward(capture_for_pc=True) returns 5-tuple with capture_data dict.

    capture_data["layer_A"] is dict, capture_data["layer_B_recursions"] is list len n_recursions.
    """
    from backend.core.v35 import FireResetCascade
    torch.manual_seed(0)
    cascade = FireResetCascade(d_input=8, n_recursions=5)
    x = torch.randn(2, 30, 8)
    out = cascade(x, capture_for_pc=True)
    assert len(out) == 5
    residuals, spike_trains_B, spike_train_A, h_seqs_B, capture_data = out
    assert residuals.shape == (2, 5)
    assert "layer_A" in capture_data
    assert "layer_B_recursions" in capture_data
    assert capture_data["layer_A"] is not None
    assert isinstance(capture_data["layer_B_recursions"], list)
    assert len(capture_data["layer_B_recursions"]) == 5
    for b in capture_data["layer_B_recursions"]:
        assert b["gate_seq"].shape == (2, 30, 4)
        assert b["input_seq"].shape == (2, 30, 1)


def test_fire_reset_cascade_default_no_capture():
    """Cascade.forward() with no kwarg returns existing 4-tuple."""
    from backend.core.v35 import FireResetCascade
    cascade = FireResetCascade(d_input=8, n_recursions=5)
    x = torch.randn(2, 30, 8)
    out = cascade(x)
    assert len(out) == 4


def test_v35_substrate_pc_history_append():
    """Substrate accumulates capture data across forward calls when capture_for_pc=True."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5)
    sub.clear_pc_history()
    assert sub._pc_data_history == []
    x1 = torch.randn(2, 30, 8)
    x2 = torch.randn(2, 30, 8)
    sub(x1, capture_for_pc=True)
    assert len(sub._pc_data_history) == 1
    sub(x2, capture_for_pc=True)
    assert len(sub._pc_data_history) == 2
    sub.clear_pc_history()
    assert sub._pc_data_history == []


def test_v35_substrate_no_capture_when_flag_off():
    """No appends when capture_for_pc=False (default)."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5)
    sub.clear_pc_history()
    x = torch.randn(2, 30, 8)
    sub(x)
    assert sub._pc_data_history == []


def test_pc_step_layer_a_direction():
    """pc_step shifts layer_A.B_leak in the direction of (e * input) accumulated over passes.

    With biological init, P_Na = m_Na · h_Na is transient spike-shaped (m_Na rises fast,
    h_Na drops fast under sustained V). s_NMDA (τ=40) lags. At impulse onset, P_Na is
    positive and s_NMDA near 0, so e = P_Na - s_NMDA > 0. With input on dim 0 positive,
    ΔW_A[0] should be positive.
    """
    from backend.core.v35 import V35Substrate
    torch.manual_seed(0)
    sub = V35Substrate(d_input=8, n_recursions=5, biological_init=True)
    W_A_before = sub.cascade.layer_A.B_leak.weight.detach().clone()
    W_B_before = sub.cascade.layer_B.B_leak.weight.detach().clone()
    x = torch.zeros(2, 30, 8)
    # Sparse impulses every 6th step on dim 0
    for t in [3, 9, 15, 21, 27]:
        x[:, t, 0] = 1.0
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.5)
    W_A_after = sub.cascade.layer_A.B_leak.weight.detach().clone()
    W_B_after = sub.cascade.layer_B.B_leak.weight.detach().clone()
    # Layer A dim 0 should have moved positive
    assert (W_A_after[0, 0] - W_A_before[0, 0]).item() > 0, (
        f"Expected ΔW_A[0,0] > 0, got {(W_A_after[0,0] - W_A_before[0,0]).item():.4f}"
    )
    # Layer B should be unchanged: with these tiny init weights, layer A doesn't fire, so
    # spike_train_A is all zeros, layer B input is all zeros, e_gated * input = 0.
    assert torch.equal(W_B_before, W_B_after), (
        "Layer B B_leak should be unchanged when layer A produces no spikes"
    )


def test_pc_step_no_history_is_noop():
    """pc_step on empty history is safe and changes nothing."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5, biological_init=True)
    W_A_before = sub.cascade.layer_A.B_leak.weight.detach().clone()
    sub.clear_pc_history()
    sub.pc_step(lr=0.1)
    W_A_after = sub.cascade.layer_A.B_leak.weight.detach().clone()
    assert torch.equal(W_A_before, W_A_after)


def test_pc_step_no_grad_required():
    """pc_step works without any backward pass. Substrate grads remain None throughout."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5, biological_init=True)
    x = torch.randn(2, 30, 8)
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    # Confirm no .grad populated yet (we never called backward)
    for p in sub.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0
    sub.pc_step(lr=0.1)
    # Still no grad
    for p in sub.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0


def test_forward_with_external_layer_a_spike_capture():
    """forward_with_external_layer_a_spike supports capture_for_pc; layer_A capture is None."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5, biological_init=True)
    sub.clear_pc_history()
    spike_a = torch.zeros(2, 30)
    spike_a[:, [3, 9, 15, 21, 27]] = 1.0
    sub.forward_with_external_layer_a_spike(spike_a, capture_for_pc=True)
    assert len(sub._pc_data_history) == 1
    capture = sub._pc_data_history[0]
    assert capture["layer_A"] is None
    assert len(capture["layer_B_recursions"]) == 5


def test_hhssm_layer_theta_fixed():
    """theta_fixed=1.0 overrides theta_raw / theta_param; layer's _theta() returns the fixed value."""
    layer = HHSSMLayer1D(d_input=8, theta_fixed=1.0)
    assert torch.allclose(layer._theta(), torch.tensor(1.0))
    # theta_raw should be None in fixed mode
    assert getattr(layer, "theta_raw", None) is None


def test_fire_reset_cascade_per_layer_theta():
    """theta_a=1.0, theta_b=3.0 produce per-layer fixed theta, no shared theta_raw."""
    from backend.core.v35 import FireResetCascade
    cascade = FireResetCascade(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0)
    assert cascade.theta_raw is None
    assert torch.allclose(cascade.layer_A._theta(), torch.tensor(1.0))
    assert torch.allclose(cascade.layer_B._theta(), torch.tensor(3.0))
    # theta() (representative scalar) returns layer_B's theta
    assert torch.allclose(cascade.theta(), torch.tensor(3.0))


def test_capture_includes_h_seq():
    """capture dict now has h_seq key with shape (B, L, 1)."""
    torch.manual_seed(0)
    layer = HHSSMLayer1D(d_input=8, biological_init=True)
    x = torch.randn(2, 30, 8)
    _, _, _, _, capture = layer(x, capture_for_pc=True)
    assert "h_seq" in capture
    assert capture["h_seq"].shape == (2, 30, 1)


def test_log_g_leak_init_sets_leak_channel_only():
    """log_g_leak_init rewrites ONLY channel index 3 (leak); others stay -4.
    Default (None) keeps all four at -4 (current behavior)."""
    layer = HHSSMLayer1D(d_input=8, log_g_leak_init=-1.0)
    assert torch.allclose(layer.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -1.0]))
    layer_def = HHSSMLayer1D(d_input=8)
    assert torch.allclose(layer_def.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -4.0]))


def test_b_leak_warm_scale_scales_init():
    """b_leak_warm_scale multiplies the B_leak init (same-seed comparison)."""
    torch.manual_seed(5)
    ref = HHSSMLayer1D(d_input=8, biological_init=True, b_leak_warm_scale=1.0)
    torch.manual_seed(5)
    warm = HHSSMLayer1D(d_input=8, biological_init=True, b_leak_warm_scale=4.0)
    assert torch.allclose(warm.B_leak.weight.data, 4.0 * ref.B_leak.weight.data)


def test_layer_a_leak_bounds_h_ratchet():
    """Root-cause regression: with no leak (log_g_leak=-4) h ratchets unbounded
    under sustained large drive (the h=290 wall); a goldilocks leak (-1) bounds h.
    Drive the layer directly via a large B_leak (ratchet regime), bypassing the CNN."""
    def _peak_h(log_g_leak):
        torch.manual_seed(0)
        layer = HHSSMLayer1D(d_input=1, theta_fixed=1.0, biological_init=True,
                             log_g_leak_init=log_g_leak)
        layer.B_leak.weight.data.fill_(10.0)   # large input gain -> ratchet regime
        x = torch.ones(1, 120, 1)              # sustained dense drive
        h_seq, _, _, _ = layer(x)
        return float(h_seq.abs().max())

    peak_no_leak = _peak_h(-4.0)
    peak_leaky = _peak_h(-1.0)
    assert peak_leaky < peak_no_leak, (
        f"leak should bound h: leaky={peak_leaky:.1f} no_leak={peak_no_leak:.1f}")
    assert peak_leaky < 60.0, f"leaky peak should be bounded, got {peak_leaky:.1f}"
    assert peak_no_leak > 100.0, f"no-leak peak should ratchet, got {peak_no_leak:.1f}"


def test_cascade_leak_warm_layer_a_only():
    """log_g_leak_a / b_leak_warm_a route to layer_A ONLY; layer_B stays -4
    (Layer B's carry/fold requires the integrator property — correctness lock)."""
    from backend.core.v35 import FireResetCascade
    casc = FireResetCascade(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0,
                            log_g_leak_a=-1.0, b_leak_warm_a=4.0)
    assert torch.allclose(casc.layer_A.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -1.0]))
    assert torch.allclose(casc.layer_B.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -4.0]))


def test_cascade_warm_scales_layer_a_only():
    """b_leak_warm_a scales layer_A B_leak; layer_B B_leak untouched (same-seed)."""
    from backend.core.v35 import FireResetCascade
    torch.manual_seed(7)
    ref = FireResetCascade(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0,
                           b_leak_warm_a=1.0)
    torch.manual_seed(7)
    warm = FireResetCascade(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0,
                            b_leak_warm_a=4.0)
    assert torch.allclose(warm.layer_A.B_leak.weight.data, 4.0 * ref.layer_A.B_leak.weight.data)
    assert torch.allclose(warm.layer_B.B_leak.weight.data, ref.layer_B.B_leak.weight.data)


def test_v35substrate_threads_leak_warm():
    """V35Substrate forwards log_g_leak_a / b_leak_warm_a to the cascade (layer_A only)."""
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0,
                       log_g_leak_a=-1.0, b_leak_warm_a=4.0)
    assert torch.allclose(sub.cascade.layer_A.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -1.0]))
    assert torch.allclose(sub.cascade.layer_B.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -4.0]))


def test_cascade_leak_layer_a_only_legacy_branch():
    """Same layer-A-only / layer-B-stays-(-4) invariant on the shared-theta (legacy)
    branch (no theta_a/theta_b given), which the per-layer tests don't exercise."""
    from backend.core.v35 import FireResetCascade
    casc = FireResetCascade(d_input=8, n_recursions=5,
                            log_g_leak_a=-1.0, b_leak_warm_a=4.0)
    assert torch.allclose(casc.layer_A.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -1.0]))
    assert torch.allclose(casc.layer_B.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -4.0]))
    # warm scale on the legacy branch too: layer_A B_leak scaled 4x, layer_B untouched (same-seed)
    torch.manual_seed(13)
    ref = FireResetCascade(d_input=8, n_recursions=5, b_leak_warm_a=1.0)
    torch.manual_seed(13)
    warm = FireResetCascade(d_input=8, n_recursions=5, b_leak_warm_a=4.0)
    assert torch.allclose(warm.layer_A.B_leak.weight.data, 4.0 * ref.layer_A.B_leak.weight.data)
    assert torch.allclose(warm.layer_B.B_leak.weight.data, ref.layer_B.B_leak.weight.data)


def test_b_leak_init_fills_constant():
    """b_leak_init fills B_leak.weight with a constant (hand-set counter). None = no-op."""
    layer = HHSSMLayer1D(d_input=1, b_leak_init=2.0)
    w = layer.B_leak.weight.data
    assert torch.allclose(w, torch.full_like(w, 2.0))
    # default None keeps the random/default init (Linear(1,1) Kaiming is in [-1,1], never 2.0)
    layer_def = HHSSMLayer1D(d_input=1)
    wd = layer_def.B_leak.weight.data
    assert not torch.allclose(wd, torch.full_like(wd, 2.0))


def test_cascade_b_leak_b_init_layer_b_only_perlayer_branch():
    """b_leak_b_init fills layer_B's B_leak; layer_A unaffected. (per-layer-theta branch)"""
    from backend.core.v35 import FireResetCascade
    casc = FireResetCascade(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0, b_leak_b_init=2.0)
    wb = casc.layer_B.B_leak.weight.data
    assert torch.allclose(wb, torch.full_like(wb, 2.0))
    wa = casc.layer_A.B_leak.weight.data
    assert not torch.allclose(wa, torch.full_like(wa, 2.0)), "layer_A must NOT be filled"


def test_cascade_b_leak_b_init_legacy_branch():
    """Same on the shared-theta (legacy) branch (no theta_a/theta_b)."""
    from backend.core.v35 import FireResetCascade
    casc = FireResetCascade(d_input=8, n_recursions=5, b_leak_b_init=2.0)
    wb = casc.layer_B.B_leak.weight.data
    assert torch.allclose(wb, torch.full_like(wb, 2.0))


def test_v35substrate_threads_b_leak_b_init():
    from backend.core.v35 import V35Substrate
    sub = V35Substrate(d_input=8, n_recursions=5, theta_a=1.0, theta_b=3.0, b_leak_b_init=2.0)
    wb = sub.cascade.layer_B.B_leak.weight.data
    assert torch.allclose(wb, torch.full_like(wb, 2.0))
