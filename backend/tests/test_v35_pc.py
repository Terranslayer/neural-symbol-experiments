"""V35 local-PC integration tests.

test_pc_calibration_layer_b_drift: feed a fixed N=9 spike train into layer B
  via forward_with_external_layer_a_spike, run pc_step in a loop, verify
  B_leak.weight drifts and stabilizes (P_Na - s_NMDA bipolar error -> fixed
  point per design spec 2026-05-28).

test_pc_does_not_modify_other_substrate_params: pc_step only updates B_leak,
  not gate_w / log_g / log_tau / theta_raw / etc.
"""
import math
import numpy as np
import torch
from backend.core.v35 import V35Substrate


def test_pc_calibration_layer_b_drift():
    """100 PC steps on N=9 spike train: weight drifts then stabilizes.

    e_t = P_Na_t - s_NMDA_t where P_Na = m_Na · h_Na. P_Na is spike-transient
    (positive at onset, drops as h_Na closes). s_NMDA τ=40 lags. Episode-averaged
    Σ_t e_t · spike_in should reverse sign as weight grows (more spike events ->
    sustained high V -> P_Na shuts down -> e_late < 0), giving self-balancing fixed point.
    """
    torch.manual_seed(42)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    # Force layer B B_leak weight to a known start so we can measure drift.
    sub.cascade.layer_B.B_leak.weight.data.fill_(0.5)
    L, N = 60, 9
    spike = torch.zeros(1, L)
    positions = np.linspace(0, L - 1, N, dtype=int)
    spike[0, positions.tolist()] = 1.0

    weights_history = []
    for step in range(100):
        sub.clear_pc_history()
        sub.forward_with_external_layer_a_spike(spike, capture_for_pc=True)
        sub.pc_step(lr=0.05)
        weights_history.append(sub.cascade.layer_B.B_leak.weight.item())

    w_start = weights_history[0]
    w_end = weights_history[-1]
    w_recent_std = float(np.std(weights_history[-20:]))

    # 1. Weight moved away from start by more than noise
    assert abs(w_end - 0.5) > 0.01, (
        f"Layer B weight did not drift from init 0.5 (final {w_end:.4f}). "
        "PC update has no effect?"
    )
    # 2. Recent steps stabilized (variance small)
    assert w_recent_std < 0.05, (
        f"Layer B weight still drifting after 100 steps "
        f"(last-20 std = {w_recent_std:.4f}). Not converging."
    )
    # 3. Final weight in plausible calibration range
    assert 0 < w_end < 5.0, f"Final weight {w_end:.4f} out of plausible range"
    print(f"weight trajectory: start={w_start:.4f}, final={w_end:.4f}, std-last-20={w_recent_std:.4f}")


def test_pc_does_not_modify_other_substrate_params():
    """pc_step only touches B_leak, not gate_w / log_g / log_tau / theta_raw / etc."""
    sub = V35Substrate(d_input=8, n_recursions=5, biological_init=True)
    # Snapshot all substrate params EXCEPT B_leak
    snapshots = {}
    for name, p in sub.named_parameters():
        if "B_leak" not in name:
            snapshots[name] = p.detach().clone()
    # Run several PC updates
    x = torch.randn(2, 30, 8)
    for _ in range(10):
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        sub.pc_step(lr=0.5)
    # Verify everything except B_leak unchanged
    for name, p_after in sub.named_parameters():
        if "B_leak" in name:
            continue
        assert torch.equal(snapshots[name], p_after.detach()), (
            f"Param {name} changed during pc_step (expected only B_leak to change)"
        )


def test_pc_step_h_upstream_uses_upstream_h():
    """When use_h_upstream=True, layer B PC update uses preceding layer's h_seq, not spike train.

    Sanity: layer B's PC update direction is DIFFERENT when computed with h_upstream vs spike train
    (when both signals carry distinct information).

    Uses theta_a=0.5, theta_b=0.5 with random gate init so that layer B fires enough
    times to produce a non-zero error signal (spike_in > 0). With theta=3.0 (canonical
    experiment) the cascade is dead-on-arrival so e=0 and both modes give identical
    zero update — that configuration is what h_upstream mode is designed to *fix*,
    but correctness of routing is tested here with a lower threshold where B fires.
    """
    torch.manual_seed(0)
    sub_a = V35Substrate(d_input=8, n_recursions=5, biological_init=False,
                          theta_a=0.5, theta_b=0.5)
    sub_b = V35Substrate(d_input=8, n_recursions=5, biological_init=False,
                          theta_a=0.5, theta_b=0.5)
    # Same init weights so updates only differ by PC mode
    sub_b.load_state_dict(sub_a.state_dict())

    x = torch.randn(4, 30, 8)
    sub_a.clear_pc_history()
    sub_b.clear_pc_history()
    sub_a(x, capture_for_pc=True)
    sub_b(x, capture_for_pc=True)
    # Same forward; capture data should be identical
    sub_a.pc_step(lr=0.5, use_h_upstream=False)
    sub_b.pc_step(lr=0.5, use_h_upstream=True)
    # B_leak weights should differ between sub_a and sub_b due to different multiplier signal
    diff = (sub_a.cascade.layer_B.B_leak.weight - sub_b.cascade.layer_B.B_leak.weight).abs().item()
    assert diff > 1e-6, "h_upstream mode produced identical update to spike-train mode (unexpected)"


def test_pc_step_h_upstream_does_not_modify_layer_a_update():
    """Layer A PC update is the same with or without use_h_upstream (layer A uses CNN feat regardless)."""
    torch.manual_seed(0)
    sub_a = V35Substrate(d_input=8, n_recursions=5, biological_init=False,
                          theta_a=0.5, theta_b=0.5)
    sub_b = V35Substrate(d_input=8, n_recursions=5, biological_init=False,
                          theta_a=0.5, theta_b=0.5)
    sub_b.load_state_dict(sub_a.state_dict())

    x = torch.randn(4, 30, 8)
    sub_a.clear_pc_history(); sub_b.clear_pc_history()
    sub_a(x, capture_for_pc=True); sub_b(x, capture_for_pc=True)
    sub_a.pc_step(lr=0.5, use_h_upstream=False)
    sub_b.pc_step(lr=0.5, use_h_upstream=True)
    # Layer A B_leak should be identical (h_upstream only affects layer B)
    assert torch.allclose(sub_a.cascade.layer_A.B_leak.weight, sub_b.cascade.layer_A.B_leak.weight)


def test_pc_step_freezes_log_g_updates_b_leak():
    """PC invariant: pc_step changes B_leak but NEVER log_g (incl. the Layer-A
    leak-channel init). log_g is calibrated by hand (frozen), B_leak by PC."""
    import torch
    from backend.core.v35 import V35Substrate
    torch.manual_seed(0)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True,
                       log_g_leak_a=-1.0, b_leak_warm_a=4.0)
    sub.cascade.layer_A.ip_bias.data.fill_(2.0)  # ensure Layer A fires -> nonzero update
    log_g_A_before = sub.cascade.layer_A.log_g.detach().clone()
    log_g_B_before = sub.cascade.layer_B.log_g.detach().clone()
    wA_before = sub.cascade.layer_A.B_leak.weight.detach().clone()
    x = torch.zeros(2, 60, 1)
    x[:, ::6, 0] = 1.0  # periodic pulses
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.5, modulator=torch.tensor([1.0, 1.0]))
    assert torch.equal(sub.cascade.layer_A.log_g, log_g_A_before), "log_g (A) must stay frozen"
    assert torch.equal(sub.cascade.layer_B.log_g, log_g_B_before), "log_g (B) must stay frozen"
    assert not torch.allclose(sub.cascade.layer_A.B_leak.weight, wA_before), "B_leak (A) must update"


def test_threshold_a_plus_freeze_b_freezes_whole_substrate():
    """threshold-layer-a path (layer_A capture None) + freeze_layer_b -> pc_step changes NO substrate param."""
    import torch
    from backend.core.v35 import V35Substrate
    torch.manual_seed(0)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True, b_leak_b_init=2.0)
    spike_a = torch.zeros(2, 30)
    spike_a[:, [3, 9, 15, 21, 27]] = 1.0
    snap = {n: p.detach().clone() for n, p in sub.named_parameters()}
    sub.clear_pc_history()
    sub.forward_with_external_layer_a_spike(spike_a, capture_for_pc=True)
    sub.pc_step(lr=0.5, modulator=torch.tensor([1.0, 1.0]), freeze_layer_b=True)
    for n, p in sub.named_parameters():
        assert torch.equal(p, snap[n]), f"{n} changed but substrate should be fully frozen"


def test_v35_pc_training_step_isolated():
    """Full training step in PC mode: heads update via SGD, substrate via pc_step.

    Verifies: (a) substrate.B_leak weights changed by PC, (b) heads weights changed by SGD,
    (c) substrate non-B_leak params unchanged, (d) substrate.grad is zero after zero_grad.
    """
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    from backend.core.scene import SceneConfig
    import torch.nn.functional as F

    torch.manual_seed(0)
    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5,
        v35_local_pc=True, v35_pc_lr=0.5,
        successor_predict_k_max=4,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    agent.train()  # ensure training mode for capture
    # Build optimizer over heads only
    substrate_param_ids = {id(p) for p in agent.v35_substrate.parameters()}
    head_params = [p for p in agent.parameters() if id(p) not in substrate_param_ids]
    optimizer = torch.optim.Adam(head_params, lr=1e-3)
    # Snapshot before step
    B_leak_A_before = agent.v35_substrate.cascade.layer_A.B_leak.weight.detach().clone()
    gate_w_before = agent.v35_substrate.cascade.layer_A.gate_w.detach().clone()
    head_w_before = agent.v35_compare_attn.cls[0].weight.detach().clone()
    # One training step
    B = 4
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    ci = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent.v35_substrate.clear_pc_history()
    agent(inputs, ci)
    logits = agent._v35_successor_logits
    targets = torch.randint(0, logits.shape[-1], (B,))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    agent.v35_substrate.pc_step(lr=agent_cfg.v35_pc_lr)
    agent.v35_substrate.zero_grad()
    optimizer.zero_grad()
    # Checks
    B_leak_A_after = agent.v35_substrate.cascade.layer_A.B_leak.weight.detach()
    gate_w_after = agent.v35_substrate.cascade.layer_A.gate_w.detach()
    head_w_after = agent.v35_compare_attn.cls[0].weight.detach()
    assert not torch.equal(B_leak_A_before, B_leak_A_after), \
        "PC step did not change layer_A.B_leak"
    assert torch.equal(gate_w_before, gate_w_after), \
        "gate_w should be frozen but changed"
    assert not torch.equal(head_w_before, head_w_after), \
        "SGD did not update compare_attn weights"
    # Substrate grads zeroed
    for p in agent.v35_substrate.parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0, \
            "Substrate grad should be zero after substrate.zero_grad()"
