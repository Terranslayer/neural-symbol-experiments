import torch
from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
from backend.core.scene import SceneConfig


def _build_v35_agent():
    """Smallest viable V35 agent for shape/grad smoke testing."""
    scene_cfg = SceneConfig.successor_prediction_preset(
        L=30, complex_world=True, k_max=2,
    )
    # Successor preset defaults: K=1, T_gap=20, W=5. Override T_gap=0 for shorter episode.
    scene_cfg.T_gap = 0
    # Canonical V35 uses --successor-bidirectional which sets this to False
    # in train_phase1.py. Match here so the fixture mirrors canonical use.
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3,
        quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5,
        dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5,
        successor_predict_k_max=4,  # 2*k_max=4 for bidirectional k_max=2
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    return agent, scene_cfg, agent_cfg


def test_v35_forward_end_to_end():
    """Build a tiny V35 agent and run forward on a synthetic episode.
    Verify return contract + scratch alphabet + successor logits shape.
    """
    torch.manual_seed(0)
    agent, scene_cfg, _ = _build_v35_agent()
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    compare_logits, scratch, hidden_at_compares = agent(inputs, compare_indices)
    assert compare_logits.shape == (B, scene_cfg.K, 3)
    assert scratch.shape == (B, scene_cfg.W)
    # Scratch values quantize-STE'd to alphabet {0, 0.5, 1.0}
    for v in scratch.detach().flatten().tolist():
        assert min(abs(v - q) for q in (0.0, 0.5, 1.0)) < 1e-4
    # Successor logits stashed on agent
    assert agent._v35_successor_logits is not None
    assert agent._v35_successor_logits.shape == (B, 4)  # 2*k_max


def test_v35_forward_backward_loss():
    """Full forward + CE loss + backward through cascade + heads.
    Verifies the closed gradient loop: loss -> successor_logits -> compare_attn ->
    cascade(beta) + cascade(readback) <- scratch_alpha (via STE) <- write_attn <-
    cascade(alpha) <- CNN <- alpha signal.
    """
    torch.manual_seed(0)
    agent, scene_cfg, _ = _build_v35_agent()
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    logits = agent._v35_successor_logits
    targets = torch.randint(0, logits.shape[-1], (B,))
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    # Theta gets gradient
    assert agent.v35_substrate.cascade.theta_raw.grad is not None
    assert agent.v35_substrate.cascade.theta_raw.grad.abs().item() > 0
    # Layer A and Layer B gate params get gradient
    assert agent.v35_substrate.cascade.layer_A.gate_w.grad is not None
    assert agent.v35_substrate.cascade.layer_B.gate_w.grad is not None
    # Write attention queries get gradient
    assert agent.v35_write_attn.slot_queries.grad is not None
    # CNN encoder gets gradient (SM closure verification)
    for conv in agent.dp_convs:
        assert conv.weight.grad is not None
        assert conv.weight.grad.abs().sum().item() > 0


def test_v35_collect_path_returns_real_preds():
    """compute_successor_preds should return V35 predictions (not None) when
    agent has _v35_successor_logits stashed. This bypasses the
    successor_predict_head check.
    """
    from backend.evaluation.collect import compute_successor_preds
    torch.manual_seed(0)
    agent, scene_cfg, _ = _build_v35_agent()
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    # Bidirectional class mapping (matches V29 / collect _k_target_from_metas):
    #   n_classes = 4, k_max_val = 2
    #   k=-2 → class 0,  k=-1 → class 1
    #   k=+1 → class 2,  k=+2 → class 3
    metas = [
        {"beta_counts": [3], "N_alpha": 1},  # k=+2 → class 3
        {"beta_counts": [1], "N_alpha": 2},  # k=-1 → class 1
    ]
    result = compute_successor_preds(agent, metas, device=torch.device("cpu"))
    assert result is not None, "compute_successor_preds returned None; V35 fast-path missing"
    preds, targets, n_classes = result
    assert preds.shape == (B,)
    assert targets.shape == (B,)
    assert n_classes == 4  # 2*k_max=4 for bidirectional k_max=2
    assert targets[0].item() == 3  # k=+2 → class 3
    assert targets[1].item() == 1  # k=-1 → class 1


def test_v35_forward_food_pattern_spkA_equals_N():
    """Under detect='food_pattern', the alpha-pass Layer-A spike train (stashed at
    _v35_spike_train_A) has exactly N = count_foods(alpha signal) spikes."""
    import numpy as np
    from backend.core.scene import build_episode, sample_beta_count, count_foods
    torch.manual_seed(0)
    scene_cfg = SceneConfig.successor_prediction_preset(L=40, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
        v35_threshold_layer_a=True, v35_layer_a_detect="food_pattern",
    )
    agent = MambaAgent(agent_cfg, scene_cfg).eval()
    rng = np.random.default_rng(1)
    N = 5
    beta_counts = [sample_beta_count(N, 5, scene_cfg, rng)]
    inp, _l, _c, _m = build_episode(N, beta_counts, scene_cfg, rng)
    inputs = inp.unsqueeze(0)
    compare_indices = torch.zeros(1, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    L = scene_cfg.L
    assert count_foods(inp[:L, 0].numpy()) == N           # episode really has N foods
    assert agent._v35_spike_train_A is not None
    assert int(agent._v35_spike_train_A.sum().item()) == N


def test_v35_forward_level_detect_back_compat():
    """detect='level' (default) under threshold bypass still runs and produces logits."""
    torch.manual_seed(0)
    agent, scene_cfg, _ = _build_v35_agent()              # default detect='level'
    agent.agent_cfg.v35_threshold_layer_a = True          # enable bypass on the level path
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3).abs().clamp(max=1.0)
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    assert agent._v35_successor_logits is not None


def _build_v35_readback_agent(readback_direct: bool):
    """food_pattern + per-cell write head agent, with readback_direct toggle."""
    scene_cfg = SceneConfig.successor_prediction_preset(L=40, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
        v35_threshold_layer_a=True, v35_layer_a_detect="food_pattern",
        v35_write_per_cell=True, v35_readback_direct=readback_direct,
    )
    return MambaAgent(agent_cfg, scene_cfg), scene_cfg


def _v35_write_head_grads(agent, W):
    return [agent.v35_write_attn.mlps[k][0].weight.grad for k in range(W)]


def test_v35_readback_direct_gradient_reaches_write_head():
    """With readback_direct, task CE gradient reaches every per-cell write head MLP."""
    torch.manual_seed(0)
    agent, scene_cfg = _build_v35_readback_agent(readback_direct=True)
    B = 2
    inputs = torch.randn(B, scene_cfg.total_timesteps, 3).abs().clamp(max=1.0)
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    logits = agent._v35_successor_logits
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, logits.shape[-1], (B,)))
    agent.zero_grad()
    loss.backward()
    grads = _v35_write_head_grads(agent, scene_cfg.W)
    assert all(g is not None for g in grads), "write head MLP got no grad with readback_direct"
    assert any(float(g.abs().sum()) > 0 for g in grads), "write head MLP grad all zero with readback_direct"


def test_v35_hard_readback_blocks_write_head_gradient():
    """Without readback_direct (hard-threshold readback), the write head MLPs get NO task
    gradient -- documents the bug the flag fixes (CE-only loss isolates the readback path)."""
    torch.manual_seed(0)
    agent, scene_cfg = _build_v35_readback_agent(readback_direct=False)
    B = 2
    inputs = torch.randn(B, scene_cfg.total_timesteps, 3).abs().clamp(max=1.0)
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    logits = agent._v35_successor_logits
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, logits.shape[-1], (B,)))
    agent.zero_grad()
    loss.backward()
    grads = _v35_write_head_grads(agent, scene_cfg.W)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads), \
        "hard-threshold readback unexpectedly passed gradient to the write head"


def _build_v35_agent_pc():
    """Build V35 agent with v35_local_pc=True for PC-mode integration tests."""
    scene_cfg = SceneConfig.successor_prediction_preset(
        L=30, complex_world=True, k_max=2,
    )
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3,
        quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5,
        dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5,
        v35_local_pc=True, v35_pc_lr=0.1,
        successor_predict_k_max=4,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    return agent, scene_cfg, agent_cfg


def test_v35_forward_pc_mode_captures_history():
    """In PC mode, _v35_forward populates substrate._pc_data_history with 3 entries (alpha, readback, beta)."""
    torch.manual_seed(0)
    agent, scene_cfg, _ = _build_v35_agent_pc()
    agent.train()
    agent.v35_substrate.clear_pc_history()
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    # 3 passes: alpha + readback + beta
    assert len(agent.v35_substrate._pc_data_history) == 3
    for pass_data in agent.v35_substrate._pc_data_history:
        assert pass_data["layer_A"] is not None
        assert len(pass_data["layer_B_recursions"]) == 5


def test_v35_forward_non_pc_mode_no_capture():
    """In non-PC mode (v35_local_pc=False), no capture occurs."""
    torch.manual_seed(0)
    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, v35_local_pc=False,
        successor_predict_k_max=4,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    agent.v35_substrate.clear_pc_history()
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    assert agent.v35_substrate._pc_data_history == []


def test_v35_biological_init_when_pc_enabled():
    """v35_local_pc=True triggers biological init on substrate layers."""
    agent, _, _ = _build_v35_agent_pc()
    layer_A = agent.v35_substrate.cascade.layer_A
    expected_gate_w = torch.tensor([2.0, 2.0, 1.0, 0.5])
    assert torch.allclose(layer_A.gate_w.data, expected_gate_w)


def test_v35_agent_threads_layer_a_leak_warm():
    """MambaAgentConfig v35_log_g_leak_a / v35_warm_b_leak_a reach layer_A only;
    layer_B keeps log_g all -4."""
    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
        v35_local_pc=True, v35_theta_a=1.0, v35_theta_b=3.0,
        v35_log_g_leak_a=-1.0, v35_warm_b_leak_a=4.0,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    la = agent.v35_substrate.cascade.layer_A
    lb = agent.v35_substrate.cascade.layer_B
    assert torch.allclose(la.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -1.0]))
    assert torch.allclose(lb.log_g.data, torch.tensor([-4.0, -4.0, -4.0, -4.0]))


def test_v35_agent_warm_scales_layer_a_b_leak():
    """Integration: v35_warm_b_leak_a scales layer_A's B_leak init through
    config -> substrate (4x vs the default warm=1.0 build, same seed). Closes the
    gap where only log_g threading was asserted."""
    def _build(warm):
        scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
        scene_cfg.T_gap = 0
        scene_cfg.successor_only_positive = False
        agent_cfg = MambaAgentConfig(
            d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
            read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
            use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
            v35_local_pc=True, v35_theta_a=1.0, v35_theta_b=3.0,
            v35_warm_b_leak_a=warm,
        )
        torch.manual_seed(0)  # seed immediately before construction so the two builds match
        return MambaAgent(agent_cfg, scene_cfg)

    warm_agent = _build(4.0)
    ref_agent = _build(1.0)
    wa = warm_agent.v35_substrate.cascade.layer_A.B_leak.weight.data
    ref = ref_agent.v35_substrate.cascade.layer_A.B_leak.weight.data
    assert torch.allclose(wa, 4.0 * ref), "v35_warm_b_leak_a should scale layer_A B_leak init 4x"


def test_v35_agent_b_leak_b_init_fills_layer_b():
    """MambaAgentConfig v35_b_leak_b_init fills layer_B B_leak; layer_A unaffected."""
    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
        v35_local_pc=True, v35_theta_a=1.0, v35_theta_b=3.0,
        v35_b_leak_b_init=2.0,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    lb = agent.v35_substrate.cascade.layer_B.B_leak.weight.data
    assert torch.allclose(lb, torch.full_like(lb, 2.0))
    la = agent.v35_substrate.cascade.layer_A.B_leak.weight.data
    assert not torch.allclose(la, torch.full_like(la, 2.0))


def test_v35_agent_log_g_b_pins_layer_b_channels_off():
    """v35_log_g_b fills ALL layer_B log_g channels (timing-invariant form) and freezes them;
    layer_A log_g untouched."""
    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
        v35_local_pc=True, v35_theta_a=1.0, v35_theta_b=3.0,
        v35_b_leak_b_init=2.0, v35_log_g_b=-20.0,
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    lg = agent.v35_substrate.cascade.layer_B.log_g
    assert torch.allclose(lg.data, torch.full_like(lg.data, -20.0)), "log_g_b must fill all 4 channels"
    assert lg.requires_grad is False, "hand-set log_g_b must be frozen (requires_grad False)"
    la = agent.v35_substrate.cascade.layer_A.log_g.data
    assert not torch.allclose(la, torch.full_like(la, -20.0)), "layer_A log_g must be untouched"


def test_v35_recon_decoder_built_and_stashes_residuals():
    """v35_recon_aux_lambda>0 builds the recon decoder and a forward stashes
    _v35_residuals_alpha (B, W); lambda=0 -> no decoder. Guards the silent-skip failure
    mode where the recon loss term never fires."""
    import numpy as np
    from backend.core.scene import build_episode, sample_beta_count

    def _make_agent(recon_lambda):
        scene_cfg = SceneConfig.successor_prediction_preset(L=60, complex_world=True, k_max=2)
        scene_cfg.T_gap = 0
        scene_cfg.successor_only_positive = False
        agent_cfg = MambaAgentConfig(
            d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
            read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
            use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
            v35_local_pc=True, v35_theta_a=1.0, v35_theta_b=3.0,
            v35_b_leak_b_init=2.0, v35_log_g_b=-20.0,
            v35_threshold_layer_a=True, v35_layer_a_detect="food_pattern",
            v35_write_per_cell=True, v35_readback_direct=True,
            v35_recon_aux_lambda=recon_lambda,
        )
        return MambaAgent(agent_cfg, scene_cfg), scene_cfg

    agent0, _ = _make_agent(0.0)
    assert agent0.v35_recon_decoder is None, "lambda=0 must not build a decoder"

    agent, scene_cfg = _make_agent(1.0)
    assert agent.v35_recon_decoder is not None, "lambda>0 must build the recon decoder"
    rng = np.random.default_rng(0)
    N_alpha = 4
    beta = [sample_beta_count(N_alpha, 5, scene_cfg, rng)]
    inp, _lbl, cidx, _meta = build_episode(N_alpha, beta, scene_cfg, rng)
    agent.eval()
    with torch.no_grad():
        agent(inp.unsqueeze(0), cidx.unsqueeze(0))
    res = agent._v35_residuals_alpha
    assert res is not None and res.shape == (1, scene_cfg.W), "forward must stash residuals_alpha (B, W)"
    recon = agent.v35_recon_decoder(agent._scratch_alpha_for_aux)
    assert recon.shape == (1, scene_cfg.W), "decoder must map scratch (B,W) -> residuals (B,W)"
