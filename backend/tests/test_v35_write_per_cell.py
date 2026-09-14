"""V35 per-cell write readout tests (route A, spec 2026-06-02).
Run on the pod (project rule: feedback_tests_on_pod)."""
import torch

from backend.core.v35_heads import V35WritePerCellReadout


def test_per_cell_decoupling():
    """raw[:, k] depends ONLY on residuals[:, k]: perturbing residuals[:, j!=k]
    leaves raw[:, k] bit-identical, while raw[:, j] changes."""
    torch.manual_seed(0)
    m = V35WritePerCellReadout(n_slots=5, hidden=8, quantize_levels=3)
    res = torch.randn(4, 5)
    raw0, _ = m(res)
    j = 2
    res2 = res.clone()
    res2[:, j] = res2[:, j] + 3.7          # perturb only column j
    raw1, _ = m(res2)
    for k in range(5):
        if k == j:
            assert not torch.equal(raw1[:, k], raw0[:, k]), f"cell {k}=j must change"
        else:
            assert torch.equal(raw1[:, k], raw0[:, k]), \
                f"cell {k} must be unchanged by perturbing res[:, {j}]"


def test_output_shape_and_alphabet():
    torch.manual_seed(0)
    m = V35WritePerCellReadout(n_slots=5, hidden=8, quantize_levels=3)
    res = torch.randn(7, 5)
    raw, scratch = m(res)
    assert raw.shape == (7, 5)
    assert scratch.shape == (7, 5)
    uniq = set(torch.unique(scratch).tolist())
    assert uniq.issubset({0.0, 0.5, 1.0}), f"scratch must be in alphabet, got {uniq}"


def test_gradient_flows_through_ste():
    """Backward from scratch reaches every per-cell MLP (quantize-STE passes grad)."""
    torch.manual_seed(0)
    m = V35WritePerCellReadout(n_slots=5, hidden=8, quantize_levels=3)
    res = torch.randn(4, 5)
    _raw, scratch = m(res)
    scratch.sum().backward()
    for k in range(5):
        for p in m.mlps[k].parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, \
                f"mlp[{k}] must receive STE gradient"


def test_write_per_cell_flag_selects_head():
    """v35_write_per_cell=True builds V35WritePerCellReadout; False builds V35WriteAttention."""
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    from backend.core.scene import SceneConfig
    from backend.core.v35_heads import V35WritePerCellReadout, V35WriteAttention

    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    base = dict(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
    )
    agent_pc = MambaAgent(MambaAgentConfig(**base, v35_write_per_cell=True), scene_cfg)
    assert isinstance(agent_pc.v35_write_attn, V35WritePerCellReadout)

    agent_attn = MambaAgent(MambaAgentConfig(**base, v35_write_per_cell=False), scene_cfg)
    assert isinstance(agent_attn.v35_write_attn, V35WriteAttention)


def test_strict_false_warmstart_swaps_write_head():
    """strict=False load of an attention-write-head state_dict into a per-cell agent:
    substrate/CNN/compare keys load; the mismatched write-head keys are skipped, leaving
    the per-cell write head fresh. This is the warm-start path the per-cell run uses
    (--init-from-ckpt-nonstrict)."""
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    from backend.core.scene import SceneConfig

    scene_cfg = SceneConfig.successor_prediction_preset(L=30, complex_world=True, k_max=2)
    scene_cfg.T_gap = 0
    scene_cfg.successor_only_positive = False
    base = dict(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5, successor_predict_k_max=4,
    )
    agent_attn = MambaAgent(MambaAgentConfig(**base, v35_write_per_cell=False), scene_cfg)
    agent_pc = MambaAgent(MambaAgentConfig(**base, v35_write_per_cell=True), scene_cfg)

    result = agent_pc.load_state_dict(agent_attn.state_dict(), strict=False)
    # write-head keys mismatch: attention's params are "unexpected", per-cell mlps "missing".
    assert any("v35_write_attn" in k for k in result.unexpected_keys), result.unexpected_keys
    assert any("v35_write_attn.mlps" in k for k in result.missing_keys), result.missing_keys
    # substrate loaded: a substrate param in the per-cell agent now equals the attention agent's.
    a = dict(agent_attn.named_parameters())
    p = dict(agent_pc.named_parameters())
    key = "v35_substrate.cascade.layer_A.B_leak.weight"
    assert torch.equal(p[key], a[key]), "substrate B_leak should have loaded from the ckpt"
