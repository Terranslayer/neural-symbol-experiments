# -*- coding: utf-8 -*-
"""TDD: V35 PSS scratch-vs-scratch forward (spec 2026-06-05 §5).

PSS routes beta through its OWN scratch (write -> readback) so NO fresh-magnitude
beta reaches the compare head — severing the leak where a near beta hands the head
alpha's high-order info. Canonical/runnable path = threshold-Layer-A + readback-direct.
"""
import torch
from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
from backend.core.scene import SceneConfig
from backend.training.train_phase1 import balanced_n_classes


def _build_pss_agent(pss=True):
    k_max = 5
    n_classes = balanced_n_classes(k_max, far_bands=1)  # 13
    scene_cfg = SceneConfig.successor_balanced_preset(L=30, complex_world=True, k_max=k_max)
    scene_cfg.T_gap = 0
    scene_cfg.successor_far_bands = 1
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5,
        successor_predict_k_max=n_classes,
        v35_threshold_layer_a=True, v35_layer_a_detect="food_pattern",
        v35_readback_direct=True,
        v35_pss_readback_beta=pss,
    )
    return MambaAgent(agent_cfg, scene_cfg), scene_cfg


def _run(agent, scene_cfg, B=2, seed=0):
    torch.manual_seed(seed)
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    return agent._v35_successor_logits


def test_pss_emits_logits_and_beta_scratch():
    agent, scene_cfg = _build_pss_agent(pss=True)
    logits = _run(agent, scene_cfg)
    assert logits.shape == (2, 13)
    sb = agent._scratch_beta_for_aux            # beta written to its OWN scratch
    assert sb is not None and sb.shape == (2, scene_cfg.W)
    for v in sb.detach().flatten().tolist():    # quantized to alphabet
        assert min(abs(v - q) for q in (0.0, 0.5, 1.0)) < 1e-4
    assert sb.grad_fn is not None               # beta write is in the autograd graph


def test_pss_off_leaves_beta_scratch_none():
    agent, scene_cfg = _build_pss_agent(pss=False)
    logits = _run(agent, scene_cfg)
    assert logits.shape == (2, 13)
    assert getattr(agent, "_scratch_beta_for_aux", None) is None


def test_pss_write_head_receives_gradient():
    agent, scene_cfg = _build_pss_agent(pss=True)
    logits = _run(agent, scene_cfg)
    logits.sum().backward()
    grads = [p.grad for p in agent.v35_write_attn.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum().item() > 0 for g in grads)


def test_pss_lesion_zero_beta_changes_logits():
    agent, scene_cfg = _build_pss_agent(pss=True)
    base = _run(agent, scene_cfg, seed=0).detach().clone()
    agent._v35_lesion = "zero_beta"             # in-forward lesion hook (spec §6 D7)
    lesioned = _run(agent, scene_cfg, seed=0).detach().clone()
    assert not torch.allclose(base, lesioned, atol=1e-6)
