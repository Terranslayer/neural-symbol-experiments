# -*- coding: utf-8 -*-
"""V35 PSS leak-kill — balanced config plumbs n_classes through the forward (spec §3.4).

Characterization/regression test: a V35 agent built for the balanced preset
(successor_predict_k_max = balanced_n_classes = 13) emits 13-class successor logits.
The argparse/dispatch glue is validated end-to-end by the Stage-0 smoke (task 8).
"""
import torch
from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
from backend.core.scene import SceneConfig
from backend.training.train_phase1 import balanced_n_classes


def test_balanced_v35_forward_emits_n_classes_logits():
    torch.manual_seed(0)
    k_max = 5
    n_classes = balanced_n_classes(k_max, far_bands=1)   # 2*5+1+2 = 13
    assert n_classes == 13
    scene_cfg = SceneConfig.successor_balanced_preset(L=30, complex_world=True, k_max=k_max)
    scene_cfg.T_gap = 0
    scene_cfg.successor_far_bands = 1
    assert scene_cfg.beta_range_preset == "balanced"
    agent_cfg = MambaAgentConfig(
        d_model=8, input_dim=3, quantize_levels=3, quantize_range="unit",
        read_cnn_kernel=5, dual_pathway=True, dp_kernels=(5, 21, 51),
        use_v35=True, v35_n_recursions=5,
        successor_predict_k_max=n_classes,   # 13 (what train_phase1 wires for balanced)
    )
    agent = MambaAgent(agent_cfg, scene_cfg)
    B = 2
    T = scene_cfg.total_timesteps
    inputs = torch.randn(B, T, 3) * 0.1
    compare_indices = torch.zeros(B, scene_cfg.K, dtype=torch.long)
    agent(inputs, compare_indices)
    assert agent._v35_successor_logits is not None
    assert agent._v35_successor_logits.shape == (B, n_classes)
