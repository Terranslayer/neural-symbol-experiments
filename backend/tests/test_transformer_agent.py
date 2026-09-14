# -*- coding: utf-8 -*-
"""Tests for backend/core/transformer_agent.py — recurrent Transformer agent."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.transformer_agent import (
    TransformerAgent,
    TransformerAgentConfig,
    _gate_ste,
)
from backend.core.scene import SceneConfig, sample_training_batch


@pytest.fixture
def scene_cfg():
    return SceneConfig()


@pytest.fixture
def agent_cfg_plain():
    return TransformerAgentConfig(use_event_gate=False)


@pytest.fixture
def agent_cfg_gated():
    return TransformerAgentConfig(use_event_gate=True)


@pytest.fixture
def batch(scene_cfg):
    rng = np.random.default_rng(seed=42)
    return sample_training_batch(4, 5, scene_cfg, rng)


class TestConstruction:
    def test_plain_param_count(self, scene_cfg, agent_cfg_plain):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        n = a.num_parameters()
        assert 1500 < n < 2200, f"Unexpected param count: {n}"

    def test_gated_param_count(self, scene_cfg, agent_cfg_gated):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_gated, scene_cfg)
        n = a.num_parameters()
        assert 1500 < n < 2200, f"Unexpected param count: {n}"

    def test_has_required_modules(self, scene_cfg, agent_cfg_plain):
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        assert isinstance(a.input_proj, nn.Linear)
        assert isinstance(a.write_head, nn.Linear)
        assert isinstance(a.compare_head, nn.Linear)
        assert a.gate_net is None

    def test_gate_net_exists_when_enabled(self, scene_cfg, agent_cfg_gated):
        a = TransformerAgent(agent_cfg_gated, scene_cfg)
        assert a.gate_net is not None


class TestForward:
    def test_output_shapes_plain(self, scene_cfg, agent_cfg_plain, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        inputs, _, cidx, _ = batch
        logits, scratch, h_c = a(inputs, cidx)
        B = inputs.shape[0]
        assert logits.shape == (B, scene_cfg.K, 3)
        assert scratch.shape == (B, scene_cfg.W)
        assert h_c.shape == (B, scene_cfg.K, a.agent_cfg.d_model)

    def test_output_shapes_gated(self, scene_cfg, agent_cfg_gated, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_gated, scene_cfg)
        inputs, _, cidx, _ = batch
        logits, scratch, h_c = a(inputs, cidx)
        B = inputs.shape[0]
        assert logits.shape == (B, scene_cfg.K, 3)
        assert scratch.shape == (B, scene_cfg.W)

    def test_logits_finite(self, scene_cfg, agent_cfg_plain, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        inputs, _, cidx, _ = batch
        logits, scratch, _ = a(inputs, cidx)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(scratch).all()

    def test_determinism(self, scene_cfg, agent_cfg_plain, batch):
        inputs, _, cidx, _ = batch
        torch.manual_seed(42)
        a1 = TransformerAgent(agent_cfg_plain, scene_cfg)
        torch.manual_seed(42)
        a2 = TransformerAgent(agent_cfg_plain, scene_cfg)
        l1, _, _ = a1(inputs, cidx)
        l2, _, _ = a2(inputs, cidx)
        assert torch.allclose(l1, l2)


class TestGateSTE:
    def test_forward_is_binary(self):
        x = torch.tensor([-3.0, -0.1, 0.0, 0.1, 3.0], requires_grad=True)
        g = _gate_ste(x)
        # Forward values should be in {0, 1}
        rounded = g.detach().round()
        assert torch.all((rounded == 0) | (rounded == 1))
        # Check correctness: sigmoid(0.1) > 0.5 → 1, sigmoid(-0.1) < 0.5 → 0
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(g.detach(), expected, atol=1e-4)

    def test_backward_passes_gradient(self):
        x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        g = _gate_ste(x)
        loss = g.sum()
        loss.backward()
        # Gradient should be sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) (non-zero)
        assert x.grad is not None
        assert torch.all(x.grad > 0)


class TestScratchPadLoop:
    def test_scratch_pad_feeds_into_state(self, scene_cfg, agent_cfg_plain, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        inputs, _, cidx, _ = batch

        h_log: list = []

        def hook(mod, inp, out):
            # Capture y after each timestep; use cell output at pos 0
            h_log.append(out[:, 0].detach().clone())

        handle = a.cell.register_forward_hook(hook)
        _ = a(inputs, cidx)
        h_before = list(h_log)
        h_log.clear()

        with torch.no_grad():
            a.write_head.weight.add_(torch.randn_like(a.write_head.weight) * 5)
            a.write_head.bias.add_(torch.randn_like(a.write_head.bias) * 5)

        _ = a(inputs, cidx)
        h_after = list(h_log)
        handle.remove()

        # Check cell output differs at read timesteps (first read block)
        read_start = scene_cfg.compare_block_start(0)
        for t in range(read_start, read_start + scene_cfg.W):
            diff = (h_before[t] - h_after[t]).abs().max().item()
            assert diff > 1e-3, (
                f"Cell output at read-phase t={t} unchanged (diff={diff}) — "
                "scratch pad not wired into Transformer input"
            )


class TestGradientFlow:
    def test_gradient_to_write_head_plain(self, scene_cfg, agent_cfg_plain, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        inputs, labels, cidx, _ = batch
        logits, _, _ = a(inputs, cidx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        loss.backward()
        assert a.write_head.weight.grad is not None
        assert a.write_head.weight.grad.norm().item() > 0

    def test_gradient_to_write_head_gated(self, scene_cfg, agent_cfg_gated, batch):
        torch.manual_seed(0)
        a = TransformerAgent(agent_cfg_gated, scene_cfg)
        inputs, labels, cidx, _ = batch
        logits, _, _ = a(inputs, cidx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        loss.backward()
        assert a.write_head.weight.grad is not None
        assert a.write_head.weight.grad.norm().item() > 0
        # Also check gate_net got gradient via STE
        assert a.gate_net.weight.grad is not None
        assert a.gate_net.weight.grad.norm().item() > 0


class TestOverfit:
    def test_plain_can_overfit_tiny(self, scene_cfg, agent_cfg_plain):
        torch.manual_seed(0)
        rng = np.random.default_rng(seed=0)
        a = TransformerAgent(agent_cfg_plain, scene_cfg)
        inputs, labels, cidx, _ = sample_training_batch(
            batch_size=8, n_max_stage=3, config=scene_cfg, rng=rng
        )
        optimizer = torch.optim.Adam(a.parameters(), lr=3e-3)

        best_acc = 0.0
        for step in range(400):
            logits, _, _ = a(inputs, cidx)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), labels.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(a.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                best_acc = max(best_acc, acc)
        assert best_acc > 0.7, (
            f"Transformer cell failed to overfit 8 samples: best_acc={best_acc:.3f}"
        )
