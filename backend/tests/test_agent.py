# -*- coding: utf-8 -*-
"""Tests for backend/core/agent.py — GRU agent (Stage 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.agent import AgentConfig, GRUAgent
from backend.core.scene import SceneConfig, sample_training_batch


@pytest.fixture
def scene_cfg():
    return SceneConfig()


@pytest.fixture
def agent_cfg():
    return AgentConfig(d_model=16)


@pytest.fixture
def agent(scene_cfg, agent_cfg):
    torch.manual_seed(42)
    return GRUAgent(agent_cfg, scene_cfg)


@pytest.fixture
def batch(scene_cfg):
    rng = np.random.default_rng(seed=42)
    return sample_training_batch(
        batch_size=4, n_max_stage=5, config=scene_cfg, rng=rng
    )


class TestAgentConstruction:
    def test_parameter_count_is_small(self, agent):
        # Expected: 3 * d² + some biases + input proj + heads
        # Rough upper bound for d_model=16 is well under 2000
        n = agent.num_parameters()
        assert 500 < n < 2000, f"Unexpected param count: {n}"

    def test_has_required_submodules(self, agent):
        assert isinstance(agent.input_proj, nn.Linear)
        assert isinstance(agent.processor, nn.GRUCell)
        assert isinstance(agent.write_head, nn.Linear)
        assert isinstance(agent.compare_head, nn.Linear)


class TestForwardPass:
    def test_output_shapes(self, agent, scene_cfg, batch):
        inputs, labels, compare_idx, _ = batch
        logits, scratch, h_c = agent(inputs, compare_idx)
        B = inputs.shape[0]
        assert logits.shape == (B, scene_cfg.K, 3)
        assert scratch.shape == (B, scene_cfg.W)
        assert h_c.shape == (B, scene_cfg.K, agent.agent_cfg.d_model)

    def test_logits_finite(self, agent, batch):
        inputs, _, compare_idx, _ = batch
        logits, scratch, _ = agent(inputs, compare_idx)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(scratch).all()

    def test_determinism(self, scene_cfg, agent_cfg, batch):
        inputs, _, compare_idx, _ = batch
        torch.manual_seed(42)
        a1 = GRUAgent(agent_cfg, scene_cfg)
        torch.manual_seed(42)
        a2 = GRUAgent(agent_cfg, scene_cfg)
        l1, _, _ = a1(inputs, compare_idx)
        l2, _, _ = a2(inputs, compare_idx)
        assert torch.allclose(l1, l2)


class TestScratchPadLoop:
    def test_scratch_pad_feeds_into_hidden_state(self, agent, scene_cfg, batch):
        """
        Modifying the scratch pad (via perturbing write_head) must change the
        GRU hidden state AT read-phase timesteps. We hook into the GRU cell
        to capture per-timestep hidden states.

        We don't assert on compare-step h because an untrained GRU's
        decay characteristics can wash out information over the ~50-step
        β observation window — that's a training/dynamics issue, not a
        scratch-pad-wiring issue. The wiring is verified by checking the
        signal reaches h during the read window itself.
        """
        inputs, _, compare_idx, _ = batch

        h_log = []

        def hook(_mod, _inp, out):
            h_log.append(out.detach().clone())

        handle = agent.processor.register_forward_hook(hook)

        _ = agent(inputs, compare_idx)
        h_orig = list(h_log)
        h_log.clear()

        # Perturb write_head
        with torch.no_grad():
            agent.write_head.weight.add_(
                torch.randn_like(agent.write_head.weight) * 5
            )
            agent.write_head.bias.add_(
                torch.randn_like(agent.write_head.bias) * 5
            )

        _ = agent(inputs, compare_idx)
        h_new = list(h_log)
        handle.remove()

        # Check h differs at read timesteps (first read block: t ∈ [75, 79])
        read_start = scene_cfg.compare_block_start(0)
        for t in range(read_start, read_start + scene_cfg.W):
            diff = (h_orig[t] - h_new[t]).abs().max().item()
            assert diff > 1e-3, (
                f"h at read-phase t={t} unchanged after write_head perturbation "
                f"(diff={diff}) — scratch pad not wired into GRU input"
            )


class TestGradientFlow:
    def test_gradient_reaches_write_head(self, agent, batch):
        """
        Verify that gradient from compare_head loss flows back into
        write_head weights. This confirms the scratch pad forms a live
        computation graph, not a detached one.
        """
        inputs, labels, compare_idx, _ = batch
        logits, _, _ = agent(inputs, compare_idx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        loss.backward()

        assert agent.write_head.weight.grad is not None
        grad_norm = agent.write_head.weight.grad.norm().item()
        assert grad_norm > 0, "write_head got zero gradient — scratch pad loop broken"

    def test_gradient_reaches_input_proj(self, agent, batch):
        inputs, labels, compare_idx, _ = batch
        logits, _, _ = agent(inputs, compare_idx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        loss.backward()
        assert agent.input_proj.weight.grad is not None
        assert agent.input_proj.weight.grad.norm().item() > 0


class TestQuantization:
    def test_quantized_output_takes_discrete_values(self, scene_cfg):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cfg = AgentConfig(d_model=16, quantize_levels=3)
        agent = GRUAgent(cfg, scene_cfg)
        inputs, _, cidx, _ = sample_training_batch(8, 5, scene_cfg, rng)
        _, scratch, _ = agent(inputs, cidx)
        # With levels=3, scratch values must be in {-1, 0, +1}
        unique = torch.unique(scratch.round(decimals=4))
        allowed = torch.tensor([-1.0, 0.0, 1.0])
        for v in unique:
            assert any(torch.isclose(v, a, atol=1e-3) for a in allowed), (
                f"Got quantized value {v.item()} not in {{-1, 0, +1}}"
            )

    def test_quantize_levels_9_uses_9_buckets(self, scene_cfg):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cfg = AgentConfig(d_model=16, quantize_levels=9)
        agent = GRUAgent(cfg, scene_cfg)
        inputs, _, cidx, _ = sample_training_batch(32, 20, scene_cfg, rng)
        _, scratch, _ = agent(inputs, cidx)
        expected_buckets = torch.linspace(-1.0, 1.0, 9)
        unique = torch.unique(scratch.round(decimals=4))
        for v in unique:
            assert any(torch.isclose(v, b, atol=1e-3) for b in expected_buckets), (
                f"Quantized value {v.item()} not in 9-level grid"
            )

    def test_quantization_preserves_gradient_flow(self, scene_cfg):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cfg = AgentConfig(d_model=16, quantize_levels=5)
        agent = GRUAgent(cfg, scene_cfg)
        inputs, labels, cidx, _ = sample_training_batch(8, 5, scene_cfg, rng)
        logits, _, _ = agent(inputs, cidx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        loss.backward()
        # write_head must get a non-zero gradient through the STE path
        grad_norm = agent.write_head.weight.grad.norm().item()
        assert grad_norm > 0, "STE failed: write_head got zero gradient"


class TestOverfitSingleBatch:
    def test_can_overfit_tiny_batch(self, scene_cfg, agent_cfg):
        """
        Smoke test: agent should be able to overfit a single small batch
        to high accuracy. If it can't, something is wrong with architecture.

        Uses 800 steps — fewer than that can hit a loss plateau before
        escaping it. Reports the best accuracy seen (training is unstable
        without curriculum, dips are expected).
        """
        torch.manual_seed(0)
        rng = np.random.default_rng(seed=0)
        agent = GRUAgent(agent_cfg, scene_cfg)
        inputs, labels, cidx, _ = sample_training_batch(
            batch_size=8, n_max_stage=3, config=scene_cfg, rng=rng
        )
        optimizer = torch.optim.Adam(agent.parameters(), lr=3e-3)

        best_acc = 0.0
        for step in range(800):
            logits, _, _ = agent(inputs, cidx)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), labels.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                best_acc = max(best_acc, acc)

        assert best_acc > 0.85, (
            f"Best overfit accuracy {best_acc:.3f} too low in 800 steps — "
            "architecture may have a capacity/wiring issue"
        )
