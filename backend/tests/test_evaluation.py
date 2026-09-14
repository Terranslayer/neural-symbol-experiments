# -*- coding: utf-8 -*-
"""Smoke tests for backend/evaluation/ modules."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.agent import AgentConfig, GRUAgent
from backend.core.scene import SceneConfig
from backend.evaluation.collect import collect_notes_and_predictions
from backend.evaluation.compositional import compositional_probe_score
from backend.evaluation.extrapolation import extrapolation_accuracy
from backend.evaluation.mi_matrix import base_scanning_mi, mutual_information_matrix
from backend.evaluation.topo_sim import topographic_similarity


@pytest.fixture
def scene_cfg():
    return SceneConfig()


@pytest.fixture
def agent(scene_cfg):
    torch.manual_seed(42)
    return GRUAgent(AgentConfig(), scene_cfg)


@pytest.fixture
def rollout(agent, scene_cfg):
    return collect_notes_and_predictions(
        agent, scene_cfg, n_max=10, num_episodes=64,
        device=torch.device("cpu"), seed=0,
    )


class TestCollect:
    def test_rollout_shapes(self, rollout, scene_cfg):
        assert rollout.num_episodes == 64
        assert rollout.n_alpha.shape == (64,)
        assert rollout.notes.shape == (64, scene_cfg.W)
        assert rollout.predictions.shape == (64, scene_cfg.K)
        assert rollout.labels.shape == (64, scene_cfg.K)

    def test_n_alpha_in_range(self, rollout):
        assert rollout.n_alpha.min() >= 1
        assert rollout.n_alpha.max() <= 10

    def test_accuracy_is_float_in_range(self, rollout):
        assert 0.0 <= rollout.accuracy <= 1.0


class TestTopoSim:
    def test_returns_finite_score(self, rollout):
        result = topographic_similarity(rollout.n_alpha, rollout.notes)
        assert "topo_sim" in result
        assert "p_value" in result
        # Not NaN with real data
        assert not np.isnan(result["topo_sim"]) or rollout.notes.std() < 1e-12

    def test_identical_notes_return_zero(self):
        n = np.array([1, 2, 3, 4, 5])
        notes = np.ones((5, 3))  # all identical
        result = topographic_similarity(n, notes)
        assert result["topo_sim"] == 0.0

    def test_perfect_correspondence(self):
        # When notes = f(N) with varying direction per N, topo_sim near 1.
        # (If we used scalar multiples like [n, 2n, -n], cosine distance
        # would be 0 for all pairs since they share the same direction.)
        n = np.arange(1, 21).astype(np.float64)
        notes = np.stack([n, np.sin(n), np.cos(n)], axis=1)
        result = topographic_similarity(n, notes, metric="euclidean")
        assert result["topo_sim"] > 0.9


class TestCompositional:
    def test_output_structure(self, rollout, scene_cfg):
        result = compositional_probe_score(rollout.n_alpha, rollout.notes, cv_folds=3)
        assert len(result["per_position_r2"]) == scene_cfg.W
        assert len(result["marginal_gains"]) == scene_cfg.W
        assert 0.0 <= result["composition_index"] <= 1.0

    def test_single_position_suffices(self):
        # If N is linearly encoded in first note, later positions should add 0 gain
        n = np.arange(1, 101)
        notes = np.stack([
            n.astype(np.float64),       # carries all the info
            np.random.randn(100),       # noise
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100),
        ], axis=1)
        result = compositional_probe_score(n, notes, cv_folds=3)
        assert result["per_position_r2"][0] > 0.95
        # composition_index should be small (not much to gain beyond position 0)
        assert result["composition_index"] < 0.5

    def test_distributed_encoding(self):
        # Each position carries a different digit — composition_index should be high
        rng = np.random.default_rng(0)
        n = rng.integers(1, 100, size=200)
        digits = np.stack([
            n % 10,            # ones
            (n // 10) % 10,    # tens
            np.zeros_like(n),
            np.zeros_like(n),
            np.zeros_like(n),
        ], axis=1).astype(np.float64)
        # Add small noise so probes don't get 1.0 trivially
        notes = digits + rng.normal(0, 0.05, size=digits.shape)
        result = compositional_probe_score(n, notes, cv_folds=3)
        # With true place-value encoding, position 1 adds substantial info
        assert result["marginal_gains"][1] > 0.1


class TestMIMatrix:
    def test_shape(self, rollout, scene_cfg):
        result = mutual_information_matrix(
            rollout.n_alpha, rollout.notes, b=5, n_digits=3
        )
        mi = result["mi_matrix"]
        assert mi.shape == (scene_cfg.W, 3)
        assert 0.0 <= result["diagonal_score"] <= 1.0

    def test_invalid_base_raises(self, rollout):
        with pytest.raises(ValueError):
            mutual_information_matrix(rollout.n_alpha, rollout.notes, b=1)

    def test_base_scanning_returns_winner(self, rollout):
        result = base_scanning_mi(
            rollout.n_alpha, rollout.notes,
            candidate_bases=[2, 3, 5, 7],
        )
        assert result["best_base"] in [2, 3, 5, 7]
        assert "all_alignments" in result
        assert 0.0 <= result["best_alignment"] <= 1.0


class TestWeberCurve:
    def test_curve_shape_and_fields(self, agent, scene_cfg):
        from backend.evaluation.weber import compute_weber_curve
        curve = compute_weber_curve(
            agent=agent,
            scene_cfg=scene_cfg,
            n_range=range(3, 8),
            num_trials_per_n=10,
            device=torch.device("cpu"),
            seed=0,
        )
        assert len(curve.n_values) == 5
        assert len(curve.accuracy) == 5
        for acc in curve.accuracy:
            assert 0.0 <= acc <= 1.0

    def test_flatness_score_bounded(self, agent, scene_cfg):
        from backend.evaluation.weber import compute_weber_curve
        curve = compute_weber_curve(
            agent=agent,
            scene_cfg=scene_cfg,
            n_range=range(3, 8),
            num_trials_per_n=10,
            device=torch.device("cpu"),
            seed=0,
        )
        # flatness is 1 - std/mean, can be negative for very variable curves
        score = curve.flatness_score()
        assert score <= 1.0

    def test_weber_slope_returns_float(self, agent, scene_cfg):
        from backend.evaluation.weber import compute_weber_curve
        curve = compute_weber_curve(
            agent=agent,
            scene_cfg=scene_cfg,
            n_range=range(3, 8),
            num_trials_per_n=10,
            device=torch.device("cpu"),
            seed=0,
        )
        slope = curve.weber_slope()
        assert isinstance(slope, float)


class TestExtrapolation:
    def test_untrained_agent_gives_finite_accuracy(self, agent, scene_cfg):
        result = extrapolation_accuracy(
            agent=agent,
            scene_cfg=scene_cfg,
            n_train_max=5,
            n_test_ranges={"near": range(6, 11)},
            device=torch.device("cpu"),
            num_episodes_per_range=20,
            seed=0,
        )
        assert "near" in result
        assert 0.0 <= result["near"] <= 1.0
