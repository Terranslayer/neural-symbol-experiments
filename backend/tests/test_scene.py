# -*- coding: utf-8 -*-
"""Tests for backend/core/scene.py — 1D scene generator (Stage 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.scene import (
    ALPHA_EQUAL,
    ALPHA_GREATER,
    ALPHA_LESS,
    SceneConfig,
    _scan_world_complex,
    build_episode,
    comparison_label,
    count_foods,
    is_food_position,
    sample_beta_count,
    sample_training_batch,
)


@pytest.fixture
def config():
    return SceneConfig()


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


class TestSceneConfig:
    def test_total_timesteps(self, config):
        # L + W + T_gap + K × (W + L + 1)
        expected = 50 + 5 + 20 + 8 * (5 + 50 + 1)
        assert config.total_timesteps == expected

    def test_phase_boundaries(self, config):
        assert config.alpha_end == 50
        assert config.write_end == 55
        assert config.forget_end == 75

    def test_compare_block_starts_are_sequential(self, config):
        starts = [config.compare_block_start(i) for i in range(config.K)]
        diffs = np.diff(starts)
        assert all(d == config.W + config.L + 1 for d in diffs)


class TestBuildEpisode:
    def test_output_shapes(self, config, rng):
        inp, lbl, cidx, meta = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        assert inp.shape == (config.total_timesteps, 3)
        assert lbl.shape == (config.K,)
        assert cidx.shape == (config.K,)
        assert inp.dtype == torch.float32
        assert lbl.dtype == torch.int64

    def test_alpha_phase_markers(self, config, rng):
        inp, _, _, _ = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        assert torch.all(inp[0:config.L, 1] == 1)
        assert torch.all(inp[0:config.L, 2] == 0)

    def test_write_and_forget_markers(self, config, rng):
        inp, _, _, _ = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        assert torch.all(inp[config.L:config.forget_end, 1] == 0)
        assert torch.all(inp[config.L:config.forget_end, 2] == 0)
        assert torch.all(inp[config.L:config.forget_end, 0] == 0)

    def test_read_notes_markers_are_placeholders(self, config, rng):
        inp, _, _, meta = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        for read_start, beta_start in meta["read_slices"]:
            assert torch.all(inp[read_start:beta_start, 1] == 1)
            assert torch.all(inp[read_start:beta_start, 2] == 1)
            assert torch.all(inp[read_start:beta_start, 0] == 0)

    def test_beta_phase_markers(self, config, rng):
        inp, _, _, _ = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        for i in range(config.K):
            read_start, beta_start, compare_step = config.compare_block_phases(i)
            assert torch.all(inp[beta_start:compare_step, 1] == 1)
            assert torch.all(inp[beta_start:compare_step, 2] == 1)

    def test_compare_step_markers(self, config, rng):
        inp, _, cidx, _ = build_episode(
            N_alpha=3, beta_counts=[3] * config.K, config=config, rng=rng
        )
        for i in range(config.K):
            step = cidx[i].item()
            assert inp[step, 0] == 0
            assert inp[step, 1] == 0
            assert inp[step, 2] == 1

    def test_alpha_signal_has_correct_object_peaks(self, rng):
        quiet_config = SceneConfig(noise_std=0.0)
        inp, _, _, _ = build_episode(
            N_alpha=5, beta_counts=[5] * quiet_config.K, config=quiet_config, rng=rng
        )
        alpha_signal = inp[0:quiet_config.L, 0]
        peaks = (alpha_signal > 0.5).sum().item()
        assert peaks == 5

    def test_labels_are_correct(self, config, rng):
        inp, lbl, _, _ = build_episode(
            N_alpha=5,
            beta_counts=[3, 5, 7, 3, 5, 7, 5, 1],
            config=config,
            rng=rng,
        )
        expected = [
            ALPHA_GREATER, ALPHA_EQUAL, ALPHA_LESS, ALPHA_GREATER,
            ALPHA_EQUAL, ALPHA_LESS, ALPHA_EQUAL, ALPHA_GREATER,
        ]
        assert lbl.tolist() == expected

    def test_determinism_same_seed(self, config):
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        inp1, lbl1, _, _ = build_episode(3, [3] * config.K, config, rng1)
        inp2, lbl2, _, _ = build_episode(3, [3] * config.K, config, rng2)
        assert torch.equal(inp1, inp2)
        assert torch.equal(lbl1, lbl2)

    def test_wrong_beta_count_raises(self, config, rng):
        with pytest.raises(ValueError):
            build_episode(3, [3, 3], config=config, rng=rng)

    def test_too_many_objects_for_world(self, config, rng):
        with pytest.raises(ValueError):
            build_episode(100, [50] * config.K, config=config, rng=rng)


class TestComparisonLabel:
    def test_greater(self):
        assert comparison_label(5, 3) == ALPHA_GREATER

    def test_less(self):
        assert comparison_label(3, 5) == ALPHA_LESS

    def test_equal(self):
        assert comparison_label(5, 5) == ALPHA_EQUAL


class TestBetaSampling:
    def test_equal_produces_same_N(self, rng):
        cfg = SceneConfig(equal_weight=1.0, near_weight=0.0, far_weight=0.0)
        for _ in range(20):
            n_beta = sample_beta_count(10, 20, cfg, rng)
            assert n_beta == 10

    def test_near_produces_close_N(self, rng):
        cfg = SceneConfig(equal_weight=0.0, near_weight=1.0, far_weight=0.0, near_range=(1, 3))
        for _ in range(50):
            n_beta = sample_beta_count(10, 100, cfg, rng)
            assert 1 <= abs(n_beta - 10) <= 3

    def test_far_produces_values_in_range(self, rng):
        cfg = SceneConfig(equal_weight=0.0, near_weight=0.0, far_weight=1.0, far_min=10)
        for _ in range(50):
            n_beta = sample_beta_count(10, 100, cfg, rng)
            assert 1 <= n_beta <= 100

    def test_clips_to_range(self, rng):
        cfg = SceneConfig()
        for _ in range(50):
            n_beta = sample_beta_count(5, 10, cfg, rng)
            assert 1 <= n_beta <= 10


class TestBatchSampling:
    def test_batch_shapes(self, config):
        rng = np.random.default_rng(seed=42)
        inputs, labels, cidx, metas = sample_training_batch(
            batch_size=4, n_max_stage=10, config=config, rng=rng
        )
        assert inputs.shape == (4, config.total_timesteps, 3)
        assert labels.shape == (4, config.K)
        assert cidx.shape == (4, config.K)
        assert len(metas) == 4

    def test_different_episodes_differ(self, config):
        rng = np.random.default_rng(seed=42)
        _, _, _, metas = sample_training_batch(
            batch_size=8, n_max_stage=10, config=config, rng=rng
        )
        alphas = [m["N_alpha"] for m in metas]
        assert len(set(alphas)) > 1


class TestDiscriminationPreset:
    def test_preset_values(self):
        cfg = SceneConfig.discrimination_preset()
        assert cfg.equal_weight == 0.3
        assert cfg.near_weight == 0.5
        assert cfg.near_range == (1, 1)
        assert cfg.far_weight == 0.2
        # weights sum to 1
        assert abs(cfg.equal_weight + cfg.near_weight + cfg.far_weight - 1.0) < 1e-9

    def test_near_samples_always_differ_by_one(self):
        cfg = SceneConfig.discrimination_preset()
        rng = np.random.default_rng(seed=0)
        # force near branch
        near_only = SceneConfig(
            equal_weight=0.0, near_weight=1.0, far_weight=0.0,
            near_range=cfg.near_range,
        )
        diffs = set()
        for _ in range(200):
            n_beta = sample_beta_count(10, 50, near_only, rng)
            diffs.add(abs(n_beta - 10))
        assert diffs == {1}, f"Near should only give +/-1, got {diffs}"

    def test_preset_distribution_proportions(self):
        cfg = SceneConfig.discrimination_preset()
        rng = np.random.default_rng(seed=42)
        N_alpha = 20
        n_max = 40
        equal_count = near_count = far_count = 0
        trials = 2000
        for _ in range(trials):
            n_beta = sample_beta_count(N_alpha, n_max, cfg, rng)
            diff = abs(n_beta - N_alpha)
            if diff == 0:
                equal_count += 1
            elif diff == 1:
                near_count += 1
            else:
                far_count += 1
        # Approximate checks (Monte Carlo, +/- 5% tolerance)
        assert 0.25 < equal_count / trials < 0.35
        assert 0.45 < near_count / trials < 0.55
        assert 0.15 < far_count / trials < 0.25


class TestComplexWorld:
    """Tests for _scan_world_complex (Stage 2 Phase 3)."""

    def _rng(self, seed=0):
        return np.random.default_rng(seed=seed)

    def test_count_matches_N(self):
        for N in [0, 1, 3, 5, 10, 20, 40]:
            for seed in range(5):
                sig = _scan_world_complex(N, L=150, rng=self._rng(seed))
                assert len(sig) == 150
                assert count_foods(sig) == N, (
                    f"N={N} seed={seed}: rule-based count={count_foods(sig)}"
                )

    def test_values_in_unit_range(self):
        for N in [0, 5, 20]:
            sig = _scan_world_complex(N, L=150, rng=self._rng(N))
            assert sig.min() >= 0.0 - 1e-6
            assert sig.max() <= 1.0 + 1e-6

    def test_distractor_never_counted_as_food(self):
        # Force 2 distractors, 0 foods, 0 mid-noise: should still count=0.
        for seed in range(10):
            sig = _scan_world_complex(
                0, L=150, rng=self._rng(seed), n_distractors=2, n_mid_noise=0
            )
            assert count_foods(sig) == 0

    def test_mid_noise_never_counted_as_food(self):
        # 5 mid-noise cells, no foods, no distractors: count should be 0.
        for seed in range(10):
            sig = _scan_world_complex(
                0, L=150, rng=self._rng(seed), n_distractors=0, n_mid_noise=5
            )
            assert count_foods(sig) == 0

    def test_background_dominates(self):
        # With N=0 and no extras, most cells in [0, 0.2].
        sig = _scan_world_complex(
            0, L=200, rng=self._rng(0), n_distractors=0, n_mid_noise=0
        )
        assert (sig <= 0.2).mean() > 0.99

    def test_food_rule_is_consistent(self):
        sig = _scan_world_complex(5, L=150, rng=self._rng(42))
        # Every True from is_food_position should be reflected in the count.
        manual = sum(is_food_position(sig, i) for i in range(len(sig)))
        assert manual == count_foods(sig) == 5

    def test_capacity_failure_raises(self):
        # L too small for N: should raise rather than silently truncate.
        with pytest.raises((ValueError, RuntimeError)):
            _scan_world_complex(40, L=50, rng=self._rng(0))

    def test_integrates_with_sample_training_batch(self):
        cfg = SceneConfig.discrimination_preset(L=150, complex_world=True)
        rng = self._rng(7)
        inp, labels, cidx, Ns = sample_training_batch(4, 10, cfg, rng)
        assert inp.shape[-1] == 3
        # Channel 0 values should span full [0, 1] range across alpha scans.
        alpha_window = inp[:, :cfg.L, 0]
        assert alpha_window.max().item() > 0.8
        assert alpha_window.min().item() < 0.3


class TestZeroPriorCompliance:
    def test_no_hardcoded_digits_in_source(self):
        scene_path = Path(__file__).resolve().parents[1] / "core" / "scene.py"
        source = scene_path.read_text(encoding="utf-8")
        import re
        forbidden = [
            r"decimal", r"binary", r"base\s*\d+",
            r"\bzero\b", r"\b0x", r"\b0b", r"radix",
        ]
        for pattern in forbidden:
            matches = re.findall(pattern, source, flags=re.IGNORECASE)
            assert not matches, f"Found forbidden prior pattern {pattern!r} in scene.py: {matches}"
