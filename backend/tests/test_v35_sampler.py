# -*- coding: utf-8 -*-
"""TDD: V35 PSS leak-kill — balanced beta sampling + SCALE regime (spec 2026-06-05 §3.1-3.3).

Four regimes per episode: EQUAL (0.10) + NEAR (0.30, |k|<=k_max, place 0) +
SCALE (0.30, offsets near theta^k for each place k -> fine decision at every place)
+ FAR (0.30, |k|>k_max up to n_max -> high-band membership). The defining property
vs the pure successor preset (|k|<=k_max): a substantial fraction of pairs have
|delta| > k_max, which is what forces alpha's HIGH digits to be stored.
"""
import numpy as np
from backend.core.scene import SceneConfig, sample_beta_count


def _sample_deltas(config, N_alpha, n_max, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([sample_beta_count(N_alpha, n_max, config, rng) - N_alpha
                     for _ in range(n)])


def test_balanced_preset_four_regime_weights():
    cfg = SceneConfig.successor_balanced_preset(k_max=5)
    assert abs(cfg.equal_weight - 0.10) < 1e-6
    assert abs(cfg.near_weight - 0.30) < 1e-6
    assert abs(cfg.scale_weight - 0.30) < 1e-6
    assert abs(cfg.far_weight - 0.30) < 1e-6
    assert cfg.near_range == (1, 5)
    assert cfg.far_min == 6                     # k_max + 1, disjoint from near (no gap)
    assert cfg.successor_only_positive is False  # bidirectional required
    assert cfg.beta_range_preset == "balanced"


def test_balanced_sampling_produces_far_mass_beyond_k_max():
    # The point vs pure successor: substantial mass at |delta| > k_max (forces high digits).
    cfg = SceneConfig.successor_balanced_preset(k_max=5)
    ad = np.abs(_sample_deltas(cfg, N_alpha=45, n_max=90))
    equal_frac = (ad == 0).mean()
    near_frac = ((ad >= 1) & (ad <= 5)).mean()
    beyond_frac = (ad > 5).mean()
    assert 0.06 < equal_frac < 0.14            # ~equal_weight 0.10
    assert near_frac > 0.20                      # NEAR + SCALE place-1
    assert beyond_frac > 0.35                    # FAR + SCALE places>=2 (pure successor=0)


def test_scale_regime_clusters_near_theta_powers():
    # Force SCALE-only: offsets concentrate near theta^k (3, 9, 27, ...), a signature
    # neither NEAR (1..k_max) nor uniform-FAR produces.
    cfg = SceneConfig.successor_balanced_preset(
        k_max=5, equal_weight=0.0, near_weight=0.0, scale_weight=1.0, far_weight=0.0,
    )
    d = np.abs(_sample_deltas(cfg, N_alpha=45, n_max=90))
    place2 = ((d >= 6) & (d <= 12)).mean()       # theta^2=9 +/- 3
    place3 = ((d >= 18) & (d <= 36)).mean()      # theta^3=27 +/- 9
    assert place2 > 0.12
    assert place3 > 0.10


def test_existing_successor_preset_unchanged_by_scale_field():
    # Regression: scale_weight defaults 0, so the pure successor preset is unaffected.
    cfg = SceneConfig.successor_prediction_preset(k_max=5)
    assert getattr(cfg, "scale_weight", 0.0) == 0.0
    ad = np.abs(_sample_deltas(cfg, N_alpha=15, n_max=30))
    # successor_only_positive=True, near_range=(1,5) -> all |delta| in [1,5], no far/scale
    assert (ad >= 1).all() and (ad <= 5).all()
