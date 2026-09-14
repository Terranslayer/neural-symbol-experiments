"""Unit tests for v35.food_pattern_spike_gate (deterministic object detector)."""
import numpy as np
import torch

from backend.core.scene import _scan_world_complex, count_foods
from backend.core.v35 import food_pattern_spike_gate


def test_gate_count_and_positions_match_count_foods():
    """On real complex_world signals, gate sum == N == count_foods and spikes
    land exactly at the food centers, for N across the curriculum range."""
    rng = np.random.default_rng(0)
    for N in (1, 3, 5, 10, 30):
        L = 2 * N + 41  # feasible (need L >= 2N+1) + room for noise/distractors
        sig, centers = _scan_world_complex(N, L, rng, return_centers=True)
        assert count_foods(sig) == N  # scene invariant
        x = torch.from_numpy(sig).unsqueeze(0)  # (1, L) float32
        spikes = food_pattern_spike_gate(x, 0.9, 0.8)
        assert spikes.shape == (1, L)
        assert int(spikes.sum().item()) == N
        fired = sorted(torch.nonzero(spikes[0]).flatten().tolist())
        assert fired == sorted(centers)


def test_gate_rejects_distractors():
    """isolated_spike (dead neighbors) and weak_neighbor (one weak neighbor) have
    a food-level center but are not foods -> gate stays silent."""
    L = 12
    sig = torch.zeros(1, L)
    sig[0, 2], sig[0, 3], sig[0, 4] = 0.05, 0.95, 0.05      # isolated_spike at c=3
    sig[0, 7], sig[0, 8], sig[0, 9] = 0.60, 0.95, 0.85      # weak_neighbor at c=8 (left 0.6<0.8)
    spikes = food_pattern_spike_gate(sig, 0.9, 0.8)
    assert int(spikes.sum().item()) == 0


def test_gate_boundary_safe():
    """A high cell at index 0 (no left neighbor) must not fire; a valid food at c=2 must."""
    L = 6
    sig = torch.zeros(1, L)
    sig[0, 0] = 0.95                                         # boundary high cell, no left neighbor
    sig[0, 1], sig[0, 2], sig[0, 3] = 0.85, 0.95, 0.85      # food at c=2
    spikes = food_pattern_spike_gate(sig, 0.9, 0.8)
    fired = torch.nonzero(spikes[0]).flatten().tolist()
    assert fired == [2]
