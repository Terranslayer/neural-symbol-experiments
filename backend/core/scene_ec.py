"""V32 EC scene: symmetric 2-agent (N_A, N_B) with signed-δ 5-class label.

Design constraints (see plan doc § 6, 'Sampling design'):
1. δ = N_B - N_A ∈ {-2, -1, 0, +1, +2}.
2. N_A uniformly sampled from [N_min+2, N_max-2] so any δ ∈ ±2 gives
   N_B in [N_min, N_max] without rejection (no edge-clip bias).
3. δ uniform on {-2,...,+2} → class balance.
4. 50% probability swap (A↔B) so N_A and N_B marginal distributions match
   exactly. Without swap, N_A is in narrower range than N_B.
5. Signal generation inherits `_scan_world_complex` from scene.py (padding-
   aware, food-center spacing, distractors, exclusion zones).
6. Each agent's δ label is computed from THAT agent's POV:
       delta_A = N_A - N_B (own minus partner)
       delta_B = N_B - N_A = -delta_A
   Class = delta + DELTA_CLASS_OFFSET (so range [0, 4]).

Sample dict format:
    {
        "N_A": int, "N_B": int,
        "signal_A": np.ndarray (L,) float32,
        "signal_B": np.ndarray (L,) float32,
        "delta_class_A": int in [0, 4],
        "delta_class_B": int in [0, 4],
    }
"""
from dataclasses import dataclass

import numpy as np

from backend.core.scene import _scan_world_complex


# δ class offset: class index = delta + offset, so class ∈ [0, 4].
DELTA_CLASS_OFFSET = 2  # δ=-2 → class 0; δ=+2 → class 4
DELTA_RANGE = (-2, 2)
N_DELTA_CLASSES = 5


@dataclass
class ECSceneConfig:
    """EC scene generation config.

    N_A is uniformly sampled from [N_min + 2, N_max - 2]; with δ ∈ ±2 this
    gives N_B in [N_min, N_max] without rejection bias.

    L must satisfy L ≥ 2 * N_max + 1 (per scene.py:284 spacing constraint).
    """
    N_min: int = 1
    N_max: int = 30
    L: int = 200
    swap_prob: float = 0.5


def sample_ec_episode(cfg: ECSceneConfig, rng: np.random.Generator) -> dict:
    """Sample one EC episode.

    Steps:
    1. N_A uniform [N_min + 2, N_max - 2]
    2. δ uniform {-2, -1, 0, +1, +2}
    3. N_B = N_A + δ ∈ [N_min, N_max] guaranteed by N_A range
    4. With prob swap_prob, swap (N_A, N_B) → δ sign flips; ensures marginal
       symmetry across many episodes
    5. Generate signal_A and signal_B independently with their N quantities
    6. Compute per-agent POV delta classes
    """
    n_a_lo = cfg.N_min + 2
    n_a_hi = cfg.N_max - 2
    if n_a_hi < n_a_lo:
        raise ValueError(
            f"N range [{cfg.N_min}, {cfg.N_max}] too narrow for δ ∈ ±2. "
            f"Need N_max - N_min ≥ 4."
        )
    N_A = int(rng.integers(n_a_lo, n_a_hi + 1))  # inclusive on both ends
    delta = int(rng.integers(DELTA_RANGE[0], DELTA_RANGE[1] + 1))
    N_B = N_A + delta

    if rng.random() < cfg.swap_prob:
        N_A, N_B = N_B, N_A

    # Generate signals (independent, both use complex world generator)
    signal_A = _scan_world_complex(N_A, cfg.L, rng)
    signal_B = _scan_world_complex(N_B, cfg.L, rng)

    delta_A = N_A - N_B
    delta_B = N_B - N_A
    return {
        "N_A": N_A,
        "N_B": N_B,
        "signal_A": signal_A,
        "signal_B": signal_B,
        "delta_class_A": delta_A + DELTA_CLASS_OFFSET,
        "delta_class_B": delta_B + DELTA_CLASS_OFFSET,
    }


def sample_ec_batch(
    cfg: ECSceneConfig, batch_size: int, rng: np.random.Generator,
) -> dict:
    """Collate batch_size episodes into batched tensors.

    Returns dict with keys:
        N_A, N_B: (B,) int
        signal_A, signal_B: (B, L) float32
        delta_class_A, delta_class_B: (B,) int
    """
    eps = [sample_ec_episode(cfg, rng) for _ in range(batch_size)]
    return {
        "N_A": np.array([e["N_A"] for e in eps], dtype=np.int64),
        "N_B": np.array([e["N_B"] for e in eps], dtype=np.int64),
        "signal_A": np.stack([e["signal_A"] for e in eps]),
        "signal_B": np.stack([e["signal_B"] for e in eps]),
        "delta_class_A": np.array([e["delta_class_A"] for e in eps], dtype=np.int64),
        "delta_class_B": np.array([e["delta_class_B"] for e in eps], dtype=np.int64),
    }
