# -*- coding: utf-8 -*-
"""
1D scene generator for Stage 1 multi-comparison task.

Builds a complete episode as a 3-channel input tensor with phase markers,
plus ground-truth comparison labels. The scratch pad (notes) positions in
the input tensor are placeholders (zeros) — the agent fills them during
forward pass.

Prior-free design: no digit/numeral concepts are hardcoded. The opaque
integer N passed in is treated as an arbitrary quantity of items to
scatter in space — the agent never sees N directly.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch


# Phase code values for (is_input_valid, post_gap_flag) channels.
PHASE_OBSERVE_ALPHA = (1, 0)
PHASE_WRITE = (0, 0)
PHASE_FORGET = (0, 0)  # same marker as write — both are "pre-gap blank"
PHASE_READ_NOTES = (1, 1)
PHASE_OBSERVE_BETA = (1, 1)
PHASE_COMPARE = (0, 1)

# Comparison class indices.
ALPHA_GREATER = 0
ALPHA_LESS = 1
ALPHA_EQUAL = 2


@dataclass
class SceneConfig:
    L: int = 50           # world length (timesteps)
    W: int = 5            # scratch pad length
    T_gap: int = 20       # forgetting interval
    K: int = 8            # number of comparisons
    noise_std: float = 0.1
    far_weight: float = 0.4
    near_weight: float = 0.4
    equal_weight: float = 0.2
    near_range: Tuple[int, int] = (1, 3)
    far_min: int = 10
    far_max: int = 0  # 0 = no cap (use n_max_stage). >0 = cap |δ| at this value.
    # V24: when True, sample_beta_count forces positive sign (β = α + k always).
    # Used by successor_prediction task: predict k from (α, β=α+k).
    successor_only_positive: bool = False
    complex_world: bool = False  # if True, use _scan_world_complex (peak + neighbors)
    # 2026-06-17: clean (decoy-free) world -> _scan_world_complex with n_distractors=0, n_mid_noise=0.
    # Isolates the food-vs-distractor discrimination: on a clean world a LEARNED counter is near-exact
    # (es_pretrain --clean-scene 31/31), so this enables the scoped counting-de-scaffold V35 test.
    clean_world: bool = False
    alpha_distribution: str = "uniform"  # "uniform" | "log_uniform"
    # log_uniform: log(N) ~ U[0, log(n_max+1)]; over-samples small N, sparsifies large N.
    # Used to break the beta-prior shortcut where uniform alpha lets the model
    # predict comparison from beta + global prior on alpha alone.
    # V35 PSS leak-kill (spec 2026-06-05 §3): SCALE regime weight (per-scale hard
    # negatives near theta^k) + which beta-range preset is active (inspect/dispatch tag).
    # scale_weight defaults 0 so existing presets keep the old equal/near/far 3-way roll.
    scale_weight: float = 0.0
    beta_range_preset: str = "successor"
    successor_far_bands: int = 1  # FAR ordinal bands per direction (1=COARSE, sign-only far)

    @staticmethod
    def trio_preset(L: int = 50, complex_world: bool = False) -> "SceneConfig":
        """
        Beta sampling that kills the uniform-alpha prior shortcut:
            1/3 equal (N_beta == N_alpha)
            1/3 N_beta = N_alpha + 1
            1/3 N_beta = N_alpha - 1  (clipped to [1, n_max])
        With no far cases, knowing beta tells almost nothing about whether alpha
        is larger/smaller — predicting any of (>, <, =) with uniform alpha is
        ~1/3 baseline. Forces model to use h_alpha for emergence above 0.333.
        """
        return SceneConfig(
            L=L,
            equal_weight=1.0 / 3.0,
            near_weight=2.0 / 3.0,
            near_range=(1, 1),
            far_weight=0.0,
            complex_world=complex_world,
        )

    @staticmethod
    def trio_wide_preset(
        L: int = 50, complex_world: bool = False, gap_max: int = 4
    ) -> "SceneConfig":
        """
        Wide-trio: β = α + uniform({-gap_max..+gap_max}).

        Achieved by reusing the equal/near/far mixture:
          equal_weight = 1 / (2*gap_max + 1)  (the gap=0 case)
          near_weight  = 2*gap_max / (2*gap_max + 1)
          near_range   = (1, gap_max)         (|diff| ∈ {1..gap_max}, sign random)
        With those weights, every offset δ ∈ {-gap_max..+gap_max} has
        marginal probability 1/(2*gap_max+1).

        Targets the gap-head + error^4 supervised loss and the K-option top-1
        RL design: provides a 9-level signed gap distribution rather than 3-level.
        """
        denom = 2 * gap_max + 1
        return SceneConfig(
            L=L,
            equal_weight=1.0 / denom,
            near_weight=(2.0 * gap_max) / denom,
            near_range=(1, gap_max),
            far_weight=0.0,
            complex_world=complex_world,
        )

    @staticmethod
    def trio_wide_extended_preset(
        L: int = 50, complex_world: bool = False,
        gap_max_near: int = 4, far_min: int = 5, far_max: int = 15,
        far_weight: float = 0.4, equal_weight: float = 0.05,
    ) -> "SceneConfig":
        """
        Extended trio_wide: same near-mixture but with a far-range tail
        |δ| ∈ [far_min, far_max]. Designed to make mod-5 head test real
        modular learning (not E-pair confound) by including δ=±5/±10/±15
        same-mod-5 non-E pairs in the training distribution.

        Default mixture:
          5%  equal (δ=0)
          55% near (|δ| ∈ [1,4])
          40% far  (|δ| ∈ [5,15])
        """
        near_weight = 1.0 - equal_weight - far_weight
        return SceneConfig(
            L=L,
            equal_weight=equal_weight,
            near_weight=near_weight,
            near_range=(1, gap_max_near),
            far_weight=far_weight,
            far_min=far_min,
            far_max=far_max,
            complex_world=complex_world,
        )

    @staticmethod
    def successor_prediction_preset(
        L: int = 200, complex_world: bool = True, k_max: int = 5,
    ) -> "SceneConfig":
        """V24 task pivot: episode = (α with N_α spikes, β with N_β=N_α+k spikes).
        Task is to predict k ∈ [1, k_max]. K=1 single compare block. Forces
        β = α + k via successor_only_positive=True. No E pairs, no far cases.
        """
        return SceneConfig(
            L=L,
            K=1,
            equal_weight=0.0,
            near_weight=1.0,
            near_range=(1, k_max),
            far_weight=0.0,
            successor_only_positive=True,
            complex_world=complex_world,
        )

    @staticmethod
    def successor_balanced_preset(
        L: int = 200, complex_world: bool = True, k_max: int = 5,
        equal_weight: float = 0.10, near_weight: float = 0.30,
        scale_weight: float = 0.30, far_weight: float = 0.30,
    ) -> "SceneConfig":
        """V35 PSS leak-kill (spec 2026-06-05 §3.1-3.3). Four-regime mixture:
          EQUAL (a==b consistency / collapse probe)
          NEAR  (|k| in [1,k_max], place 0 -> fine low-digit precision)
          SCALE (offsets near theta^k for each place k -> fine decision at EVERY place;
                 the per-scale hard negative that forces high-cell fineness)
          FAR   (|k| in [k_max+1, n_max], high-band membership; a far beta cannot
                 substitute for alpha's stored magnitude)
        Bidirectional (successor_only_positive=False). far_min=k_max+1 is disjoint
        from near_range so labels are never aliased. far_max=0 => far_hi tracks
        n_max_stage each curriculum stage (full range)."""
        assert abs(equal_weight + near_weight + scale_weight + far_weight - 1.0) < 1e-6
        return SceneConfig(
            L=L, K=1,
            equal_weight=equal_weight,
            near_weight=near_weight,
            scale_weight=scale_weight,
            far_weight=far_weight,
            near_range=(1, k_max),
            far_min=k_max + 1, far_max=0,
            successor_only_positive=False,
            complex_world=complex_world,
            beta_range_preset="balanced",
        )

    @staticmethod
    def binary_balanced_preset(
        L: int = 50, complex_world: bool = False, gap_max: int = 4
    ) -> "SceneConfig":
        """
        50/50 equal vs non-equal balanced binary preset (designed for rl_perblock loss):
            50% equal:    N_beta == N_alpha
            50% non-equal: N_beta = N_alpha + delta, delta uniform over {-gap_max..-1, +1..+gap_max}

        Why 50/50: removes the prior shortcut where 'always say not-equal' wins
        when equal_weight is small (e.g. trio_wide's 1/9 equal would give random
        accuracy 8/9 = 0.89 for 'always not-equal'). At 50/50, both 'always-equal'
        and 'always-not-equal' give 0.5 accuracy → policy MUST use alpha info.

        Used by --task binary + --loss-type rl_perblock for K independent binary
        REINFORCE decisions per episode (vs rl_top1's K-coupled categorical).
        """
        return SceneConfig(
            L=L,
            equal_weight=0.5,
            near_weight=0.5,
            near_range=(1, gap_max),
            far_weight=0.0,
            complex_world=complex_world,
        )

    @staticmethod
    def discrimination_preset(L: int = 50, complex_world: bool = False) -> "SceneConfig":
        """
        Beta-sampling distribution that stress-tests N-precision:
            30% equal (N_beta == N_alpha)
            50% hard-negative: |N_beta - N_alpha| == 1
            20% easy-negative: |N_beta - N_alpha| >= far_min
        Breaks magnitude/log encoding because log(N) vs log(N+1) collapses
        for large N (Weber's law), but ground-truth label still asks for
        exact match.
        """
        return SceneConfig(
            L=L,
            equal_weight=0.3,
            near_weight=0.5,
            near_range=(1, 1),  # exactly +/-1
            far_weight=0.2,
            complex_world=complex_world,
        )

    @property
    def total_timesteps(self) -> int:
        return self.L + self.W + self.T_gap + self.K * (self.W + self.L + 1)

    @property
    def alpha_end(self) -> int:
        return self.L

    @property
    def write_end(self) -> int:
        return self.L + self.W

    @property
    def forget_end(self) -> int:
        return self.L + self.W + self.T_gap

    def compare_block_start(self, i: int) -> int:
        """Start timestep of the i-th comparison block (0-indexed)."""
        return self.forget_end + i * (self.W + self.L + 1)

    def compare_block_phases(self, i: int) -> Tuple[int, int, int]:
        """Return (read_start, beta_start, compare_step) for the i-th block."""
        start = self.compare_block_start(i)
        return start, start + self.W, start + self.W + self.L


def _scan_world(N: int, L: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Produce a 1D scan signal of length L with N objects at random positions.

    Objects are placed at distinct integer positions (no stacking).
    Object signal ≈ 1.0 + noise, background ≈ 0.0 + noise.
    """
    if N > L:
        raise ValueError(f"Cannot place {N} distinct objects in length-{L} line")
    truth = np.zeros(L, dtype=np.float32)
    if N > 0:
        positions = rng.choice(L, size=N, replace=False)
        truth[positions] = 1.0
    signal = truth + rng.normal(0.0, noise_std, size=L).astype(np.float32)
    return signal


# -- Complex world (Stage 2 Phase 3) ------------------------------------------
# World design:
#   * background  uniform [0.0, 0.2]  (every cell)
#   * food        center uniform [0.9, 1.0], neighbors uniform [0.8, 0.9]
#   * distractor  3 types, high-but-not-food patterns; 0-2 per scan
#   * mid-noise   3-5 sparse cells uniform [0.2, 0.8]
# Food ground-truth rule (used for N labelling only, not fed to agent):
#     signal[i] >= FOOD_CENTER_THRESH
#   AND signal[i-1] >= FOOD_NEIGHBOR_THRESH
#   AND signal[i+1] >= FOOD_NEIGHBOR_THRESH

FOOD_CENTER_THRESH = 0.9
FOOD_NEIGHBOR_THRESH = 0.8
MID_NOISE_LOW = 0.2
MID_NOISE_HIGH = 0.8
BACKGROUND_LOW = 0.0
BACKGROUND_HIGH = 0.2


def is_food_position(signal: np.ndarray, i: int) -> bool:
    """Check if position i of signal is a food center per the ground-truth rule."""
    if i <= 0 or i >= len(signal) - 1:
        return False
    return (
        signal[i] >= FOOD_CENTER_THRESH
        and signal[i - 1] >= FOOD_NEIGHBOR_THRESH
        and signal[i + 1] >= FOOD_NEIGHBOR_THRESH
    )


def count_foods(signal: np.ndarray) -> int:
    """Count food centers by the ground-truth rule across the scan."""
    return sum(1 for i in range(1, len(signal) - 1) if is_food_position(signal, i))


def _sample_food_center_positions(
    N: int, L: int, rng: np.random.Generator
) -> List[int]:
    """Pick N food centers in [1, L-2] with min-spacing=2 so neighbor zones don't
    collide (adjacent centers are allowed via shared-boundary packing, but
    distance=1 causes one cell to serve as both center-left-of-A and center-B).

    Uses a compressed-range transform: place N indices in a compressed range
    of size (range - (N-1)), then expand by adding k to the k-th sorted pick.
    Guaranteed feasible iff L >= 2N + 1.
    """
    if N == 0:
        return []
    lo, hi = 1, L - 2
    range_size = hi - lo + 1  # positions available for centers
    compressed = range_size - (N - 1)
    if compressed < N:
        raise ValueError(
            f"L={L} cannot fit N={N} foods with min spacing 2 (need L >= 2N+1)"
        )
    raw = np.sort(rng.choice(compressed, size=N, replace=False))
    return [int(lo + int(raw[k]) + k) for k in range(N)]


def _place_distractor(
    signal: np.ndarray,
    exclusion: set,
    rng: np.random.Generator,
) -> bool:
    """Try to place one distractor at a random clean 3-cell region.

    3 types (equal prob):
      'weak_center':   center in [0.75, 0.9), neighbors in [0.8, 0.95]  -> fails center
      'weak_neighbor': center in [0.9, 1.0], one neighbor in [0.5, 0.8), other in [0.8, 0.95] -> fails neighbor
      'isolated_spike':center in [0.9, 1.0], both neighbors in [0.0, 0.2] -> fails neighbor

    Placement reserves a 5-cell exclusion zone (center +- 2) so neighbour cells
    cannot be assembled into a spurious food by later operations.

    Returns True if placed, False if no free region found.
    """
    L = len(signal)
    kind = rng.choice(["weak_center", "weak_neighbor", "isolated_spike"])
    for _ in range(30):
        c = int(rng.integers(2, L - 2))
        buffer_cells = set(range(c - 2, c + 3))
        if buffer_cells & exclusion:
            continue
        if kind == "weak_center":
            signal[c] = rng.uniform(0.75, 0.9 - 1e-4)
            signal[c - 1] = rng.uniform(0.8, 0.95)
            signal[c + 1] = rng.uniform(0.8, 0.95)
        elif kind == "weak_neighbor":
            signal[c] = rng.uniform(FOOD_CENTER_THRESH, 1.0)
            weak_side = int(rng.choice([-1, 1]))
            signal[c + weak_side] = rng.uniform(0.5, FOOD_NEIGHBOR_THRESH - 1e-4)
            signal[c - weak_side] = rng.uniform(0.8, 0.95)
        else:  # isolated_spike
            signal[c] = rng.uniform(FOOD_CENTER_THRESH, 1.0)
            signal[c - 1] = rng.uniform(0.0, 0.2)
            signal[c + 1] = rng.uniform(0.0, 0.2)
        exclusion.update(buffer_cells)
        return True
    return False


def _scan_world_complex(
    N: int,
    L: int,
    rng: np.random.Generator,
    n_mid_noise: Optional[int] = None,
    n_distractors: Optional[int] = None,
    max_retries: int = 50,
    return_centers: bool = False,
):
    """Complex-world scan generator (Stage 2 Phase 3+).

    Layout:
      1. Background fill (all cells uniform [0, 0.2])
      2. N foods at spaced random centers (center [0.9, 1.0], neighbors [0.8, 0.9]).
         Each food reserves a 5-cell exclusion zone (center +- 2) to prevent
         adjacent placements from creating spurious food patterns via shared
         high-neighbor cells.
      3. n_distractors in [0, 2] (uniform) placed with their own 5-cell exclusion
      4. n_mid_noise in [3, 5] (uniform) scattered in free cells, value [0.2, 0.8]

    Validates count_foods(signal) == N; retries up to max_retries on mismatch.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    for attempt in range(max_retries):
        signal = rng.uniform(BACKGROUND_LOW, BACKGROUND_HIGH, size=L).astype(np.float32)

        # Exclusion tracks cells that cannot be touched by later placements
        # (includes a +-2 buffer around foods/distractors to block spurious
        # food patterns formed by boundary cells).
        exclusion: set = set()
        food_centers = _sample_food_center_positions(N, L, rng)
        for c in food_centers:
            signal[c] = rng.uniform(FOOD_CENTER_THRESH, 1.0)
            signal[c - 1] = rng.uniform(FOOD_NEIGHBOR_THRESH, FOOD_CENTER_THRESH)
            signal[c + 1] = rng.uniform(FOOD_NEIGHBOR_THRESH, FOOD_CENTER_THRESH)
            for k in range(max(0, c - 2), min(L, c + 3)):
                exclusion.add(k)

        n_d = n_distractors if n_distractors is not None else int(rng.integers(0, 3))
        for _ in range(n_d):
            _place_distractor(signal, exclusion, rng)

        # Mid-noise must not land in exclusion zones either (a mid-noise cell
        # at distance 1 from a food neighbour cannot create a food because its
        # own value is < 0.9, but we keep the invariant clean).
        n_m = n_mid_noise if n_mid_noise is not None else int(rng.integers(3, 6))
        free = [p for p in range(L) if p not in exclusion]
        if len(free) >= n_m and n_m > 0:
            picks = rng.choice(free, size=n_m, replace=False)
            for p in picks:
                signal[p] = rng.uniform(MID_NOISE_LOW, MID_NOISE_HIGH)

        if count_foods(signal) == N:
            sig = signal.astype(np.float32)
            if return_centers:
                return sig, list(int(c) for c in food_centers)
            return sig

    raise RuntimeError(
        f"Could not generate scan with exactly N={N} foods in L={L} "
        f"after {max_retries} retries"
    )


def sample_beta_count(
    N_alpha: int,
    n_max_stage: int,
    config: SceneConfig,
    rng: np.random.Generator,
) -> int:
    """
    Sample a β count N_β given α count N_alpha and current curriculum max.

    Mixture:
        far:   |N_β - N_α| in [far_min, n_max_stage], direction random
        near:  |N_β - N_α| in near_range, direction random
        equal: N_β == N_α

    Rejection sampling: candidates outside [1, n_max_stage] (or equal to N_alpha
    when non-equal mixture chose) are rejected and resampled. This avoids the
    edge-clipping bias that previously made α=1/α=n_max samples accidentally
    equal half the time when n_max was small.
    """
    roll = rng.random()
    if roll < config.equal_weight:
        return N_alpha
    # V35 PSS leak-kill (spec 2026-06-05 §3.1): 4-way roll EQUAL/NEAR/SCALE/FAR.
    # scale_weight defaults 0 -> collapses to the old equal/near/far 3-way (no regression).
    roll2 = roll - config.equal_weight
    if roll2 < config.near_weight:                              # NEAR (place 0)
        gap_lo, gap_hi = config.near_range
    elif roll2 < config.near_weight + config.scale_weight:      # SCALE (places 1..K, near theta^k)
        THETA = 3                                               # the q=3 radix (token-clean identifier)
        max_place = 1
        while THETA ** (max_place + 1) <= n_max_stage:
            max_place += 1
        k_place = int(rng.integers(1, max_place + 1))           # 1..max_place
        centre = THETA ** k_place
        w_k = max(1, centre // THETA)
        gap_lo, gap_hi = max(1, centre - w_k), centre + w_k
    else:                                                       # FAR (high-band membership)
        gap_lo = config.far_min
        gap_hi = max(config.far_min, n_max_stage)
        if config.far_max > 0:
            gap_hi = min(gap_hi, config.far_max)
    # Rejection sampling: keep rolling until candidate is in range AND ≠ N_alpha.
    # V24: when successor_only_positive, sign is always +1 (β = α + k).
    for _ in range(50):
        diff = rng.integers(gap_lo, gap_hi + 1)
        if config.successor_only_positive:
            sign = 1
        else:
            sign = rng.choice([-1, 1])
        candidate = N_alpha + sign * int(diff)
        if 1 <= candidate <= n_max_stage and candidate != N_alpha:
            return int(candidate)
    # Fallback when n_max + alpha + gap_range force no valid candidate:
    # uniform over valid β values excluding N_alpha. If no valid β (n_max=1),
    # return N_alpha (degenerate case).
    valid = [b for b in range(1, n_max_stage + 1) if b != N_alpha]
    if not valid:
        return N_alpha
    return int(rng.choice(valid))


def comparison_label(N_alpha: int, N_beta: int) -> int:
    if N_alpha > N_beta:
        return ALPHA_GREATER
    if N_alpha < N_beta:
        return ALPHA_LESS
    return ALPHA_EQUAL


def build_episode(
    N_alpha: int,
    beta_counts: List[int],
    config: SceneConfig,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Build one complete episode.

    Returns:
        input_tensor: shape (T, 3) float32
            [:, 0] = signal_value (scratch pad positions are placeholder zeros)
            [:, 1] = is_input_valid (0/1)
            [:, 2] = post_gap_flag (0/1)
        labels: shape (K,) int64  —— comparison class per block
        compare_indices: shape (K,) int64 —— timestep index where compare_head reads
        meta: dict with diagnostic info (positions, phase boundaries, etc.)
    """
    if len(beta_counts) != config.K:
        raise ValueError(f"Expected {config.K} beta counts, got {len(beta_counts)}")

    T = config.total_timesteps
    inp = np.zeros((T, 3), dtype=np.float32)

    # α observation phase
    alpha_centers: List[int] = []
    if config.complex_world:
        _cw_kw = dict(n_distractors=0, n_mid_noise=0) if getattr(config, "clean_world", False) else {}
        alpha_signal, alpha_centers = _scan_world_complex(
            N_alpha, config.L, rng, return_centers=True, **_cw_kw
        )
    else:
        alpha_signal = _scan_world(N_alpha, config.L, config.noise_std, rng)
    inp[0:config.L, 0] = alpha_signal
    inp[0:config.L, 1] = 1  # valid
    inp[0:config.L, 2] = 0  # pre-gap

    # Write phase: signal channel stays 0 (agent's write_head output, not
    # fed back into input; scratch_pad is managed in agent forward pass)
    inp[config.L:config.write_end, 1] = 0
    inp[config.L:config.write_end, 2] = 0

    # Forget phase: same marker as write (valid=0, post_gap=0)
    inp[config.write_end:config.forget_end, 1] = 0
    inp[config.write_end:config.forget_end, 2] = 0

    # Comparison blocks
    beta_signals = []
    beta_centers_list: List[List[int]] = []
    for i, N_beta in enumerate(beta_counts):
        read_start, beta_start, compare_step = config.compare_block_phases(i)

        # Read notes phase: placeholder in signal channel; agent fills it.
        inp[read_start:beta_start, 1] = 1  # valid
        inp[read_start:beta_start, 2] = 1  # post-gap

        # β observation phase
        if config.complex_world:
            beta_signal, beta_ctrs = _scan_world_complex(
                N_beta, config.L, rng, return_centers=True,
                **(dict(n_distractors=0, n_mid_noise=0) if getattr(config, "clean_world", False) else {})
            )
            beta_centers_list.append(beta_ctrs)
        else:
            beta_signal = _scan_world(N_beta, config.L, config.noise_std, rng)
            beta_centers_list.append([])
        inp[beta_start:compare_step, 0] = beta_signal
        inp[beta_start:compare_step, 1] = 1  # valid
        inp[beta_start:compare_step, 2] = 1  # post-gap
        beta_signals.append(beta_signal)

        # Compare step
        inp[compare_step, 0] = 0
        inp[compare_step, 1] = 0
        inp[compare_step, 2] = 1

    labels = np.array([comparison_label(N_alpha, nb) for nb in beta_counts], dtype=np.int64)
    compare_indices = np.array(
        [config.compare_block_phases(i)[2] for i in range(config.K)],
        dtype=np.int64,
    )

    meta = {
        "N_alpha": N_alpha,
        "beta_counts": list(beta_counts),
        "alpha_end": config.alpha_end,
        "write_end": config.write_end,
        "forget_end": config.forget_end,
        "compare_indices": compare_indices.tolist(),
        "write_slice": (config.L, config.write_end),
        "read_slices": [
            config.compare_block_phases(i)[:2] for i in range(config.K)
        ],
        "alpha_centers": alpha_centers,
        "beta_centers": beta_centers_list,
    }

    return (
        torch.from_numpy(inp),
        torch.from_numpy(labels),
        torch.from_numpy(compare_indices),
        meta,
    )


def sample_training_batch(
    batch_size: int,
    n_max_stage: int,
    config: SceneConfig,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    """
    Sample a batch of episodes for the given curriculum stage.

    n_max_stage is the current maximum N for α sampling (curriculum-dependent).

    Returns stacked tensors suitable for batched forward pass:
        inputs: (B, T, 3)
        labels: (B, K)
        compare_indices: (B, K)
        metas: list of per-episode meta dicts
    """
    inputs, labels, compare_idx, metas = [], [], [], []
    log_max = np.log(n_max_stage + 1)
    for _ in range(batch_size):
        if config.alpha_distribution == "log_uniform":
            # log(N) ~ U[0, log(n_max+1)]; over-samples small N
            n_raw = int(np.exp(rng.uniform(0.0, log_max)))
            N_alpha = max(1, min(n_raw, n_max_stage))
        else:
            N_alpha = int(rng.integers(1, n_max_stage + 1))
        beta_counts = [
            sample_beta_count(N_alpha, n_max_stage, config, rng)
            for _ in range(config.K)
        ]
        inp, lbl, cidx, meta = build_episode(N_alpha, beta_counts, config, rng)
        inputs.append(inp)
        labels.append(lbl)
        compare_idx.append(cidx)
        metas.append(meta)
    return (
        torch.stack(inputs),
        torch.stack(labels),
        torch.stack(compare_idx),
        metas,
    )
