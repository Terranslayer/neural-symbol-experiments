# -*- coding: utf-8 -*-
"""
Fair base alignment — symbol-and-position permutation invariant metric for
testing whether emergent scratch encoding aligns with base-b digit decomposition.

REPLACES the older `base_scanning_mi` diagonal-score metric (removed
2026-04-29). The old metric had a structural bias toward bases with fewer
digits (e.g. base=10 with n_digits=2 had a 5×2 MI matrix where only 2
diagonal cells counted, vs base=3 with 5×5 matrix needing 5 diagonal
cells to align). This made base=10 win for thermometer encodings even
when partial base-3 alignment was visually obvious.

NEW metric: for each candidate base b,
  1. Decompose N into base-b digits.
  2. Compute MI between every (note_pos, digit_pos) pair.
  3. Find the optimal 1-to-1 position assignment via Hungarian algorithm
     (scipy.optimize.linear_sum_assignment).
  4. Sum MI of best-matched pairs, normalize by sum of digit entropies for
     the matched digits → alignment ∈ [0, 1].

Properties:
  - Symbol permutation invariant (MI is invariant to relabeling).
  - Position permutation invariant (Hungarian finds best pairing).
  - Normalized by chance/maximum (alignment = 1.0 means perfect base-b match).
  - Fair across bases: each base scored on its own information-theoretic max.

IMPORTANT: base is an EXPERIMENTAL FINDING here, not a prior. Agents
are never told about bases during training. This analysis tool uses
base decomposition only after training, as one way to interpret what
emerged.
"""
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.optimize import linear_sum_assignment


def _digitize(n_values: np.ndarray, b: int, n_digits: int) -> np.ndarray:
    """
    Decompose each integer n into n_digits digits in base b (least-significant first).

    Returns (num_samples, n_digits) int array.
    """
    out = np.zeros((n_values.shape[0], n_digits), dtype=np.int64)
    current = n_values.astype(np.int64).copy()
    for k in range(n_digits):
        out[:, k] = current % b
        current = current // b
    return out


def _digit_entropy(digit_col: np.ndarray) -> float:
    """Empirical entropy of a digit column (in nats)."""
    if digit_col.std() < 1e-12:
        return 0.0
    _, counts = np.unique(digit_col, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def mutual_information_matrix(
    n_alpha: np.ndarray,
    notes: np.ndarray,
    b: int,
    n_digits: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Compute the MI matrix for a single base b.

    Args:
        n_alpha: (num_episodes,) int N values
        notes: (num_episodes, W) float
        b: candidate base (>= 2)
        n_digits: how many digit positions to consider. If None, use the
            minimum needed to represent max(n_alpha).
        seed: passed to mutual_info_regression for reproducibility.

    Returns dict with:
        mi_matrix: (W, n_digits) array — MI[j][k] between notes[:, j] and digit k
        base: int — the base used
        n_digits: int
    """
    if b < 2:
        raise ValueError(f"Base must be >= 2, got {b}")

    num, W = notes.shape
    max_n = int(n_alpha.max()) if num > 0 else 1
    if n_digits is None:
        k = 1
        while b ** k <= max_n and k < 20:
            k += 1
        n_digits = k

    digits = _digitize(n_alpha, b, n_digits)

    mi = np.zeros((W, n_digits), dtype=np.float64)
    for k in range(n_digits):
        if digits[:, k].std() < 1e-12:
            continue
        # mutual_info_regression: continuous features (notes), discrete target (digits)
        mi_col = mutual_info_regression(
            notes, digits[:, k].astype(np.float64), random_state=seed
        )
        mi[:, k] = mi_col

    return {
        "mi_matrix": mi,
        "base": b,
        "n_digits": n_digits,
    }


def fair_base_alignment(
    n_alpha: np.ndarray,
    notes: np.ndarray,
    b: int,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Symbol-and-position permutation invariant alignment between scratch notes
    and base-b digits of N.

    Algorithm:
      1. Compute MI matrix between (note_pos, digit_pos) pairs.
      2. Pad to square via zero-pad if W != n_digits.
      3. Hungarian algorithm finds best 1-to-1 (pos -> digit) assignment
         maximizing total MI.
      4. Normalize: alignment = total_assigned_MI / sum(entropy of matched digits)

    Returns:
      alignment ∈ [0, 1]: 1.0 = perfect base-b decomposition, 0.0 = no alignment.
      best_pairs: list of (note_pos, digit_pos) for valid assignments.
    """
    res = mutual_information_matrix(n_alpha, notes, b, seed=seed)
    mi = res["mi_matrix"]
    n_digits = res["n_digits"]
    W = notes.shape[1]

    # Hungarian: maximize MI -> minimize -MI. Pad to square.
    n = max(W, n_digits)
    cost = np.zeros((n, n), dtype=np.float64)
    cost[:W, :n_digits] = -mi
    row_ind, col_ind = linear_sum_assignment(cost)

    # Collect valid (within-bounds) pairs and their MI contributions.
    total_mi = 0.0
    pairs = []
    used_digit_cols: List[int] = []
    for r, c in zip(row_ind, col_ind):
        if int(r) < W and int(c) < n_digits:
            total_mi += float(mi[int(r), int(c)])
            pairs.append((int(r), int(c)))
            used_digit_cols.append(int(c))

    # Normalize: divide by sum of entropies of matched digits (theoretical max MI).
    digits = _digitize(n_alpha, b, n_digits)
    max_mi = sum(_digit_entropy(digits[:, k]) for k in used_digit_cols)
    alignment = total_mi / max_mi if max_mi > 1e-9 else 0.0

    return {
        "base": b,
        "alignment": float(alignment),
        "total_mi": float(total_mi),
        "max_mi": float(max_mi),
        "n_digits": n_digits,
        "best_pairs": pairs,
        "mi_matrix": mi,
    }


def base_scanning_mi(
    n_alpha: np.ndarray,
    notes: np.ndarray,
    candidate_bases: Optional[List[int]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Scan candidate bases and return the best-aligning one using fair alignment.

    NOTE: This used to compute a strict-diagonal score (j == k cells) which had
    a systematic bias toward fewer-digit (larger) bases. Replaced 2026-04-29
    with `fair_base_alignment`: Hungarian-assigned MI normalized by digit entropy.

    Args:
        n_alpha: (num_episodes,) int
        notes: (num_episodes, W) float
        candidate_bases: list of bases to try. Default: {2..10}.

    Returns dict with:
        best_base: int — the base with highest fair alignment
        best_alignment: float ∈ [0, 1]
        all_alignments: {base: alignment} for every candidate
        best_pairs: assignment for the best base
        best_mi_matrix: MI matrix for the best base
    """
    if candidate_bases is None:
        candidate_bases = list(range(2, 11))

    all_alignments: Dict[int, float] = {}
    best_full = None
    best_score = -1.0
    best_b = candidate_bases[0]

    for b in candidate_bases:
        result = fair_base_alignment(n_alpha, notes, b, seed=seed)
        all_alignments[b] = result["alignment"]
        if result["alignment"] > best_score:
            best_score = result["alignment"]
            best_b = b
            best_full = result

    return {
        "best_base": best_b,
        "best_alignment": best_score,
        "all_alignments": all_alignments,
        "best_pairs": best_full["best_pairs"] if best_full else [],
        "best_mi_matrix": best_full["mi_matrix"] if best_full else None,
    }
