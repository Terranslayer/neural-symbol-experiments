# -*- coding: utf-8 -*-
"""V35 PSS leak-kill detection (spec 2026-06-05 §6).

Distinguish, from a trained V35 codebook, three outcomes:
  (A) base-theta PLACE-VALUE : positional decomposition; local carry on N->N+1
  (B) arbitrary 1-to-1 LOOKUP cipher : ~1-to-1 but unstructured (no carry locality)
  (C) coarse THERMOMETER / COLLAPSE : few codes / dead cells / monotone single axis

Pure functions over a per-N modal codebook {N: (cell levels...)} so they are unit-testable
on synthetic codebooks (and double as the D10 synthetic-null). Cell values may be floats in
{0,0.5,1.0} or ints {0,1,2}; both are normalized to integer levels here.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _to_levels(code: Sequence[float]) -> Tuple[int, ...]:
    """Map a cell-value code (floats in {0,0.5,1.0} or ints {0,1,2}) to integer levels."""
    out = []
    for v in code:
        fv = float(v)
        if fv <= 0.25:
            out.append(0)
        elif fv <= 0.75:
            out.append(1)
        else:
            out.append(2)
    return tuple(out)


def carry_consistency(modal_by_n: Dict[int, Sequence[float]], theta: int = 3) -> Dict[str, float]:
    """D3 (PRIMARY base-theta-vs-cipher discriminator).

    Place-value: code(N+1)-code(N) is sparse and boundary-localized — mostly the low cell
    flips, and a higher cell changes only when N+1 crosses a theta^k boundary. A cipher's
    increments are dense and not boundary-aligned.

    Returns: mean_increment_hamming (place-value small ~1), low_cell_only_frac (high for
    place-value), boundary_hit_frac (fraction of higher-cell changes landing on a theta^k
    boundary; ~1 for place-value, ~chance for a cipher), n_pairs.
    """
    Ns = sorted(modal_by_n)
    if not Ns:
        return {"mean_increment_hamming": float("nan"), "low_cell_only_frac": float("nan"),
                "boundary_hit_frac": float("nan"), "n_pairs": 0}
    W = len(modal_by_n[Ns[0]])
    inc_hamming: List[int] = []
    higher_changes = 0
    higher_at_boundary = 0
    low_only = 0
    n_pairs = 0
    for N in Ns:
        if (N + 1) not in modal_by_n:
            continue
        n_pairs += 1
        a = np.array(_to_levels(modal_by_n[N]))
        b = np.array(_to_levels(modal_by_n[N + 1]))
        changed = a != b
        inc_hamming.append(int(changed.sum()))
        higher_changed = bool(changed[1:].any())
        if higher_changed:
            higher_changes += 1
            on_boundary = any(((N + 1) % (theta ** k) == 0) for k in range(1, W))
            if on_boundary:
                higher_at_boundary += 1
        if changed[0] and not higher_changed:
            low_only += 1
    return {
        "mean_increment_hamming": float(np.mean(inc_hamming)) if inc_hamming else float("nan"),
        "low_cell_only_frac": (low_only / n_pairs) if n_pairs else float("nan"),
        "boundary_hit_frac": (higher_at_boundary / higher_changes) if higher_changes else float("nan"),
        "n_pairs": n_pairs,
    }


def high_cell_fineness(modal_by_n: Dict[int, Sequence[float]], theta: int = 3,
                       n_cells: int = None) -> Dict[int, int]:
    """D2 (go/no-go). For each HIGH cell k>=2, count the distinct levels the modal code's
    cell k takes across N in that cell's band [theta^k, theta^(k+1)). A fine (place-value)
    cell uses >=2 levels across its band; a thermometer/dead cell uses 1.
    """
    Ns = sorted(modal_by_n)
    if not Ns:
        return {}
    W = n_cells if n_cells is not None else len(modal_by_n[Ns[0]])
    res: Dict[int, int] = {}
    for k in range(2, W):
        lo, hi = theta ** k, theta ** (k + 1)
        levels = set()
        for N in Ns:
            if lo <= N < hi:
                levels.add(_to_levels(modal_by_n[N])[k])
        res[k] = len(levels)
    return res


def per_cell_level_usage(codes: Sequence[Sequence[float]], n_cells: int,
                         min_frac: float = 0.05, dom_frac: float = 0.90) -> Dict[int, dict]:
    """D1 (collapse check). Over all sampled codes, per cell: distinct levels with >min_frac
    occupancy, the dominant-level fraction, and dead/degenerate flags.
      dead  = levels_used <= 1                 (a frozen cell -> outcome C)
      degen = levels_used <= 2 or max_frac>=dom_frac  (thermometer-ish)
    """
    arr = np.array([_to_levels(c) for c in codes])
    res: Dict[int, dict] = {}
    for k in range(n_cells):
        col = arr[:, k]
        _, counts = np.unique(col, return_counts=True)
        fracs = counts / counts.sum()
        used = int((fracs > min_frac).sum())
        max_frac = float(fracs.max())
        res[k] = {"levels_used": used, "max_frac": max_frac,
                  "dead": used <= 1, "degen": (used <= 2) or (max_frac >= dom_frac)}
    return res


def residual_fidelity(scratch_cell_by_n: Dict[int, Sequence[float]],
                      residual_cell_by_n: Dict[int, Sequence[float]]) -> Dict[int, float]:
    """D4. Per cell k, Spearman(scratch_cell_k(N), substrate_residual_k(N)) over N. The
    substrate residuals are the known radix-theta ground truth (scaffolded); a faithful
    write head keeps each cell a monotone function of its residual (toward A). Scrambled /
    low correlation = cipher or collapse. Returns per-cell Spearman rho.
    """
    Ns = sorted(set(scratch_cell_by_n) & set(residual_cell_by_n))
    if len(Ns) < 3:
        return {}
    W = len(scratch_cell_by_n[Ns[0]])
    res: Dict[int, float] = {}
    for k in range(W):
        s = np.array([float(scratch_cell_by_n[N][k]) for N in Ns])
        r = np.array([float(residual_cell_by_n[N][k]) for N in Ns])
        if s.std() < 1e-9 or r.std() < 1e-9:
            res[k] = float("nan")
            continue
        rs = np.argsort(np.argsort(s)).astype(float)
        rr = np.argsort(np.argsort(r)).astype(float)
        res[k] = float(np.corrcoef(rs, rr)[0, 1])
    return res


def classify_codebook(modal_by_n: Dict[int, Sequence[float]], n_cells: int,
                      theta: int = 3) -> Dict[str, object]:
    """Combine D1/D2/D3 into a coarse {A: place-value, B: cipher, C: thermometer/collapse}
    verdict. Heuristic — corroborate with the full inspect report, not a sole oracle.
    """
    codes = [modal_by_n[N] for N in sorted(modal_by_n)]
    usage = per_cell_level_usage(codes, n_cells)
    carry = carry_consistency(modal_by_n, theta)
    fine = high_cell_fineness(modal_by_n, theta, n_cells)
    n_dead = sum(1 for k in usage if usage[k]["dead"])
    high_fine = [v for k, v in fine.items()]
    place_like = (
        n_dead == 0
        and carry.get("boundary_hit_frac", 0) is not None
        and (carry.get("boundary_hit_frac") or 0) > 0.7
        and (carry.get("mean_increment_hamming") or 9) < 1.5
    )
    if n_dead > 0:
        verdict = "C_thermometer_or_collapse"
    elif place_like and high_fine and min(high_fine) >= 2:
        verdict = "A_place_value"
    else:
        verdict = "B_cipher"
    return {"verdict": verdict, "n_dead_cells": n_dead, "carry": carry,
            "high_cell_fineness": fine, "per_cell_usage": usage}
