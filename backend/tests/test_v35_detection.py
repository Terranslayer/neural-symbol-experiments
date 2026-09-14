# -*- coding: utf-8 -*-
"""TDD: V35 detection helpers (spec 2026-06-05 §6) — distinguish base-theta place-value
vs lookup cipher vs thermometer/collapse from a per-N modal codebook.

D3 carry-consistency is the PRIMARY base-theta-vs-cipher discriminator: place-value has a
defining algebraic signature on N->N+1 (sparse, boundary-localized carry) a cipher can't fake.
"""
import numpy as np
from backend.evaluation.v35_detection import (
    carry_consistency, high_cell_fineness, per_cell_level_usage,
)


def _base3_code(N, W=5):
    """Perfect radix-3 positional code: cell k = (N // 3^k) % 3, mapped to {0,0.5,1.0}."""
    digs, x = [], N
    for _ in range(W):
        digs.append(x % 3); x //= 3
    return tuple(d / 2.0 for d in digs)


def test_carry_consistency_base3_vs_cipher():
    Ns = list(range(1, 41))
    base3 = {N: _base3_code(N) for N in Ns}
    codes = [base3[N] for N in Ns]
    perm = list(range(len(Ns)))
    perm = perm[7:] + perm[:7]                       # fixed rotation -> unstructured cipher
    cipher = {Ns[i]: codes[perm[i]] for i in range(len(Ns))}

    b = carry_consistency(base3, theta=3)
    c = carry_consistency(cipher, theta=3)

    # base-3: sparse, boundary-localized carries
    assert b["mean_increment_hamming"] < 2.0
    assert b["low_cell_only_frac"] > 0.5
    assert b["boundary_hit_frac"] > 0.9
    # cipher: denser increments, carries not boundary-aligned
    assert c["mean_increment_hamming"] > b["mean_increment_hamming"]
    assert c["boundary_hit_frac"] < 0.6


def test_high_cell_fineness_base3_active():
    base3 = {N: _base3_code(N) for N in range(1, 91)}
    fine = high_cell_fineness(base3, theta=3)
    # high cells use >=2 distinct levels across their band (a thermometer/dead cell uses 1)
    assert fine[2] >= 2          # band [9,27)
    assert fine[3] >= 2          # band [27,81)


def test_per_cell_level_usage_flags_dead_and_alive():
    base3 = {N: _base3_code(N) for N in range(1, 28)}   # 1..27
    codes = [base3[N] for N in range(1, 28)]
    usage = per_cell_level_usage(codes, n_cells=5)
    assert usage[0]["levels_used"] == 3 and not usage[0]["dead"]   # cell0 cycles 0/1/2
    assert usage[4]["dead"]                                         # 3^4=81 > 27 -> never active
