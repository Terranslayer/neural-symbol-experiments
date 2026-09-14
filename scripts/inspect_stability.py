# -*- coding: utf-8 -*-
# DIAG_FOR: any | ANSWERS: codebook | INPUTS: <ckpt>
"""Stability + full distinct-code listing for a Phase-1 GRU checkpoint.

For each N value sampled, shows:
  - top-K most common code tuples and their frequency
  - total #distinct tuples seen for this N
  - "stability index" = freq of modal tuple (closer to 1.0 = more deterministic)

Also dumps the FULL distinct-code list ranked by total observation count, so
we can see codes beyond the top-10 collisions that the basic inspector lists.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.inspect_checkpoint import load_agent, collect_notes_per_N
from backend.core.scene import SceneConfig


def levels_for(qrange: str, q: int) -> np.ndarray:
    if qrange == "unit":
        return np.linspace(0.0, 1.0, q)
    return np.linspace(-1.0, 1.0, q)


def per_N_stability(per_N, levels: np.ndarray, n_max: int, top_k: int = 3, sample_step: int = 5):
    """For each sampled N, show the top-K most-common codes and their frequencies."""
    print("\n" + "=" * 110)
    print(f"PER-N STABILITY (top-{top_k} codes per N, with frequency)")
    print("=" * 110)
    print(f"{'N':>4} {'#samples':>8} {'#distinct':>9} {'stability':>10} | top codes (code: freq)")
    print("-" * 110)
    for N in [1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, n_max + 1, sample_step)):
        if N not in per_N:
            continue
        arrs = np.stack(per_N[N], axis=0)
        codes = [tuple(int(l) for l in np.abs(a[:, None] - levels[None, :]).argmin(axis=1)) for a in arrs]
        cnt = Counter(codes)
        n_total = len(codes)
        n_distinct = len(cnt)
        most = cnt.most_common(top_k)
        stability = most[0][1] / n_total
        codes_str = "  ".join(f"{c}:{v/n_total:.2f}" for c, v in most)
        print(f"{N:>4} {n_total:>8} {n_distinct:>9} {stability:>10.2f} | {codes_str}")


def all_distinct_codes(per_N, levels: np.ndarray, top_to_show: int = 50):
    """Dump the full list of distinct codes ranked by total occurrence count."""
    code_to_Ns = defaultdict(Counter)  # code -> Counter(N: count)
    for N, lst in per_N.items():
        for arr in lst:
            tup = tuple(int(l) for l in np.abs(arr[:, None] - levels[None, :]).argmin(axis=1))
            code_to_Ns[tup][N] += 1
    # Sort by total occurrence, descending
    ranked = sorted(code_to_Ns.items(), key=lambda kv: -sum(kv[1].values()))
    print("\n" + "=" * 110)
    print(f"ALL DISTINCT CODES RANKED BY TOTAL FREQUENCY (top {top_to_show})")
    print("=" * 110)
    print(f"{'rank':>4} {'code':>20} {'#samples':>9} {'#unique_N':>10} {'N range (min..max)':>20} | top-3 N")
    print("-" * 110)
    for rank, (code, Ns) in enumerate(ranked[:top_to_show], 1):
        total = sum(Ns.values())
        n_unique = len(Ns)
        n_min = min(Ns.keys())
        n_max_v = max(Ns.keys())
        top_Ns = ", ".join(f"N={n}({c})" for n, c in Ns.most_common(3))
        print(f"{rank:>4} {str(code):>20} {total:>9} {n_unique:>10} {n_min:>4}..{n_max_v:<13} | {top_Ns}")
    print(f"\nTotal distinct codes observed: {len(ranked)}")
    return ranked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--q", type=int, required=True)
    p.add_argument("--range", choices=["unit", "symmetric"], default="unit")
    p.add_argument("--L", type=int, default=250)
    p.add_argument("--complex", action="store_true")
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pfc-at-write", action="store_true")
    p.add_argument("--cell-type", default="gru")
    p.add_argument("--top-codes", type=int, default=50)
    args = p.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        Path(args.ckpt), q=args.q, qrange=args.range, L=args.L,
        complex_world=args.complex, cell_type=args.cell_type,
        pfc_at_write=args.pfc_at_write,
    )
    levels = levels_for(args.range, args.q)
    per_N = collect_notes_per_N(agent, scene_cfg, args.n_max, args.n_batches, args.batch_size, args.seed)
    per_N_stability(per_N, levels, args.n_max)
    all_distinct_codes(per_N, levels, args.top_codes)


if __name__ == "__main__":
    main()
