# -*- coding: utf-8 -*-
"""Code → N mapping diagnostic.

Per-N: distinct codes used, dominant code, entropy of code distribution
Per-code: N range covered (collision width)
Overall: Spearman(code_rank, N), per-N consistency
"""
import argparse, sys, json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=600)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=150)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=args.q, qrange=args.qrange, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    agent.train(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=args.complex)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"  # uniform N for clean per-N stats
    rng = np.random.default_rng(0)

    n_to_codes = defaultdict(list)  # N -> list of code tuples
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _, scratch, _ = agent(inputs.to(device), ci.to(device))
        for b, m in enumerate(metas):
            n = m["N_alpha"]
            code = tuple(np.round(scratch[b].cpu().numpy(), 3).tolist())
            n_to_codes[n].append(code)

    print(f"\n=== {args.ckpt} ===")
    print(f"Eval: alpha uniform [1, {args.n_max}], {sum(len(v) for v in n_to_codes.values())} samples")

    # Per-N stats
    n_values = sorted(n_to_codes.keys())
    print(f"\n--- Per-N stats (sample of N values) ---")
    print(f"{'N':>4}  {'#samples':>9}  {'#distinct':>10}  {'most_common_code':<35}  {'most_common_freq':>5}")
    for n in n_values:
        if not (n in [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 125, 140, 150] or n % 25 == 0):
            continue
        codes = n_to_codes[n]
        if not codes:
            continue
        cnt = Counter(codes)
        mc, mc_count = cnt.most_common(1)[0]
        distinct = len(cnt)
        print(f"{n:>4}  {len(codes):>9}  {distinct:>10}  {str(list(mc)):<35}  {mc_count/len(codes):>5.2f}")

    # Per-code stats: N range
    code_to_ns = defaultdict(list)
    for n, codes in n_to_codes.items():
        for c in codes:
            code_to_ns[c].append(n)

    print(f"\n--- Top 10 most-used codes (by sample count) ---")
    code_counts = sorted(code_to_ns.items(), key=lambda x: -len(x[1]))[:10]
    for code, ns in code_counts:
        ns_arr = np.array(ns)
        print(f"  {str(list(code)):<35} count={len(ns):>4}  N range=[{ns_arr.min():>3}, {ns_arr.max():>3}] median={int(np.median(ns_arr)):>3} std={ns_arr.std():.1f}")

    # Spearman: code rank (by avg N) vs N
    code_avg_n = {c: np.mean(ns) for c, ns in code_to_ns.items()}
    sorted_codes = sorted(code_avg_n.keys(), key=lambda c: code_avg_n[c])
    code_rank = {c: i for i, c in enumerate(sorted_codes)}

    all_n = []
    all_rank = []
    for n, codes in n_to_codes.items():
        for c in codes:
            all_n.append(n)
            all_rank.append(code_rank[c])
    rho, pval = spearmanr(all_rank, all_n)
    print(f"\n--- Spearman(code_rank_by_avg_N, true_N) = {rho:.4f}  (p={pval:.2e}) ---")

    # Per-N consistency: how many distinct codes per N
    consistency = {}
    for n in n_values:
        codes = n_to_codes[n]
        if codes:
            distinct = len(set(codes))
            consistency[n] = distinct
    avg_distinct = np.mean(list(consistency.values()))
    max_distinct = max(consistency.values())
    print(f"\n--- Consistency: avg distinct codes per N = {avg_distinct:.2f}, max = {max_distinct} ---")
    print(f"    (1 = perfect 1:1 N→code mapping; >1 means model uses multiple codes for same N)")

    # Resolution: average N range per code
    resolutions = []
    for c, ns in code_to_ns.items():
        if len(ns) >= 5:  # only count codes used 5+ times
            resolutions.append(max(ns) - min(ns))
    print(f"\n--- Resolution: avg N range per common code = {np.mean(resolutions):.2f} ---")
    print(f"    (0 = perfect; high means same code maps to multiple N → low resolution)")

    print(f"\n--- Summary ---")
    print(f"  total distinct codes: {len(code_to_ns)}")
    print(f"  N range covered: [1, {args.n_max}]")
    print(f"  log2(distinct codes) = {np.log2(len(code_to_ns)):.2f} bits")
    print(f"  log2(N range)        = {np.log2(args.n_max):.2f} bits")


if __name__ == "__main__":
    main()
