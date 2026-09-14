# -*- coding: utf-8 -*-
"""Dump representative scratch notes per N value for a ckpt + per-pos stats.

Used to compare how encoding evolves across training stages.
"""
import argparse, sys, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa: E402
from scripts.inspect_checkpoint import load_agent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=81)
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    p.add_argument("--n-show-per-bucket", type=int, default=5)
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
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    all_n, all_notes = [], []
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _, scratch, _ = agent(inputs.to(device), ci.to(device))
        all_n.extend([m["N_alpha"] for m in metas])
        all_notes.append(scratch.detach().cpu().numpy())
    n_arr = np.array(all_n)
    notes = np.concatenate(all_notes, axis=0)  # (B, W)
    W = notes.shape[1]

    # Bucket by N range
    n_max = int(n_arr.max())
    buckets = [
        ("very-small (1-3)", lambda x: 1 <= x <= 3),
        ("small (4-9)", lambda x: 4 <= x <= 9),
        ("mid-low (10-20)", lambda x: 10 <= x <= 20),
        ("mid (21-40)", lambda x: 21 <= x <= 40),
        ("mid-high (41-60)", lambda x: 41 <= x <= 60),
        ("high (61-80)", lambda x: 61 <= x <= 80),
        ("max (>=81)", lambda x: x >= 81),
    ]

    print(f"\n=== ckpt: {args.ckpt} ===")
    print(f"  n_samples = {len(n_arr)}, n_max in eval = {n_max}, W = {W}")

    # Per-pos stats
    pc_N = [round(float(np.corrcoef(notes[:, j], n_arr)[0, 1]), 3) for j in range(W)]
    pc_logN = [round(float(np.corrcoef(notes[:, j], np.log(n_arr+1))[0, 1]), 3) for j in range(W)]
    pos_means = [round(float(notes[:, j].mean()), 3) for j in range(W)]
    pos_stds = [round(float(notes[:, j].std()), 3) for j in range(W)]

    code_strs = [tuple(np.round(row, 3).tolist()) for row in notes]
    distinct = len(set(code_strs))

    print(f"\n  pos_corr_N    : {pc_N}")
    print(f"  pos_corr_logN : {pc_logN}")
    print(f"  pos mean      : {pos_means}")
    print(f"  pos std       : {pos_stds}")
    print(f"  distinct codes: {distinct}")

    # Sample notes per bucket
    print(f"\n  --- Representative notes per N bucket ---")
    for name, fn in buckets:
        idx = np.where([fn(int(n)) for n in n_arr])[0]
        if len(idx) == 0:
            continue
        # Group by exact N within bucket
        by_n = defaultdict(list)
        for i in idx:
            by_n[int(n_arr[i])].append(i)
        # Sample up to n_show_per_bucket distinct N values
        n_keys = sorted(by_n.keys())
        if len(n_keys) > args.n_show_per_bucket:
            chosen_n = [n_keys[0], n_keys[len(n_keys)//2], n_keys[-1]]
        else:
            chosen_n = n_keys
        if not chosen_n:
            continue
        print(f"\n  [{name}]")
        for n_val in chosen_n:
            idxs = by_n[n_val][:3]
            for i in idxs:
                code = tuple(np.round(notes[i], 3).tolist())
                print(f"    N={n_val:>3} : {list(code)}")

    # Top-5 most common codes
    from collections import Counter
    counter = Counter(code_strs)
    print(f"\n  --- Top-5 most common codes (count) ---")
    for code, cnt in counter.most_common(5):
        # Find example N values for this code
        ns_with_code = [int(n_arr[i]) for i, c in enumerate(code_strs) if c == code]
        n_min, n_max_obs = min(ns_with_code), max(ns_with_code)
        print(f"    {list(code)}  count={cnt:>3}  N range=[{n_min}, {n_max_obs}]")


if __name__ == "__main__":
    main()
