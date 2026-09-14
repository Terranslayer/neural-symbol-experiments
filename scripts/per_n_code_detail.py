"""Per-N detailed code listing — top codes for each N value individually.

Shows for each N: top-5 most common scratch codes + their freq.
Reveals whether N has a "signature code" (good 1-to-1) or scattered.
"""
import argparse, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    p.add_argument("--cnn-pfc-only", action="store_true")
    p.add_argument("--v7-mode", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=args.q, qrange=args.qrange, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
        cnn_pfc_only=args.cnn_pfc_only, v7_mode=args.v7_mode,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=args.complex)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    n_to_codes = defaultdict(list)
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

    print(f"\n=== {args.ckpt.name} per-N detail ===\n")
    print(f"{'N':>3}  {'#sam':>5}  {'#dist':>5}  top-{args.top_k} codes")
    print("-" * 90)
    for n in sorted(n_to_codes.keys()):
        codes = n_to_codes[n]
        cnt = Counter(codes)
        top = cnt.most_common(args.top_k)
        n_total = len(codes)
        n_dist = len(cnt)
        # First line: header
        first_code, first_freq = top[0]
        first_line = f"{n:>3}  {n_total:>5}  {n_dist:>5}  {str(list(first_code)):<30} {first_freq/n_total:>5.2f}"
        print(first_line)
        for code, freq in top[1:]:
            print(f"  {'':>17}{str(list(code)):<30} {freq/n_total:>5.2f}")
        print()

    # Cross-N overlap analysis
    print("\n=== Codes shared across N values (N range > 0) ===")
    code_to_n_set = defaultdict(set)
    for n, codes in n_to_codes.items():
        for c in codes:
            code_to_n_set[c].add(n)
    shared_codes = [(c, ns) for c, ns in code_to_n_set.items() if len(ns) >= 3]
    shared_codes.sort(key=lambda x: -len(x[1]))
    print(f"{'Code':<32}  N values it represents")
    for c, ns in shared_codes[:15]:
        print(f"  {str(list(c)):<30}  {sorted(ns)}")

    # Pure N codes (used for ONE N only)
    print("\n=== Pure-N codes (used for exactly 1 N value, 5+ times) ===")
    pure_codes = [(c, list(ns)[0], cnt) for c, ns in code_to_n_set.items() if len(ns) == 1]
    pure_codes_with_freq = []
    for c, n, _ in pure_codes:
        freq = Counter(n_to_codes[n])[c]
        if freq >= 5:
            pure_codes_with_freq.append((c, n, freq))
    pure_codes_with_freq.sort(key=lambda x: -x[2])
    for c, n, f in pure_codes_with_freq[:15]:
        print(f"  {str(list(c)):<30}  N={n}  count={f}")


if __name__ == "__main__":
    main()
