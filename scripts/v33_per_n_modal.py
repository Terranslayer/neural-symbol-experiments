# scripts/v33_per_n_modal.py
"""V33 per-N modal scratch tuple.

DIAG_FOR: v33

Pulls scratch codes for every N in a sweep range and reports:
  - modal scratch tuple per N (most-frequent 5-cell pattern)
  - distinct codes count
  - per-cell value distribution {0, 0.5, 1.0}
  - max code share for any single N (mid-N collapse detector)

Usage:
  python scripts/v33_per_n_modal.py --ckpt <path> --task successor_prediction \\
    --n-min 5 --n-max 90 --n-samples 30
"""
import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--task", default="successor_prediction")
    parser.add_argument("--L", type=int, required=True,
                        help="Episode length (must match training --curriculum-L)")
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--n-max", type=int, default=90)
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Episodes per N value")
    args = parser.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
    )
    device = next(agent.parameters()).device
    scene_cfg.task = args.task
    agent.train(False)

    rng = np.random.default_rng(42)

    per_n_codes = collections.defaultdict(list)
    with torch.no_grad():
        for N in range(args.n_min, args.n_max + 1):
            # sample_training_batch samples N uniformly in [1, n_max_stage].
            # To pin N_alpha=N exactly, we call it with n_max_stage=N and
            # alpha_distribution="uniform" so every sample hits N.
            saved_dist = scene_cfg.alpha_distribution
            scene_cfg.alpha_distribution = "uniform"
            inputs, _, ci, metas = sample_training_batch(
                batch_size=args.n_samples, n_max_stage=N, config=scene_cfg, rng=rng,
            )
            scene_cfg.alpha_distribution = saved_dist

            _, scratch_q, _ = agent(inputs.to(device), ci.to(device))
            scratch_q_cpu = scratch_q.cpu()
            for b in range(args.n_samples):
                # Round to 1 decimal to canonicalize 0/0.5/1.0
                code = tuple(round(float(v), 1) for v in scratch_q_cpu[b])
                per_n_codes[N].append(code)

    print(f"\n{'N':>4} | {'modal_code':<25} | {'modal_share':<10} | distinct")
    print("-" * 60)
    all_codes = set()
    max_share = 0.0
    for N in sorted(per_n_codes.keys()):
        codes = per_n_codes[N]
        counts = collections.Counter(codes)
        modal_code, modal_count = counts.most_common(1)[0]
        share = modal_count / len(codes)
        distinct = len(counts)
        all_codes.update(codes)
        max_share = max(max_share, share)
        modal_str = "(" + ",".join(f"{v:.1f}" for v in modal_code) + ")"
        print(f"{N:>4} | {modal_str:<25} | {share:<10.2f} | {distinct}")

    print(f"\nTotal distinct codes across N range: {len(all_codes)}")
    print(f"Max modal share (mid-N collapse detector): {max_share:.2f}")


if __name__ == "__main__":
    main()
