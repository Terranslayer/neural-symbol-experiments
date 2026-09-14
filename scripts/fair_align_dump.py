# -*- coding: utf-8 -*-
"""
Quick fair_base_alignment dump for one ckpt.
Outputs JSON line with best_base, best_alignment, all_alignments.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa: E402
from backend.evaluation.mi_matrix import base_scanning_mi  # noqa: E402
from scripts.inspect_checkpoint import load_agent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=81)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
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

    rng = np.random.default_rng(args.seed)
    all_n: list = []
    all_notes: list = []

    for _ in range(args.n_batches):
        inputs, _, compare_idx, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        inputs = inputs.to(device)
        compare_idx = compare_idx.to(device)
        with torch.no_grad():
            _, scratch_pad, _ = agent(inputs, compare_idx)
        all_n.extend([m["N_alpha"] for m in metas])
        all_notes.append(scratch_pad.detach().cpu().numpy())

    n_arr = np.array(all_n)
    notes_arr = np.concatenate(all_notes, axis=0)
    scan = base_scanning_mi(n_arr, notes_arr, candidate_bases=[2, 3, 4, 5, 6, 7, 8, 10])

    print(json.dumps({
        "ckpt": str(args.ckpt),
        "n_max": args.n_max,
        "n_samples": len(n_arr),
        "best_base": int(scan["best_base"]),
        "best_alignment": float(scan["best_alignment"]),
        "all_alignments": {int(k): float(v) for k, v in scan["all_alignments"].items()},
    }))


if __name__ == "__main__":
    main()
