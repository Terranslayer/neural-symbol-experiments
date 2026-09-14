# -*- coding: utf-8 -*-
"""Per-pos correlation of scratch with N + log(N) + distinct-codes count."""
import argparse
import json
import sys
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
    p.add_argument("--n-batches", type=int, default=8)
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
    n = np.array(all_n)
    notes = np.concatenate(all_notes, axis=0)  # (N, W)

    W = notes.shape[1]
    pos_corr_N = [float(np.corrcoef(notes[:, j], n)[0, 1]) for j in range(W)]
    log_n = np.log(n + 1)
    pos_corr_logN = [float(np.corrcoef(notes[:, j], log_n)[0, 1]) for j in range(W)]

    # distinct codes (preserve fractional levels — round to 3 decimals to dedupe noise)
    code_strs = [tuple(np.round(row, 3).tolist()) for row in notes]
    distinct = len(set(code_strs))

    out = {
        "ckpt": str(args.ckpt),
        "n_max": args.n_max,
        "n_samples": len(n),
        "distinct_codes": distinct,
        "pos_corr_N": [round(x, 3) for x in pos_corr_N],
        "pos_corr_logN": [round(x, 3) for x in pos_corr_logN],
        "notes_min": float(notes.min()),
        "notes_max": float(notes.max()),
        "notes_mean_per_pos": [round(float(notes[:, j].mean()), 3) for j in range(W)],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
