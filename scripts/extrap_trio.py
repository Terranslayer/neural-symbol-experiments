# -*- coding: utf-8 -*-
"""Custom extrapolation: trio-style beta (alpha +/- 1 or =) on far N values.

This is a proper test of "can the model extrapolate the trio compare task
to N values it never saw during training?". Default extrapolation uses
uniform beta which collapses with saturated codes; this version forces
beta to be near alpha so the question is precise comparison."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.inspect_checkpoint import load_agent
from backend.core.scene import build_episode


def trio_extrap_test(agent, scene_cfg, n_ranges, n_episodes_per_range=128, seed=0):
    rng = np.random.default_rng(seed)
    device = next(agent.parameters()).device
    out = {}
    for name, n_range in n_ranges.items():
        n_vals = list(n_range)
        if not n_vals:
            out[name] = float("nan")
            continue
        correct = total = 0
        with torch.no_grad():
            for _ in range(n_episodes_per_range):
                N_alpha = int(rng.choice(n_vals))
                n_max_feasible = (scene_cfg.L - 1) // 2
                beta_counts = []
                for _ in range(scene_cfg.K):
                    roll = rng.random()
                    if roll < 1/3:
                        nb = N_alpha
                    elif roll < 2/3:
                        nb = min(N_alpha + 1, n_max_feasible)
                    else:
                        nb = max(N_alpha - 1, 1)
                    beta_counts.append(nb)
                inp, lbl, cidx, _ = build_episode(N_alpha, beta_counts, scene_cfg, rng)
                inp = inp.unsqueeze(0).to(device)
                cidx = cidx.unsqueeze(0).to(device)
                logits, _, _ = agent(inp, cidx)
                preds = logits.argmax(-1).cpu().squeeze(0)
                correct += (preds == lbl).sum().item()
                total += lbl.numel()
        out[name] = correct / total
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", choices=["unit", "symmetric"], default="unit")
    p.add_argument("--L", type=int, default=250)
    p.add_argument("--complex", action="store_true")
    p.add_argument("--n-train-max", type=int, default=100)
    p.add_argument("--pfc-at-write", action="store_true")
    p.add_argument("--pfc-compare-drop-raw", action="store_true")
    p.add_argument("--episodes", type=int, default=128)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        Path(args.ckpt), q=args.q, qrange=args.range, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    n_max_feasible = (scene_cfg.L - 1) // 2  # min spacing 2 requires L >= 2N+1
    far_lower = min(int(args.n_train_max * 1.2) + 1, n_max_feasible)
    far_upper = n_max_feasible
    test_ranges = {
        "in_range": range(1, args.n_train_max + 1),
    }
    if args.n_train_max + 1 < far_lower:
        test_ranges["near_extrap"] = range(args.n_train_max + 1, far_lower + 1)
    if far_lower < far_upper:
        test_ranges["far_extrap"] = range(far_lower + 1, far_upper + 1)
    out = trio_extrap_test(agent, scene_cfg, test_ranges,
                           n_episodes_per_range=args.episodes)
    print()
    for k, v in out.items():
        n_range = test_ranges[k]
        print(f"  {k:14s} (N={n_range.start}..{n_range.stop-1}): trio_acc={v:.3f}")


if __name__ == "__main__":
    main()
