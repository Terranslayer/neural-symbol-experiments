# -*- coding: utf-8 -*-
# DIAG_FOR: any | ANSWERS: lesion | INPUTS: <ckpt>
"""Lesion test: measure how much alpha-info actually flows via h_alpha at compare time.

Procedure for each checkpoint:
  1. Sample a batch of episodes; record correct labels.
  2. FULL forward: normal accuracy.
  3. SHUFFLED-alpha: replace alpha-phase inputs across batch (random pairing). Scratch
     gets written from the wrong alpha; h_alpha at compare time encodes the wrong alpha.
     beta-phase and labels stay original. If accuracy stays high → model is NOT
     using h_alpha at compare; it's using beta-only prior shortcut. If accuracy drops
     to ~chance (1/3) → h_alpha carries the answer.
  4. ZERO-alpha: replace alpha-phase with zeros. Hardest version of the lesion.
     Residual accuracy = pure beta-only prior baseline.

Reports per checkpoint:
  full_acc, shuffled_acc, zero_acc, h_alpha_contribution = full - shuffled
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.inspect_checkpoint import load_agent
from backend.core.scene import sample_training_batch


def evaluate(agent, scene_cfg, n_max, n_batches, batch_size, seed, mode, loss_type="ce"):
    rng = np.random.default_rng(seed)
    forget_end = scene_cfg.L + scene_cfg.W + scene_cfg.T_gap
    device = next(agent.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inputs, labels, cidx, _ = sample_training_batch(
                batch_size, n_max, scene_cfg, rng,
            )
            B = inputs.shape[0]
            if mode == "full":
                inp_use = inputs
            elif mode == "shuffle":
                perm = torch.randperm(B)
                inp_use = inputs.clone()
                inp_use[:, :forget_end, :] = inputs[perm, :forget_end, :]
            elif mode == "zero":
                inp_use = inputs.clone()
                inp_use[:, :forget_end, :] = 0.0
            else:
                raise ValueError(mode)
            logits, _, _ = agent(inp_use.to(device), cidx.to(device))
            logits = logits.cpu()
            if loss_type in ("regression", "gap_mse4"):
                pred_scalar = logits[..., 0]
                preds = torch.full(pred_scalar.shape, 2, dtype=torch.long)
                preds = torch.where(pred_scalar > 0.5, torch.zeros_like(preds), preds)
                preds = torch.where(pred_scalar < -0.5, torch.ones_like(preds), preds)
            elif loss_type == "rl_top1":
                # rl_top1: lesion measures shift in preference-score argmax accuracy
                # vs. gap=0. With per-block dilution from extra options, this is a
                # "did the picked block have N_β = N_α?" metric.
                scores = logits[..., 0]
                # Without explicit metas in this script, we just use argmax index;
                # caller compares against label==EQUAL position. For full reward eval,
                # use train_phase1.validate_stage instead.
                preds = torch.full(scores.shape, 2, dtype=torch.long)
            else:
                preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", choices=["unit", "symmetric"], default="unit")
    p.add_argument("--L", type=int, default=250)
    p.add_argument("--complex", action="store_true")
    p.add_argument("--n-max", type=int, default=100)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pfc-at-write", action="store_true")
    p.add_argument("--pfc-compare-drop-raw", action="store_true",
                   help="Set if checkpoint was trained with --pfc-compare-drop-raw")
    p.add_argument("--task", choices=["compare", "discrim", "trio", "trio_wide"], default="discrim",
                   help="Beta sampling distribution to use for evaluation (must match training)")
    p.add_argument("--trio-wide-gap-max", type=int, default=4,
                   help="(--task trio_wide) Maximum |N_β - N_α| offset; gap distribution is "
                        "uniform over {-gap_max..+gap_max}. Default 4 matches training spec.")
    p.add_argument("--alpha-dist", choices=["uniform", "log_uniform"], default="uniform",
                   help="Alpha sampling distribution for evaluation (must match training)")
    p.add_argument("--loss-type",
                   choices=["ce", "ordinal", "regression", "gap_mse4", "rl_top1"],
                   default="ce",
                   help="Match training loss. Scalar-output heads (regression, gap_mse4) use "
                        "channel-0 threshold (>0.5=GREATER, <-0.5=LESS, else=EQUAL). "
                        "rl_top1: lesion accuracy is a coarse proxy (no per-block label).")
    args = p.parse_args()

    agent, _, _ = load_agent(
        Path(args.ckpt), q=args.q, qrange=args.range, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    # Build scene_cfg matching the training task (load_agent always returns discrim default)
    from backend.core.scene import SceneConfig
    if args.task == "trio":
        scene_cfg = SceneConfig.trio_preset(L=args.L, complex_world=args.complex)
    elif args.task == "trio_wide":
        scene_cfg = SceneConfig.trio_wide_preset(
            L=args.L, complex_world=args.complex, gap_max=args.trio_wide_gap_max)
    elif args.task == "discrim":
        scene_cfg = SceneConfig.discrimination_preset(L=args.L, complex_world=args.complex)
    else:
        scene_cfg = SceneConfig(L=args.L, complex_world=args.complex)
    scene_cfg.alpha_distribution = args.alpha_dist
    print(f"  eval task={args.task} alpha_dist={args.alpha_dist}")

    print()
    full = evaluate(agent, scene_cfg, args.n_max, args.n_batches,
                    args.batch_size, args.seed, "full", loss_type=args.loss_type)
    shuffled = evaluate(agent, scene_cfg, args.n_max, args.n_batches,
                        args.batch_size, args.seed, "shuffle", loss_type=args.loss_type)
    zeroed = evaluate(agent, scene_cfg, args.n_max, args.n_batches,
                      args.batch_size, args.seed, "zero", loss_type=args.loss_type)
    print(f"  full          : {full:.3f}")
    print(f"  shuffled-alpha    : {shuffled:.3f}  (replaces alpha scene with random one)")
    print(f"  zeroed-alpha      : {zeroed:.3f}  (replaces alpha scene with zeros)")
    print(f"  h_alpha contrib.  : {full - shuffled:.3f}  (= full - shuffled)")
    print(f"  beta-only floor  : {zeroed:.3f}  (= pure beta prior baseline)")


if __name__ == "__main__":
    main()
