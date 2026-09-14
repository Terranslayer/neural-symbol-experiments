# -*- coding: utf-8 -*-
"""
Confusion matrix + per-gap accuracy + codebook usage + cycle_loss diagnostic.

Runs on a saved ckpt (no training disruption). Outputs:
  - 3x3 confusion matrix (true G/L/E vs pred G/L/E)
  - Per-gap accuracy for gap ∈ {0..4}
  - Per-position scratch slot value distribution
  - Cycle loss + order loss separated values
"""
import argparse, sys, json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, build_episode, sample_beta_count  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=600)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=150)
    p.add_argument("--n-batches", type=int, default=16)
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
    cfg.alpha_distribution = "log_uniform"  # match training
    rng = np.random.default_rng(args.seed)

    # 3x3 confusion: rows=true, cols=pred; classes G=0, L=1, E=2
    confusion = np.zeros((3, 3), dtype=int)
    # per-gap acc: gap from 0 to 4
    gap_correct = {0:0, 1:0, 2:0, 3:0, 4:0}
    gap_total   = {0:0, 1:0, 2:0, 3:0, 4:0}
    # collect scratches
    all_scratch = []
    # collect cycle_loss and order_loss
    cycle_losses = []
    order_losses = []

    for _ in range(args.n_batches):
        episodes = []
        for _ in range(args.batch_size):
            N_alpha = int(rng.integers(1, args.n_max + 1))
            beta_counts = [
                sample_beta_count(N_alpha, args.n_max, cfg, rng)
                for _ in range(cfg.K)
            ]
            try:
                inp, lbl, cidx, meta = build_episode(N_alpha, beta_counts, cfg, rng)
                episodes.append((inp, lbl, cidx, meta))
            except ValueError:
                continue
        if not episodes:
            continue
        inputs = torch.stack([e[0] for e in episodes]).to(device)
        labels = torch.stack([e[1] for e in episodes]).to(device)  # (B, K)
        compare_idx = torch.stack([e[2] for e in episodes]).to(device)
        metas = [e[3] for e in episodes]

        with torch.no_grad():
            logits, scratch_pad, _ = agent(inputs, compare_idx)

        # logits: (B, K, 3) — predict argmax
        preds = logits.argmax(dim=-1)  # (B, K)
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        scratch_np = scratch_pad.cpu().numpy()  # (B, W)
        all_scratch.append(scratch_np)

        # Confusion matrix
        for b in range(labels_np.shape[0]):
            for k in range(labels_np.shape[1]):
                t, p = int(labels_np[b, k]), int(preds_np[b, k])
                confusion[t, p] += 1
                # Per-gap from meta
                N_alpha = metas[b]["N_alpha"]
                N_beta = metas[b]["beta_counts"][k]
                gap = abs(N_alpha - N_beta)
                if gap > 4:
                    continue
                gap_total[gap] += 1
                if t == p:
                    gap_correct[gap] += 1

        # Cycle loss + order loss
        cy_loss = getattr(agent, "_last_cycle_loss", None)
        if cy_loss is not None:
            cycle_losses.append(float(cy_loss.item()))
        order_loss = F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1)
        )
        order_losses.append(float(order_loss.item()))

    # ===== Reports =====
    print(f"\n{'='*70}")
    print(f"Diagnostic: {args.ckpt}")
    print(f"  alpha range [1, {args.n_max}], {sum(gap_total.values())} total compare instances")
    print(f"{'='*70}\n")

    print("--- 3x3 Confusion Matrix (true rows × pred cols) ---")
    print(f"{'':>10} {'pred_G':>8} {'pred_L':>8} {'pred_E':>8} {'total':>8} {'recall':>8}")
    class_names = ["G", "L", "E"]
    total_correct = 0
    total = 0
    for i, name in enumerate(class_names):
        row_total = confusion[i].sum()
        row_diag = confusion[i, i]
        recall = row_diag / max(row_total, 1)
        total_correct += row_diag
        total += row_total
        print(f"{'true_'+name:>10} {confusion[i,0]:>8d} {confusion[i,1]:>8d} {confusion[i,2]:>8d} {row_total:>8d} {recall:>8.3f}")
    print(f"  Overall accuracy: {total_correct/max(total,1):.4f}")
    print(f"  Class precision: G={confusion[0,0]/max(confusion[:,0].sum(),1):.3f}  "
          f"L={confusion[1,1]/max(confusion[:,1].sum(),1):.3f}  "
          f"E={confusion[2,2]/max(confusion[:,2].sum(),1):.3f}")

    print("\n--- Per-gap Accuracy ---")
    for g in sorted(gap_total.keys()):
        if gap_total[g] > 0:
            acc = gap_correct[g] / gap_total[g]
            print(f"  gap={g}: {acc:.4f}  ({gap_correct[g]}/{gap_total[g]})")

    print("\n--- Loss components ---")
    print(f"  L_order (CE): {np.mean(order_losses):.4f}")
    if cycle_losses:
        print(f"  L_cycle:      {np.mean(cycle_losses):.4f}")
    else:
        print("  L_cycle:      N/A (cycle_loss_lambda=0 or not enabled)")

    print("\n--- Codebook usage (per scratch slot) ---")
    notes = np.concatenate(all_scratch, axis=0)  # (N, W)
    W = notes.shape[1]
    for j in range(W):
        col = notes[:, j]
        unique, cnt = np.unique(np.round(col, 2), return_counts=True)
        total = cnt.sum()
        dist = ", ".join(f"{u:.2f}={c/total*100:.1f}%" for u, c in zip(unique, cnt))
        print(f"  pos {j}: {dist}")

    code_strs = [tuple(np.round(row, 3).tolist()) for row in notes]
    distinct = len(set(code_strs))
    print(f"  distinct codes: {distinct}")


if __name__ == "__main__":
    main()
