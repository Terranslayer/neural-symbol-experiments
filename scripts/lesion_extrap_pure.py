# -*- coding: utf-8 -*-
"""
Pure-extrap lesion: alpha restricted to extrap range only.

Measures TRUE alpha-emergence at extrap range, not mixed in-dist + extrap.
Reports full / shuffled-alpha / zeroed-alpha at the SPECIFIED N_alpha range.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, build_episode, sample_beta_count  # noqa: E402
from scripts.inspect_checkpoint import load_agent  # noqa: E402


def _build_gap_target(metas, device):
    return torch.tensor(
        [[m["N_alpha"] - nb for nb in m["beta_counts"]] for m in metas],
        dtype=torch.float32, device=device,
    )


def _threshold_acc(logits, gap_target):
    pred = logits[..., 0]
    pred_class = torch.full_like(pred, fill_value=2, dtype=torch.long)
    pred_class = torch.where(pred > 0.5, torch.zeros_like(pred_class), pred_class)
    pred_class = torch.where(pred < -0.5, torch.ones_like(pred_class), pred_class)
    true_class = torch.full_like(gap_target, fill_value=2, dtype=torch.long)
    true_class = torch.where(gap_target > 0, torch.zeros_like(true_class), true_class)
    true_class = torch.where(gap_target < 0, torch.ones_like(true_class), true_class)
    return float((pred_class == true_class).float().mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-min", type=int, required=True, help="alpha lower bound (inclusive)")
    p.add_argument("--n-max", type=int, required=True, help="alpha upper bound (inclusive)")
    p.add_argument("--beta-max", type=int, default=None, help="beta upper bound; default = n_max")
    p.add_argument("--beta-mode", choices=["uniform", "trio_wide"], default="trio_wide",
                   help="'trio_wide': beta = alpha+-gap (matches training); 'uniform': legacy")
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
    beta_max = args.beta_max if args.beta_max else args.n_max

    rng = np.random.default_rng(args.seed)

    # 3 conditions: full / shuffled-alpha / zeroed-alpha
    full_correct = shuf_correct = zero_correct = 0
    total = 0

    for _ in range(args.n_batches):
        episodes = []
        for _ in range(args.batch_size):
            N_alpha = int(rng.integers(args.n_min, args.n_max + 1))
            if args.beta_mode == "trio_wide":
                beta_counts = [
                    sample_beta_count(N_alpha, beta_max, cfg, rng)
                    for _ in range(cfg.K)
                ]
            else:
                beta_counts = rng.integers(1, beta_max + 1, size=cfg.K).tolist()
            try:
                inp, lbl, cidx, meta = build_episode(N_alpha, beta_counts, cfg, rng)
                episodes.append((inp, lbl, cidx, meta))
            except ValueError:
                continue
        if not episodes:
            continue
        inputs = torch.stack([e[0] for e in episodes]).to(device)
        compare_idx = torch.stack([e[2] for e in episodes]).to(device)
        metas = [e[3] for e in episodes]
        gap_target = _build_gap_target(metas, device)

        # FULL
        with torch.no_grad():
            logits, _, _ = agent(inputs, compare_idx)
        full_acc = _threshold_acc(logits, gap_target)

        # SHUFFLED-α: replace α phase with random alpha from same range
        shuf_inputs = inputs.clone()
        L = cfg.L
        # build random α scenes for shuffle replacement
        shuf_alpha_scenes = []
        for _ in range(len(episodes)):
            n_a_random = int(rng.integers(args.n_min, args.n_max + 1))
            beta_counts_dummy = [1] * cfg.K  # not used
            try:
                ri, _, _, _ = build_episode(n_a_random, beta_counts_dummy, cfg, rng)
                shuf_alpha_scenes.append(ri[:L])
            except ValueError:
                shuf_alpha_scenes.append(inputs[0, :L].cpu())
        shuf_alpha_t = torch.stack(shuf_alpha_scenes).to(device)
        shuf_inputs[:, :L, :] = shuf_alpha_t

        with torch.no_grad():
            logits_shuf, _, _ = agent(shuf_inputs, compare_idx)
        shuf_acc = _threshold_acc(logits_shuf, gap_target)

        # ZEROED-α: zero out α phase
        zero_inputs = inputs.clone()
        zero_inputs[:, :L, 0] = 0
        with torch.no_grad():
            logits_zero, _, _ = agent(zero_inputs, compare_idx)
        zero_acc = _threshold_acc(logits_zero, gap_target)

        n_samples = gap_target.numel()
        full_correct += full_acc * n_samples
        shuf_correct += shuf_acc * n_samples
        zero_correct += zero_acc * n_samples
        total += n_samples

    full_acc = full_correct / total
    shuf_acc = shuf_correct / total
    zero_acc = zero_correct / total

    print(f"  ckpt: {args.ckpt}")
    print(f"  alpha range: [{args.n_min}, {args.n_max}]  (PURE — no in-dist mix)")
    print(f"  beta sampling: {args.beta_mode} (max={beta_max})")
    print(f"  total samples: {total}")
    print(f"")
    print(f"  full              : {full_acc:.4f}")
    print(f"  shuffled-alpha    : {shuf_acc:.4f}  ← α-presence baseline")
    print(f"  zeroed-alpha      : {zero_acc:.4f}  ← β-only floor")
    print(f"")
    print(f"  full - zero  (α total contribution)  : {full_acc - zero_acc:+.4f}")
    print(f"  full - shuf  (α VALUE encoding)      : {full_acc - shuf_acc:+.4f}  ← TRUE N-extrap signal")
    print(f"  shuf - zero  (α PRESENCE encoding)   : {shuf_acc - zero_acc:+.4f}")


if __name__ == "__main__":
    main()
