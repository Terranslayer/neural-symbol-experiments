"""V5b internal diagnostic: probe pre-pool CNN responses for hidden HAT structure.

Hypothesis: pool (max/mean) is monotonic in N. If CNN learns "exactly N spike"
detector internally, max/mean would still report monotonic. Need richer pool
statistics to find HAT.

For each channel, after training, compute per-N statistics on raw conv output:
  - max         (current MaxPool, monotonic if cascade)
  - mean        (current MeanPool, monotonic if density signal)
  - top3_mean   (mean of top-3 strongest responses)
  - top10_mean  (mean of top-10)
  - count_above_50pct_global_max (# positions firing strong)
  - std         (response variance across T)
  - sparsity    (fraction of T with response > 1e-3)

Any of these statistic showing HAT (peak at N=2, low at N=1, N=3) → CNN does
have hidden cardinality detector, but pool kills it. Then fix is attention pool
or learned pool. None showing HAT → CNN truly cannot do spindle in single layer.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v5b model
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.subitizing_v5b import SubitizingV5b, generate_signal, make_batch  # noqa


def collect_raw_responses(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    """Returns dict: per-N (n_per_N, C, T) tensor of raw conv responses (post-ReLU)."""
    model.train(False)
    out = {}
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            _, _, _, feats = model(x)  # (B, 24, T)
            out[int(N)] = feats.cpu().numpy()
    return out


def compute_per_channel_stats(responses_per_N, k_top=(3, 10), threshold_frac=0.5):
    """For each channel and each N, compute multiple aggregation statistics.

    Returns: dict statistic -> (C, K_n) numpy array.
    """
    n_choices = sorted(responses_per_N.keys())
    sample_resp = responses_per_N[n_choices[0]]
    C, T = sample_resp.shape[1], sample_resp.shape[2]
    K_n = len(n_choices)

    stats = {
        "max": np.zeros((C, K_n)),
        "mean": np.zeros((C, K_n)),
        "std": np.zeros((C, K_n)),
        "sparsity": np.zeros((C, K_n)),
    }
    for k in k_top:
        stats[f"top{k}_mean"] = np.zeros((C, K_n))
    stats[f"count_above_50pct"] = np.zeros((C, K_n))

    # Global max per channel (across all N and all samples) for normalization
    global_max_per_ch = np.zeros(C)
    for N in n_choices:
        r = responses_per_N[N]  # (B, C, T)
        for c in range(C):
            global_max_per_ch[c] = max(global_max_per_ch[c], r[:, c, :].max())

    for nj, N in enumerate(n_choices):
        r = responses_per_N[N]  # (B, C, T)
        for c in range(C):
            chans = r[:, c, :]  # (B, T)
            stats["max"][c, nj] = chans.max(axis=-1).mean()
            stats["mean"][c, nj] = chans.mean(axis=-1).mean()
            stats["std"][c, nj] = chans.std(axis=-1).mean()
            stats["sparsity"][c, nj] = (chans > 1e-3).mean()
            for k in k_top:
                topk = np.partition(chans, -k, axis=-1)[:, -k:]  # (B, k)
                stats[f"top{k}_mean"][c, nj] = topk.mean()
            thresh = threshold_frac * global_max_per_ch[c]
            if thresh > 1e-6:
                stats["count_above_50pct"][c, nj] = (chans > thresh).sum(axis=-1).mean()
    return stats, global_max_per_ch


def is_hat(row, eps=1e-3):
    """Strict HAT: peak at internal index, strictly above both neighbors."""
    if len(row) < 3:
        return False
    pmax = int(np.argmax(row))
    return 0 < pmax < len(row) - 1 and row[pmax] > row[pmax - 1] + eps and row[pmax] > row[pmax + 1] + eps


def report_stats(stats, n_choices, channels_per_scale):
    K_n = len(n_choices)
    C = stats["max"].shape[0]
    cps = channels_per_scale
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]

    # Summary first
    print(f"\n{'='*72}")
    print("=== HAT count per statistic (across {} channels) ===".format(C))
    print(f"{'='*72}")
    for stat_name in ["max", "mean", "top3_mean", "top10_mean", "std", "sparsity", "count_above_50pct"]:
        arr = stats[stat_name]
        hats = sum(is_hat(arr[c]) for c in range(C))
        # also report monotonic up/down for context
        mono_up = sum(1 for c in range(C) if np.all(np.diff(arr[c]) >= -1e-3) and not np.all(np.diff(arr[c]) <= 1e-3))
        mono_dn = sum(1 for c in range(C) if np.all(np.diff(arr[c]) <= 1e-3) and not np.all(np.diff(arr[c]) >= -1e-3))
        const = sum(1 for c in range(C) if np.all(np.abs(np.diff(arr[c])) < 1e-3))
        print(f"  {stat_name:>22}: HAT={hats:>2}/{C}  mono_up={mono_up:>2}  mono_dn={mono_dn:>2}  const={const:>2}")

    # Per-stat detailed channel matrix (only print if any HAT)
    for stat_name in ["max", "mean", "top3_mean", "top10_mean", "std", "sparsity", "count_above_50pct"]:
        arr = stats[stat_name]
        any_hat = any(is_hat(arr[c]) for c in range(C))
        if not any_hat:
            continue
        print(f"\n--- {stat_name}: detailed (HAT channels marked) ---")
        for sc_idx in range(3):
            print(f"  [{scale_names[sc_idx]}]")
            for ci in range(cps):
                gci = sc_idx * cps + ci
                row = "  ".join(f"{arr[gci, j]:>+7.4f}" for j in range(K_n))
                tag = " <-- HAT" if is_hat(arr[gci]) else ""
                print(f"    ch{ci}: [{row}]{tag}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    n_choices = np.arange(args.n_min, args.n_max + 1)
    K_n = len(n_choices)

    sig_kwargs = dict(
        spike_width=args.spike_width,
        min_distance=args.min_distance,
        noise_std=args.noise_std,
    )

    model = SubitizingV5b(
        num_classes=K_n,
        channels_per_scale=args.channels_per_scale,
        head_hidden=tuple(int(h) for h in args.head_hidden.split(",")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"Train v5b for {args.epochs} ep, then probe internal CNN (seed={args.seed})")

    for ep in range(args.epochs):
        model.train(True)
        ep_acc = 0.0
        for _ in range(args.steps_per_epoch):
            x, y = make_batch(args.batch_size, args.T, n_choices, rng, **sig_kwargs)
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_acc += (logits.argmax(-1) == y).float().mean().item()
        ep_acc /= args.steps_per_epoch
        if (ep + 1) % args.log_every == 0:
            print(f"  ep {ep+1:>3}: acc={ep_acc:.4f}")

    print(f"\n=== Probing pre-pool CNN responses (n_per_N={args.n_per_N}) ===")
    rng_chk = np.random.default_rng(args.seed + 1000)
    responses = collect_raw_responses(model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs)
    stats, global_max = compute_per_channel_stats(responses)
    report_stats(stats, n_choices, args.channels_per_scale)

    # Save
    out = {
        "config": vars(args),
        "n_choices": n_choices.tolist(),
        "stats": {k: v.tolist() for k, v in stats.items()},
        "global_max_per_ch": global_max.tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=3)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--spike-width", type=int, default=5)
    p.add_argument("--min-distance", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--channels-per-scale", type=int, default=8)
    p.add_argument("--head-hidden", type=str, default="64,32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/v5b_internal_s42.json")
    args = p.parse_args()
    run(args)
