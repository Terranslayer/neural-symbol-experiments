"""V5c: Replace pool with learned attention pool.

Hypothesis (from internal probe): pool kills hidden HAT structure that exists
in count-above-threshold view. Attention pool with learnable query can
discover non-monotonic aggregation rules.

Architecture vs V5b:
  - Same multi-scale CNN (k=5/21/51, 8ch each, 24 total)
  - REPLACE MaxPool/MeanPool with AttentionPool:
      * num_queries = 4 learnable query tokens
      * MultiheadAttention(d=24, num_heads=4) — queries attend over T positions
      * Output (B, 4, 24) -> flatten -> (B, 96)
  - MLP head: Linear(96, 64) -> ReLU -> Linear(64, 32) -> ReLU -> Linear(32, 3)

Diagnostic:
  - Per-output-dim tuning (96 dims = 4 queries x 24 channels)
  - HAT count per query
  - Per-query attention pattern (attention weights vs T position) for each N
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.subitizing_v5b import generate_signal, make_batch  # noqa


class AttentionPool(nn.Module):
    def __init__(self, d, num_queries=4, num_heads=4):
        super().__init__()
        assert d % num_heads == 0, f"d={d} must be divisible by num_heads={num_heads}"
        self.num_queries = num_queries
        self.d = d
        self.queries = nn.Parameter(torch.randn(1, num_queries, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)

    def forward(self, x, return_weights=False):
        # x: (B, d, T) — channels-first conv output
        x = x.transpose(1, 2)  # (B, T, d)
        B = x.shape[0]
        Q = self.queries.expand(B, -1, -1)  # (B, num_queries, d)
        out, weights = self.attn(Q, x, x, need_weights=return_weights, average_attn_weights=True)
        # out: (B, num_queries, d)
        # weights: (B, num_queries, T) if return_weights, else None
        return out, weights


class SubitizingV5c(nn.Module):
    def __init__(self, num_classes=3, channels_per_scale=8, num_queries=4, num_heads=4, head_hidden=(64, 32)):
        super().__init__()
        self.cnn_short = nn.Conv1d(1, channels_per_scale, kernel_size=5, padding=2)
        self.cnn_medium = nn.Conv1d(1, channels_per_scale, kernel_size=21, padding=10)
        self.cnn_long = nn.Conv1d(1, channels_per_scale, kernel_size=51, padding=25)
        self.activation = nn.ReLU()
        d = channels_per_scale * 3
        self.attn_pool = AttentionPool(d, num_queries, num_heads)
        head_input = num_queries * d
        layers = []
        prev = head_input
        for h in head_hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)
        self.channels_per_scale = channels_per_scale
        self.num_queries = num_queries
        self.d = d

    def forward(self, signal, return_attn=False):
        x = signal.unsqueeze(1)
        feats = torch.cat([
            self.activation(self.cnn_short(x)),
            self.activation(self.cnn_medium(x)),
            self.activation(self.cnn_long(x)),
        ], dim=1)  # (B, d, T)
        pooled, attn_weights = self.attn_pool(feats, return_weights=return_attn)
        # pooled: (B, num_queries, d)
        flat = pooled.flatten(1)  # (B, num_queries * d)
        logits = self.head(flat)
        return logits, pooled, attn_weights, feats


def is_hat(row, eps=1e-3):
    if len(row) < 3:
        return False
    pmax = int(np.argmax(row))
    return 0 < pmax < len(row) - 1 and row[pmax] > row[pmax - 1] + eps and row[pmax] > row[pmax + 1] + eps


def diagnose(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    """Per-output-dim tuning + per-query attention patterns."""
    model.train(False)
    K_n = len(n_choices)
    Q = model.num_queries
    d = model.d
    # per-output (Q*d) tuning
    out_tuning = np.zeros((Q, d, K_n))
    out_std = np.zeros((Q, d, K_n))
    # per-query attention weight histograms
    attn_per_N = {}
    correct_per_N = np.zeros(K_n, dtype=int)
    total_per_N = np.zeros(K_n, dtype=int)
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            logits, pooled, attn_w, _ = model(x, return_attn=True)
            # pooled: (B, Q, d)
            out_tuning[:, :, nj] = pooled.mean(0).cpu().numpy()
            out_std[:, :, nj] = pooled.std(0).cpu().numpy()
            attn_per_N[int(N)] = attn_w.cpu().numpy()  # (B, Q, T)
            preds = logits.argmax(-1).cpu().numpy()
            correct_per_N[nj] = (preds == nj).sum()
            total_per_N[nj] = len(preds)
    return {
        "out_tuning": out_tuning,
        "out_std": out_std,
        "attn_per_N": attn_per_N,
        "correct_per_N": correct_per_N,
        "total_per_N": total_per_N,
    }


def report(diag, n_choices, channels_per_scale, T):
    K_n = len(n_choices)
    out_tuning = diag["out_tuning"]  # (Q, d, K_n)
    Q = out_tuning.shape[0]
    d = out_tuning.shape[1]
    cps = channels_per_scale
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]

    print(f"\n--- Per-N accuracy ---")
    for ni, N in enumerate(n_choices):
        c, t = diag["correct_per_N"][ni], diag["total_per_N"][ni]
        print(f"  N={N}: {c}/{t} = {c/max(t,1):.3f}")

    # HAT summary across all Q*d output features
    print(f"\n=== HAT count per query (each query has {d} channel-outputs) ===")
    hat_counts = []
    for q in range(Q):
        hats = sum(is_hat(out_tuning[q, c]) for c in range(d))
        mono_up = sum(1 for c in range(d) if np.all(np.diff(out_tuning[q, c]) >= -1e-3) and not np.all(np.diff(out_tuning[q, c]) <= 1e-3))
        mono_dn = sum(1 for c in range(d) if np.all(np.diff(out_tuning[q, c]) <= 1e-3) and not np.all(np.diff(out_tuning[q, c]) >= -1e-3))
        const = sum(1 for c in range(d) if np.all(np.abs(np.diff(out_tuning[q, c])) < 1e-3))
        print(f"  Query {q}: HAT={hats:>2}/{d}  mono_up={mono_up:>2}  mono_dn={mono_dn:>2}  const={const:>2}")
        hat_counts.append(hats)

    # Print HAT-shaped output details
    total_hat = sum(hat_counts)
    print(f"\n=== TOTAL HAT-shaped outputs: {total_hat}/{Q*d} ===")

    if total_hat > 0:
        print(f"\n--- HAT-shaped output details ---")
        for q in range(Q):
            for c in range(d):
                row = out_tuning[q, c]
                if is_hat(row):
                    sc_idx = c // cps
                    ch_idx = c % cps
                    pmax = int(np.argmax(row))
                    sorted_resp = np.sort(row)[::-1]
                    peak_2nd = sorted_resp[0] - sorted_resp[1]
                    rng_ = row.max() - row.min()
                    pct = peak_2nd / max(abs(row.max()), 1e-9) * 100
                    print(f"  Q{q} {scale_names[sc_idx]} ch{ch_idx}: [{row[0]:>+7.3f}, {row[1]:>+7.3f}, {row[2]:>+7.3f}]"
                          f"  peak-2nd={peak_2nd:+.4f}  rng={rng_:.3f}  peakedness%={pct:.1f}")

    # Attention pattern summary: where does each query attend per N?
    print(f"\n=== Attention pattern (mean over batch) per query x N ===")
    for q in range(Q):
        print(f"\n  Query {q}:")
        for nj, N in enumerate(n_choices):
            attn = diag["attn_per_N"][int(N)][:, q, :].mean(0)  # (T,)
            # Top-5 attention positions
            top_pos = np.argsort(-attn)[:5]
            top_w = attn[top_pos]
            print(f"    N={N}: top-5 positions={top_pos.tolist()}, weights={[f'{w:.3f}' for w in top_w]}")
            # Entropy of attention (low = focused, high = uniform)
            p = attn / max(attn.sum(), 1e-9)
            ent = -(p * np.log(p + 1e-9)).sum()
            print(f"          attn entropy={ent:.3f}  (uniform={np.log(T):.3f})")

    return {
        "hat_per_query": hat_counts,
        "total_hat": total_hat,
        "per_N_acc": (diag["correct_per_N"] / np.maximum(diag["total_per_N"], 1)).tolist(),
    }


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

    model = SubitizingV5c(
        num_classes=K_n,
        channels_per_scale=args.channels_per_scale,
        num_queries=args.num_queries,
        num_heads=args.num_heads,
        head_hidden=tuple(int(h) for h in args.head_hidden.split(",")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V5c: K_n={K_n} (N in {list(n_choices)}), T={args.T}, Q={args.num_queries}, heads={args.num_heads}")
    print(f"Total params: {n_params}, head_hidden={args.head_hidden}")
    print(f"sig_kwargs: {sig_kwargs}")

    history = []
    for ep in range(args.epochs):
        model.train(True)
        ep_ce, ep_acc = 0.0, 0.0
        for _ in range(args.steps_per_epoch):
            x, y = make_batch(args.batch_size, args.T, n_choices, rng, **sig_kwargs)
            x, y = x.to(device), y.to(device)
            logits, _, _, _ = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_ce += loss.item()
            ep_acc += (logits.argmax(-1) == y).float().mean().item()
        ep_ce /= args.steps_per_epoch
        ep_acc /= args.steps_per_epoch
        history.append({"ep": ep, "ce": ep_ce, "acc": ep_acc})
        if (ep + 1) % args.log_every == 0:
            print(f"  ep {ep+1:>3}: ce={ep_ce:.4f} acc={ep_acc:.4f}")

    print(f"\n=== Diagnostic (n_per_N={args.n_per_N}) ===")
    rng_chk = np.random.default_rng(args.seed + 1000)
    diag = diagnose(model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs)
    summary = report(diag, n_choices, args.channels_per_scale, args.T)

    out = {
        "config": vars(args),
        "n_params": n_params,
        "n_choices": n_choices.tolist(),
        "out_tuning": diag["out_tuning"].tolist(),
        "correct_per_N": diag["correct_per_N"].tolist(),
        "total_per_N": diag["total_per_N"].tolist(),
        "summary": summary,
        "history_last5": history[-5:],
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
    p.add_argument("--num-queries", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-hidden", type=str, default="64,32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/v5c_results.json")
    args = p.parse_args()
    run(args)
