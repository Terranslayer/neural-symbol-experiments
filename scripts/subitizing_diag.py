"""CNN internal potential diagnostic.

After training V5b briefly (acc=1.0 by ep 50), examine the trained CNN's
internal features WITHOUT pool-related architectural changes. Goal: distinguish
between three failure hypotheses:

  H1 (training/sampling): CNN never sees enough multi-spike pattern variety
       → fix data, not architecture.
  H2 (architectural capacity): CNN can't represent cardinality features even
       with perfect data → need different cell type / readout.
  H3 (pool destroys signal): CNN has the info but pool collapses it
       → fix pool primitive (V5d already partially tested).

Diagnostics:
  Part A — INFORMATION PROBES:
    A1. Linear probe on full feats (24×100 = 2400 dim) → predict N.
        If acc=100%: info IS present in CNN, downstream is bottleneck.
    A2. Per-channel linear probe (each ch's full T response, 100 dim → 3 class).
        Reports per-channel discrimination accuracy → which channels carry N info.
    A3. Per-position linear probe (each pos's 24-channel response → 3 class).
        Finds spatially "informative" positions, regardless of channel.
    A4. MLP probe on full feats — upper bound on extractable info.

  Part B — SYNTHETIC PATTERN TEST:
    Feed deterministic spike configurations and dump per-channel responses:
      - 1 spike at center, edge
      - 2 spikes: tight (8 apart), medium (30), wide (80)
      - 3 spikes: tight, spread, asymmetric
    Identifies channels that respond differently to "exactly N" vs "more than N".

  Part C — CONV KERNEL INSPECTION:
    Print learned kernel weights (the long k=51 channels — most likely to have
    multi-spike patterns). Visualize as 1D weight curves.

  Part D — TRAINING DATA STATS:
    Sample 1000 instances per N, report position diversity, pairwise distance
    histogram, "tight cluster" vs "spread" mode coverage.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.subitizing_v5b import SubitizingV5b, generate_signal, make_batch  # noqa


def collect_feats(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    """Returns: (n_per_N * K_n, C, T) feats, (n_per_N * K_n,) labels."""
    model.train(False)
    feats_list, labels_list = [], []
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            _, _, _, feats = model(x)  # (B, C, T)
            feats_list.append(feats.cpu().numpy())
            labels_list.extend([nj] * n_per_N)
    return np.concatenate(feats_list, axis=0), np.array(labels_list)


def part_A_probes(feats, labels, n_choices, channels_per_scale):
    """Information probes."""
    print(f"\n{'='*72}\n=== PART A: Information Probes ===\n{'='*72}")
    K_n = len(n_choices)
    N_total, C, T = feats.shape
    print(f"Feats: {N_total} samples × {C} ch × {T} positions")

    # A1: Linear probe on full flattened features
    print(f"\n--- A1: Linear probe on full feats (dim={C*T}) ---")
    X = feats.reshape(N_total, -1)
    # train/test split
    rs = np.random.RandomState(0)
    idx = rs.permutation(N_total)
    split = int(0.8 * N_total)
    tr, te = idx[:split], idx[split:]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], labels[tr])
    acc_tr = clf.score(X[tr], labels[tr])
    acc_te = clf.score(X[te], labels[te])
    print(f"  Linear (logistic): train={acc_tr:.3f}  test={acc_te:.3f}")

    # A4: MLP probe (small)
    print(f"\n--- A4: MLP probe (64-32 hidden) on full feats ---")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
    mlp.fit(X[tr], labels[tr])
    print(f"  MLP: train={mlp.score(X[tr], labels[tr]):.3f}  test={mlp.score(X[te], labels[te]):.3f}")

    # A2: Per-channel probe (each channel's T response → predict N)
    print(f"\n--- A2: Per-channel linear probe (one channel's T={T} dim) ---")
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]
    per_ch_acc = np.zeros(C)
    for c in range(C):
        Xc = feats[:, c, :]  # (N_total, T)
        clf_c = LogisticRegression(max_iter=500, C=1.0)
        clf_c.fit(Xc[tr], labels[tr])
        per_ch_acc[c] = clf_c.score(Xc[te], labels[te])
    print("  channel_id  test_acc  scale")
    for sc_idx in range(3):
        print(f"  [{scale_names[sc_idx]}]")
        for ci in range(channels_per_scale):
            gci = sc_idx * channels_per_scale + ci
            star = "  <-- best" if per_ch_acc[gci] == per_ch_acc.max() else ""
            print(f"    ch{ci} (gci={gci:>2}): {per_ch_acc[gci]:.3f}{star}")
    # most informative channel
    top3 = np.argsort(-per_ch_acc)[:3]
    print(f"\n  Top-3 informative channels: {top3.tolist()} (accs={per_ch_acc[top3].tolist()})")

    # A3: Per-position probe (each position's C-channel vector → predict N)
    print(f"\n--- A3: Per-position linear probe (one position's C={C} dim) ---")
    per_pos_acc = np.zeros(T)
    for t in range(T):
        Xt = feats[:, :, t]  # (N_total, C)
        clf_t = LogisticRegression(max_iter=500, C=1.0)
        clf_t.fit(Xt[tr], labels[tr])
        per_pos_acc[t] = clf_t.score(Xt[te], labels[te])
    # report top-10 positions
    top_pos = np.argsort(-per_pos_acc)[:10]
    print(f"  Top-10 positions: {top_pos.tolist()}")
    print(f"  Accuracies:       {[f'{a:.3f}' for a in per_pos_acc[top_pos]]}")
    print(f"  Mean acc across positions: {per_pos_acc.mean():.3f}, max: {per_pos_acc.max():.3f}")

    return {
        "A1_full_linear_test": acc_te,
        "A4_full_mlp_test": float(mlp.score(X[te], labels[te])),
        "A2_per_ch_acc": per_ch_acc.tolist(),
        "A3_per_pos_acc": per_pos_acc.tolist(),
        "top3_channels": top3.tolist(),
    }


def part_B_synthetic(model, T, device):
    """Synthetic pattern test."""
    print(f"\n{'='*72}\n=== PART B: Synthetic Pattern Test ===\n{'='*72}")
    SPIKE = np.array([0.3, 0.7, 1.0, 0.7, 0.3], dtype=np.float32)

    def make(positions):
        s = np.zeros(T, dtype=np.float32)
        for p in positions:
            s[p-2:p+3] = np.maximum(s[p-2:p+3], SPIKE)
        return s

    patterns = {
        "1@center": [50],
        "1@left": [10],
        "1@right": [90],
        "2@tight (8 apart)": [40, 48],
        "2@medium (30)": [30, 60],
        "2@wide (80)": [10, 90],
        "3@tight": [40, 48, 56],
        "3@spread": [20, 50, 80],
        "3@asymmetric": [10, 18, 90],
    }

    # Build batch
    sigs = np.stack([make(p) for p in patterns.values()])
    x = torch.from_numpy(sigs).to(device)
    model.train(False)
    with torch.no_grad():
        _, _, _, feats = model(x)  # (P, C, T)
    feats_np = feats.cpu().numpy()
    P, C, _ = feats_np.shape

    # Per pattern, per channel, take max + mean response
    print(f"\n--- Pattern × channel: max response (per channel max over T) ---")
    print(f"{'pattern':<22}" + "".join(f"  ch{c:>2}" for c in range(C)))
    for pi, name in enumerate(patterns.keys()):
        max_resp = feats_np[pi].max(axis=-1)
        row = "  ".join(f"{r:>5.2f}" for r in max_resp)
        print(f"  {name:<20}{row}")

    print(f"\n--- Pattern × channel: mean response (per channel mean over T) ---")
    print(f"{'pattern':<22}" + "".join(f"  ch{c:>2}" for c in range(C)))
    for pi, name in enumerate(patterns.keys()):
        mean_resp = feats_np[pi].mean(axis=-1)
        row = "  ".join(f"{r:>5.2f}" for r in mean_resp)
        print(f"  {name:<20}{row}")

    # Find channels with HAT-like behavior across the synthetic patterns
    # Define "exactly 2" group vs "exactly 3" group
    pat_names = list(patterns.keys())
    n1_idxs = [i for i, n in enumerate(pat_names) if n.startswith("1")]
    n2_idxs = [i for i, n in enumerate(pat_names) if n.startswith("2")]
    n3_idxs = [i for i, n in enumerate(pat_names) if n.startswith("3")]
    print(f"\n--- HAT detection on synthetic patterns ---")
    for stat_name, agg_fn in [("max", lambda x: x.max(axis=-1)),
                                ("mean", lambda x: x.mean(axis=-1))]:
        responses = agg_fn(feats_np)  # (P, C)
        n1_avg = responses[n1_idxs].mean(0)
        n2_avg = responses[n2_idxs].mean(0)
        n3_avg = responses[n3_idxs].mean(0)
        hat_chans = []
        for c in range(C):
            if n2_avg[c] > n1_avg[c] + 0.01 and n2_avg[c] > n3_avg[c] + 0.01:
                hat_chans.append(c)
        print(f"  {stat_name}: HAT channels={hat_chans} ({len(hat_chans)}/{C})")
        if hat_chans:
            for c in hat_chans:
                print(f"    ch{c}: N1={n1_avg[c]:.3f}, N2={n2_avg[c]:.3f}, N3={n3_avg[c]:.3f}")


def part_C_kernels(model, channels_per_scale):
    """Inspect learned conv kernel weights."""
    print(f"\n{'='*72}\n=== PART C: Conv Kernel Inspection ===\n{'='*72}")
    for name, layer in [("short k=5", model.cnn_short), ("medium k=21", model.cnn_medium), ("long k=51", model.cnn_long)]:
        # weights shape: (C_out, C_in=1, K)
        W = layer.weight.detach().cpu().numpy()
        print(f"\n--- {name} kernel weights (shape {W.shape}) ---")
        for c in range(channels_per_scale):
            w = W[c, 0, :]
            # summary stats
            wmax = w.max()
            wmin = w.min()
            wsum = w.sum()
            wabs = np.abs(w).sum()
            # find peaks (positive lobes) — count them
            pos_runs = 0
            in_pos = False
            for v in w:
                if v > 0.05:  # threshold
                    if not in_pos:
                        pos_runs += 1
                        in_pos = True
                else:
                    in_pos = False
            neg_runs = 0
            in_neg = False
            for v in w:
                if v < -0.05:
                    if not in_neg:
                        neg_runs += 1
                        in_neg = True
                else:
                    in_neg = False
            print(f"  ch{c}: max={wmax:+.3f} min={wmin:+.3f} sum={wsum:+.3f} pos_lobes={pos_runs} neg_lobes={neg_runs}")


def part_D_data_stats(T, n_choices, n_samples, rng, **sig_kwargs):
    """Training distribution diagnostics."""
    print(f"\n{'='*72}\n=== PART D: Training Data Statistics ===\n{'='*72}")
    for N in n_choices:
        positions_seen = []
        pairwise_dists = []
        spans = []
        for _ in range(n_samples):
            sig = generate_signal(int(N), T, rng=rng, **sig_kwargs)
            # extract spike positions (peaks)
            positions = []
            for t in range(2, T-2):
                if sig[t] > 0.5 and sig[t] >= sig[t-1] and sig[t] >= sig[t+1]:
                    positions.append(t)
            positions = sorted(positions)[:N]
            positions_seen.append(positions)
            if len(positions) >= 2:
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        pairwise_dists.append(positions[j] - positions[i])
                spans.append(positions[-1] - positions[0])
        # position entropy (per slot)
        if positions_seen:
            pos_arrs = np.array([p + [-1]*(N - len(p)) for p in positions_seen])
            print(f"\n  N={N}: {len(positions_seen)} samples")
            for slot in range(N):
                slot_pos = pos_arrs[:, slot][pos_arrs[:, slot] >= 0]
                hist, _ = np.histogram(slot_pos, bins=10)
                p_norm = hist / max(hist.sum(), 1)
                ent = -(p_norm * np.log(p_norm + 1e-9)).sum()
                print(f"    slot {slot}: range=[{slot_pos.min():>3}, {slot_pos.max():>3}], entropy={ent:.3f} (max={np.log(10):.3f})")
            if pairwise_dists:
                pd_arr = np.array(pairwise_dists)
                print(f"    pairwise distance: mean={pd_arr.mean():.1f}, median={int(np.median(pd_arr))}, min={pd_arr.min()}, max={pd_arr.max()}")
                # tight (<=15) vs spread (>=30)
                tight_frac = (pd_arr <= 15).mean()
                spread_frac = (pd_arr >= 30).mean()
                print(f"    tight (<=15): {tight_frac:.2%}, spread (>=30): {spread_frac:.2%}")


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

    # Quick training
    model = SubitizingV5b(
        num_classes=K_n,
        channels_per_scale=args.channels_per_scale,
        head_hidden=tuple(int(h) for h in args.head_hidden.split(",")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"Train v5b briefly (seed={args.seed}, {args.epochs} ep)")
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

    # Diagnostics
    rng_chk = np.random.default_rng(args.seed + 1000)
    feats, labels = collect_feats(model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs)
    summary = part_A_probes(feats, labels, n_choices, args.channels_per_scale)
    part_B_synthetic(model, args.T, device)
    part_C_kernels(model, args.channels_per_scale)
    rng_d = np.random.default_rng(args.seed + 2000)
    part_D_data_stats(args.T, n_choices, args.n_data_samples, rng_d, **sig_kwargs)

    out = {
        "config": vars(args),
        "summary": summary,
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
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--n-per-N", type=int, default=300)
    p.add_argument("--n-data-samples", type=int, default=1000)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/diag_results.json")
    args = p.parse_args()
    run(args)
