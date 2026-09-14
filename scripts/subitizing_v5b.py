"""V5b: Root-cause fix of V5.

Changes vs V5:
  1. T=100 (was 300), min_distance=8 (was 30) → kernel=51 comfortably contains
     N_max=3 spikes (16-step minimum span << 51-step receptive field). CNN can
     finally LEARN multi-spike patterns instead of just per-spike detectors.
  2. MaxPool + MeanPool concat (24+24 = 48 features). MeanPool gives N-density
     signal; MaxPool keeps "is spike present" signal. Both available to head.
  3. Deeper head: Linear(48,64) → ReLU → Linear(64,32) → ReLU → Linear(32,3).
     Two-layer ReLU MLP can piecewise-fit hat function from monotonic input.
  4. steps_per_epoch=100 (was 50) — more multi-spike configuration variety.
  5. Per-N accuracy report (was missing in V5).
  6. Per-spread-mode position sampling: random "spread factor" per sample to
     ensure CNN sees both clustered and spread multi-spike configs.
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPIKE_TEMPLATE = np.array([0.3, 0.7, 1.0, 0.7, 0.3], dtype=np.float32)


def generate_signal(N, T, spike_width=5, min_distance=8, noise_std=0.05, rng=None):
    """Generate (T,) signal with N spikes. Position sampling uses random 'spread'
    to ensure both clustered and spread configurations appear in training."""
    if rng is None:
        rng = np.random.default_rng()
    sig = np.clip(rng.standard_normal(T) * noise_std, 0, 1).astype(np.float32) * 0.1

    if N == 0:
        return sig

    # spread factor: 0 = max clustered (use min_distance), 1 = max spread
    # mixing both in training gives CNN richer multi-spike pattern variety
    spread = float(rng.random())
    eff_min = min_distance
    # for cluster mode: cap usable_T to reduce spread
    usable_T = T - 2 * spike_width
    if spread < 0.5 and N > 1:
        # cluster mode: pack spikes within a small window
        cluster_window = int(eff_min * (N - 1) + rng.integers(0, max(1, usable_T // 4)))
        cluster_window = min(cluster_window, usable_T)
        cs_high = T - spike_width - cluster_window
        if cs_high <= spike_width:
            # cluster too big to fit in T — fall back to spread mode
            placed = []
        else:
            cluster_start = int(rng.integers(spike_width, cs_high))
            placed = []
            attempts = 0
            while len(placed) < N and attempts < 1000:
                pos = cluster_start + int(rng.integers(0, cluster_window))
                if all(abs(pos - p) >= eff_min for p in placed):
                    placed.append(pos)
                attempts += 1
    else:
        # spread mode: positions uniform across full T
        placed = []
        attempts = 0
        while len(placed) < N and attempts < 1000:
            pos = int(rng.integers(spike_width, T - spike_width))
            if all(abs(pos - p) >= eff_min for p in placed):
                placed.append(pos)
            attempts += 1

    # If failed to place all spikes (e.g., cluster too tight), fallback to spread mode
    if len(placed) < N:
        placed = []
        attempts = 0
        while len(placed) < N and attempts < 2000:
            pos = int(rng.integers(spike_width, T - spike_width))
            if all(abs(pos - p) >= eff_min for p in placed):
                placed.append(pos)
            attempts += 1

    for pos in placed:
        sig[pos - 2: pos + 3] = np.maximum(sig[pos - 2: pos + 3], SPIKE_TEMPLATE)
    return sig


def make_batch(batch_size, T, n_choices, rng, **sig_kwargs):
    Ns = rng.choice(n_choices, size=batch_size)
    sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for N in Ns])
    return (
        torch.from_numpy(sigs),
        torch.from_numpy(Ns - n_choices[0]).long(),
    )


class SubitizingV5b(nn.Module):
    def __init__(self, num_classes=3, channels_per_scale=8, head_hidden=(64, 32)):
        super().__init__()
        self.cnn_short = nn.Conv1d(1, channels_per_scale, kernel_size=5, padding=2)
        self.cnn_medium = nn.Conv1d(1, channels_per_scale, kernel_size=21, padding=10)
        self.cnn_long = nn.Conv1d(1, channels_per_scale, kernel_size=51, padding=25)
        self.activation = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.meanpool = nn.AdaptiveAvgPool1d(1)
        total_ch = channels_per_scale * 3 * 2  # max + mean concat
        layers = []
        prev = total_ch
        for h in head_hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)
        self.channels_per_scale = channels_per_scale

    def forward(self, signal):
        x = signal.unsqueeze(1)
        feats = torch.cat([
            self.activation(self.cnn_short(x)),
            self.activation(self.cnn_medium(x)),
            self.activation(self.cnn_long(x)),
        ], dim=1)  # (B, 24, T)
        max_pooled = self.maxpool(feats).squeeze(-1)   # (B, 24)
        mean_pooled = self.meanpool(feats).squeeze(-1) # (B, 24)
        pooled = torch.cat([max_pooled, mean_pooled], dim=-1)  # (B, 48)
        logits = self.head(pooled)
        return logits, max_pooled, mean_pooled, feats


def diagnose_tuning(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    """Per-channel tuning curves for BOTH max and mean pool channels."""
    model.train(False)
    K_n = len(n_choices)
    cps = model.channels_per_scale
    total_ch = cps * 3
    max_tuning = np.zeros((total_ch, K_n))
    mean_tuning = np.zeros((total_ch, K_n))
    max_std = np.zeros((total_ch, K_n))
    mean_std = np.zeros((total_ch, K_n))
    correct_per_N = np.zeros(K_n, dtype=int)
    total_per_N = np.zeros(K_n, dtype=int)
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            logits, max_p, mean_p, _ = model(x)
            max_tuning[:, nj] = max_p.mean(0).cpu().numpy()
            mean_tuning[:, nj] = mean_p.mean(0).cpu().numpy()
            max_std[:, nj] = max_p.std(0).cpu().numpy()
            mean_std[:, nj] = mean_p.std(0).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            correct_per_N[nj] = (preds == nj).sum()
            total_per_N[nj] = len(preds)
    return {
        "max_tuning": max_tuning, "mean_tuning": mean_tuning,
        "max_std": max_std, "mean_std": mean_std,
        "correct_per_N": correct_per_N, "total_per_N": total_per_N,
    }


def report_tuning(diag, n_choices, channels_per_scale):
    K_n = len(n_choices)
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]
    cps = channels_per_scale

    for pool_name, key in [("MAX-POOL", "max_tuning"), ("MEAN-POOL", "mean_tuning")]:
        tunings = diag[key]
        total_ch = tunings.shape[0]
        print(f"\n--- {pool_name} per-channel tuning ({total_ch} ch x {K_n} N) ---")
        for sc_idx in range(3):
            print(f"\n  [{scale_names[sc_idx]}]")
            for ci in range(cps):
                gci = sc_idx * cps + ci
                row = "  ".join(f"{tunings[gci, j]:>+6.3f}" for j in range(K_n))
                pref = int(np.argmax(tunings[gci]))
                sorted_resp = np.sort(tunings[gci])[::-1]
                peakedness = sorted_resp[0] - sorted_resp[1]
                shape = ""
                # check hat: peak strictly at internal N
                pmax = pref
                if 0 < pmax < K_n - 1 and tunings[gci, pmax] > tunings[gci, pmax-1] + 1e-3 and tunings[gci, pmax] > tunings[gci, pmax+1] + 1e-3:
                    shape = "HAT"
                elif np.all(np.diff(tunings[gci]) >= -1e-3):
                    shape = "mono_up"
                elif np.all(np.diff(tunings[gci]) <= 1e-3):
                    shape = "mono_dn"
                else:
                    shape = "other"
                print(f"    ch{ci}: [{row}]  pref=N{n_choices[pref]}  peak-2nd={peakedness:+.3f}  {shape}")

        pref_counter = Counter(int(np.argmax(tunings[c])) for c in range(total_ch))
        print(f"\n  {pool_name} preference distribution:")
        for ni, N in enumerate(n_choices):
            print(f"    N={N}: {pref_counter.get(ni, 0)} channels")
        hat_count = 0
        for c in range(total_ch):
            pmax = int(np.argmax(tunings[c]))
            if 0 < pmax < K_n - 1:
                if (tunings[c, pmax] > tunings[c, pmax-1] + 1e-3
                        and tunings[c, pmax] > tunings[c, pmax+1] + 1e-3):
                    hat_count += 1
        mono_up = sum(1 for c in range(total_ch) if np.all(np.diff(tunings[c]) >= -1e-3) and not np.all(np.diff(tunings[c]) <= 1e-3))
        mono_dn = sum(1 for c in range(total_ch) if np.all(np.diff(tunings[c]) <= 1e-3) and not np.all(np.diff(tunings[c]) >= -1e-3))
        const = sum(1 for c in range(total_ch) if np.all(np.abs(np.diff(tunings[c])) < 1e-3))
        print(f"  HAT-shaped: {hat_count}/{total_ch}, mono_up: {mono_up}, mono_dn: {mono_dn}, constant: {const}")

    print(f"\n--- Per-N accuracy ---")
    for ni, N in enumerate(n_choices):
        c, t = diag["correct_per_N"][ni], diag["total_per_N"][ni]
        print(f"  N={N}: {c}/{t} = {c/max(t,1):.3f}")

    return {
        "max_hat": sum(1 for c in range(diag["max_tuning"].shape[0])
                       for pmax in [int(np.argmax(diag["max_tuning"][c]))]
                       if 0 < pmax < K_n - 1
                       and diag["max_tuning"][c, pmax] > diag["max_tuning"][c, pmax-1] + 1e-3
                       and diag["max_tuning"][c, pmax] > diag["max_tuning"][c, pmax+1] + 1e-3),
        "mean_hat": sum(1 for c in range(diag["mean_tuning"].shape[0])
                        for pmax in [int(np.argmax(diag["mean_tuning"][c]))]
                        if 0 < pmax < K_n - 1
                        and diag["mean_tuning"][c, pmax] > diag["mean_tuning"][c, pmax-1] + 1e-3
                        and diag["mean_tuning"][c, pmax] > diag["mean_tuning"][c, pmax+1] + 1e-3),
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

    model = SubitizingV5b(
        num_classes=K_n,
        channels_per_scale=args.channels_per_scale,
        head_hidden=tuple(int(h) for h in args.head_hidden.split(",")),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V5b: K_n={K_n} (N in {list(n_choices)}), T={args.T}, channels_per_scale={args.channels_per_scale}")
    print(f"Total params: {n_params}, head_hidden={args.head_hidden}")
    print(f"sig_kwargs: {sig_kwargs}")
    print(f"steps_per_epoch={args.steps_per_epoch}, batch_size={args.batch_size}")

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

    print(f"\n=== Tuning diagnostic (n_per_N={args.n_per_N}) ===")
    rng_chk = np.random.default_rng(args.seed + 1000)
    diag = diagnose_tuning(model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs)
    summary = report_tuning(diag, n_choices, args.channels_per_scale)

    out = {
        "config": vars(args),
        "n_params": n_params,
        "n_choices": n_choices.tolist(),
        "max_tuning": diag["max_tuning"].tolist(),
        "mean_tuning": diag["mean_tuning"].tolist(),
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
    p.add_argument("--head-hidden", type=str, default="64,32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/v5b_results.json")
    args = p.parse_args()
    run(args)
