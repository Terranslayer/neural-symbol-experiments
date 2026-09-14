"""V5: CNN-only subitizing test.

Multi-scale Conv1d (k=5/21/51, 8ch each) + ReLU + AdaptiveMaxPool + MLP head.
Goal: test whether CNN with large receptive field self-organizes into spindle-like
"cardinality detectors" for N in {1, 2, 3}.

Spec from user (2026-05-03 V5 design).
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPIKE_TEMPLATE = np.array([0.3, 0.7, 1.0, 0.7, 0.3], dtype=np.float32)


def generate_signal(N, T, spike_width=5, min_distance=30, noise_std=0.05, rng=None):
    """Generate (T,) 1D float signal with N spikes at random non-overlapping positions."""
    if rng is None:
        rng = np.random.default_rng()
    sig = np.clip(rng.standard_normal(T) * noise_std, 0, 1).astype(np.float32) * 0.1
    placed = []
    attempts = 0
    while len(placed) < N and attempts < 1000:
        pos = int(rng.integers(spike_width, T - spike_width))
        if all(abs(pos - p) >= min_distance for p in placed):
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


class SubitizingV5(nn.Module):
    def __init__(self, num_classes=3, channels_per_scale=8):
        super().__init__()
        self.cnn_short = nn.Conv1d(1, channels_per_scale, kernel_size=5, padding=2)
        self.cnn_medium = nn.Conv1d(1, channels_per_scale, kernel_size=21, padding=10)
        self.cnn_long = nn.Conv1d(1, channels_per_scale, kernel_size=51, padding=25)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        total_ch = channels_per_scale * 3
        self.head = nn.Sequential(
            nn.Linear(total_ch, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        self.channels_per_scale = channels_per_scale

    def forward(self, signal):
        x = signal.unsqueeze(1)
        feats = torch.cat([
            self.activation(self.cnn_short(x)),
            self.activation(self.cnn_medium(x)),
            self.activation(self.cnn_long(x)),
        ], dim=1)
        pooled = self.pool(feats).squeeze(-1)
        logits = self.head(pooled)
        return logits, pooled, feats


def diagnose_tuning(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    """Per-channel tuning curves (channels x N)."""
    model.train(False)
    K_n = len(n_choices)
    cps = model.channels_per_scale
    total_ch = cps * 3
    tunings = np.zeros((total_ch, K_n))
    tunings_std = np.zeros((total_ch, K_n))
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            _, pooled, _ = model(x)
            tunings[:, nj] = pooled.mean(0).cpu().numpy()
            tunings_std[:, nj] = pooled.std(0).cpu().numpy()
    return tunings, tunings_std


def report_tuning(tunings, n_choices, channels_per_scale):
    K_n = len(n_choices)
    total_ch = tunings.shape[0]

    print(f"\n--- Per-channel tuning matrix ({total_ch} ch x {K_n} N) ---")
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]
    for sc_idx in range(3):
        print(f"\n  [{scale_names[sc_idx]}]")
        for ci in range(channels_per_scale):
            global_ci = sc_idx * channels_per_scale + ci
            row = "  ".join(f"{tunings[global_ci, j]:>+6.3f}" for j in range(K_n))
            pref = int(np.argmax(tunings[global_ci]))
            sorted_resp = np.sort(tunings[global_ci])[::-1]
            peakedness = sorted_resp[0] - sorted_resp[1]
            shape = ""
            if K_n >= 3:
                # check hat shape: peak at middle, drop on both sides
                mid = K_n // 2
                if pref == mid and tunings[global_ci, mid] > tunings[global_ci, 0] + 1e-3 and tunings[global_ci, mid] > tunings[global_ci, -1] + 1e-3:
                    shape = "HAT"
                elif np.all(np.diff(tunings[global_ci]) >= -1e-3):
                    shape = "mono_up"
                elif np.all(np.diff(tunings[global_ci]) <= 1e-3):
                    shape = "mono_dn"
                else:
                    shape = "other"
            print(f"    ch{ci}: [{row}]  pref=N{n_choices[pref]}  peakedness={peakedness:+.3f}  shape={shape}")

    pref_counter = Counter(int(np.argmax(tunings[c])) for c in range(total_ch))
    print(f"\n--- Preference distribution across {total_ch} channels ---")
    for ni, N in enumerate(n_choices):
        print(f"  N={N}: {pref_counter.get(ni, 0)} channels prefer this N")

    # Spindle analysis: count true HAT-shaped channels (peak at non-edge N)
    hat_count = 0
    for c in range(total_ch):
        pref = int(np.argmax(tunings[c]))
        if 0 < pref < K_n - 1:
            if (tunings[c, pref] > tunings[c, pref - 1] + 1e-3
                    and tunings[c, pref] > tunings[c, pref + 1] + 1e-3):
                hat_count += 1
    print(f"\n--- True HAT-shaped channels (peak at internal N): {hat_count}/{total_ch} ---")

    mono_up = sum(1 for c in range(total_ch) if np.all(np.diff(tunings[c]) >= -1e-3))
    mono_dn = sum(1 for c in range(total_ch) if np.all(np.diff(tunings[c]) <= 1e-3))
    print(f"--- Monotonic channels: up={mono_up}, down={mono_dn} ---")

    return {
        "preferences": dict(pref_counter),
        "hat_count": hat_count,
        "mono_up": mono_up,
        "mono_dn": mono_dn,
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

    model = SubitizingV5(num_classes=K_n, channels_per_scale=args.channels_per_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V5: K_n={K_n} (N in {list(n_choices)}), T={args.T}, channels_per_scale={args.channels_per_scale}")
    print(f"Total params: {n_params}")
    print(f"sig_kwargs: {sig_kwargs}")

    history = []
    for ep in range(args.epochs):
        model.train(True)
        ep_ce, ep_acc = 0.0, 0.0
        for _ in range(args.steps_per_epoch):
            x, y = make_batch(args.batch_size, args.T, n_choices, rng, **sig_kwargs)
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
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
    tunings, tunings_std = diagnose_tuning(
        model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs,
    )
    diag = report_tuning(tunings, n_choices, args.channels_per_scale)

    out = {
        "config": vars(args),
        "n_params": n_params,
        "n_choices": n_choices.tolist(),
        "tunings": tunings.tolist(),
        "tunings_std": tunings_std.tolist(),
        "diagnostic": diag,
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
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--spike-width", type=int, default=5)
    p.add_argument("--min-distance", type=int, default=30)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--channels-per-scale", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/v5_results.json")
    args = p.parse_args()
    run(args)
