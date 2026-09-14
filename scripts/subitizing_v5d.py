"""V5d: Soft-threshold pool — directly expose count-above-threshold mechanism.

Internal probe (v5b_internal) found 1 HAT in count_above_50pct stat (long ch1
[+6.13, +8.14, +7.78], peakedness 0.36). Hypothesis: making this stat the pool
primitive — instead of max/mean — exposes the hidden HAT structure.

Per channel: learnable threshold tau_c, learnable sharpness beta_c.
  output_c = mean_t sigmoid(beta_c * (response_{c,t} - tau_c))

This is differentiable count-above-threshold (normalized to [0, 1]).

If output_c can be non-monotonic in N (peaks at intermediate N), then:
  - For N=1: few responses above threshold -> low count
  - For N=2: optimal multi-spike pattern -> peak count
  - For N=3: 3rd spike interference -> count drops

This is the geometric mechanism we hypothesized for long ch1.
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


class SoftThresholdPool(nn.Module):
    """Per-channel learnable threshold + sharpness. Output = mean of sigmoid responses."""
    def __init__(self, num_channels, init_tau=0.1, init_beta=10.0):
        super().__init__()
        self.tau = nn.Parameter(torch.full((num_channels,), init_tau))
        self.log_beta = nn.Parameter(torch.full((num_channels,), float(np.log(init_beta))))

    def forward(self, x):
        # x: (B, C, T)
        beta = self.log_beta.exp().clamp(0.1, 200.0)  # avoid extreme sharpness
        # Broadcast: (B, C, T) - (C, 1) per channel
        diff = x - self.tau.view(1, -1, 1)
        soft_count = torch.sigmoid(beta.view(1, -1, 1) * diff)  # (B, C, T)
        return soft_count.mean(-1)  # (B, C)


class SubitizingV5d(nn.Module):
    def __init__(self, num_classes=3, channels_per_scale=8, head_hidden=(64, 32),
                 use_max=True, use_mean=True, use_softthresh=True):
        super().__init__()
        self.cnn_short = nn.Conv1d(1, channels_per_scale, kernel_size=5, padding=2)
        self.cnn_medium = nn.Conv1d(1, channels_per_scale, kernel_size=21, padding=10)
        self.cnn_long = nn.Conv1d(1, channels_per_scale, kernel_size=51, padding=25)
        self.activation = nn.ReLU()
        d = channels_per_scale * 3
        self.use_max = use_max
        self.use_mean = use_mean
        self.use_softthresh = use_softthresh
        if use_max:
            self.maxpool = nn.AdaptiveMaxPool1d(1)
        if use_mean:
            self.meanpool = nn.AdaptiveAvgPool1d(1)
        if use_softthresh:
            self.softthresh = SoftThresholdPool(d)
        head_input = d * (int(use_max) + int(use_mean) + int(use_softthresh))
        layers = []
        prev = head_input
        for h in head_hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)
        self.channels_per_scale = channels_per_scale
        self.d = d

    def forward(self, signal):
        x = signal.unsqueeze(1)
        feats = torch.cat([
            self.activation(self.cnn_short(x)),
            self.activation(self.cnn_medium(x)),
            self.activation(self.cnn_long(x)),
        ], dim=1)  # (B, d, T)
        outs = []
        outs_dict = {}
        if self.use_max:
            mp = self.maxpool(feats).squeeze(-1)
            outs.append(mp)
            outs_dict["max"] = mp
        if self.use_mean:
            mep = self.meanpool(feats).squeeze(-1)
            outs.append(mep)
            outs_dict["mean"] = mep
        if self.use_softthresh:
            sp = self.softthresh(feats)
            outs.append(sp)
            outs_dict["softthresh"] = sp
        pooled = torch.cat(outs, dim=-1)
        logits = self.head(pooled)
        return logits, outs_dict, feats


def is_hat(row, eps=1e-3):
    if len(row) < 3:
        return False
    pmax = int(np.argmax(row))
    return 0 < pmax < len(row) - 1 and row[pmax] > row[pmax - 1] + eps and row[pmax] > row[pmax + 1] + eps


def diagnose(model, n_choices, T, n_per_N, device, rng, **sig_kwargs):
    model.train(False)
    K_n = len(n_choices)
    d = model.d
    pool_keys = []
    if model.use_max: pool_keys.append("max")
    if model.use_mean: pool_keys.append("mean")
    if model.use_softthresh: pool_keys.append("softthresh")
    tunings = {k: np.zeros((d, K_n)) for k in pool_keys}
    correct_per_N = np.zeros(K_n, dtype=int)
    total_per_N = np.zeros(K_n, dtype=int)
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for _ in range(n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            logits, outs_dict, _ = model(x)
            for k in pool_keys:
                tunings[k][:, nj] = outs_dict[k].mean(0).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            correct_per_N[nj] = (preds == nj).sum()
            total_per_N[nj] = len(preds)
    return {
        "tunings": tunings,
        "correct_per_N": correct_per_N,
        "total_per_N": total_per_N,
    }


def report(diag, n_choices, channels_per_scale, model):
    K_n = len(n_choices)
    cps = channels_per_scale
    scale_names = ["short(k=5)", "medium(k=21)", "long(k=51)"]

    print(f"\n--- Per-N accuracy ---")
    for ni, N in enumerate(n_choices):
        c, t = diag["correct_per_N"][ni], diag["total_per_N"][ni]
        print(f"  N={N}: {c}/{t} = {c/max(t,1):.3f}")

    summary = {}
    for k, tunings in diag["tunings"].items():
        d = tunings.shape[0]
        hats = sum(is_hat(tunings[c]) for c in range(d))
        mono_up = sum(1 for c in range(d) if np.all(np.diff(tunings[c]) >= -1e-3) and not np.all(np.diff(tunings[c]) <= 1e-3))
        mono_dn = sum(1 for c in range(d) if np.all(np.diff(tunings[c]) <= 1e-3) and not np.all(np.diff(tunings[c]) >= -1e-3))
        const = sum(1 for c in range(d) if np.all(np.abs(np.diff(tunings[c])) < 1e-3))
        summary[k] = dict(hat=hats, mono_up=mono_up, mono_dn=mono_dn, const=const)
        print(f"\n  {k}: HAT={hats}/{d}  mono_up={mono_up}  mono_dn={mono_dn}  const={const}")

    # Detail soft-threshold (the new primitive)
    if "softthresh" in diag["tunings"]:
        st = diag["tunings"]["softthresh"]
        d = st.shape[0]
        print(f"\n--- SOFT-THRESHOLD per-channel tuning (HAT marked) ---")
        for sc_idx in range(3):
            print(f"  [{scale_names[sc_idx]}]")
            for ci in range(cps):
                gci = sc_idx * cps + ci
                row = st[gci]
                tag = " <-- HAT" if is_hat(row) else ""
                tau = model.softthresh.tau[gci].item()
                beta = model.softthresh.log_beta[gci].exp().item()
                print(f"    ch{ci}: [{row[0]:>+6.4f}, {row[1]:>+6.4f}, {row[2]:>+6.4f}]  tau={tau:+.3f} beta={beta:.2f}{tag}")

    return summary


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

    model = SubitizingV5d(
        num_classes=K_n,
        channels_per_scale=args.channels_per_scale,
        head_hidden=tuple(int(h) for h in args.head_hidden.split(",")),
        use_max=args.use_max,
        use_mean=args.use_mean,
        use_softthresh=args.use_softthresh,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V5d: K_n={K_n} (N in {list(n_choices)}), T={args.T}")
    print(f"Pool: max={args.use_max}, mean={args.use_mean}, softthresh={args.use_softthresh}")
    print(f"Total params: {n_params}, head_hidden={args.head_hidden}")

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

    print(f"\n=== Diagnostic (n_per_N={args.n_per_N}) ===")
    rng_chk = np.random.default_rng(args.seed + 1000)
    diag = diagnose(model, n_choices, args.T, args.n_per_N, device, rng_chk, **sig_kwargs)
    summary = report(diag, n_choices, args.channels_per_scale, model)

    out = {
        "config": vars(args),
        "n_params": n_params,
        "n_choices": n_choices.tolist(),
        "tunings": {k: v.tolist() for k, v in diag["tunings"].items()},
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
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=30)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--use-max", action="store_true", default=False)
    p.add_argument("--use-mean", action="store_true", default=False)
    p.add_argument("--use-softthresh", action="store_true", default=True)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/v5d_results.json")
    args = p.parse_args()
    run(args)
