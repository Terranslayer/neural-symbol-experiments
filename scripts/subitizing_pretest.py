"""Subitizing pretest: K independent small Mambas -> K scalars -> spindle tuning?

Spec from user (2026-05-03):
  - K=3 small Mambas (d_model=8, d_state=8), each outputs 1 scalar
  - Train via downstream classification (option C) + diversity reg (option B)
  - No direct supervision per Mamba - let competition self-organize
  - Diagnose tuning curve shape: spindle (each peaks at unique N) vs step (thermometer)

Reference: Nieder/Dehaene number-selective neurons in PFC.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def make_signal(N, L, rng):
    s = np.zeros(L, dtype=np.float32)
    pos = rng.choice(L, size=N, replace=False)
    s[pos] = 1.0
    return s


def make_batch(batch_size, L, n_choices, rng):
    Ns = rng.choice(n_choices, size=batch_size)
    sigs = np.stack([make_signal(int(N), L, rng) for N in Ns])
    return (
        torch.from_numpy(sigs).unsqueeze(-1),
        torch.from_numpy(Ns - n_choices[0]).long(),
    )


class SmallMamba(nn.Module):
    def __init__(self, d_model=8, d_state=8, d_conv=4, expand=2, readout_hidden=0):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if readout_hidden > 0:
            self.out = nn.Sequential(
                nn.Linear(d_model, readout_hidden),
                nn.GELU(),
                nn.Linear(readout_hidden, 1),
            )
        else:
            self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.mamba(h)
        return self.out(h[:, -1]).squeeze(-1)


class SubitizingModule(nn.Module):
    def __init__(self, K=3, d_model=8, d_state=8, seed=0, readout_hidden=0, use_classifier=True):
        super().__init__()
        self.K = K
        self.use_classifier = use_classifier
        self.mambas = nn.ModuleList()
        for k in range(K):
            torch.manual_seed(seed + k * 100)
            self.mambas.append(SmallMamba(d_model, d_state, readout_hidden=readout_hidden))
        torch.manual_seed(seed)
        if use_classifier:
            self.classifier = nn.Linear(K, K)
        else:
            self.classifier = None

    def forward(self, x):
        scalars = torch.stack([m(x) for m in self.mambas], dim=-1)
        if self.classifier is not None:
            logits = self.classifier(scalars)
        else:
            logits = scalars  # scalars-as-logits
        return logits, scalars


def diversity_loss(scalars):
    x = scalars - scalars.mean(0, keepdim=True)
    std = x.std(0, keepdim=True) + 1e-6
    xn = x / std
    K = xn.shape[1]
    corr = (xn.T @ xn) / xn.shape[0]
    off_diag = corr - torch.eye(K, device=corr.device)
    return off_diag.abs().sum() / max(K * (K - 1), 1)


def variance_loss(scalars, target_std=1.0):
    """VICReg-style: penalize per-Mamba std below target. Forces all channels alive."""
    std = scalars.std(0) + 1e-6
    return F.relu(target_std - std).mean()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    n_choices = np.arange(args.n_min, args.n_max + 1)
    K = len(n_choices)

    model = SubitizingModule(K=K, d_model=args.d_model, d_state=args.d_state, seed=args.seed, readout_hidden=args.readout_hidden, use_classifier=not args.no_classifier).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Subitizing pretest: K={K} (N in {list(n_choices)}), d_model={args.d_model}, L={args.L}")
    print(f"Total params: {n_params}  (per-Mamba ~{(n_params - K*K - K) // K})")
    print(f"div_lambda={args.div_lambda}, lr={args.lr}, wd={args.wd}")

    history = []
    for ep in range(args.epochs):
        model.train(True)
        ep_ce, ep_div, ep_var, ep_acc = 0.0, 0.0, 0.0, 0.0
        for _ in range(args.steps_per_epoch):
            x, y = make_batch(args.batch_size, args.L, n_choices, rng)
            x, y = x.to(device), y.to(device)
            logits, scalars = model(x)
            l_ce = F.cross_entropy(logits, y)
            l_div = diversity_loss(scalars)
            l_var = variance_loss(scalars, target_std=args.target_std)
            loss = l_ce + args.div_lambda * l_div + args.var_lambda * l_var
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_ce += l_ce.item()
            ep_div += l_div.item()
            ep_var += l_var.item()
            ep_acc += (logits.argmax(-1) == y).float().mean().item()
        ep_ce /= args.steps_per_epoch
        ep_div /= args.steps_per_epoch
        ep_var /= args.steps_per_epoch
        ep_acc /= args.steps_per_epoch
        history.append({"ep": ep, "ce": ep_ce, "div": ep_div, "var": ep_var, "acc": ep_acc})
        if (ep + 1) % args.log_every == 0:
            print(f"  ep {ep+1:>3}: ce={ep_ce:.3f} div={ep_div:.3f} var={ep_var:.3f} acc={ep_acc:.3f}")

    # ===== Tuning curve diagnostic =====
    print(f"\n=== Tuning curves (avg scalar per N, n_per_N={args.n_per_N}) ===")
    model.train(False)
    rng_chk = np.random.default_rng(args.seed + 1000)
    tunings = np.zeros((K, K))
    tunings_std = np.zeros((K, K))
    with torch.no_grad():
        for nj, N in enumerate(n_choices):
            sigs = np.stack([make_signal(int(N), args.L, rng_chk) for _ in range(args.n_per_N)])
            x = torch.from_numpy(sigs).unsqueeze(-1).to(device)
            _, scalars = model(x)
            tunings[:, nj] = scalars.mean(0).cpu().numpy()
            tunings_std[:, nj] = scalars.std(0).cpu().numpy()

    print("          " + "".join(f"  N={n:>2}    " for n in n_choices))
    for i in range(K):
        row = "  ".join(f"{tunings[i, j]:>+7.3f}" for j in range(K))
        print(f"  Mamba_{i+1}: {row}")
    print("  --- (std) ---")
    for i in range(K):
        row = "  ".join(f"{tunings_std[i, j]:>+7.3f}" for j in range(K))
        print(f"  Mamba_{i+1}: {row}")

    print("\n--- Per-Mamba preferred N (argmax across N) ---")
    pref_per_mamba = []
    for i in range(K):
        pref = int(np.argmax(tunings[i]))
        pref_per_mamba.append(pref)
        print(f"  Mamba_{i+1} peaks at N={n_choices[pref]}  (response={tunings[i, pref]:+.3f})")

    print("\n--- Tuning shape diagnosis ---")
    if len(set(pref_per_mamba)) == K:
        print(f"  PERMUTATION: each Mamba peaks at unique N -> spindle-like emerged")
    else:
        print(f"  COLLISION: prefs={pref_per_mamba} -> not spindle (some Mambas redundant)")

    monotonic_count = 0
    for i in range(K):
        diffs = np.diff(tunings[i])
        if np.all(diffs >= -1e-3) or np.all(diffs <= 1e-3):
            monotonic_count += 1
    print(f"  monotonic Mambas: {monotonic_count}/{K}  (high = thermometer-like)")

    print("\n--- Per-Mamba peakedness (peak - second-best) ---")
    for i in range(K):
        sorted_resp = np.sort(tunings[i])[::-1]
        peakedness = sorted_resp[0] - sorted_resp[1]
        print(f"  Mamba_{i+1}: peak={sorted_resp[0]:+.3f}  2nd={sorted_resp[1]:+.3f}  delta={peakedness:+.3f}")

    raw_correct, raw_total = 0, 0
    cls_correct, cls_total = 0, 0
    rng_chk2 = np.random.default_rng(args.seed + 2000)
    with torch.no_grad():
        for _ in range(20):
            x, y = make_batch(args.batch_size, args.L, n_choices, rng_chk2)
            x, y = x.to(device), y.to(device)
            logits, scalars = model(x)
            raw_correct += (scalars.argmax(-1) == y).sum().item()
            cls_correct += (logits.argmax(-1) == y).sum().item()
            raw_total += y.numel()
            cls_total += y.numel()
    raw_acc = raw_correct / raw_total
    cls_acc = cls_correct / cls_total
    print(f"\n--- Final accuracy ---")
    print(f"  argmax(scalars):    {raw_acc:.3f}")
    print(f"  classifier head:    {cls_acc:.3f}")
    if abs(raw_acc - cls_acc) < 0.02:
        print(f"  -> spindle structure directly usable (no classifier rescaling needed)")
    else:
        print(f"  -> classifier doing nontrivial linear combo (delta={cls_acc-raw_acc:+.3f})")

    out = {
        "config": vars(args),
        "n_params": n_params,
        "tunings": tunings.tolist(),
        "tunings_std": tunings_std.tolist(),
        "n_choices": n_choices.tolist(),
        "preferred_N_per_mamba": pref_per_mamba,
        "monotonic_count": monotonic_count,
        "argmax_acc": raw_acc,
        "classifier_acc": cls_acc,
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
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--d-state", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--div-lambda", type=float, default=0.1)
    p.add_argument("--var-lambda", type=float, default=1.0)
    p.add_argument("--target-std", type=float, default=1.0)
    p.add_argument("--readout-hidden", type=int, default=0, help="MLP readout hidden dim (0 = linear readout)")
    p.add_argument("--no-classifier", action="store_true", help="Treat scalars directly as logits (winner-take-all flavor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--n-per-N", type=int, default=200)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/subitizing_results.json")
    args = p.parse_args()
    run(args)
