"""ANS capacity test: at what N_max does Mamba lose ±1 precision?

Question (user 2026-05-04): "At what range can Mamba achieve exact N
discrimination?" This determines the boundary between subitizing-style exact
perception and symbolic-system-required regime.

Architecture (matches M6 backbone, minus PFC/scratch):
  Conv1d(1, 16, k=5, padding=2) -> ReLU
  Mamba(d_model=16, d_state=16, layers=2)
  take last token state
  Linear(16, N_max) classifier
  CE loss

Sweep N_max ∈ {3, 5, 7, 10, 15, 20}. For each:
  - Train 250 epochs (plateaus quickly)
  - Per-N accuracy
  - Off-by-1 / off-by-2 / further error distribution
  - PCA on hidden states (PC1 var ratio, PC1↔N spearman)
  - Within-N std vs adjacent-N gap on PC1 (resolution metric)

Sub-questions:
  - At which N does adjacent-N PC1 gap < within-N std (becomes inseparable)?
  - At which N does per-N acc drop below 95%?
  - Does Spearman(PC1, log(N)) > Spearman(PC1, N) (Fechner law confirmed)?
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from mamba_ssm import Mamba

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.subitizing_v5b import generate_signal  # noqa


def make_batch(batch_size, T, N_max, rng, **sig_kwargs):
    Ns = rng.integers(1, N_max + 1, size=batch_size)
    sigs = np.stack([generate_signal(int(N), T, rng=rng, **sig_kwargs) for N in Ns])
    return (
        torch.from_numpy(sigs),
        torch.from_numpy(Ns - 1).long(),  # 0-indexed labels
    )


class ANS_Mamba(nn.Module):
    def __init__(self, N_max, d_model=16, d_state=16, num_layers=2, k=5):
        super().__init__()
        self.encoder = nn.Conv1d(1, d_model, kernel_size=k, padding=k // 2)
        self.activation = nn.ReLU()
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, N_max)
        self.d_model = d_model

    def forward(self, signal, return_h=False):
        # signal: (B, T)
        x = signal.unsqueeze(1)  # (B, 1, T)
        h = self.activation(self.encoder(x))  # (B, d, T)
        h = h.transpose(1, 2)  # (B, T, d)
        for m in self.mamba_layers:
            h = m(h)
        last_h = h[:, -1, :]  # (B, d)
        logits = self.classifier(last_h)
        if return_h:
            return logits, last_h
        return logits


def train_one(N_max, args, device, sig_kwargs):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = ANS_Mamba(N_max, d_model=args.d_model, d_state=args.d_state,
                      num_layers=args.num_layers, k=args.kernel).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  --- N_max={N_max}, params={n_params} ---")

    for ep in range(args.epochs):
        model.train(True)
        ep_acc = 0.0
        for _ in range(args.steps_per_epoch):
            x, y = make_batch(args.batch_size, args.T, N_max, rng, **sig_kwargs)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_acc += (logits.argmax(-1) == y).float().mean().item()
        ep_acc /= args.steps_per_epoch
        if (ep + 1) % args.log_every == 0:
            print(f"    ep {ep+1:>3}: acc={ep_acc:.4f}")

    return model


def diagnose(model, N_max, args, device, sig_kwargs):
    model.train(False)
    rng_chk = np.random.default_rng(args.seed + 1000)
    per_N_correct = np.zeros(N_max, dtype=int)
    per_N_total = np.zeros(N_max, dtype=int)
    per_N_off = np.zeros((N_max, 2 * N_max + 1), dtype=int)  # off by [-N_max, +N_max]
    H_list, N_list = [], []
    with torch.no_grad():
        for ni, N_val in enumerate(range(1, N_max + 1)):
            sigs = np.stack([generate_signal(int(N_val), args.T, rng=rng_chk, **sig_kwargs)
                             for _ in range(args.n_per_N)])
            x = torch.from_numpy(sigs).to(device)
            logits, h = model(x, return_h=True)
            preds = logits.argmax(-1).cpu().numpy()
            true = ni
            per_N_correct[ni] = (preds == true).sum()
            per_N_total[ni] = len(preds)
            for p in preds:
                off = p - true  # signed offset
                idx = off + N_max  # shift to [0, 2*N_max]
                per_N_off[ni, idx] += 1
            H_list.append(h.cpu().numpy())
            N_list.extend([N_val] * args.n_per_N)

    H = np.concatenate(H_list, axis=0)
    N_arr = np.array(N_list)

    # PCA
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var_ratio = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio /= var_ratio.sum()
    cum_var = np.cumsum(var_ratio)
    PC1 = Hc @ Vt[0]  # (N_total,)

    rho_N, _ = spearmanr(PC1, N_arr)
    rho_logN, _ = spearmanr(PC1, np.log(N_arr))

    # Within-N std + adjacent-N PC1 gap
    pc1_per_n = {n: PC1[N_arr == n] for n in range(1, N_max + 1)}
    within_std = np.mean([v.std() for v in pc1_per_n.values()])
    means = [pc1_per_n[n].mean() for n in range(1, N_max + 1)]
    adj_gaps = np.abs(np.diff(means))
    resolution_ratio = adj_gaps / max(within_std, 1e-9)  # how many sigma between adj N

    return {
        "N_max": N_max,
        "overall_acc": float(per_N_correct.sum() / per_N_total.sum()),
        "per_N_acc": (per_N_correct / np.maximum(per_N_total, 1)).tolist(),
        "per_N_off_distribution": per_N_off.tolist(),
        "PC1_var_ratio": float(var_ratio[0]),
        "cum_var_99": int(np.argmax(cum_var >= 0.99) + 1),
        "PC1_vs_N_spearman": float(rho_N),
        "PC1_vs_logN_spearman": float(rho_logN),
        "PC1_within_std": float(within_std),
        "PC1_adj_gaps": adj_gaps.tolist(),
        "PC1_resolution_ratio": resolution_ratio.tolist(),
    }


def report(diag):
    N_max = diag["N_max"]
    print(f"\n  === N_max={N_max} results ===")
    print(f"  Overall acc: {diag['overall_acc']:.4f}")
    print(f"  PC1 var: {diag['PC1_var_ratio']:.3f}, cum_var_99 PCs: {diag['cum_var_99']}")
    print(f"  PC1↔N spearman: {diag['PC1_vs_N_spearman']:+.3f}")
    print(f"  PC1↔log(N) spearman: {diag['PC1_vs_logN_spearman']:+.3f}")
    print(f"  Within-N PC1 std: {diag['PC1_within_std']:.3f}")
    print(f"  Per-N accuracy:")
    for ni, acc in enumerate(diag["per_N_acc"]):
        N_val = ni + 1
        adj_gap = diag["PC1_adj_gaps"][ni - 1] if ni > 0 else float("nan")
        res = diag["PC1_resolution_ratio"][ni - 1] if ni > 0 else float("nan")
        marker = " <-- below 0.95" if acc < 0.95 else ""
        gap_str = f"adj_gap={adj_gap:.3f}, res={res:.2f}σ" if ni > 0 else "(first N)"
        print(f"    N={N_val:>2}: acc={acc:.3f}  {gap_str}{marker}")
    print(f"  Off-by-k summary (averaged over N):")
    off = np.array(diag["per_N_off_distribution"])  # (N_max, 2*N_max+1)
    n_total = off.sum(axis=1)
    off_norm = off / np.maximum(n_total[:, None], 1)
    # column index k+N_max corresponds to off=k
    for k in range(-3, 4):
        idx = k + N_max
        if 0 <= idx < off.shape[1]:
            avg = off_norm[:, idx].mean()
            print(f"    off by {k:+d}: {avg:.3f}")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sig_kwargs = dict(
        spike_width=args.spike_width,
        min_distance=args.min_distance,
        noise_std=args.noise_std,
    )
    n_max_list = [int(n) for n in args.n_max_list.split(",")]
    print(f"ANS capacity sweep: N_max in {n_max_list}, T={args.T}")
    print(f"  d_model={args.d_model}, d_state={args.d_state}, layers={args.num_layers}, k={args.kernel}")
    print(f"  sig_kwargs={sig_kwargs}, epochs/cfg={args.epochs}")

    all_results = []
    for N_max in n_max_list:
        model = train_one(N_max, args, device, sig_kwargs)
        diag = diagnose(model, N_max, args, device, sig_kwargs)
        report(diag)
        all_results.append(diag)
        del model
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*72}")
    print(f"=== ANS capacity sweep summary ===")
    print(f"{'='*72}")
    print(f"  {'N_max':>6}  {'overall_acc':>11}  {'PC1_var':>8}  {'min_per_N':>10}  {'min_res(σ)':>11}")
    for d in all_results:
        per_N = d["per_N_acc"]
        min_acc = min(per_N)
        res = d["PC1_resolution_ratio"]
        min_res = min(res) if res else float("nan")
        print(f"  {d['N_max']:>6}  {d['overall_acc']:>11.4f}  {d['PC1_var_ratio']:>8.3f}  {min_acc:>10.3f}  {min_res:>11.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "results": all_results}, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-max-list", type=str, default="3,5,7,10,15,20")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--spike-width", type=int, default=5)
    p.add_argument("--min-distance", type=int, default=12)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-per-N", type=int, default=300)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/ans_capacity.json")
    args = p.parse_args()
    run(args)
