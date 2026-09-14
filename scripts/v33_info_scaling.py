# scripts/v33_info_scaling.py
"""V33 substrate PC1 information-scaling diagnostic.

DIAG_FOR: v33

Answers: how does the information carried by the substrate's dominant 1D mode
(PC1 of h_final) scale with N?

For each N in a dense range, sample many episodes, capture h_final, PCA across
all samples, then report:
  - mean(PC1)(N) and std(PC1)(N)
  - local discriminability d'(N -> N+1) = |Δμ| / sqrt((σ_a²+σ_b²)/2)
  - mutual information I(PC1; N) via histogram (bits)
  - scaling fit: mean(PC1) ~ {linear, log(N), sqrt(N)} with R²
  - Weber check: σ(N) vs N proportional?
  - effective discriminable levels in [N_min, N_max] (≈ 2^I)

Usage:
  python scripts/v33_info_scaling.py --ckpt <path> --L 200 \\
    --n-train 30 --n-far 60 --samples-per-n 64
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import sample_training_batch
from scripts.inspect_checkpoint import load_agent
from scripts.v33_full_diag import capture_v33_internals


def collect_h_finals(agent, scene_cfg, n_values, samples_per_n, device, seed=42):
    """For each N, sample episodes and capture h_final ∈ R^d. Returns (H, Ns)
    where H is (total, d) and Ns is (total,)."""
    rng = np.random.default_rng(seed)
    all_h, all_n = [], []
    for N in n_values:
        n_collected = 0
        while n_collected < samples_per_n:
            take = min(32, samples_per_n - n_collected)
            with torch.no_grad():
                saved = scene_cfg.alpha_distribution
                scene_cfg.alpha_distribution = "uniform"
                # constrain alpha to exactly this N by setting n_max_stage=N
                # and rejecting non-matches
                inputs, _, _, metas = sample_training_batch(
                    batch_size=take * 3, n_max_stage=N, config=scene_cfg, rng=rng,
                )
                scene_cfg.alpha_distribution = saved
                # keep only episodes where N_alpha == N exactly
                keep_idx = [i for i, m in enumerate(metas) if m["N_alpha"] == N]
                if not keep_idx:
                    continue
                keep_idx = keep_idx[:take]
                sub_inputs = inputs[keep_idx]
                cap = capture_v33_internals(agent, sub_inputs, device)
                h_final = cap["h"][:, -1, :].numpy()  # (b, d)
                for b in range(h_final.shape[0]):
                    all_h.append(h_final[b])
                    all_n.append(N)
                n_collected += len(keep_idx)
    return np.stack(all_h, axis=0), np.array(all_n)


def per_n_stats(scores, Ns, n_values):
    """For each N in n_values, return mean/std/n of `scores` restricted to that N."""
    out = {}
    for N in n_values:
        mask = Ns == N
        if not mask.any():
            continue
        s = scores[mask]
        out[N] = (float(s.mean()), float(s.std()), int(mask.sum()))
    return out


def fit_scaling(n_values, means):
    """Fit means(N) to {linear, log, sqrt} via least squares. Returns dict
    with coeffs + R² per model."""
    N = np.array(n_values, dtype=float)
    y = np.array(means, dtype=float)
    out = {}
    for name, x in [("linear", N), ("log", np.log(N)), ("sqrt", np.sqrt(N))]:
        X = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        out[name] = {"a": float(coef[0]), "b": float(coef[1]), "r2": float(r2)}
    return out


def mutual_info_pc1(pc1, Ns, n_bins=20):
    """I(PC1; N) in bits via 2D histogram. PC1 binned into n_bins,
    N kept as discrete categories."""
    p1_min, p1_max = pc1.min(), pc1.max()
    edges = np.linspace(p1_min - 1e-9, p1_max + 1e-9, n_bins + 1)
    bin_idx = np.digitize(pc1, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    n_unique = sorted(np.unique(Ns))
    n_to_row = {n: i for i, n in enumerate(n_unique)}
    M = np.zeros((len(n_unique), n_bins), dtype=float)
    for n, b in zip(Ns, bin_idx):
        M[n_to_row[int(n)], b] += 1.0
    P = M / M.sum()
    Pn = P.sum(axis=1, keepdims=True)
    Pb = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P / (Pn * Pb)
        log_ratio = np.where(P > 0, np.log2(ratio + 1e-30), 0.0)
        mi = float((P * log_ratio).sum())
    H_n = float(-(Pn[Pn > 0] * np.log2(Pn[Pn > 0])).sum())
    H_b = float(-(Pb[Pb > 0] * np.log2(Pb[Pb > 0])).sum())
    return mi, H_n, H_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--task", default="successor_prediction")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--n-train", type=int, default=30, help="dense range upper bound")
    parser.add_argument("--n-far", type=int, default=60, help="sparse extrap range")
    parser.add_argument("--samples-per-n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
    )
    device = next(agent.parameters()).device
    scene_cfg.task = args.task
    agent.train(False)
    if agent.v33_substrate is None:
        print("ERROR: not a V33 ckpt")
        return

    # dense train range + sparse extrap
    dense = list(range(1, args.n_train + 1))
    sparse = list(range(args.n_train + 5, args.n_far + 1, 5))
    n_values = dense + sparse

    print(f"Collecting h_final for N in {n_values}, {args.samples_per_n} eps each...")
    H, Ns = collect_h_finals(agent, scene_cfg, n_values, args.samples_per_n,
                             device, seed=args.seed)
    print(f"  collected {H.shape[0]} total samples, h_dim={H.shape[1]}")

    # PCA across all samples
    H_centered = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    pc_scores = U * S  # (total, d)
    var_frac = (S ** 2) / (S ** 2).sum()

    # participation ratio = (Σλ)² / Σλ²  (effective number of dims)
    lam = S ** 2
    part_ratio = float((lam.sum() ** 2) / (lam ** 2).sum())
    print(f"\nVariance fractions: PC1={var_frac[0]:.4f} PC2={var_frac[1]:.4f} "
          f"PC3={var_frac[2]:.4f}")
    print(f"Effective dim (participation ratio): {part_ratio:.3f}")

    pc1 = pc_scores[:, 0]
    pc2 = pc_scores[:, 1]
    rho1, _ = spearmanr(pc1, Ns)
    rho2, _ = spearmanr(pc2, Ns)
    print(f"Spearman ρ(PC1, N) = {rho1:.3f},  ρ(PC2, N) = {rho2:.3f}")

    # per-N mean ± std on PC1
    stats = per_n_stats(pc1, Ns, n_values)

    print("\n=== PC1 per-N stats (μ ± σ, count) ===")
    print(f"{'N':>4} | {'μ(PC1)':>9} | {'σ(PC1)':>8} | {'n':>4}")
    means_train = []
    stds_train = []
    for N in dense:
        if N in stats:
            mu, sd, n = stats[N]
            print(f"{N:>4} | {mu:>+9.4f} | {sd:>8.4f} | {n:>4}")
            means_train.append(mu)
            stds_train.append(sd)
    print("---- (sparse extrap range) ----")
    for N in sparse:
        if N in stats:
            mu, sd, n = stats[N]
            print(f"{N:>4} | {mu:>+9.4f} | {sd:>8.4f} | {n:>4}")

    # local d-prime: |μ(N+1) - μ(N)| / sqrt((σ²+σ²)/2)
    print("\n=== Local discriminability d'(N -> N+1) ===")
    d_prime_arr = []
    for i in range(len(dense) - 1):
        N_a, N_b = dense[i], dense[i + 1]
        if N_a in stats and N_b in stats:
            mu_a, sd_a, _ = stats[N_a]
            mu_b, sd_b, _ = stats[N_b]
            pooled_sd = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2 + 1e-12)
            dp = abs(mu_b - mu_a) / pooled_sd
            d_prime_arr.append(dp)
    if d_prime_arr:
        print(f"  d' across N=1..30: mean={np.mean(d_prime_arr):.3f}  "
              f"min={np.min(d_prime_arr):.3f}  max={np.max(d_prime_arr):.3f}")
        # check decay: corr(d', N)
        Ns_pair = np.array(dense[:len(d_prime_arr)])
        rho_dp, _ = spearmanr(Ns_pair, d_prime_arr)
        print(f"  Spearman ρ(N, d') = {rho_dp:+.3f}  "
              f"(negative → discriminability falls with N, Weber-like)")

    # Weber check: σ(N) vs N
    if means_train and stds_train:
        sigma_arr = np.array(stds_train)
        N_arr = np.array(dense[:len(stds_train)])
        rho_sn, _ = spearmanr(N_arr, sigma_arr)
        # linear fit σ = k*N + c
        X = np.column_stack([N_arr, np.ones_like(N_arr, dtype=float)])
        k, c = np.linalg.lstsq(X, sigma_arr, rcond=None)[0]
        sigma_hat = X @ np.array([k, c])
        r2_sig = 1 - ((sigma_arr - sigma_hat) ** 2).sum() / max(((sigma_arr - sigma_arr.mean()) ** 2).sum(), 1e-12)
        print(f"\n=== Weber check ===")
        print(f"  Spearman ρ(N, σ(PC1)) = {rho_sn:+.3f}")
        print(f"  σ ≈ {k:+.5f} * N + {c:+.5f}   (R²={r2_sig:.3f})")
        if abs(k) > 1e-6:
            cv_low = sigma_arr[0] / max(abs(means_train[0]), 1e-6)
            cv_high = sigma_arr[-1] / max(abs(means_train[-1]), 1e-6)
            print(f"  CV: N={dense[0]}: σ/|μ|={cv_low:.3f}   "
                  f"N={dense[len(stds_train)-1]}: σ/|μ|={cv_high:.3f}")

    # Scaling fits on mean(PC1)(N)
    if means_train:
        fits = fit_scaling(dense[:len(means_train)], means_train)
        print("\n=== Scaling fit: μ(PC1) vs N ===")
        for name, p in fits.items():
            print(f"  {name:>6}:  μ ≈ {p['a']:+.5f} * f(N) + {p['b']:+.5f}   "
                  f"R²={p['r2']:.4f}")

    # I(PC1; N) — discrete N, binned PC1
    train_mask = np.isin(Ns, dense)
    mi_train, Hn_train, _ = mutual_info_pc1(pc1[train_mask], Ns[train_mask],
                                             n_bins=20)
    print("\n=== Mutual information I(PC1; N) — training range N=1..{} ===".format(
        args.n_train))
    print(f"  I(PC1; N) = {mi_train:.3f} bits   "
          f"(H(N) = {Hn_train:.3f} bits → capacity floor)")
    print(f"  Effective distinguishable N-levels along PC1: 2^I ≈ {2 ** mi_train:.1f}")
    print(f"  vs total N values in range: {len(dense)} → "
          f"info ratio = {mi_train / max(Hn_train, 1e-9):.3f}")

    # I(PC2; N) for comparison
    mi_pc2, _, _ = mutual_info_pc1(pc2[train_mask], Ns[train_mask], n_bins=20)
    print(f"  I(PC2; N) = {mi_pc2:.3f} bits  (orthogonal residual carrier)")


if __name__ == "__main__":
    main()
