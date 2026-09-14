# scripts/v33_recursion_observe.py
"""V33 substrate-as-iterated-operator observation.

DIAG_FOR: v33

Takes a trained V33 ckpt and iterates the HH-SSM substrate on its own output
K times, observing trajectory. Bypasses CNN encoder after iter 0 — recursion
form: h^(k+1) = substrate(h^(k), state=...).

Two state-handling modes (run both for direct comparison):
  reset : each iter, substrate(state=None) — fresh init each call
  carry : state propagated across iters — universal-transformer-like ponder

Per-N trajectory metrics:
  - ‖Δh‖₂ across iters (convergence / cycle / divergence diagnostic)
  - ‖h‖₂ growth (divergence early warning)
  - PC1 score at t=L (projected into iter-0 PCA basis, fixed across iters)
  - between-N / within-N variance ratio (N-information retained vs erased)

Outcomes (expected):
  A. ‖Δh‖→0, N-dep fixed point — recursion meaningful, K* via threshold
  B. ‖Δh‖→0, N-indep fixed point — recursion erases N info
  C. limit cycle — periodic dynamics, possibly mod-q-like
  D. divergence — substrate not self-loop compatible without retraining

NOTE (5/22): cont50 ckpt was trained for single-pass. iter ≥ 1 inputs are OOD
(h-distribution, not CNN-feat distribution). Negative result doesn't kill the
recursive-substrate-architecture idea — just shows untrained recursion fails.

Usage:
  python scripts/v33_recursion_observe.py --ckpt <path> --L 200 \\
    --n-values "1,5,10,15,20,25,30,45,60" --K 10 --samples-per-n 32 \\
    --mode both
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import sample_training_batch
from scripts.inspect_checkpoint import load_agent


def encode_with_cnn(agent, raw_alpha_inputs):
    """Run CNN encoder once to get initial substrate input. raw is (B, L, 3),
    output is (B, L, d_model)."""
    x = raw_alpha_inputs.transpose(1, 2)
    conv_outs = [conv(x) for conv in agent.dp_convs]
    feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
    return agent.dp_proj(feat)


def iterate_substrate(agent, x_init, K, mode):
    """Iterate substrate K times on its own output.

    Returns dict with keys:
      h_seqs:        list of length K+1, each (B, L, d) — h^(0), h^(1), ..., h^(K)
                     where h^(0) = x_init (CNN features) and h^(k) = substrate(h^(k-1))
      states:        list of length K+1, state dicts (None for h^(0))
    """
    substrate = agent.v33_substrate
    h_seqs = [x_init]
    states = [None]
    state = None
    cur = x_init
    for k in range(K):
        if mode == "reset":
            h_next, st_next = substrate(cur, state=None)
        elif mode == "carry":
            h_next, st_next = substrate(cur, state=state)
            state = st_next
        else:
            raise ValueError(f"unknown mode {mode}")
        h_seqs.append(h_next)
        states.append(st_next)
        cur = h_next
    return {"h_seqs": h_seqs, "states": states}


def collect_per_n_iter_stats(agent, scene_cfg, n_values, samples_per_n, K, mode,
                              device, seed=42):
    """For each N, run encode + K-iter substrate. Collect per-iter h_final stats.

    Returns dict:
      n_to_iter_h_final:  {N: list-of-K+1 arrays (samples, d)}
      n_to_delta_norm:    {N: list-of-K arrays (samples,) — ‖h^(k)-h^(k-1)‖ at t=L}
      n_to_h_norm:        {N: list-of-K+1 arrays (samples,) — ‖h^(k)‖ at t=L}
    """
    rng = np.random.default_rng(seed)
    n_to_iter_h_final = {N: [[] for _ in range(K + 1)] for N in n_values}
    n_to_delta_norm = {N: [[] for _ in range(K)] for N in n_values}
    n_to_h_norm = {N: [[] for _ in range(K + 1)] for N in n_values}

    for N in n_values:
        collected = 0
        attempts = 0
        while collected < samples_per_n and attempts < 30:
            attempts += 1
            take = max(8, min(32, samples_per_n - collected))
            saved = scene_cfg.alpha_distribution
            scene_cfg.alpha_distribution = "uniform"
            with torch.no_grad():
                inputs, _, _, metas = sample_training_batch(
                    batch_size=take * 3, n_max_stage=N, config=scene_cfg, rng=rng,
                )
            scene_cfg.alpha_distribution = saved
            keep_idx = [i for i, m in enumerate(metas) if m["N_alpha"] == N]
            if not keep_idx:
                continue
            keep_idx = keep_idx[:take]
            sub_inputs = inputs[keep_idx].to(device)

            L = scene_cfg.L
            alpha_raw = sub_inputs[:, :L, :]
            with torch.no_grad():
                x0 = encode_with_cnn(agent, alpha_raw)             # (B, L, d)
                out = iterate_substrate(agent, x0, K, mode)
            h_seqs = out["h_seqs"]                                   # K+1 tensors

            for k in range(K + 1):
                h_final = h_seqs[k][:, -1, :].detach().cpu().numpy()  # (B, d)
                n_to_iter_h_final[N][k].append(h_final)
                n_to_h_norm[N][k].append(np.linalg.norm(h_final, axis=-1))
            for k in range(1, K + 1):
                dh = (h_seqs[k][:, -1, :] - h_seqs[k - 1][:, -1, :]
                      ).detach().cpu().numpy()
                n_to_delta_norm[N][k - 1].append(np.linalg.norm(dh, axis=-1))
            collected += len(keep_idx)

    # consolidate lists → arrays
    for N in n_values:
        for k in range(K + 1):
            n_to_iter_h_final[N][k] = np.concatenate(n_to_iter_h_final[N][k], axis=0)
            n_to_h_norm[N][k] = np.concatenate(n_to_h_norm[N][k])
        for k in range(K):
            n_to_delta_norm[N][k] = np.concatenate(n_to_delta_norm[N][k])
    return n_to_iter_h_final, n_to_delta_norm, n_to_h_norm


def compute_iter0_pca(n_to_iter_h_final, n_values):
    """PCA basis from concatenated iter-0 h_finals across all N."""
    all_iter0 = np.concatenate([n_to_iter_h_final[N][0] for N in n_values], axis=0)
    mean = all_iter0.mean(axis=0, keepdims=True)
    H_c = all_iter0 - mean
    U, S, Vt = np.linalg.svd(H_c, full_matrices=False)
    return mean.squeeze(0), Vt, S


def project_to_pc(h, mean, Vt):
    """Project (n_samples, d) -> (n_samples, d) PC scores using given basis."""
    return (h - mean) @ Vt.T


def between_within_variance(per_n_h):
    """per_n_h: {N: (samples, d)}. Returns ratio between/within for d-dim mean.
    Higher = more N-separable."""
    Ns = list(per_n_h.keys())
    samples_per_n = [per_n_h[N].shape[0] for N in Ns]
    means = np.stack([per_n_h[N].mean(axis=0) for N in Ns], axis=0)  # (#N, d)
    grand_mean = means.mean(axis=0, keepdims=True)
    between = ((means - grand_mean) ** 2).sum(axis=-1).mean()
    within = np.mean([((per_n_h[N] - per_n_h[N].mean(axis=0, keepdims=True)) ** 2
                       ).sum(axis=-1).mean() for N in Ns])
    return between / max(within, 1e-12)


def report_mode(agent, scene_cfg, n_values, K, samples_per_n, mode, device, seed):
    print(f"\n{'='*70}\n=== MODE: {mode}   (K={K}, samples_per_n={samples_per_n}) ===\n{'='*70}")
    n2hf, n2dn, n2hn = collect_per_n_iter_stats(
        agent, scene_cfg, n_values, samples_per_n, K, mode, device, seed,
    )

    # PCA basis from iter 0 (encoded CNN features fed to substrate)
    mean0, Vt0, S0 = compute_iter0_pca(n2hf, n_values)
    var_frac_pc1 = (S0 ** 2 / (S0 ** 2).sum())[0]
    print(f"\nIter-0 PCA basis: PC1 captures {var_frac_pc1:.3f} of iter-0 variance")

    # Per-N per-iter: PC1 mean & std at t=L, ‖h‖, Δh norm
    print(f"\n--- per-iter ‖Δh‖₂ at t=L (mean across N range) ---")
    print(f"{'iter':>4} | " + " | ".join(f"{'N=':>2}{N:<3}" for N in n_values))
    for k in range(K):
        row = [f"{np.mean(n2dn[N][k]):>6.3f}" for N in n_values]
        print(f"{k+1:>4} | " + " | ".join(row))

    print(f"\n--- per-iter ‖h‖₂ at t=L (mean across N) ---")
    print(f"{'iter':>4} | " + " | ".join(f"{'N=':>2}{N:<3}" for N in n_values))
    for k in range(K + 1):
        row = [f"{np.mean(n2hn[N][k]):>6.2f}" for N in n_values]
        print(f"{k:>4} | " + " | ".join(row))

    # PC1 score per iter
    print(f"\n--- μ(PC1 of iter-0 basis) per iter [t=L] ---")
    print(f"{'iter':>4} | " + " | ".join(f"{'N=':>2}{N:<3}" for N in n_values))
    for k in range(K + 1):
        row = []
        for N in n_values:
            pc = project_to_pc(n2hf[N][k], mean0, Vt0)[:, 0]
            row.append(f"{pc.mean():>+6.2f}")
        print(f"{k:>4} | " + " | ".join(row))

    # Between-N / within-N variance ratio per iter
    # measures how N-separable the representation is at each iter
    print(f"\n--- between-N / within-N variance ratio per iter ---")
    print(f"  (higher = more N-separable; trending → 0 = N info erased)")
    print(f"{'iter':>4} | ratio")
    for k in range(K + 1):
        per_n = {N: n2hf[N][k] for N in n_values}
        r = between_within_variance(per_n)
        print(f"{k:>4} | {r:>8.3f}")

    # Convergence check: does ‖Δh‖ → 0?
    delta_first = np.mean([np.mean(n2dn[N][0]) for N in n_values])
    delta_last = np.mean([np.mean(n2dn[N][K - 1]) for N in n_values])
    print(f"\n  Δh convergence: iter1 = {delta_first:.3f}, iter{K} = {delta_last:.3f}")
    if delta_last < 0.1 * delta_first:
        print(f"  → CONVERGENT (Δh dropped >10×)")
    elif delta_last > 2.0 * delta_first:
        print(f"  → DIVERGENT (Δh grew >2×)")
    else:
        print(f"  → INCONCLUSIVE (mild drift / cycle / stagnation)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--task", default="successor_prediction")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--n-values", default="1,5,10,15,20,25,30,45,60")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--samples-per-n", type=int, default=32)
    parser.add_argument("--mode", choices=["reset", "carry", "both"], default="both")
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

    n_values = [int(x) for x in args.n_values.split(",")]
    modes = ["reset", "carry"] if args.mode == "both" else [args.mode]

    for mode in modes:
        report_mode(agent, scene_cfg, n_values, args.K, args.samples_per_n,
                    mode, device, args.seed)


if __name__ == "__main__":
    main()
