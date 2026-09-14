"""PCA on h_alpha (hidden state encoding N) to test 1D thermometer hypothesis.

For uniform N in [1, n_max], collect h_alpha (mean over W write positions, the
cycle-loss target). Do PCA on (n_samples, d_model). Report:
  - cumulative variance per PC
  - PC1 score vs N (correlation, monotonicity)
  - effective dimensionality (95% variance)
  - 2D scatter of (PC1, PC2) colored by N (saved as png-able array)
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=600)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=200)
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    p.add_argument("--v7-mode", action="store_true",
                   help="Load V7 mode (dual_pathway + Mamba on readback)")
    p.add_argument("--probe", choices=["h_for_cycle", "mamba_pure"], default="mamba_pure",
                   help="Which hidden state to PCA: h_for_cycle (PFC+Mamba mix) "
                        "or mamba_pure (h_A[L-1], pure Mamba state at α-scan end)")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=args.q, qrange=args.qrange, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
        v7_mode=args.v7_mode,
    )
    agent.train(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=args.complex)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    # Hook on last Mamba block to capture h_A_mamba (pure Mamba state at α-scan)
    mamba_capture = []
    def mamba_hook(module, input, output):
        # output shape: (B, T_seg, d). Capture state at end of α scan (index L-1)
        # but only on first call per forward (α encode); skip readback/β encodes.
        if len(mamba_capture) == 0:  # only first call
            mamba_capture.append(output[:, args.L - 1, :].detach().cpu())
    if args.probe == "mamba_pure" and hasattr(agent, "mamba_blocks"):
        last_mamba = agent.mamba_blocks[-1]
        hook_handle = last_mamba.register_forward_hook(mamba_hook)
    else:
        hook_handle = None

    H_list = []     # (n_samples, d_model)
    N_list = []     # (n_samples,)

    for _ in range(args.n_batches):
        mamba_capture.clear()
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))
        if args.probe == "mamba_pure":
            if not mamba_capture:
                print("ERROR: hook didn't fire. Check mamba_blocks structure.")
                return
            h_alpha = mamba_capture[0]
        else:
            h_alpha = getattr(agent, "_last_h_alpha_for_cycle", None)
            if h_alpha is None:
                h_alpha = getattr(agent, "_last_h_for_write_mean", None)
            if h_alpha is None:
                print("ERROR: no h_alpha attribute on agent.")
                return
        # h_alpha may be cpu tensor (from hook) or gpu tensor (from stash). Handle both.
        if hasattr(h_alpha, 'cpu'):
            H_list.append(h_alpha.cpu().numpy() if h_alpha.is_cuda else h_alpha.numpy())
        else:
            H_list.append(np.asarray(h_alpha))
        for m in metas:
            N_list.append(m["N_alpha"])

    H = np.concatenate(H_list, axis=0)  # (N_total, d)
    N = np.array(N_list, dtype=int)
    print(f"\n=== {args.ckpt} ===")
    print(f"Collected: {H.shape[0]} samples, d_model={H.shape[1]}, N range [1, {args.n_max}]")

    # PCA via SVD on centered H
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio = var / var.sum()
    cum_var = np.cumsum(var_ratio)

    print("\n--- PCA variance ---")
    print(f"  {'PC':>3}  {'var_ratio':>10}  {'cum_var':>10}")
    for i in range(len(var_ratio)):
        print(f"  {i+1:>3}  {var_ratio[i]:>10.4f}  {cum_var[i]:>10.4f}")

    eff_dim_50 = int(np.argmax(cum_var >= 0.50)) + 1
    eff_dim_90 = int(np.argmax(cum_var >= 0.90)) + 1
    eff_dim_99 = int(np.argmax(cum_var >= 0.99)) + 1
    print(f"\n--- Effective dim ---")
    print(f"  50% var:  {eff_dim_50:>2} PCs")
    print(f"  90% var:  {eff_dim_90:>2} PCs")
    print(f"  99% var:  {eff_dim_99:>2} PCs")
    print(f"  (1 = pure 1D thermometer; >5 = rich high-D representation)")

    # Project H onto first 5 PCs
    PC = Hc @ Vt.T[:, :5]  # (N_total, 5)

    print("\n--- PC vs N correlation ---")
    for i in range(5):
        rho_p, _ = spearmanr(PC[:, i], N)
        r_lin = np.corrcoef(PC[:, i], N)[0, 1]
        r_log = np.corrcoef(PC[:, i], np.log(N + 1e-3))[0, 1]
        print(f"  PC{i+1}: spearman={rho_p:>6.3f}  pearson(N)={r_lin:>6.3f}  pearson(logN)={r_log:>6.3f}")

    # Per-N cluster compactness (within-N std vs between-N std on PC1)
    n_unique = sorted(set(N.tolist()))
    pc1_per_n = {n: PC[N == n, 0] for n in n_unique}
    within_std = np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1])
    between_std = np.std([v.mean() for v in pc1_per_n.values() if len(v) > 0])
    print(f"\n--- PC1 clustering ---")
    print(f"  within-N std (avg):  {within_std:.4f}")
    print(f"  between-N std:       {between_std:.4f}")
    print(f"  ratio (between/within): {between_std/max(within_std,1e-9):.2f}")
    print(f"  (>1 = N-distinguishable; >3 = clean separation; <1 = N collapsed)")

    # PC1 monotonicity: sort N, walk through means, count inversions
    sorted_n = sorted(n_unique)
    mean_pc1 = [pc1_per_n[n].mean() for n in sorted_n]
    direction = 1 if mean_pc1[-1] > mean_pc1[0] else -1
    inversions = 0
    for i in range(len(mean_pc1) - 1):
        if direction * (mean_pc1[i+1] - mean_pc1[i]) < 0:
            inversions += 1
    print(f"\n--- PC1 monotonicity ---")
    print(f"  inversions: {inversions} / {len(mean_pc1)-1} adjacent N pairs")
    print(f"  (0 = perfectly monotone; high = noisy / non-monotone)")

    # Resolution at low/high N (gap between adjacent N means)
    sorted_pc1 = np.array(mean_pc1)
    diffs = np.abs(np.diff(sorted_pc1))
    print(f"\n--- Resolution at adjacent N (PC1 distance) ---")
    print(f"  N=1..10:    avg gap = {diffs[:9].mean():.4f}  (within_std ref: {within_std:.4f})")
    if len(diffs) > 50:
        print(f"  N=50..60:   avg gap = {diffs[49:59].mean():.4f}")
    if len(diffs) > 100:
        print(f"  N=100..110: avg gap = {diffs[99:109].mean():.4f}")
    if len(diffs) > 150:
        print(f"  N=150..160: avg gap = {diffs[149:159].mean():.4f}")
    print(f"  (gap < within_std means adjacent N are not separable on PC1)")


if __name__ == "__main__":
    main()
