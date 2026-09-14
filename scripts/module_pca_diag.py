"""Multi-module PCA on V7s/V8/V8_kWTA: probe 1D-thermometer geometry per module.

For each probe (Mamba state, PFC pre-kWTA, PFC post-kWTA, CNN summary):
  - Run PCA across samples grouped by N
  - Report PC1 var ratio, PC2 var, top-K cumulative
  - PC1 ↔ N spearman + log spearman
  - Within-N std vs adj-N gap (resolution)
  - Inversions

Compares whether kWTA broke the 1D collapse seen in V7s stage 2 (Mamba PC1 90%, 45% inversions).
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa
from backend.core.mamba_agent import kwta_topk  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


def pca_report(name, H, N_arr):
    """Run PCA on (n_samples, d) matrix H with N labels."""
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var_ratio = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio /= var_ratio.sum()
    cum_var = np.cumsum(var_ratio)
    PC1 = Hc @ Vt[0]
    PC2 = Hc @ Vt[1] if len(Vt) > 1 else np.zeros_like(PC1)

    rho1_n, _ = spearmanr(PC1, N_arr)
    rho1_logn, _ = spearmanr(PC1, np.log(N_arr + 1e-3))
    rho2_n, _ = spearmanr(PC2, N_arr) if PC2.std() > 1e-9 else (float('nan'), None)

    pc1_per_n = {n: PC1[N_arr == n] for n in np.unique(N_arr)}
    within_std = float(np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1]))
    means = np.array([pc1_per_n[n].mean() for n in sorted(pc1_per_n.keys())])
    adj_gaps = np.abs(np.diff(means))
    direction = 1 if means[-1] > means[0] else -1
    inversions = int(sum(1 for i in range(len(means) - 1) if direction * (means[i+1] - means[i]) < 0))
    inv_pct = inversions / max(len(means) - 1, 1) * 100

    print(f"\n--- {name} (d={H.shape[1]}, n={H.shape[0]}) ---")
    eff_dim_99 = int(np.argmax(cum_var >= 0.99) + 1)
    print(f"  PC1 var: {var_ratio[0]:.3f}  PC2: {var_ratio[1] if len(var_ratio)>1 else 0:.3f}  PC3: {var_ratio[2] if len(var_ratio)>2 else 0:.3f}  cum99: {eff_dim_99} PCs")
    print(f"  PC1 ↔ N spearman: {rho1_n:+.3f}  ↔ log(N): {rho1_logn:+.3f}")
    if PC2.std() > 1e-9:
        print(f"  PC2 ↔ N spearman: {rho2_n:+.3f}")
    print(f"  PC1 within-N std: {within_std:.4f}")
    print(f"  PC1 adj-N gap: mean={adj_gaps.mean():.4f}, min={adj_gaps.min():.4f}")
    print(f"  PC1 ratio (gap/std): {adj_gaps.mean()/max(within_std,1e-9):.3f}  (>1 = separable)")
    print(f"  PC1 inversions: {inversions}/{len(means)-1} = {inv_pct:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--v7-mode", action="store_true", default=True)
    p.add_argument("--kwta-k", type=int, default=0,
                   help="If non-zero, also probe post-kWTA PFC")
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=args.v7_mode,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # Hook 1: Mamba α-scan end state (last mamba block output)
    mamba_captured = []
    def mamba_hook(module, inp, out):
        # Only capture first call (α scan); ignore readback/β
        if len(mamba_captured) == 0:
            # out: (B, T, d) — take last position
            mamba_captured.append(out[:, -1, :].detach())
    agent.mamba_blocks[-1].register_forward_hook(mamba_hook)

    # Hook 2: PFC compare output (h_α_pfc) — pre-kWTA
    pfc_compare_captured = []
    def pfc_hook(module, inp, out):
        # Encoder output (B, n_tokens, d). Token 0 = h_α_pfc.
        # Filter for compare-shape: 4 tokens for V7 mode (h_α, h_β, cnn_α_rb, cnn_β_sum)
        if out.shape[1] == 4:
            pfc_compare_captured.append(out[:, 0, :].detach())
    if isinstance(agent.pfc_compare, nn.TransformerEncoder):
        # Force slow path so hook fires
        agent.train(True)
        agent.pfc_compare.register_forward_hook(pfc_hook)
    else:
        agent.pfc_compare.register_forward_hook(pfc_hook)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = scene_cfg.K
    cfg.equal_weight = 0.20
    cfg.near_weight = 0.80
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    n_to_mamba = defaultdict(list)
    n_to_pfc = defaultdict(list)
    n_to_pfc_kwta = defaultdict(list)
    n_to_cnn_sum = defaultdict(list)

    for _ in range(args.n_batches):
        mamba_captured.clear()
        pfc_compare_captured.clear()

        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))

        # Mamba: only first capture (α scan end)
        if mamba_captured:
            m = mamba_captured[0].cpu().numpy()  # (B, d)
        else:
            print("WARN: no Mamba capture")
            return

        # PFC: take first compare-block output (4 tokens)
        if not pfc_compare_captured:
            print("WARN: no PFC compare capture")
            return
        # First compare block per batch
        pfc_first = pfc_compare_captured[0]  # (B, d)
        pfc_np = pfc_first.cpu().numpy()

        # Post-kWTA (if applicable): apply kwta_topk on pfc output
        if args.kwta_k > 0:
            pfc_kwta = kwta_topk(pfc_first, k=args.kwta_k).cpu().numpy()

        # CNN α summary (stashed by agent during forward)
        cnn_sum = agent._last_cnn_alpha_summary.cpu().numpy() if agent._last_cnn_alpha_summary is not None else None

        for b, meta in enumerate(metas):
            n = meta["N_alpha"]
            n_to_mamba[n].append(m[b])
            n_to_pfc[n].append(pfc_np[b])
            if args.kwta_k > 0:
                n_to_pfc_kwta[n].append(pfc_kwta[b])
            if cnn_sum is not None:
                n_to_cnn_sum[n].append(cnn_sum[b])

    # Build matrices
    def build(n_to):
        H = np.array([h for n in sorted(n_to.keys()) for h in n_to[n]])
        N = np.array([n for n in sorted(n_to.keys()) for h in n_to[n]])
        return H, N

    print(f"\n{'='*72}\n=== Multi-module PCA: {args.ckpt.name}\n{'='*72}")

    H, N = build(n_to_mamba)
    pca_report("Mamba α-scan end state (h_A[-1])", H, N)

    H, N = build(n_to_pfc)
    pca_report("PFC compare h_α_pfc (pre-kWTA)", H, N)

    if args.kwta_k > 0:
        H, N = build(n_to_pfc_kwta)
        pca_report(f"PFC compare h_α_pfc (post-kWTA k={args.kwta_k})", H, N)

    if n_to_cnn_sum:
        H, N = build(n_to_cnn_sum)
        pca_report("CNN α summary (cnn_α_summary)", H, N)


if __name__ == "__main__":
    main()
