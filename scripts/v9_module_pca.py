"""V9 (cnn_pfc_only) multi-module PCA.

V9 PFC compare seq is 2 tokens (h_α=cnn_α_summary, h_β=cnn_β_summary), not 4.
Mamba is bypassed entirely (cnn_pfc_only=True), so no Mamba hook.

Probes:
  - CNN α summary (cardinality detector input)
  - PFC compare h_α_pfc (post-self-attn, pre-kWTA)
  - PFC compare h_α_pfc (post-kWTA)
  - scratch_pad code (5-pos quantized output)
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

from backend.core.scene import SceneConfig, sample_training_batch
from backend.core.mamba_agent import kwta_topk
from scripts.inspect_checkpoint import load_agent


def pca_report(name, H, N_arr):
    if H.shape[0] < 4:
        print(f"\n--- {name}: too few samples ({H.shape[0]}), skip ---")
        return
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var_ratio = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio /= var_ratio.sum()
    cum_var = np.cumsum(var_ratio)
    PC1 = Hc @ Vt[0]
    PC2 = Hc @ Vt[1] if len(Vt) > 1 else np.zeros_like(PC1)

    rho1_n, _ = spearmanr(PC1, N_arr)
    rho1_logn, _ = spearmanr(PC1, np.log(N_arr + 1e-3))
    rho2_n = spearmanr(PC2, N_arr)[0] if PC2.std() > 1e-9 else float('nan')

    pc1_per_n = {n: PC1[N_arr == n] for n in np.unique(N_arr)}
    within_std = float(np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1]))
    means = np.array([pc1_per_n[n].mean() for n in sorted(pc1_per_n.keys())])
    adj_gaps = np.abs(np.diff(means))
    direction = 1 if means[-1] > means[0] else -1
    inversions = int(sum(1 for i in range(len(means) - 1) if direction * (means[i+1] - means[i]) < 0))
    inv_pct = inversions / max(len(means) - 1, 1) * 100

    eff_dim_99 = int(np.argmax(cum_var >= 0.99) + 1)
    print(f"\n--- {name} (d={H.shape[1]}, n={H.shape[0]}, Ns={len(np.unique(N_arr))}) ---")
    print(f"  PC1 var: {var_ratio[0]:.3f}  PC2: {var_ratio[1] if len(var_ratio)>1 else 0:.3f}  PC3: {var_ratio[2] if len(var_ratio)>2 else 0:.3f}  cum99: {eff_dim_99} PCs")
    print(f"  PC1 ↔ N spearman: {rho1_n:+.3f}  ↔ log(N): {rho1_logn:+.3f}")
    if PC2.std() > 1e-9:
        print(f"  PC2 ↔ N spearman: {rho2_n:+.3f}")
    print(f"  PC1 within-N std: {within_std:.4f}")
    if len(adj_gaps) > 0:
        print(f"  PC1 adj-N gap: mean={adj_gaps.mean():.4f}, min={adj_gaps.min():.4f}")
        print(f"  PC1 ratio (gap/std): {adj_gaps.mean()/max(within_std,1e-9):.3f}  (>1 = separable)")
    print(f"  PC1 inversions: {inversions}/{len(means)-1} = {inv_pct:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--kwta-k", type=int, default=5)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=True,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # Hook PFC compare encoder output (2 tokens for V9: h_α, h_β)
    pfc_captures = []
    def pfc_hook(module, inp, out):
        # out: (B, T, d). T should be 2 for V9
        if out.shape[1] == 2:
            pfc_captures.append(out[:, 0, :].detach().clone())  # token 0 = h_α_pfc

    # Force slow path so hook fires (TransformerEncoder fastpath skips hooks in eval)
    agent.train(True)
    agent.pfc_compare.register_forward_hook(pfc_hook)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.equal_weight = 0.20
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    n_to_cnn = defaultdict(list)
    n_to_pfc = defaultdict(list)
    n_to_pfc_kwta = defaultdict(list)
    n_to_scratch = defaultdict(list)

    for _ in range(args.n_batches):
        pfc_captures.clear()
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _, scratch, _ = agent(inputs.to(device), ci.to(device))

        if not pfc_captures:
            print("WARN: PFC hook never fired (shape[1] != 2). Got nothing.")
            return

        # First compare block per batch
        pfc_first = pfc_captures[0]  # (B, d)
        pfc_np = pfc_first.cpu().numpy()
        pfc_kwta = kwta_topk(pfc_first, k=args.kwta_k).cpu().numpy()

        cnn_sum = agent._last_cnn_alpha_summary.cpu().numpy() if agent._last_cnn_alpha_summary is not None else None
        scr_np = scratch.cpu().numpy()

        for b, meta in enumerate(metas):
            n = meta["N_alpha"]
            n_to_pfc[n].append(pfc_np[b])
            n_to_pfc_kwta[n].append(pfc_kwta[b])
            n_to_scratch[n].append(scr_np[b])
            if cnn_sum is not None:
                n_to_cnn[n].append(cnn_sum[b])

    def build(n_to):
        H = np.array([h for n in sorted(n_to.keys()) for h in n_to[n]])
        N = np.array([n for n in sorted(n_to.keys()) for h in n_to[n]])
        return H, N

    print(f"\n{'='*72}\n=== V9 multi-module PCA: {args.ckpt.name}\n{'='*72}")

    if n_to_cnn:
        H, N = build(n_to_cnn)
        pca_report("CNN α summary", H, N)
    H, N = build(n_to_pfc)
    pca_report("PFC compare h_α_pfc (pre-kWTA)", H, N)
    H, N = build(n_to_pfc_kwta)
    pca_report(f"PFC compare h_α_pfc (post-kWTA k={args.kwta_k})", H, N)
    H, N = build(n_to_scratch)
    pca_report("scratch_pad code (5 floats)", H, N)


if __name__ == "__main__":
    main()
