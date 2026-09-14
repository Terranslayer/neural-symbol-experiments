# DIAG_FOR: v25-v19 | ANSWERS: pca,write_head_raw,cell_decouple | INPUTS: <ckpt>
"""V25-V19 module-internal representation analysis.

Probes per-N representation in:
  1. CNN multi-scale α summary (cnn_α_summary)
  2. LRU α-scan end state (last block out[:, -1, :])
  3. PFC compare h_α_pfc (post self-attn token 0)
  4. PFC compare h_β_pfc (token 1)
  5. compare_in (h_α_pfc − h_β_pfc) — fed to RBF compare head
  6. Write-head raw pre-quantize (per-cell, by N)
  7. Cell-cell Pearson on raw pre-quantize

For each PCA: PC1/2/3 var ratio, PC1↔N spearman, within-N std vs adj-N gap,
inversion count.

V25-V19 specifics:
  - dp_scratch_skip_mamba=True → readback uses CNN only
  - cnn_pfc_only=False → β segment uses LRU+CNN
  - pfc_compare_drop_raw_scratch=True → seq_c = [h_α, h_β] (2 tokens, no scratch)
  - kwta_k=5 applied AFTER mean over tokens
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def pca_report(name, H, N_arr):
    if H.shape[0] < 2:
        print(f"\n--- {name}: not enough samples (n={H.shape[0]}) ---")
        return
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = (S ** 2) / max(H.shape[0] - 1, 1)
    var /= max(var.sum(), 1e-12)
    cum = np.cumsum(var)
    PC1 = Hc @ Vt[0]
    PC2 = Hc @ Vt[1] if len(Vt) > 1 else np.zeros_like(PC1)
    PC3 = Hc @ Vt[2] if len(Vt) > 2 else np.zeros_like(PC1)

    rho1_n, _ = spearmanr(PC1, N_arr) if PC1.std() > 1e-9 else (float('nan'), None)
    rho1_logn, _ = spearmanr(PC1, np.log(N_arr + 1e-3)) if PC1.std() > 1e-9 else (float('nan'), None)
    rho2_n = float('nan')
    if PC2.std() > 1e-9:
        rho2_n, _ = spearmanr(PC2, N_arr)
    rho3_n = float('nan')
    if PC3.std() > 1e-9:
        rho3_n, _ = spearmanr(PC3, N_arr)

    pc1_per_n = {n: PC1[N_arr == n] for n in np.unique(N_arr)}
    within = float(np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1])) if any(len(v) > 1 for v in pc1_per_n.values()) else float('nan')
    means = np.array([pc1_per_n[n].mean() for n in sorted(pc1_per_n.keys())])
    if len(means) > 1:
        gaps = np.abs(np.diff(means))
        direction = 1 if means[-1] > means[0] else -1
        inv = sum(1 for i in range(len(means)-1) if direction*(means[i+1]-means[i]) < 0)
        inv_pct = inv / max(len(means)-1, 1) * 100
        gap_mean = gaps.mean()
        gap_min = gaps.min()
    else:
        inv, inv_pct, gap_mean, gap_min = 0, 0, 0, 0

    eff99 = int(np.argmax(cum >= 0.99) + 1)
    print(f"\n--- {name} (d={H.shape[1]}, n={H.shape[0]}) ---")
    print(f"  PC1 var: {var[0]:.3f}  PC2: {var[1] if len(var)>1 else 0:.3f}  PC3: {var[2] if len(var)>2 else 0:.3f}  cum99: {eff99} PCs")
    print(f"  PC1 ↔ N: {rho1_n:+.3f}  ↔ log(N): {rho1_logn:+.3f}")
    print(f"  PC2 ↔ N: {rho2_n:+.3f}    PC3 ↔ N: {rho3_n:+.3f}")
    print(f"  PC1 within-N std: {within:.4f}")
    print(f"  PC1 adj-N gap mean: {gap_mean:.4f}  min: {gap_min:.4f}  ratio gap/std: {gap_mean/max(within,1e-9):.3f}")
    print(f"  PC1 inversions: {inv}/{len(means)-1} = {inv_pct:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # === Hooks ===
    # LRU/Mamba α-scan end state. mamba_blocks[-1] runs on every _encode_segment
    # call (α-seg, readback IF not dp_scratch_skip_mamba, β-seg). For V25-V19 with
    # dp_scratch_skip_mamba=True, only α and β go through LRU. We want α scan end.
    block_calls = []
    def block_hook(module, inp, out):
        # out: (B, T, d). For α scan T=L; β scan T=L+1; readback skipped under V25.
        block_calls.append(out[:, -1, :].detach().cpu())
    agent.mamba_blocks[-1].register_forward_hook(block_hook)

    # PFC compare encoder output (token 0 = h_α_pfc, token 1 = h_β_pfc).
    # V25-V19 uses 2-token compare (no v7_compare since dp_scratch_skip_mamba=True).
    pfc_compare_calls = []
    pfc_compare_module = agent.pfc_compare if agent.pfc_compare is not None else agent.pfc
    def pfc_hook(module, inp, out):
        if out.shape[1] == 2:
            pfc_compare_calls.append(out.detach().cpu())  # full (B, 2, d)
    pfc_compare_module.register_forward_hook(pfc_hook)

    # write_head raw pre-quantize: stash via monkey-patch on SequentialWriteHead
    from backend.core.mamba_agent import SequentialWriteHead
    write_raw_calls = []
    if isinstance(agent.write_head, SequentialWriteHead):
        orig_fwd = agent.write_head.forward
        def wrapped(h_pfc, qlevels, qrange):
            raw, q = orig_fwd(h_pfc, qlevels, qrange)
            write_raw_calls.append(raw.detach().cpu())  # (B, W)
            return raw, q
        agent.write_head.forward = wrapped

    # Setup scene with broader N coverage for PCA
    cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True) if hasattr(SceneConfig, "trio_wide_extended_preset") else SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = scene_cfg.K
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(args.seed)

    n_to_lru_alpha = defaultdict(list)
    n_to_pfc_a = defaultdict(list)
    n_to_pfc_b = defaultdict(list)
    n_to_compare_in = defaultdict(list)
    n_to_cnn_sum = defaultdict(list)
    n_to_write_raw = defaultdict(list)
    write_raw_per_cell = []  # all writes, for cell-cell pearson

    for batch_i in range(args.n_batches):
        block_calls.clear()
        pfc_compare_calls.clear()
        write_raw_calls.clear()

        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))

        # block_calls: 1 (α) + K (β per compare block). First = α.
        if not block_calls:
            print("WARN: no LRU capture")
            return
        lru_alpha = block_calls[0].numpy()  # (B, d)

        # write_raw: 1 (α write) + (compare blocks * virtual β write) per episode.
        # Take first = α write.
        if not write_raw_calls:
            print("WARN: no write_head capture (not SequentialWriteHead?)")
        else:
            wr = write_raw_calls[0].numpy()  # (B, W)
            for b, meta in enumerate(metas):
                n_to_write_raw[meta["N_alpha"]].append(wr[b])
                write_raw_per_cell.append(wr[b])

        # PFC compare: K calls per episode (one per compare block). Take first.
        if pfc_compare_calls:
            pfc0 = pfc_compare_calls[0].numpy()  # (B, 2, d)
            h_a_pfc = pfc0[:, 0, :]
            h_b_pfc = pfc0[:, 1, :]
            for b, meta in enumerate(metas):
                n_a = meta["N_alpha"]
                n_to_pfc_a[n_a].append(h_a_pfc[b])
                n_to_compare_in[n_a].append(h_a_pfc[b] - h_b_pfc[b])
                # h_β_pfc keyed by N_β of the block-0 β
                n_b = meta["beta_counts"][0]
                n_to_pfc_b[n_b].append(h_b_pfc[b])

        # CNN α summary stashed by agent
        if agent._last_cnn_alpha_summary is not None:
            cnn = agent._last_cnn_alpha_summary.cpu().numpy()
            for b, meta in enumerate(metas):
                n_to_cnn_sum[meta["N_alpha"]].append(cnn[b])

        for b, meta in enumerate(metas):
            n_to_lru_alpha[meta["N_alpha"]].append(lru_alpha[b])

    def build(n_to):
        keys = sorted(n_to.keys())
        H = np.array([h for n in keys for h in n_to[n]])
        N = np.array([n for n in keys for h in n_to[n]])
        return H, N

    print(f"\n{'='*72}\n=== V25-V19 module PCA: {args.ckpt.name}\n=== n batches={args.n_batches} bs={args.batch_size} n_max={args.n_max}\n{'='*72}")

    if n_to_cnn_sum:
        H, N = build(n_to_cnn_sum)
        pca_report("CNN α summary (multi-scale CNN, gradient-active)", H, N)
    if n_to_lru_alpha:
        H, N = build(n_to_lru_alpha)
        pca_report("LRU α-scan end state (block[-1] last timestep)", H, N)
    if n_to_pfc_a:
        H, N = build(n_to_pfc_a)
        pca_report("PFC compare h_α_pfc (token 0 post self-attn)", H, N)
    if n_to_pfc_b:
        H, N = build(n_to_pfc_b)
        pca_report("PFC compare h_β_pfc (token 1, indexed by N_β)", H, N)
    if n_to_compare_in:
        H, N = build(n_to_compare_in)
        pca_report("compare_in = h_α_pfc − h_β_pfc (indexed by N_α)", H, N)

    # === Write-head raw pre-quantize per-cell analysis ===
    if n_to_write_raw:
        print(f"\n{'='*72}\n=== Write-head raw pre-quantize (5 cells)\n{'='*72}")
        # Per-cell mean/std + corr with N
        Ns = sorted(n_to_write_raw.keys())
        all_n, all_w = [], []
        for n in Ns:
            for v in n_to_write_raw[n]:
                all_n.append(n); all_w.append(v)
        all_n = np.array(all_n)  # (M,)
        all_w = np.array(all_w)  # (M, W=5)
        W = all_w.shape[1]
        print(f"  total writes: {len(all_n)}, N range [{all_n.min()},{all_n.max()}], cells={W}")
        for c in range(W):
            wc = all_w[:, c]
            r_n, _ = spearmanr(wc, all_n) if wc.std() > 1e-9 else (float('nan'), None)
            r_logn, _ = spearmanr(wc, np.log(all_n + 1e-3)) if wc.std() > 1e-9 else (float('nan'), None)
            mean = wc.mean()
            std = wc.std()
            qmin, q25, q50, q75, qmax = np.quantile(wc, [0, 0.25, 0.5, 0.75, 1.0])
            print(f"  cell{c}: mean={mean:+.3f} std={std:.3f}  Q[0,25,50,75,100]=[{qmin:+.2f},{q25:+.2f},{q50:+.2f},{q75:+.2f},{qmax:+.2f}]  ρ(N)={r_n:+.3f}  ρ(logN)={r_logn:+.3f}")

        print(f"\n  Cell-cell Pearson on raw pre-quantize (W×W):")
        print("       " + "  ".join(f"c{c}" for c in range(W)))
        for i in range(W):
            row = []
            for j in range(W):
                if all_w[:, i].std() < 1e-9 or all_w[:, j].std() < 1e-9:
                    row.append("nan")
                else:
                    r, _ = pearsonr(all_w[:, i], all_w[:, j])
                    row.append(f"{r:+.2f}")
            print(f"  c{i}  " + "  ".join(f"{r:>5}" for r in row))


if __name__ == "__main__":
    main()
