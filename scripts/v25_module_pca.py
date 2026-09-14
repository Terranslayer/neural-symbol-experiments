# DIAG_FOR: v25;v25-v19;v25-v22;v25-v24;v26 | ANSWERS: pca,write_head_raw,cell_decouple,successor_function | INPUTS: <ckpt> --task
"""V25 family module-internal representation analysis (V25-V19/V22/V24/V26).

Probes per-N representation in:
  1. CNN multi-scale α summary
  2. LRU α-scan end state
  3. PFC compare h_α_pfc / h_β_pfc (post self-attn token 0/1)
  4. compare_in (= h_α_pfc − h_β_pfc)
  5. Write-head raw pre-quantize (per-cell, by N) + cell-cell Pearson
  6. PFC-at-write `h_for_write` (the input to write_head, captured pre-write)
  7. Successor function output (V25-V22 only) raw S(sc_α) vs S(sc_β)
  8. Refiner-internal trajectory (V26 only — sample iter 0/1/2 sc per N)

Supports tasks: trio_wide_extended (V25-V19/V22), successor_prediction (V25-V24/V26).
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
from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl


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
    rho1_n = float('nan'); rho1_logn = float('nan'); rho2_n = float('nan'); rho3_n = float('nan')
    if PC1.std() > 1e-9:
        rho1_n, _ = spearmanr(PC1, N_arr)
        rho1_logn, _ = spearmanr(PC1, np.log(N_arr + 1e-3))
    if PC2.std() > 1e-9:
        rho2_n, _ = spearmanr(PC2, N_arr)
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
        gap_mean = gaps.mean(); gap_min = gaps.min()
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
    p.add_argument("--task", choices=["trio_wide_extended", "successor_prediction"], required=True)
    p.add_argument("--k-max", type=int, default=5, help="successor task k_max")
    p.add_argument("--jsonl", type=Path, default=None,
                   help="Path to training run.jsonl for restoring agent_cfg "
                        "(needed for ckpts saved before agent_cfg-in-ckpt fix)")
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    if args.jsonl is not None:
        restore_agent_cfg_from_jsonl(agent, args.jsonl)
    agent.train(False)
    device = next(agent.parameters()).device

    # === Hooks ===
    block_calls = []
    def block_hook(module, inp, out):
        block_calls.append(out[:, -1, :].detach().cpu())
    agent.mamba_blocks[-1].register_forward_hook(block_hook)

    pfc_compare_calls = []
    pfc_compare_module = agent.pfc_compare if agent.pfc_compare is not None else agent.pfc
    def pfc_hook(module, inp, out):
        if out.shape[1] == 2:
            pfc_compare_calls.append(out.detach().cpu())
    pfc_compare_module.register_forward_hook(pfc_hook)

    # write_head raw pre-quantize: SequentialWriteHead wrapper
    from backend.core.mamba_agent import SequentialWriteHead
    write_raw_calls = []
    h_for_write_calls = []
    if isinstance(agent.write_head, SequentialWriteHead):
        orig_fwd = agent.write_head.forward
        def wrapped(h_pfc, qlevels, qrange):
            h_for_write_calls.append(h_pfc.detach().cpu())  # input to write_head
            raw, q = orig_fwd(h_pfc, qlevels, qrange)
            write_raw_calls.append(raw.detach().cpu())
            return raw, q
        agent.write_head.forward = wrapped

    # Successor function output (V25-V22): wrap to capture S output
    succ_fn = getattr(agent, "successor", None)
    succ_in_calls = []; succ_out_calls = []
    if succ_fn is not None:
        orig_succ = succ_fn.forward
        def succ_wrapped(x):
            succ_in_calls.append(x.detach().cpu())
            out = orig_succ(x)
            succ_out_calls.append(out.detach().cpu())
            return out
        succ_fn.forward = succ_wrapped

    # === Scene setup ===
    if args.task == "trio_wide_extended":
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
        cfg.K = scene_cfg.K
    else:
        cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
        cfg.K = 1
    cfg.alpha_distribution = "uniform"
    # Override agent's stored scene_cfg so forward path uses correct K for slicing
    agent.scene_cfg = cfg
    rng = np.random.default_rng(args.seed)

    # === Collectors ===
    n_to_lru_alpha = defaultdict(list)
    n_to_pfc_a = defaultdict(list); n_to_pfc_b = defaultdict(list)
    n_to_compare_in = defaultdict(list)
    n_to_cnn_sum = defaultdict(list)
    n_to_h_for_write = defaultdict(list)
    n_to_write_raw = defaultdict(list)
    succ_pairs = []  # list of (sc_in, sc_out) for V25-V22

    for batch_i in range(args.n_batches):
        block_calls.clear(); pfc_compare_calls.clear()
        write_raw_calls.clear(); h_for_write_calls.clear()
        succ_in_calls.clear(); succ_out_calls.clear()

        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))

        if not block_calls:
            continue
        lru_alpha = block_calls[0].numpy()  # first call = α scan

        # write head: first call = α write. Subsequent = virtual β writes per K block.
        if write_raw_calls:
            wr_a = write_raw_calls[0].numpy()
            hw_a = h_for_write_calls[0].numpy()
            for b, meta in enumerate(metas):
                n_to_write_raw[meta["N_alpha"]].append(wr_a[b])
                n_to_h_for_write[meta["N_alpha"]].append(hw_a[b])

        if pfc_compare_calls:
            pfc0 = pfc_compare_calls[0].numpy()  # (B, 2, d)
            h_a = pfc0[:, 0, :]; h_b = pfc0[:, 1, :]
            for b, meta in enumerate(metas):
                n_a = meta["N_alpha"]
                n_to_pfc_a[n_a].append(h_a[b])
                n_to_compare_in[n_a].append(h_a[b] - h_b[b])
                n_b = meta["beta_counts"][0]
                n_to_pfc_b[n_b].append(h_b[b])

        if agent._last_cnn_alpha_summary is not None:
            cnn = agent._last_cnn_alpha_summary.cpu().numpy()
            for b, meta in enumerate(metas):
                n_to_cnn_sum[meta["N_alpha"]].append(cnn[b])

        for b, meta in enumerate(metas):
            n_to_lru_alpha[meta["N_alpha"]].append(lru_alpha[b])

        # Successor function (V25-V22): pairs in order. Most often called once
        # per α-write + one per K block = K+1 calls. Take first = α path.
        if succ_in_calls and succ_out_calls:
            for s_in, s_out in zip(succ_in_calls, succ_out_calls):
                succ_pairs.append((s_in.numpy(), s_out.numpy()))

    def build(n_to):
        keys = sorted(n_to.keys())
        H = np.array([h for n in keys for h in n_to[n]])
        N = np.array([n for n in keys for h in n_to[n]])
        return H, N

    print(f"\n{'='*72}\n=== V25 module PCA: {args.ckpt.name}\n=== task={args.task}  n_batches={args.n_batches}  bs={args.batch_size}  n_max={args.n_max}\n{'='*72}")

    if n_to_cnn_sum:
        H, N = build(n_to_cnn_sum)
        pca_report("CNN α summary (multi-scale)", H, N)
    if n_to_lru_alpha:
        H, N = build(n_to_lru_alpha)
        pca_report("LRU α-scan end state", H, N)
    if n_to_h_for_write:
        H, N = build(n_to_h_for_write)
        pca_report("h_for_write (write_head input from PFC-at-write)", H, N)
    if n_to_pfc_a:
        H, N = build(n_to_pfc_a)
        pca_report("PFC compare h_α_pfc (token 0, indexed by N_α)", H, N)
    if n_to_pfc_b:
        H, N = build(n_to_pfc_b)
        pca_report("PFC compare h_β_pfc (token 1, indexed by N_β)", H, N)
    if n_to_compare_in:
        H, N = build(n_to_compare_in)
        pca_report("compare_in = h_α_pfc − h_β_pfc (indexed by N_α)", H, N)

    # Write-head raw pre-quantize per-cell
    if n_to_write_raw:
        print(f"\n{'='*72}\n=== Write-head raw pre-quantize (5 cells)\n{'='*72}")
        Ns = sorted(n_to_write_raw.keys())
        all_n, all_w = [], []
        for n in Ns:
            for v in n_to_write_raw[n]:
                all_n.append(n); all_w.append(v)
        all_n = np.array(all_n); all_w = np.array(all_w); W = all_w.shape[1]
        print(f"  total writes: {len(all_n)}, N range [{all_n.min()},{all_n.max()}], cells={W}")
        for c in range(W):
            wc = all_w[:, c]
            r_n = float('nan'); r_logn = float('nan')
            if wc.std() > 1e-9:
                r_n, _ = spearmanr(wc, all_n); r_logn, _ = spearmanr(wc, np.log(all_n + 1e-3))
            qmin, q25, q50, q75, qmax = np.quantile(wc, [0, 0.25, 0.5, 0.75, 1.0])
            print(f"  cell{c}: mean={wc.mean():+.3f} std={wc.std():.3f}  Q[0,25,50,75,100]=[{qmin:+.2f},{q25:+.2f},{q50:+.2f},{q75:+.2f},{qmax:+.2f}]  ρ(N)={r_n:+.3f}  ρ(logN)={r_logn:+.3f}")
        print(f"\n  Cell-cell Pearson:")
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

    # Successor function output analysis (V25-V22)
    if succ_pairs:
        print(f"\n{'='*72}\n=== Successor function S: input vs output\n{'='*72}")
        # Aggregate: residual stats, output collapse check
        sc_ins = np.concatenate([p[0] for p in succ_pairs], axis=0)  # (M, W)
        sc_outs = np.concatenate([p[1] for p in succ_pairs], axis=0)  # (M, W)
        residual = sc_outs - sc_ins
        print(f"  total succ calls aggregated: {sc_ins.shape[0]}")
        print(f"  input  mean per-cell: {sc_ins.mean(0)}")
        print(f"  output mean per-cell: {sc_outs.mean(0)}")
        print(f"  residual (out-in) per-cell mean: {residual.mean(0)}, std: {residual.std(0)}")
        print(f"  output cell-wise std (M): {sc_outs.std(0)}  → if all near 0, S learned to ignore input")


if __name__ == "__main__":
    main()
