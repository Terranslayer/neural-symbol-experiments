# DIAG_FOR: v27 | ANSWERS: pca,layer_trace,write_head_raw,cell_decouple,codebook | INPUTS: <ckpt> --L --n-max
"""V27 full internal-representation diag.

Uses _cot_alpha_stash / _cot_beta_stash from agent forward to access
per-chunk CNN tokens, LRU h, and PFC state evolution.

Reports:
  1. CNN α chunks (per chunk PC1↔N) — 4 chunks × per-chunk PCA
  2. LRU h per chunk (per chunk ρ(N), trajectory across chunks)
  3. PFC state per chunk (chain-of-thought evolution)
  4. Final pfc_state (= write_head input) PCA
  5. Write_head raw output per cell + cell-cell Pearson
  6. Per-cell ρ(N) on raw + on quantized
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
from backend.core.mamba_agent import CoTPFC
from scripts.inspect_checkpoint import load_agent


def per_dim_corr(H, N):
    rs = []
    for i in range(H.shape[1]):
        if H[:, i].std() < 1e-9:
            rs.append(float('nan'))
        else:
            r, _ = spearmanr(H[:, i], N)
            rs.append(r)
    return rs


def pca_brief(name, H, N):
    if H.shape[0] < 2:
        print(f"  {name}: too few samples"); return
    Hc = H - H.mean(0)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = S ** 2 / max(H.shape[0]-1, 1); var /= var.sum()
    PC1 = Hc @ Vt[0]; PC2 = Hc @ Vt[1] if len(Vt)>1 else np.zeros_like(PC1)
    rho1 = float('nan') if PC1.std()<1e-9 else spearmanr(PC1, N)[0]
    rho2 = float('nan') if PC2.std()<1e-9 else spearmanr(PC2, N)[0]
    pc1_per_n = {n: PC1[N == n] for n in np.unique(N)}
    within = float(np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1]))
    means = np.array([pc1_per_n[n].mean() for n in sorted(pc1_per_n.keys())])
    if len(means) > 1:
        gaps = np.abs(np.diff(means))
        direction = 1 if means[-1] > means[0] else -1
        inv = sum(1 for i in range(len(means)-1) if direction*(means[i+1]-means[i]) < 0)
        inv_pct = inv / max(len(means)-1, 1) * 100
    else:
        inv_pct = 0
    print(f"  {name}: PC1 var={var[0]:.3f} ρ={rho1:+.3f}  PC2 ρ={rho2:+.3f}  "
          f"within_std={within:.3f}  inversions={inv_pct:.0f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
    )
    device = next(agent.parameters()).device

    # Build cot_pfc + successor_predict_head if needed
    raw_state = torch.load(args.ckpt, map_location=device)
    if "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    if agent.agent_cfg.cot_pfc_n_chunks > 0 and "cot_pfc.init_state" in raw_state:
        cot = CoTPFC(
            d_model=agent.agent_cfg.d_model,
            n_chunks=agent.agent_cfg.cot_pfc_n_chunks,
            kernels=tuple(agent.agent_cfg.cot_pfc_cnn_kernels),
            cnn_channels_per_scale=agent.agent_cfg.dp_channels_per_scale,
            lru_block=agent.mamba_blocks[0],
            input_dim=agent.agent_cfg.input_dim,
        ).to(device)
        cot_state = {k.replace("cot_pfc.", ""): v for k, v in raw_state.items()
                     if k.startswith("cot_pfc.") and "lru_block" not in k}
        cot.load_state_dict(cot_state, strict=False)
        agent.cot_pfc = cot
        print(f"CoTPFC built (n_chunks={agent.agent_cfg.cot_pfc_n_chunks})")
    if "successor_predict_head.weight" in raw_state:
        succ_w = raw_state["successor_predict_head.weight"]
        head = nn.Linear(succ_w.shape[1], succ_w.shape[0]).to(device)
        head.load_state_dict({"weight": succ_w, "bias": raw_state["successor_predict_head.bias"]})
        agent.successor_predict_head = head

    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1; cfg.alpha_distribution = "uniform"; cfg.successor_only_positive = False
    agent.scene_cfg = cfg
    agent.train(False)

    n_chunks = agent.agent_cfg.cot_pfc_n_chunks
    n_to_cnn_per_chunk = [defaultdict(list) for _ in range(n_chunks)]
    n_to_lru_per_chunk = [defaultdict(list) for _ in range(n_chunks)]
    n_to_pfc_per_chunk = [defaultdict(list) for _ in range(n_chunks)]
    n_to_final_pfc = defaultdict(list)
    n_to_write_raw = defaultdict(list)
    n_to_write_q = defaultdict(list)

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))
            stash = agent._cot_alpha_stash
            cnn_chunks = stash["cnn_chunks"].cpu().numpy()    # (B, n_chunks, d)
            pfc_states = [s.cpu().numpy() for s in stash["pfc_states_per_chunk"]]  # list of (B, d)
            lru_h = [h.cpu().numpy() for h in stash["lru_h_per_chunk"]]            # list of (B, d)
            sc_a_q = agent._scratch_alpha_for_aux.cpu().numpy()       # (B, W)
            sc_a_raw = agent._scratch_alpha_raw.cpu().numpy()         # (B, W)
            final_state = pfc_states[-1]

        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            for c in range(n_chunks):
                n_to_cnn_per_chunk[c][n_a].append(cnn_chunks[b, c])
                n_to_lru_per_chunk[c][n_a].append(lru_h[c][b])
                n_to_pfc_per_chunk[c][n_a].append(pfc_states[c][b])
            n_to_final_pfc[n_a].append(final_state[b])
            n_to_write_raw[n_a].append(sc_a_raw[b])
            n_to_write_q[n_a].append(sc_a_q[b])

    def build(d):
        keys = sorted(d.keys())
        H = np.array([h for n in keys for h in d[n]])
        N = np.array([n for n in keys for h in d[n]])
        return H, N

    print(f"\n{'='*72}\n=== V27 module PCA — chain-of-thought trajectory\n{'='*72}")

    print(f"\n--- CNN per chunk (PC1 vs N_α) ---")
    for c in range(n_chunks):
        H, N = build(n_to_cnn_per_chunk[c])
        pca_brief(f"chunk {c}", H, N)

    print(f"\n--- LRU h per chunk (chain-persistent across chunks) ---")
    for c in range(n_chunks):
        H, N = build(n_to_lru_per_chunk[c])
        pca_brief(f"chunk {c}", H, N)

    print(f"\n--- PFC state per chunk (chain-of-thought evolution) ---")
    for c in range(n_chunks):
        H, N = build(n_to_pfc_per_chunk[c])
        pca_brief(f"chunk {c}", H, N)

    print(f"\n--- LRU per-dim ρ(N) at FINAL chunk (chunk {n_chunks-1}) ---")
    H, N = build(n_to_lru_per_chunk[-1])
    rs = per_dim_corr(H, N)
    active = sum(1 for i in range(H.shape[1]) if H[:, i].std() > 0.01)
    strong = sum(1 for r in rs if abs(r) > 0.4)
    print(f"  active dims (std>0.01): {active}/{H.shape[1]}")
    print(f"  strong-N dims (|ρ|>0.4): {strong}/{H.shape[1]}")
    print(f"  per-dim ρ: {[f'{r:+.2f}' for r in rs]}")

    print(f"\n--- PFC final state per-dim ρ(N) ---")
    H, N = build(n_to_final_pfc)
    rs = per_dim_corr(H, N)
    active = sum(1 for i in range(H.shape[1]) if H[:, i].std() > 0.01)
    strong = sum(1 for r in rs if abs(r) > 0.4)
    print(f"  active dims: {active}/{H.shape[1]}, strong-N dims: {strong}/{H.shape[1]}")
    print(f"  per-dim ρ: {[f'{r:+.2f}' for r in rs]}")

    print(f"\n{'='*72}\n=== Write-head raw + cell decouple\n{'='*72}")
    H, N = build(n_to_write_raw)
    W = H.shape[1]
    print(f"  total writes: {H.shape[0]}, N range [{N.min()},{N.max()}]")
    for c in range(W):
        col = H[:, c]
        rho = float('nan') if col.std()<1e-9 else spearmanr(col, N)[0]
        qmin, q25, q50, q75, qmax = np.quantile(col, [0, 0.25, 0.5, 0.75, 1.0])
        print(f"  cell{c}: mean={col.mean():+.3f} std={col.std():.3f}  "
              f"Q[0,25,50,75,100]=[{qmin:+.2f},{q25:+.2f},{q50:+.2f},{q75:+.2f},{qmax:+.2f}]  "
              f"ρ(N)={rho:+.3f}")

    print(f"\n  Cell-cell Pearson on raw:")
    print("       " + "  ".join(f"c{c}" for c in range(W)))
    for i in range(W):
        row = []
        for j in range(W):
            if H[:, i].std() < 1e-9 or H[:, j].std() < 1e-9:
                row.append("nan")
            else:
                r, _ = pearsonr(H[:, i], H[:, j])
                row.append(f"{r:+.2f}")
        print(f"  c{i}  " + "  ".join(f"{r:>5}" for r in row))


if __name__ == "__main__":
    main()
