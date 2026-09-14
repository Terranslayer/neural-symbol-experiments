# DIAG_FOR: v26 | ANSWERS: codebook | INPUTS: <ckpt>
"""V26-correct refiner trajectory diag.

For V24R / V26-correct ckpts (with `--pfc-recurrent --pfc-n-iterations 3`),
the predict_head reads REFINED scratch (after 3 iter), not the initial write.
Module PCA via agent.forward only captures the initial write_head call.
This script invokes the refiner explicitly and captures the per-iter sc evolution:

  iter 0: initial scratch (post first write_head, what module_pca sees)
  iter 1: after refiner step 1 (cnn_scratch + iter_proj + write_head)
  iter 2: after refiner step 2
  iter 3: after refiner step 3 (this is what predict_head reads)

Reports per-iter:
  - distinct codes
  - per-cell raw stats (mean, std, ρ(N))
  - cell-cell Pearson
  - modal codes per N (sample 5 N values)

Also reports refiner internals:
  - cnn_scratch output stats per iter
  - iter_proj output (refined h_pfc) per-dim ρ(N)
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
from backend.core.mamba_agent import SequentialWriteHead, RecurrentScratchRefiner
from scripts.inspect_checkpoint import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", choices=["successor_prediction", "trio_wide_extended"], default="successor_prediction")
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # inspect_checkpoint skips refiner. Build it here from raw state dict.
    raw_state = torch.load(args.ckpt, map_location=device)
    if isinstance(raw_state, dict) and "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    elif isinstance(raw_state, dict) and "model" in raw_state:
        raw_state = raw_state["model"]
    refiner_keys = [k for k in raw_state if k.startswith("recurrent_refiner.")]
    if not refiner_keys:
        print("ERROR: ckpt has no recurrent_refiner.* keys.")
        return
    cnn_w = raw_state["recurrent_refiner.cnn_scratch.weight"]
    scratch_cnn_channels = cnn_w.shape[0]
    scratch_cnn_kernel = cnn_w.shape[2]
    iter_proj_w = raw_state["recurrent_refiner.iter_proj.weight"]
    d_pfc = iter_proj_w.shape[0]
    W = agent.scene_cfg.W
    d_lru = iter_proj_w.shape[1] - d_pfc - scratch_cnn_channels * W
    n_iter = 3
    refiner = RecurrentScratchRefiner(
        W=W, d_pfc=d_pfc, d_lru=d_lru,
        scratch_cnn_kernel=scratch_cnn_kernel,
        scratch_cnn_channels=scratch_cnn_channels,
        n_iter=n_iter,
    ).to(device)
    refiner_state = {k.replace("recurrent_refiner.", ""): v for k, v in raw_state.items() if k.startswith("recurrent_refiner.")}
    refiner.load_state_dict(refiner_state)
    refiner.train(False)
    print(f"Refiner built: n_iter={n_iter}, channels={scratch_cnn_channels}, d_pfc={d_pfc}, d_lru={d_lru}")

    if args.task == "successor_prediction":
        cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
        cfg.K = 1
    else:
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
        cfg.K = scene_cfg.K
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg

    rng = np.random.default_rng(args.seed)

    # Collectors per iter (0 = initial, 1..n_iter = post-refiner-step-i)
    raw_per_iter = defaultdict(lambda: defaultdict(list))   # raw_per_iter[iter][N] = [(W,) raw]
    q_per_iter = defaultdict(lambda: defaultdict(list))     # quantized scratch
    pfc_per_iter = defaultdict(lambda: defaultdict(list))   # iter_proj output (refined h_pfc)

    qlevels = agent.agent_cfg.quantize_levels
    qrange = agent.agent_cfg.quantize_range

    for batch_i in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))
            sc_a_init = agent._scratch_alpha_for_aux
            h_a_pfc = agent._last_h_alpha_pfc_for_refine
            lru_a = agent._last_lru_h_alpha_for_refine
            if sc_a_init is None or h_a_pfc is None or lru_a is None:
                print("WARN: missing stash, skipping batch")
                continue
            for b, meta in enumerate(metas):
                q_per_iter[0][meta["N_alpha"]].append(sc_a_init[b].cpu().numpy())
            sc = sc_a_init.clone()
            pfc = h_a_pfc.clone()
            for it in range(n_iter):
                cnn_in = sc.unsqueeze(1)
                cnn_feat = refiner.cnn_scratch(cnn_in).flatten(1)
                combined = torch.cat([pfc, lru_a, cnn_feat], dim=-1)
                pfc = refiner.iter_proj(combined)
                for b, meta in enumerate(metas):
                    pfc_per_iter[it+1][meta["N_alpha"]].append(pfc[b].cpu().numpy())
                raw, q = agent.write_head(pfc, qlevels, qrange)
                for b, meta in enumerate(metas):
                    n_a = meta["N_alpha"]
                    raw_per_iter[it+1][n_a].append(raw[b].cpu().numpy())
                    q_per_iter[it+1][n_a].append(q[b].cpu().numpy())
                sc = q

    # === Reports per iter ===
    print(f"\n{'='*72}\n=== Refiner trajectory: {args.ckpt.name}\n=== task={args.task}  n_batches={args.n_batches}  bs={args.batch_size}\n{'='*72}")

    for it in range(n_iter + 1):
        if it not in q_per_iter:
            continue
        Ns = sorted(q_per_iter[it].keys())
        if not Ns:
            continue
        all_n, all_q = [], []
        for n in Ns:
            for v in q_per_iter[it][n]:
                all_n.append(n); all_q.append(v)
        all_n = np.array(all_n); all_q = np.array(all_q)
        W = all_q.shape[1]

        # Distinct quantized tuples
        tuples = [tuple(int(round(x * 2)) for x in v) for v in all_q]  # 0/0.5/1 → 0/1/2
        unique = sorted(set(tuples))
        print(f"\n--- iter {it} {'(initial)' if it == 0 else f'(refiner step {it})'} ---")
        print(f"  Distinct quantized tuples: {len(unique)} / {3**W}")
        print(f"  Top-5 tuples by count:")
        from collections import Counter
        c = Counter(tuples)
        for tup, cnt in c.most_common(5):
            print(f"    {tup}: {cnt} ({cnt/len(tuples)*100:.1f}%)")

        # Modal per N (sample 5 N values)
        print(f"  Modal per N (sample {len(Ns)} N values, showing 5 spread):")
        sample_Ns = [Ns[0], Ns[len(Ns)//4], Ns[len(Ns)//2], Ns[3*len(Ns)//4], Ns[-1]]
        for n in sample_Ns:
            tuples_n = [tuple(int(round(x * 2)) for x in v) for v in q_per_iter[it][n]]
            cnt_n = Counter(tuples_n)
            most = cnt_n.most_common(1)[0]
            print(f"    N={n:3d}: modal={most[0]}  freq={most[1]/len(tuples_n):.2f}  distinct={len(set(tuples_n))}")

        # Per-cell raw stats (only for it >= 1, since iter 0 has no raw captured)
        if it >= 1:
            all_n_r, all_r = [], []
            for n in Ns:
                for v in raw_per_iter[it][n]:
                    all_n_r.append(n); all_r.append(v)
            all_n_r = np.array(all_n_r); all_r = np.array(all_r)
            print(f"  Per-cell raw stats:")
            for c_i in range(W):
                rc = all_r[:, c_i]
                if rc.std() < 1e-9:
                    rho = float('nan')
                else:
                    rho, _ = spearmanr(rc, all_n_r)
                print(f"    cell{c_i}: mean={rc.mean():+.3f} std={rc.std():.3f} ρ(N)={rho:+.3f}")
            print(f"  Cell-cell Pearson on raw:")
            for i in range(W):
                row = []
                for j in range(W):
                    if all_r[:, i].std() < 1e-9 or all_r[:, j].std() < 1e-9:
                        row.append("nan")
                    else:
                        r, _ = pearsonr(all_r[:, i], all_r[:, j])
                        row.append(f"{r:+.2f}")
                print(f"    c{i}  " + "  ".join(f"{r:>5}" for r in row))

        # PFC per-dim ρ(N) (only for it >= 1)
        if it >= 1:
            all_n_p, all_p = [], []
            for n in Ns:
                for v in pfc_per_iter[it][n]:
                    all_n_p.append(n); all_p.append(v)
            all_p = np.array(all_p); all_n_p = np.array(all_n_p)
            d = all_p.shape[1]
            rs = []
            for di in range(d):
                if all_p[:, di].std() < 1e-9:
                    rs.append(float('nan'))
                else:
                    r, _ = spearmanr(all_p[:, di], all_n_p)
                    rs.append(r)
            strong = sum(1 for r in rs if abs(r) > 0.4)
            med = sum(1 for r in rs if 0.2 <= abs(r) <= 0.4)
            print(f"  Refined h_pfc (iter_proj output) per-dim ρ(N): strong (>0.4): {strong}/{d}, med (0.2-0.4): {med}/{d}")
            print(f"    All dim ρ(N): {[f'{r:+.2f}' for r in rs]}")


if __name__ == "__main__":
    main()
