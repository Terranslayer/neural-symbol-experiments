# DIAG_FOR: v26 | ANSWERS: codebook,cell_decouple | INPUTS: <ckpt>
"""V26-correct pipeline-consistent refiner output diag.

Replicates compute_step's V24 branch *verbatim* on a V26-correct ckpt:
  1. Load full agent INCLUDING refiner + successor_predict_head (manual build)
  2. agent.forward(inputs, ci) — same as training
  3. Pull sc_a = _scratch_alpha_for_aux, sc_b_first = _last_scratch_beta_K_q[:, 0]
  4. Run refiner on both (same args as compute_step)
  5. v24_input = torch.cat([sc_a, sc_b_first], dim=-1) — exactly what predict_head reads
  6. Dump v24_input stats: per-cell mean/std, per-cell ρ(N_α) and ρ(N_β),
     distinct tuples, predict_head accuracy on this batch

Also reports parallel "no-refiner" baseline (skip refiner step) to see what
v24_input would look like with initial scratch only.

Both eval mode (no noise) AND train mode (with noise σ=0.1) measured separately.
"""
import argparse, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from backend.core.mamba_agent import RecurrentScratchRefiner, SequentialWriteHead
from scripts.inspect_checkpoint import load_agent


def per_cell_corr(arr, n_arr):
    """arr: (N_samples, W). Returns ρ(N) per cell."""
    W = arr.shape[1]
    rs = []
    for c in range(W):
        if arr[:, c].std() < 1e-9:
            rs.append(float('nan'))
        else:
            r, _ = spearmanr(arr[:, c], n_arr)
            rs.append(r)
    return rs


def cell_pearson(arr):
    W = arr.shape[1]
    M = np.zeros((W, W))
    for i in range(W):
        for j in range(W):
            if arr[:, i].std() < 1e-9 or arr[:, j].std() < 1e-9:
                M[i, j] = float('nan')
            else:
                r, _ = pearsonr(arr[:, i], arr[:, j])
                M[i, j] = r
    return M


def report_block(label, sc, n_arr):
    """sc: (N_samples, W) numpy. Report stats."""
    print(f"\n  --- {label} ---")
    W = sc.shape[1]
    # Distinct tuples (after rounding to {0, 0.5, 1} grid → {0, 1, 2} ints)
    tuples = [tuple(int(round(x * 2)) for x in v) for v in sc]
    c = Counter(tuples)
    print(f"    distinct tuples: {len(set(tuples))} / {3**W}")
    for tup, cnt in c.most_common(3):
        print(f"      {tup}: {cnt} ({cnt/len(tuples)*100:.1f}%)")
    # Per-cell stats
    print(f"    per-cell stats:")
    for ci in range(W):
        col = sc[:, ci]
        if col.std() < 1e-9:
            rho = float('nan')
        else:
            rho, _ = spearmanr(col, n_arr)
        print(f"      cell{ci}: mean={col.mean():+.3f} std={col.std():.4f} ρ(N)={rho:+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()

    # Step 1: load agent (refiner skipped by inspect_checkpoint)
    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    device = next(agent.parameters()).device

    # Step 2: manually build refiner + successor_predict_head, load weights
    raw_state = torch.load(args.ckpt, map_location=device)
    if isinstance(raw_state, dict) and "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    cnn_w = raw_state["recurrent_refiner.cnn_scratch.weight"]
    iter_proj_w = raw_state["recurrent_refiner.iter_proj.weight"]
    scratch_cnn_channels = cnn_w.shape[0]
    scratch_cnn_kernel = cnn_w.shape[2]
    d_pfc = iter_proj_w.shape[0]
    W = agent.scene_cfg.W
    d_lru = iter_proj_w.shape[1] - d_pfc - scratch_cnn_channels * W
    refiner = RecurrentScratchRefiner(
        W=W, d_pfc=d_pfc, d_lru=d_lru,
        scratch_cnn_kernel=scratch_cnn_kernel,
        scratch_cnn_channels=scratch_cnn_channels,
        n_iter=3,
    ).to(device)
    refiner.load_state_dict({k.replace("recurrent_refiner.", ""): v
                             for k, v in raw_state.items()
                             if k.startswith("recurrent_refiner.")})
    refiner.train(False)

    succ_w = raw_state["successor_predict_head.weight"]  # (k_max, 2W)
    k_max = succ_w.shape[0]
    succ_head = nn.Linear(2 * W, k_max).to(device)
    succ_head.load_state_dict({"weight": raw_state["successor_predict_head.weight"],
                                "bias": raw_state["successor_predict_head.bias"]})
    succ_head.train(False)

    # Override scene_cfg for K=1 successor_prediction
    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg

    qlevels = agent.agent_cfg.quantize_levels
    qrange = agent.agent_cfg.quantize_range

    print(f"Refiner: n_iter={refiner.n_iter}, channels={scratch_cnn_channels}")
    print(f"successor_predict_head: Linear({2*W}, {k_max})")

    rng = np.random.default_rng(args.seed)

    for mode_name, train_mode in [("EVAL_MODE (no write_noise)", False),
                                  ("TRAIN_MODE (write_noise_std=0.1)", True)]:
        print(f"\n{'='*72}\n=== Pipeline-consistent refiner trajectory: {mode_name}\n{'='*72}")
        agent.train(train_mode)
        # Refiner / succ_head always eval
        refiner.train(False)
        succ_head.train(False)

        all_sc_a_init = []
        all_sc_b_init = []
        all_sc_a_refined = []
        all_sc_b_refined = []
        all_n_alpha = []
        all_n_beta = []
        all_k_target = []
        all_k_pred_initial = []  # if no refiner
        all_k_pred_refined = []  # actual training path

        for batch_i in range(args.n_batches):
            inputs, _, ci, metas = sample_training_batch(
                batch_size=args.batch_size, n_max_stage=args.n_max,
                config=cfg, rng=rng,
            )
            with torch.no_grad():
                _ = agent(inputs.to(device), ci.to(device))
                sc_a_init = agent._scratch_alpha_for_aux  # (B, W) quantized
                sc_b_q = agent._last_scratch_beta_K_q
                if sc_b_q is None or sc_a_init is None:
                    print("WARN: missing stash, skip batch")
                    continue
                sc_b_init = sc_b_q[:, 0, :]  # (B, W) for K=0 block
                # Refiner inputs (stash)
                h_a_pfc = agent._last_h_alpha_pfc_for_refine
                h_b_pfc = agent._last_h_beta_pfc_for_refine
                lru_a = agent._last_lru_h_alpha_for_refine
                lru_b = agent._last_lru_h_beta_for_refine
                # Run refiner exactly like compute_step
                sc_a_refined = refiner(sc_a_init, h_a_pfc, lru_a,
                                       agent.write_head, agent.scratch_pos_emb,
                                       qlevels, qrange)
                sc_b_refined = refiner(sc_b_init, h_b_pfc, lru_b,
                                       agent.write_head, agent.scratch_pos_emb,
                                       qlevels, qrange)
                # Predict head outputs
                v24_in_initial = torch.cat([sc_a_init, sc_b_init], dim=-1)
                v24_in_refined = torch.cat([sc_a_refined, sc_b_refined], dim=-1)
                k_logits_init = succ_head(v24_in_initial)
                k_logits_ref = succ_head(v24_in_refined)
                k_pred_init = k_logits_init.argmax(dim=-1).cpu().numpy()
                k_pred_ref = k_logits_ref.argmax(dim=-1).cpu().numpy()
                k_targets = np.array([m["beta_counts"][0] - m["N_alpha"] - 1 for m in metas])
                k_targets = np.clip(k_targets, 0, k_max - 1)

                all_sc_a_init.append(sc_a_init.cpu().numpy())
                all_sc_b_init.append(sc_b_init.cpu().numpy())
                all_sc_a_refined.append(sc_a_refined.cpu().numpy())
                all_sc_b_refined.append(sc_b_refined.cpu().numpy())
                all_n_alpha.extend(m["N_alpha"] for m in metas)
                all_n_beta.extend(m["beta_counts"][0] for m in metas)
                all_k_target.extend(k_targets.tolist())
                all_k_pred_initial.extend(k_pred_init.tolist())
                all_k_pred_refined.extend(k_pred_ref.tolist())

        sc_a_init_all = np.concatenate(all_sc_a_init, axis=0)
        sc_b_init_all = np.concatenate(all_sc_b_init, axis=0)
        sc_a_ref_all = np.concatenate(all_sc_a_refined, axis=0)
        sc_b_ref_all = np.concatenate(all_sc_b_refined, axis=0)
        n_alpha_arr = np.array(all_n_alpha)
        n_beta_arr = np.array(all_n_beta)
        k_target_arr = np.array(all_k_target)
        k_pred_init_arr = np.array(all_k_pred_initial)
        k_pred_ref_arr = np.array(all_k_pred_refined)

        report_block("sc_α INITIAL (post agent.forward, pre-refiner) — α path", sc_a_init_all, n_alpha_arr)
        report_block("sc_α REFINED (post 3-iter refiner) — what predict_head reads", sc_a_ref_all, n_alpha_arr)
        report_block("sc_β INITIAL (post agent.forward virtual β path)", sc_b_init_all, n_beta_arr)
        report_block("sc_β REFINED (post 3-iter refiner)", sc_b_ref_all, n_beta_arr)

        # Cell-cell Pearson on refined
        print(f"\n  --- Cell-cell Pearson on sc_α REFINED ---")
        M = cell_pearson(sc_a_ref_all)
        print("       " + "  ".join(f"c{c}" for c in range(W)))
        for i in range(W):
            print(f"    c{i}  " + "  ".join(f"{M[i,j]:+.2f}" if not np.isnan(M[i,j]) else " nan " for j in range(W)))

        # Predict head accuracy
        acc_init = (k_pred_init_arr == k_target_arr).mean()
        acc_ref = (k_pred_ref_arr == k_target_arr).mean()
        # Distribution of predictions
        pred_init_dist = Counter(k_pred_init_arr.tolist())
        pred_ref_dist = Counter(k_pred_ref_arr.tolist())
        target_dist = Counter(k_target_arr.tolist())
        print(f"\n  --- predict_head accuracy ---")
        print(f"    USING INITIAL scratch (no refiner): k_acc = {acc_init:.4f} (chance = {1/k_max:.4f})")
        print(f"    USING REFINED scratch (actual training path): k_acc = {acc_ref:.4f}")
        print(f"    Total samples: {len(k_target_arr)}")
        print(f"    Pred distribution (refined): {dict(sorted(pred_ref_dist.items()))}")
        print(f"    Target distribution: {dict(sorted(target_dist.items()))}")


if __name__ == "__main__":
    main()
