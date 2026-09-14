# DIAG_FOR: v26 | ANSWERS: codebook,cell_decouple,pca | INPUTS: <ckpt>
"""V26-correct continue (ep100) FULL diag with proper agent_cfg restoration.

After discovering load_agent missed kwta_k/use_oracle_spike_gate, ALL prior
diagnostic outputs are invalid. This script reproduces the actual training
forward path by restoring agent_cfg from run.jsonl.

Reports:
  1. validate_stage acc (sanity check vs training jsonl)
  2. Pipeline-consistent refined scratch state (sc_α / sc_β distinct codes,
     per-cell stats, cell-cell Pearson)
  3. predict_head pred distribution
"""
import argparse, sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig
from backend.core.mamba_agent import RecurrentScratchRefiner
from backend.training.train_phase1 import validate_stage, run_batch, TrainConfig, CurriculumStage
from scripts.inspect_checkpoint import load_agent
from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("jsonl", type=Path, help="path to training run.jsonl")
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=20)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    device = next(agent.parameters()).device

    # CRITICAL FIX: restore agent_cfg from training run.jsonl
    restore_agent_cfg_from_jsonl(agent, args.jsonl)

    # Manual build refiner + successor head
    raw_state = torch.load(args.ckpt, map_location=device)
    if "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    if "recurrent_refiner.cnn_scratch.weight" in raw_state:
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
            n_iter=agent.agent_cfg.pfc_recurrent_n_iter or 3,
        ).to(device)
        refiner.load_state_dict({k.replace("recurrent_refiner.", ""): v
                                 for k, v in raw_state.items()
                                 if k.startswith("recurrent_refiner.")})
        agent.recurrent_refiner = refiner
        print(f"refiner attached: n_iter={refiner.n_iter}")

    if "successor_predict_head.weight" in raw_state:
        succ_w = raw_state["successor_predict_head.weight"]
        in_dim = succ_w.shape[1]
        k_max_det = succ_w.shape[0]
        succ_head = nn.Linear(in_dim, k_max_det).to(device)
        succ_head.load_state_dict({"weight": raw_state["successor_predict_head.weight"],
                                    "bias": raw_state["successor_predict_head.bias"]})
        agent.successor_predict_head = succ_head
        print(f"successor_predict_head attached: Linear({in_dim}, {k_max_det})")

    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg

    # === Test 1: validate_stage in TRAIN MODE (matches training-time validation)
    print(f"\n{'='*72}\n=== Test 1: validate_stage acc (train mode, matches training jsonl)\n{'='*72}")
    agent.train(True)
    train_cfg = TrainConfig(batch_size=64, validation_batches=4)
    stage = CurriculumStage(name="s1", n_max=args.n_max, target_accuracy=0.7, max_epochs=100)
    accs = []
    for trial in range(10):
        rng = np.random.default_rng(10000 + trial)
        m = validate_stage(agent, cfg, train_cfg, stage, device, rng, loss_type="ce")
        accs.append(m["validation_accuracy"])
    print(f"  10 trials val_acc: mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"  per-trial: {[f'{a:.3f}' for a in accs]}")

    # === Test 2: pipeline-consistent refined scratch state (eval mode, no noise)
    print(f"\n{'='*72}\n=== Test 2: refined scratch state (eval mode, larger sample)\n{'='*72}")
    agent.train(False)
    rng = np.random.default_rng(999)
    from backend.core.scene import sample_training_batch
    all_sc_a_init, all_sc_a_ref, all_sc_b_init, all_sc_b_ref = [], [], [], []
    all_n_a, all_n_b = [], []
    all_k_target, all_k_pred = [], []
    qlevels = agent.agent_cfg.quantize_levels
    qrange = agent.agent_cfg.quantize_range
    for batch_i in range(args.n_batches):
        batch = sample_training_batch(
            batch_size=32, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            loss, k_logits, k_targets, info = run_batch(agent, batch, device, loss_type="ce")
        # k_pred + targets are returned by run_batch
        k_pred = k_logits.argmax(dim=-1).cpu().numpy()
        all_k_target.extend(k_targets.cpu().numpy().tolist())
        all_k_pred.extend(k_pred.tolist())
        # Pull stashed scratch (post-refiner is in compute_step, but agent only has initial)
        sc_a_init = agent._scratch_alpha_for_aux.cpu().numpy()
        sc_b_init = agent._last_scratch_beta_K_q[:, 0, :].cpu().numpy()
        all_sc_a_init.append(sc_a_init); all_sc_b_init.append(sc_b_init)
        # Manually run refiner to get refined
        with torch.no_grad():
            sc_a_t = agent._scratch_alpha_for_aux
            sc_b_t = agent._last_scratch_beta_K_q[:, 0, :]
            sc_a_ref = agent.recurrent_refiner(
                sc_a_t, agent._last_h_alpha_pfc_for_refine, agent._last_lru_h_alpha_for_refine,
                agent.write_head, agent.scratch_pos_emb, qlevels, qrange,
            ).cpu().numpy()
            sc_b_ref = agent.recurrent_refiner(
                sc_b_t, agent._last_h_beta_pfc_for_refine, agent._last_lru_h_beta_for_refine,
                agent.write_head, agent.scratch_pos_emb, qlevels, qrange,
            ).cpu().numpy()
        all_sc_a_ref.append(sc_a_ref); all_sc_b_ref.append(sc_b_ref)
        for m in batch[3]:
            all_n_a.append(m["N_alpha"]); all_n_b.append(m["beta_counts"][0])

    sc_a_init = np.concatenate(all_sc_a_init); sc_a_ref = np.concatenate(all_sc_a_ref)
    sc_b_init = np.concatenate(all_sc_b_init); sc_b_ref = np.concatenate(all_sc_b_ref)
    n_a_arr = np.array(all_n_a); n_b_arr = np.array(all_n_b)
    k_target_arr = np.array(all_k_target); k_pred_arr = np.array(all_k_pred)

    def report(name, sc, n_arr):
        from scipy.stats import spearmanr
        tuples = [tuple(int(round(x*2)) for x in v) for v in sc]
        c = Counter(tuples)
        print(f"\n  {name}: distinct={len(set(tuples))}/{3**sc.shape[1]}, top-5:")
        for tup, cnt in c.most_common(5):
            print(f"    {tup}: {cnt} ({cnt/len(tuples)*100:.1f}%)")
        for ci in range(sc.shape[1]):
            col = sc[:, ci]
            rho = float('nan') if col.std()<1e-9 else spearmanr(col, n_arr)[0]
            print(f"    cell{ci}: mean={col.mean():+.3f} std={col.std():.4f} ρ(N)={rho:+.3f}")

    report("sc_α INITIAL", sc_a_init, n_a_arr)
    report("sc_α REFINED", sc_a_ref, n_a_arr)
    report("sc_β INITIAL", sc_b_init, n_b_arr)
    report("sc_β REFINED", sc_b_ref, n_b_arr)

    # Pred distribution
    print(f"\n  predict_head pred dist: {dict(sorted(Counter(k_pred_arr.tolist()).items()))}")
    print(f"  k_target dist: {dict(sorted(Counter(k_target_arr.tolist()).items()))}")
    print(f"  acc on this sample: {(k_pred_arr == k_target_arr).mean():.4f}")


if __name__ == "__main__":
    main()
