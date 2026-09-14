"""V26 pipeline DIRECT call — uses train_phase1.compute_step directly.

Eliminates replication risk. Loads V26-correct ckpt with full agent (refiner +
successor_predict_head built manually), then literally calls compute_step from
train_phase1.py and reports k_acc.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from backend.core.mamba_agent import RecurrentScratchRefiner
from scripts.inspect_checkpoint import load_agent
from backend.training.train_phase1 import run_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)  # match training bs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    device = next(agent.parameters()).device

    # Manual build refiner + successor head
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
    agent.recurrent_refiner = refiner

    succ_w = raw_state["successor_predict_head.weight"]
    k_max = succ_w.shape[0]
    succ_head = nn.Linear(2 * W, k_max).to(device)
    succ_head.load_state_dict({"weight": raw_state["successor_predict_head.weight"],
                                "bias": raw_state["successor_predict_head.bias"]})
    agent.successor_predict_head = succ_head

    # Override scene_cfg to K=1 successor preset (matches training)
    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg

    print(f"Refiner attached: n_iter={refiner.n_iter}")
    print(f"successor_predict_head attached: Linear({2*W}, {k_max})")

    rng = np.random.default_rng(args.seed)
    for mode_name, train_mode in [("EVAL_MODE", False), ("TRAIN_MODE", True)]:
        agent.train(train_mode)
        print(f"\n=== {mode_name} ===")
        accs = []
        losses = []
        for batch_i in range(args.n_batches):
            batch = sample_training_batch(
                batch_size=args.batch_size, n_max_stage=args.n_max,
                config=cfg, rng=rng,
            )
            with torch.no_grad():
                loss, k_logits, k_targets, info = run_batch(
                    agent, batch, device, loss_type="ce",
                )
            acc = info.get("v24_k_acc", float('nan'))
            accs.append(acc)
            losses.append(float(loss.item()))
        mean_acc = float(np.mean(accs))
        mean_loss = float(np.mean(losses))
        print(f"  v24_k_acc mean over {args.n_batches} batches: {mean_acc:.4f}")
        print(f"  loss mean: {mean_loss:.4f}")
        print(f"  per-batch acc: {[f'{a:.3f}' for a in accs[:10]]}")


if __name__ == "__main__":
    main()
