"""V26 validate_stage direct call — reproduces training's validation exactly."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig
from backend.core.mamba_agent import RecurrentScratchRefiner
from backend.training.train_phase1 import validate_stage, TrainConfig, CurriculumStage
from scripts.inspect_checkpoint import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)  # match training data_seed default
    p.add_argument("--n-batches", type=int, default=4)  # match train_cfg.validation_batches
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    device = next(agent.parameters()).device

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
            n_iter=3,
        ).to(device)
        refiner.load_state_dict({k.replace("recurrent_refiner.", ""): v
                                 for k, v in raw_state.items()
                                 if k.startswith("recurrent_refiner.")})
        agent.recurrent_refiner = refiner
        print(f"Refiner attached: n_iter=3")
    else:
        print("No refiner in ckpt — skipping refiner attach")

    if "successor_predict_head.weight" in raw_state:
        succ_w = raw_state["successor_predict_head.weight"]
        in_dim = succ_w.shape[1]
        k_max = succ_w.shape[0]
        succ_head = nn.Linear(in_dim, k_max).to(device)
        succ_head.load_state_dict({"weight": raw_state["successor_predict_head.weight"],
                                    "bias": raw_state["successor_predict_head.bias"]})
        agent.successor_predict_head = succ_head
        print(f"successor_predict_head: Linear({in_dim}, {k_max})")
    else:
        print("No successor_predict_head — V25-V19/V22 trio task")
        k_max = args.k_max

    # CRITICAL: restore agent_cfg from training jsonl
    # ckpt path: <root>/checkpoints/<run>/stage1.pt
    # log path:  <root>/logs/<run>/run.jsonl
    jsonl_path = args.ckpt.parent.parent.parent / "logs" / args.ckpt.parent.name / "run.jsonl"
    if jsonl_path.exists():
        from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl
        restore_agent_cfg_from_jsonl(agent, jsonl_path)
    else:
        print(f"WARN: jsonl not found at {jsonl_path}")

    # Detect V29 ckpt: has v29.predict_head.weight (V29Pipeline internal head)
    is_v29 = "v29.predict_head.weight" in raw_state
    if is_v29 or "successor_predict_head.weight" in raw_state:
        cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
        cfg.K = 1
        # CRITICAL: successor_prediction_preset hardcodes successor_only_positive=True.
        # If training used --successor-bidirectional (V27/V29), preserve that from
        # the restored agent.scene_cfg (set via load_agent's auto-restore).
        cfg.successor_only_positive = getattr(agent.scene_cfg, "successor_only_positive", True)
    else:
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
        cfg.K = agent.scene_cfg.K  # use restored K (typically 8)
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    if is_v29:
        print(f"V29 ckpt detected; using successor_prediction_preset(K=1, k_max={args.k_max})")

    train_cfg = TrainConfig(
        batch_size=64,
        validation_batches=args.n_batches,
    )
    stage = CurriculumStage(name="stage1_n30", n_max=args.n_max, target_accuracy=0.7, max_epochs=50)

    # Match training: validate_stage is called from train loop where agent.train(True).
    # validate_stage doesn't toggle this. So during training validation, write_noise IS active.
    agent.train(True)
    print(f"agent.scene_cfg.K = {cfg.K}  agent.training = {agent.training}")
    print(f"agent.agent_cfg.pfc_recurrent_n_iter = {agent.agent_cfg.pfc_recurrent_n_iter}")
    print(f"successor_predict_input_level = {agent.agent_cfg.successor_predict_input_level}")

    # Run validate_stage MANY times with different seeds to see distribution
    accs = []
    losses = []
    for trial in range(10):
        rng = np.random.default_rng(args.seed + trial)
        metrics = validate_stage(agent, cfg, train_cfg, stage, device, rng, loss_type="ce")
        accs.append(metrics["validation_accuracy"])
        losses.append(metrics["validation_loss"])
        print(f"Trial {trial} (seed {args.seed + trial}): val_acc = {metrics['validation_accuracy']:.4f}  val_loss = {metrics['validation_loss']:.4f}")

    print(f"\nMean val_acc over 10 trials: {np.mean(accs):.4f}  std: {np.std(accs):.4f}")
    print(f"Mean val_loss: {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
