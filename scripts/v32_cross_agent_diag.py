"""V32 cross-agent decode diag.

DIAG_FOR: v32_multigate_ec

For each (speaker, listener) ordered pair from pool, evaluate δ accuracy on
held-out episodes. Strong compositional protocol → cross-pair accuracy high
across all pairs. Idiosyncratic protocol → diagonal pairs high, off-diag low.

Usage:
    python scripts/v32_cross_agent_diag.py --ckpt-dir checkpoints/v32_ec_default \
        --n-eval 500
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.core.ec_agent import ECAgent, ECAgentConfig
from backend.core.scene_ec import ECSceneConfig, sample_ec_batch


def load_agent(ckpt_path: Path, device: str) -> ECAgent:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["agent_cfg"]
    cfg = ECAgentConfig(**cfg_dict)
    agent = ECAgent(cfg).to(device)
    agent.load_state_dict(state["agent_state_dict"])
    agent.train(False)  # inference mode
    return agent


def evaluate_pair(speaker: ECAgent, listener: ECAgent,
                   scene_cfg: ECSceneConfig, n_eval: int, device: str,
                   rng: np.random.Generator) -> float:
    """Speaker encodes signal_A, listener encodes signal_B. Listener predicts
    own-POV δ = N_B - N_A using (listener_scratch_B, speaker_scratch_A).
    Returns accuracy.
    """
    batch = sample_ec_batch(scene_cfg, n_eval, rng)
    sig_A = torch.from_numpy(batch["signal_A"]).to(device)
    sig_B = torch.from_numpy(batch["signal_B"]).to(device)
    target_listener = torch.from_numpy(batch["delta_class_B"]).to(device)
    with torch.no_grad():
        _, sc_A_speaker = speaker.encode_signal(sig_A)
        _, sc_B_listener = listener.encode_signal(sig_B)
        logits = listener.read_scratches_predict(sc_B_listener, sc_A_speaker)
        pred = logits.argmax(dim=-1)
        acc = (pred == target_listener).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--stage", type=int, default=-1)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--n-max-stage", type=int, default=30)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    stage_files = sorted(ckpt_dir.glob("stage*_agent*.pt"))
    if not stage_files:
        raise SystemExit(f"No ckpts in {ckpt_dir}")
    if args.stage == -1:
        max_stage = max(int(p.name.split("_")[0][5:]) for p in stage_files)
    else:
        max_stage = args.stage
    agent_files = sorted(ckpt_dir.glob(f"stage{max_stage}_agent*.pt"))
    agents = [load_agent(p, args.device) for p in agent_files]
    K = len(agents)
    print(f"Loaded {K} agents from stage {max_stage}")

    scene_cfg = ECSceneConfig(N_min=1, N_max=args.n_max_stage, L=args.L)
    rng = np.random.default_rng(0)
    acc_matrix = np.zeros((K, K))
    for sp in range(K):
        for ls in range(K):
            acc = evaluate_pair(
                agents[sp], agents[ls], scene_cfg, args.n_eval, args.device, rng,
            )
            acc_matrix[sp, ls] = acc

    print("\nCross-agent accuracy matrix (rows = speaker, cols = listener):")
    print("         " + "  ".join(f"L{l:d}    " for l in range(K)))
    for sp in range(K):
        print(f"  S{sp:d}: " + "  ".join(f"{acc_matrix[sp, ls]:.3f}" for ls in range(K)))

    diag = np.diag(acc_matrix)
    off_diag = acc_matrix[~np.eye(K, dtype=bool)]
    print(f"\nDiagonal (self-pair) mean: {diag.mean():.3f}")
    print(f"Off-diagonal (cross-pair) mean: {off_diag.mean():.3f}")
    print(f"Ratio (off-diag / diag): {off_diag.mean() / max(diag.mean(), 1e-6):.3f}")
    print("(Ratio ~ 1 -> compositional / shared protocol)")
    print("(Ratio << 1 -> idiosyncratic per-agent codes)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "stage": max_stage,
                "K": K,
                "acc_matrix": acc_matrix.tolist(),
                "diag_mean": float(diag.mean()),
                "off_diag_mean": float(off_diag.mean()),
            }, f, indent=2)


if __name__ == "__main__":
    main()
