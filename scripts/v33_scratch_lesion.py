# -*- coding: utf-8 -*-
# DIAG_FOR: v33 | ANSWERS: scratch-pathway-lesion | INPUTS: <ckpt>
"""V33 SM scratch-pathway lesion.

Tests whether the successor predict_head actually uses scratch_alpha content,
or bypasses it (e.g. via PFC h memory through the substrate state continuity).

Reproduces _v33_forward (SM branch) step-by-step but intercepts scratch_alpha_q
between write_head(h_alpha) and {readback inject, predictor input}. Substitutes
according to mode:

  full         : no intervention (sanity check vs train rollout)
  zero         : scratch_alpha_q := zeros (zero content, kills both readback signal
                 and predictor sc_a input)
  shuffle      : scratch_alpha_q := scratch_alpha_q[perm], with perm a random
                 batch permutation (preserves marginal distribution of scratch
                 values, only breaks alignment with this episode's alpha)
  rand-codes   : scratch_alpha_q := uniform random ∈ alphabet (q=3 unit: {0,0.5,1.0}^W)

Interpretation:
  - If full / zero / shuffle accs are all close → predictor bypasses scratch,
    decodes alpha from h_beta or via substrate state continuity (the readback
    state still carries some alpha info even with zero/wrong scratch via the
    W-step recurrence initialized from alpha_state).
  - If zero/shuffle drop to near chance (1/n_classes) → scratch is the actual
    carrier, predictor truly reads sc_a content.
  - Intermediate drops localize how much of the signal is sc_a-content vs
    substrate-state-continuity.

Only meaningful for ckpts trained with --v33-same-medium (SM branch). Non-SM
ckpts don't have a scratch readback pathway, so lesioning sc_a only affects
the direct predictor input — script still runs but the "readback contribution"
component is moot.

Usage:
  python scripts/v33_scratch_lesion.py --ckpt <path> --L 200 \
    --n-max 30 --n-batches 8 --batch-size 32 \
    --modes full,zero,shuffle,rand-codes
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import sample_training_batch
from scripts.inspect_checkpoint import load_agent


def _quantize_random_codes(B: int, W: int, q_levels: int, q_range: str,
                           device, generator: torch.Generator) -> torch.Tensor:
    """Sample uniform random codes from the alphabet {0, 1/(q-1), ..., 1.0}
    (or symmetric variant). Returns (B, W) float tensor with values exactly on
    the alphabet grid — matches the post-quantize statistics of scratch_alpha_q."""
    idx = torch.randint(0, q_levels, (B, W), device=device, generator=generator)
    if q_range == "symmetric":
        return -1.0 + 2.0 * idx.float() / (q_levels - 1)
    return idx.float() / (q_levels - 1)


def _encode_segment(agent, seg: torch.Tensor) -> torch.Tensor:
    """seg: (B, T, input_dim) → features (B, T, d). Mirrors _v33_forward.encode()."""
    x = seg.transpose(1, 2)
    conv_outs = [conv(x) for conv in agent.dp_convs]
    feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
    return agent.dp_proj(feat)


def _k_target_from_metas(metas, scene_bidir: bool, n_classes: int,
                         device) -> torch.Tensor:
    """Reproduce class mapping from train_phase1.py:639-657."""
    k_raw = [m["beta_counts"][0] - m["N_alpha"] for m in metas]
    if scene_bidir:
        k_max_val = n_classes // 2
        def k_to_class(k):
            if k < 0:
                return max(0, k + k_max_val)
            elif k > 0:
                return min(n_classes - 1, k + k_max_val - 1)
            return 0
        return torch.tensor([k_to_class(k) for k in k_raw], device=device, dtype=torch.long)
    return torch.tensor([max(0, min(n_classes - 1, k - 1)) for k in k_raw],
                        device=device, dtype=torch.long)


def evaluate_mode(agent, scene_cfg, n_max, n_batches, batch_size, mode, seed):
    """Run lesion mode over n_batches and return accuracy."""
    device = next(agent.parameters()).device
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1000)

    same_medium = getattr(agent.agent_cfg, "v33_same_medium", False)
    q_levels = agent.agent_cfg.quantize_levels
    q_range = agent.agent_cfg.quantize_range
    W = scene_cfg.W
    L = scene_cfg.L
    K_compare = scene_cfg.K

    pred_head = agent.successor_predict_head
    n_classes = pred_head.out_features
    scene_bidir = not getattr(scene_cfg, "successor_only_positive", True)

    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            inputs, _, _, metas = sample_training_batch(
                batch_size, n_max, scene_cfg, rng,
            )
            inputs = inputs.to(device)
            B = inputs.shape[0]

            if same_medium:
                inputs = inputs.clone()
                inputs[:, :, 1:3] = 0.0

            alpha_in = inputs[:, :L, :]
            rstart, beta_start, compare_step = scene_cfg.compare_block_phases(0)
            beta_in = inputs[:, beta_start:compare_step, :]

            feat_alpha = _encode_segment(agent, alpha_in)
            h_alpha_seq, alpha_state = agent.v33_substrate(feat_alpha)
            h_alpha_for_write = h_alpha_seq[:, -1, :]

            _, scratch_alpha_q = agent.write_head(
                h_alpha_for_write, q_levels, q_range,
            )

            if mode == "full":
                sc_a = scratch_alpha_q
            elif mode == "zero":
                sc_a = torch.zeros_like(scratch_alpha_q)
            elif mode == "shuffle":
                perm = torch.randperm(B, device=device, generator=gen)
                sc_a = scratch_alpha_q[perm]
            elif mode == "rand-codes":
                sc_a = _quantize_random_codes(B, W, q_levels, q_range, device, gen)
            else:
                raise ValueError(f"unknown mode: {mode}")

            if same_medium:
                zeros_ch = torch.zeros_like(sc_a)
                seg_readback = torch.stack([sc_a, zeros_ch, zeros_ch], dim=-1)
                feat_readback = _encode_segment(agent, seg_readback)
                _, readback_state = agent.v33_substrate(feat_readback, state=alpha_state)
                feat_beta = _encode_segment(agent, beta_in)
                h_beta_seq, _ = agent.v33_substrate(feat_beta, state=readback_state)
            else:
                feat_beta = _encode_segment(agent, beta_in)
                h_beta_seq, _ = agent.v33_substrate(feat_beta, state=alpha_state)

            h_beta_for_predict = h_beta_seq[:, -1, :]
            _, sc_b = agent.write_head(h_beta_for_predict, q_levels, q_range)

            pred_input = torch.cat([sc_a, sc_b], dim=-1)
            k_logits = pred_head(pred_input)
            k_target = _k_target_from_metas(metas, scene_bidir, n_classes, device)

            preds = k_logits.argmax(dim=-1)
            correct += int((preds == k_target).sum().item())
            total += B

    return correct / max(total, 1), n_classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--L", type=int, required=True,
                   help="Episode length (must match training --curriculum-L)")
    p.add_argument("--task", default="successor_prediction")
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", default="full,zero,shuffle,rand-codes",
                   help="Comma-sep lesion modes")
    args = p.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
    )
    scene_cfg.task = args.task
    agent.train(False)

    if agent.v33_substrate is None:
        print("ERROR: ckpt is not a V33 model (agent.v33_substrate is None)")
        return
    if agent.successor_predict_head is None:
        print("ERROR: ckpt has no successor_predict_head (not a V24-task ckpt)")
        return

    same_medium = getattr(agent_cfg, "v33_same_medium", False)
    print(f"Loaded V33 ckpt: d_model={agent_cfg.v33_d_model}, "
          f"same_medium={same_medium}, q={agent_cfg.quantize_levels}/{agent_cfg.quantize_range}")
    if not same_medium:
        print("  WARN: ckpt is non-SM. scratch_alpha → readback path absent; "
              "lesion only affects direct predictor input.")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"\n=== Scratch lesion sweep (n_max={args.n_max}, "
          f"{args.n_batches}×{args.batch_size} eps/mode) ===")

    results = {}
    for mode in modes:
        acc, n_cls = evaluate_mode(
            agent, scene_cfg, args.n_max, args.n_batches, args.batch_size,
            mode, args.seed,
        )
        results[mode] = acc
        print(f"  {mode:<12}: acc = {acc:.3f}   (chance = {1.0/n_cls:.3f})")

    if "full" in results:
        base = results["full"]
        print("\n=== Contribution of scratch-α to predictor ===")
        for mode in modes:
            if mode == "full":
                continue
            drop = base - results[mode]
            print(f"  full − {mode:<12}: {drop:+.3f}")


if __name__ == "__main__":
    main()
