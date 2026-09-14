# -*- coding: utf-8 -*-
"""
INF-EXTRAP eval — does inference-time use of forward_predictor help on
out-of-distribution N values?

Tests user's hypothesis: "predictor's value is at INFERENCE-time (agent uses
imagined futures to handle unseen N), not as auxiliary training loss".

For each ckpt with a trained forward_predictor:
  1. Run normal forward → get baseline acc on extrap N range.
  2. Run "extended" forward where, at compare time, predictor outputs are
     concatenated to the raw scratch (acting as imagined-future scratch slots
     fed into PFC). Get extended acc.
  3. Compare normal vs extended.

If extended > normal on N>n_train_max → predictor genuinely helps inference.
If similar → predictor is regularizer-only (today's finding).

Usage:
  python scripts/inference_extrap_eval.py CKPT_PATH \
    --n-min 82 --n-max 99 --n-batches 8 --batch-size 64

Note: requires a ckpt trained with --world-pred-t-pred > 0 (otherwise no
forward_predictor → script aborts).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa: E402
from scripts.inspect_checkpoint import load_agent  # noqa: E402


def _build_gap_target(metas, device):
    return torch.tensor(
        [[m["N_alpha"] - nb for nb in m["beta_counts"]] for m in metas],
        dtype=torch.float32, device=device,
    )


def _threshold_acc(logits: torch.Tensor, gap_target: torch.Tensor) -> float:
    """gap_mse4 threshold accuracy: pred>0.5=GREATER, <-0.5=LESS, else=EQUAL."""
    pred = logits[..., 0]
    pred_class = torch.full_like(pred, fill_value=2, dtype=torch.long)
    pred_class = torch.where(pred > 0.5, torch.zeros_like(pred_class), pred_class)
    pred_class = torch.where(pred < -0.5, torch.ones_like(pred_class), pred_class)
    true_class = torch.full_like(gap_target, fill_value=2, dtype=torch.long)
    true_class = torch.where(gap_target > 0, torch.zeros_like(true_class), true_class)
    true_class = torch.where(gap_target < 0, torch.ones_like(true_class), true_class)
    return float((pred_class == true_class).float().mean().item())


def _extended_forward(agent, inputs, compare_idx) -> torch.Tensor:
    """Forward augmented with predictor outputs as 'imagined-future scratch slots'
    fed to PFC alongside actual scratch at compare time.
    """
    B, T, _ = inputs.shape
    cfg = agent.scene_cfg
    L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K
    d = agent.agent_cfg.d_model

    seg_a_end = L + W + T_gap
    seg_A = inputs[:, 0:seg_a_end, :]
    h_A = agent._encode_segment(seg_A)

    scratch_list = []
    for j in range(W):
        h_t = h_A[:, L + j, :]
        if agent.pfc is not None and agent.agent_cfg.pfc_at_write:
            sp_slots = []
            for k in range(W):
                if k < len(scratch_list):
                    sp_slots.append(scratch_list[k])
                else:
                    sp_slots.append(h_t.new_zeros(B, agent.agent_cfg.signal_output_dim))
            sp_w = torch.stack(sp_slots, dim=1)
            kv_w = agent.scratch_proj(sp_w) + agent.scratch_pos_emb.unsqueeze(0)
            seq_w = torch.cat([h_t.unsqueeze(1), kv_w], dim=1)
            seq_w_refined = seq_w
            for _ in range(agent.agent_cfg.pfc_iter):
                seq_w_refined = agent.pfc(seq_w_refined)
            h_for_write = seq_w_refined[:, 0, :]
        else:
            h_for_write = h_t
        raw = agent.write_head(h_for_write)
        if agent.agent_cfg.quantize_levels is not None:
            from backend.core.agent import _quantize_ste
            raw = _quantize_ste(
                raw, agent.agent_cfg.quantize_levels,
                agent.agent_cfg.quantize_range,
            )
        scratch_list.append(raw)

    h_alpha_end = h_A[:, L - 1, :]
    scratch_flat = torch.cat([s for s in scratch_list], dim=-1)
    imagined_latents = agent.forward_predictor(h_alpha_end, scratch_flat)

    compare_logits_list = []
    for i in range(K):
        rstart, beta_start, compare_step = cfg.compare_block_phases(i)

        seg_B = inputs[:, rstart:beta_start, :].clone()
        for j in range(W):
            seg_B[:, j, 0:agent.agent_cfg.signal_output_dim] = scratch_list[j]
        h_B = agent._encode_segment(seg_B)
        h_alpha = h_B[:, -1, :]

        seg_C = inputs[:, beta_start:compare_step + 1, :]
        h_C = agent._encode_segment(seg_C)
        h_beta = h_C[:, -1, :]

        if agent.pfc is not None:
            # Always include imagined_latents as new "imagined-future slots" —
            # this is the WHOLE POINT of extended forward; even with dropRaw
            # we add the imagined slots (they're not the original raw scratch).
            if agent.agent_cfg.pfc_compare_drop_raw_scratch:
                seq_c = torch.cat(
                    [h_alpha.unsqueeze(1), h_beta.unsqueeze(1), imagined_latents],
                    dim=1,
                )
            else:
                sp = torch.stack(scratch_list, dim=1)
                kv_real = agent.scratch_proj(sp) + agent.scratch_pos_emb.unsqueeze(0)
                seq_c = torch.cat(
                    [h_alpha.unsqueeze(1), h_beta.unsqueeze(1), kv_real, imagined_latents],
                    dim=1,
                )
            seq_c_refined = seq_c
            for _ in range(agent.agent_cfg.pfc_iter):
                seq_c_refined = agent.pfc(seq_c_refined)
            h_alpha_pfc = seq_c_refined[:, 0, :]
            h_beta_pfc = seq_c_refined[:, 1, :]
            compare_in = torch.cat([h_alpha_pfc, h_beta_pfc], dim=-1)
        else:
            compare_in = torch.cat([h_alpha, h_beta], dim=-1)

        if agent.gap_head is not None:
            logits = agent.gap_head(compare_in)
        else:
            logits = agent.compare_head(compare_in)
        compare_logits_list.append(logits)

    return torch.stack(compare_logits_list, dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=str)
    p.add_argument("--n-min", type=int, default=82)
    p.add_argument("--n-max", type=int, default=99)
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", choices=["unit", "symmetric"], default="unit")
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    agent, _, _ = load_agent(
        Path(args.ckpt), q=args.q, qrange=args.range, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    if agent.forward_predictor is None:
        print(
            "  ckpt has NO forward_predictor (was trained with "
            "--world-pred-t-pred 0). INF-EXTRAP cannot run on this ckpt."
        )
        return

    agent.scene_cfg.K = 8
    agent.scene_cfg.alpha_distribution = "uniform"
    agent.train(False)  # set inference mode (vs python builtin eval())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scene_cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=args.complex)
    scene_cfg.K = 8
    scene_cfg.alpha_distribution = "uniform"

    rng = np.random.default_rng(args.seed)

    normal_correct = 0
    extended_correct = 0
    total = 0
    print(f"  predictor T_pred={agent.agent_cfg.world_pred_t_pred}")
    print(f"  testing N in [{args.n_min}, {args.n_max}], task=trio_wide")

    for b_idx in range(args.n_batches):
        batch = sample_training_batch(
            batch_size=args.batch_size,
            n_max_stage=args.n_max,
            config=scene_cfg,
            rng=rng,
        )
        inputs, _, compare_idx, metas = batch
        keep = [i for i, m in enumerate(metas) if m["N_alpha"] >= args.n_min]
        if not keep:
            continue
        keep_idx = torch.tensor(keep, dtype=torch.long)
        inputs = inputs[keep_idx].to(device)
        compare_idx = compare_idx[keep_idx].to(device)
        metas_kept = [metas[i] for i in keep]
        gap_target = _build_gap_target(metas_kept, device)

        with torch.no_grad():
            normal_logits, _, _ = agent(inputs, compare_idx)
            extended_logits = _extended_forward(agent, inputs, compare_idx)

        normal_correct += _threshold_acc(normal_logits, gap_target) * gap_target.numel()
        extended_correct += _threshold_acc(extended_logits, gap_target) * gap_target.numel()
        total += gap_target.numel()

    if total == 0:
        print("  no N-in-range samples; reduce --n-min")
        return

    normal_acc = normal_correct / total
    extended_acc = extended_correct / total
    delta = extended_acc - normal_acc

    print(f"\n  normal forward       : acc = {normal_acc:.4f}  (n_compare = {total})")
    print(f"  extended (imagined)  : acc = {extended_acc:.4f}")
    print(f"  delta (ext - normal) : {delta:+.4f}")

    if abs(delta) > 0.05:
        print(f"  ** |delta| > 0.05: predictor at inference time has measurable effect")
    else:
        print(f"  delta small: predictor is regularizer-only (no inference-time benefit)")


if __name__ == "__main__":
    main()
