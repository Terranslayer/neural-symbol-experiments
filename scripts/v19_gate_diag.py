# DIAG_FOR: v19 | ANSWERS: gate_selectivity | INPUTS: <ckpt> --n-alpha
"""V19 event-gated LRU diagnostic.

Load a V19 ckpt, push a known-N α segment (and a β segment) through the agent,
hook spike_detector → spike_score and compute gate = sigmoid(score * sharpness).

Reports:
  - spike_score statistics (min/max/mean/std) per segment
  - gate statistics
  - alignment: indices where signal channel has a spike vs indices where gate ≈ 1
  - text histogram of gate values

Usage:
  python scripts/v19_gate_diag.py /tmp/v19_smoke_ckpt/smoke_stage1.pt --n-alpha 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def fmt_arr(a, prec=3):
    return f"min={a.min():.{prec}f} max={a.max():.{prec}f} mean={a.mean():.{prec}f} std={a.std():.{prec}f}"


def hist_text(vals, bins=10, lo=0.0, hi=1.0, width=40):
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    mx = max(h.max(), 1)
    out = []
    for i in range(bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        bar = "#" * int(width * h[i] / mx)
        out.append(f"  [{lo_e:.2f}-{hi_e:.2f}]  {h[i]:5d}  {bar}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--n-alpha", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--block", type=int, default=0, help="Which mamba_block to inspect")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    if not getattr(agent.agent_cfg, "lru_event_gated", False):
        print("⚠️ ckpt does NOT have lru_event_gated — this is not a V19 ckpt")
        return
    if getattr(agent, "spike_detector", None) is None or getattr(agent, "gate_sharpness", None) is None:
        print("⚠️ agent has no spike_detector / gate_sharpness (not built)")
        return

    print(f"=== gate_sharpness = {agent.gate_sharpness.item():.3f} ===")
    print(f"=== spike_detector weight: {fmt_arr(agent.spike_detector.weight.detach().cpu().numpy())} ===")
    print(f"=== spike_detector bias:   {agent.spike_detector.bias.item():.4f} ===\n")

    # we need to hook _last_spike_score; just read it after a forward pass
    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    all_score = []
    all_gate = []
    align_hits = []  # per-batch: gate at signal-spike indices

    for b in range(args.n_batches):
        inp, _, ci, metas = sample_training_batch(
            batch_size=64, n_max_stage=args.n_alpha, config=cfg, rng=rng,
        )
        # filter to only N=n_alpha
        idx = [i for i, m in enumerate(metas) if m["N_alpha"] == args.n_alpha]
        if not idx:
            continue
        idx_t = torch.tensor(idx)
        with torch.no_grad():
            agent(inp.to(device), ci.to(device))

        # _last_spike_score is set during _encode_segment_v6 for ALL segments
        # we want the alpha segment specifically. easiest: monkey-patch via re-running encode.
        # alternative: just read whatever's there (last segment = beta). For diagnostic,
        # we re-run α encode directly:
        x_alpha = inp[idx_t, : args.L + 1, :].to(device)  # alpha seg shape (B', L+1, C)
        # actually inp layout: alpha(0..L), gap, beta1, gap, beta2, ...
        # safest: just take the first L tokens as α
        x_alpha = inp[idx_t, : args.L, :].to(device)
        with torch.no_grad():
            agent._encode_segment_v6(x_alpha)
        score = agent._last_spike_score.cpu().numpy()  # (B', L)
        sharp = agent.gate_sharpness.detach().cpu().numpy()
        gate = 1.0 / (1.0 + np.exp(-score * sharp))

        all_score.append(score)
        all_gate.append(gate)

        # alignment: signal channel = inp[..., 0]; spike where signal > 0.5 (signal is in [0,1])
        sig = inp[idx_t, : args.L, 0].cpu().numpy()
        spike_mask = sig > 0.5  # (B', L)
        bg_mask = sig <= 0.1
        if spike_mask.sum() > 0:
            gate_at_spike = gate[spike_mask].mean()
        else:
            gate_at_spike = float("nan")
        if bg_mask.sum() > 0:
            gate_at_bg = gate[bg_mask].mean()
        else:
            gate_at_bg = float("nan")
        align_hits.append((gate_at_spike, gate_at_bg, spike_mask.sum(), bg_mask.sum()))

    if not all_score:
        print(f"no batches with N_alpha == {args.n_alpha}")
        return

    score = np.concatenate(all_score, axis=0)
    gate = np.concatenate(all_gate, axis=0)

    print(f"=== Aggregated over {score.shape[0]} α segments (N={args.n_alpha}, L={args.L}) ===\n")
    print(f"spike_score  : {fmt_arr(score, 4)}")
    print(f"gate         : {fmt_arr(gate, 4)}")
    print()
    print("gate histogram (0..1):")
    print(hist_text(gate.flatten(), bins=10))
    print()
    print("=== alignment (gate at signal-spike vs background) ===")
    print(f"  per-batch (gate_at_spike, gate_at_bg, n_spike, n_bg):")
    for h in align_hits:
        print(f"    {h[0]:.3f}  {h[1]:.3f}    {h[2]:5d}  {h[3]:5d}")
    if align_hits:
        gs = np.array([h[0] for h in align_hits])
        gb = np.array([h[1] for h in align_hits])
        print(f"  mean gate_at_spike = {gs.mean():.3f}")
        print(f"  mean gate_at_bg    = {gb.mean():.3f}")
        ratio = gs.mean() / max(gb.mean(), 1e-6)
        print(f"  ratio (spike/bg)   = {ratio:.2f}x")
        print()
        if ratio > 2.0:
            print("  ✅ gate is selective for signal spikes")
        elif ratio > 1.2:
            print("  ⚠️ weak selectivity — consider --detection-aux-weight")
        else:
            print("  ❌ no selectivity — gate is essentially uniform; need detection aux loss")


if __name__ == "__main__":
    main()
