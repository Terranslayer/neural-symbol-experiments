# DIAG_FOR: v19 | ANSWERS: gate_selectivity | INPUTS: <ckpt>
"""V19 Diag 2: gate per-spike-strength distribution.

Detect spike centers from signal channel (>0.9 ⇒ peak), then bin every
α-segment position into:
  peak     : within ±2 of any spike center (5-cell band)
  neighbor : within (±2, ±5] of any spike center (penumbra)
  background: rest

Report mean / median / std / 10-90 percentile of `gate` per band.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--n-alpha", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--oracle", action="store_true", help="Enable use_oracle_spike_gate at load time")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
        use_oracle_spike_gate=args.oracle,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    if getattr(agent, "gate_sharpness", None) is None:
        print("not a V19 ckpt"); return
    sharp = agent.gate_sharpness.detach().cpu().numpy()
    print(f"=== gate_sharpness = {sharp.item():.3f} ===\n")

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    band_gates = {"peak": [], "neighbor": [], "background": []}
    n_centers_per_episode = []

    for _ in range(args.n_batches):
        inp, _, ci, metas = sample_training_batch(
            batch_size=64, n_max_stage=args.n_alpha, config=cfg, rng=rng,
        )
        idx = [i for i, m in enumerate(metas) if m["N_alpha"] == args.n_alpha]
        if not idx:
            continue
        idx_t = torch.tensor(idx)
        x_alpha = inp[idx_t, : args.L, :].to(device)
        # Oracle: set per-batch alpha centers for the slice we're encoding
        if getattr(agent.agent_cfg, "use_oracle_spike_gate", False):
            agent._current_oracle_centers = [metas[i]["alpha_centers"] for i in idx]
        with torch.no_grad():
            agent._encode_segment_v6(x_alpha)
        score = agent._last_spike_score.cpu().numpy()  # (B', L)
        if getattr(agent.agent_cfg, "use_oracle_spike_gate", False):
            # Build the oracle gate the agent actually used (hard binary)
            gate = np.zeros_like(score)
            for b, ctrs in enumerate(agent._current_oracle_centers):
                for c in ctrs:
                    if 0 <= c < gate.shape[1]:
                        gate[b, c] = 1.0
        else:
            gate = 1.0 / (1.0 + np.exp(-score * sharp))    # (B', L)
        sig = inp[idx_t, : args.L, 0].numpy()          # (B', L)

        for b in range(sig.shape[0]):
            centers = np.where(sig[b] > 0.9)[0]
            n_centers_per_episode.append(len(centers))
            peak_mask = np.zeros(args.L, dtype=bool)
            penumbra_mask = np.zeros(args.L, dtype=bool)
            for c in centers:
                lo, hi = max(0, c - 2), min(args.L, c + 3)
                peak_mask[lo:hi] = True
                lo2, hi2 = max(0, c - 5), min(args.L, c + 6)
                penumbra_mask[lo2:hi2] = True
            penumbra_mask &= ~peak_mask
            background_mask = ~(peak_mask | penumbra_mask)

            band_gates["peak"].extend(gate[b, peak_mask].tolist())
            band_gates["neighbor"].extend(gate[b, penumbra_mask].tolist())
            band_gates["background"].extend(gate[b, background_mask].tolist())

    cnt_arr = np.array(n_centers_per_episode)
    print(f"N_alpha={args.n_alpha} expected centers per episode: {cnt_arr.mean():.2f} (std {cnt_arr.std():.2f}, min {cnt_arr.min()}, max {cnt_arr.max()})")
    print()
    print(f"{'band':<11} {'n':>7} {'mean':>8} {'median':>8} {'std':>8} {'p10':>8} {'p90':>8}")
    for k in ("peak", "neighbor", "background"):
        a = np.array(band_gates[k]) if band_gates[k] else np.zeros(0)
        if a.size == 0:
            print(f"  {k:<11} {0:>7} (empty)")
            continue
        print(f"  {k:<11} {a.size:>7} {a.mean():>8.4f} {np.median(a):>8.4f} {a.std():>8.4f} {np.percentile(a, 10):>8.4f} {np.percentile(a, 90):>8.4f}")

    if band_gates["peak"] and band_gates["background"]:
        ratio_pb = np.mean(band_gates["peak"]) / max(np.mean(band_gates["background"]), 1e-9)
        print()
        print(f"peak/background ratio = {ratio_pb:.2f}x")
        if ratio_pb >= 2.0:
            print("  ✅ gate selectively fires at spike peaks")
        elif ratio_pb >= 1.2:
            print("  ⚠️ weak peak selectivity")
        elif ratio_pb >= 0.8:
            print("  ❓ no real selectivity (gate ≈ uniform)")
        else:
            print("  ❌ REVERSE selectivity (gate suppressed at peaks)")


if __name__ == "__main__":
    main()
