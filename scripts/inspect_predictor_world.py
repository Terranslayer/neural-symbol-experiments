# -*- coding: utf-8 -*-
"""
Sample-inspect what each predictor's "world" looks like.

For each predictor variant in path α (E0 forward, A2 beta, A5 compare, PC top-down),
shows:
  - Input shape & sample values
  - Target shape & sample values (ground truth that supervision pushes toward)
  - The semantic meaning of the prediction

Runs locally without Mamba (uses CNN-only encoder for u[t]; the Mamba-dependent
h-vectors are reported as shape descriptions since we can't actually compute them
without pod access).

Usage:
  python scripts/inspect_predictor_world.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa: E402


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def fmt_arr(x: torch.Tensor, max_elems: int = 10, decimals: int = 3) -> str:
    flat = x.detach().cpu().float().flatten()
    n = flat.numel()
    if n <= max_elems:
        nums = flat.tolist()
    else:
        nums = flat[:max_elems].tolist()
    head = ", ".join(f"{v:+.{decimals}f}" for v in nums)
    tail = f", ... ({n - max_elems} more)" if n > max_elems else ""
    return f"[{head}{tail}]"


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # M4-canonical scene config (trio_wide, complex world).
    cfg = SceneConfig.trio_wide_preset(L=50, complex_world=True)
    cfg.K = 4  # fewer compare blocks for compact display
    cfg.alpha_distribution = "uniform"

    print(f"  Scene config: L={cfg.L}, W={cfg.W}, T_gap={cfg.T_gap}, K={cfg.K}")
    print(f"  complex_world={cfg.complex_world}, near_range={cfg.near_range}")

    # Sample 1 episode.
    inputs, labels, compare_idx, metas = sample_training_batch(
        batch_size=1, n_max_stage=20, config=cfg, rng=rng,
    )
    B, T, C = inputs.shape
    m = metas[0]
    print(f"\n  Episode shape: inputs (B,T,C)={tuple(inputs.shape)}, "
          f"N_alpha={m['N_alpha']}, beta_counts={m['beta_counts']}")
    print(f"  channels: 0=signal, 1=phase_input_valid, 2=phase_post_gap")

    banner("RAW WORLD (inputs[0, :L, :3]) — α scan: what the agent observes pre-write")
    alpha_chunk = inputs[0, :cfg.L, :]
    print(f"  shape: (L={cfg.L}, 3)")
    print(f"  signal channel min/max/mean: "
          f"{alpha_chunk[:, 0].min().item():+.3f} / "
          f"{alpha_chunk[:, 0].max().item():+.3f} / "
          f"{alpha_chunk[:, 0].mean().item():+.3f}")
    nonzero = (alpha_chunk[:, 0].abs() > 1e-6).sum().item()
    print(f"  nonzero signal timesteps in α: {nonzero}/{cfg.L} (≈ N_alpha + neighbor smear)")
    print(f"  first 30 signal values:")
    print(f"    {fmt_arr(alpha_chunk[:30, 0], max_elems=30)}")

    # Set up a small CNN front-end (random init — same arch as Mamba agent uses)
    # so we can show u[t] for the PC predictor target.
    d_model = 16
    cnn = nn.Conv1d(3, d_model, kernel_size=5, bias=True)
    with torch.no_grad():
        xt = inputs.transpose(1, 2)  # (1, 3, T)
        pad = torch.zeros(1, 3, 4)
        xtp = torch.cat([pad, xt], dim=-1)
        u = cnn(xtp).transpose(1, 2)  # (1, T, d_model)

    banner("PC: TopDownPredictor — u[t-1] -> predicted u[t]")
    print(f"  what 'world' is being predicted: CNN latent at next step")
    print(f"  input  : u[t-1]      shape=(B*T, d_model={d_model})")
    print(f"  target : u[t]        shape=(B*T, d_model={d_model})")
    print(f"  loss   : implicit — actual loss is on (u - predicted) → mamba sees error")
    print(f"")
    print(f"  Sample u[t] values (CNN output, t=0..3, dim 0..7):")
    for t in range(4):
        print(f"    u[t={t:>2}] = {fmt_arr(u[0, t, :8], max_elems=8)}")
    print(f"  Sample u[t] values at α-scan middle (t=24..27, dim 0..7):")
    for t in range(24, 28):
        print(f"    u[t={t:>2}] = {fmt_arr(u[0, t, :8], max_elems=8)}")
    print(f"")
    print(f"  ★ note: PC predictor is not really 'imagining' content — it's a")
    print(f"  surprise-extractor. Mamba processes (u - predicted), so anything")
    print(f"  the linear-MLP predictor can guess from u[t-1] is suppressed.")

    banner("E0 ForwardPredictor — (h_t, scratch=0) -> next T_pred latents h[t+1..t+T_pred]")
    print(f"  what 'world' is being predicted: Mamba hidden state trajectory (during α scan)")
    print(f"  input  : h_t (B, d={d_model}), scratch_flat (B, W*signal_dim={cfg.W * 1})  — scratch is ZERO during α (not yet written)")
    print(f"  target : h[t+1..t+T_pred]  shape=(B, T_pred, d)")
    print(f"  loss   : MSE(pred, target) — target NOT detached, so encoder also pushed to be predictable")
    print(f"")
    print(f"  Conceptually: 'given my current internal state, what will my own")
    print(f"  hidden state look like 3 steps from now?'")
    print(f"  (We can't actually compute h_t locally — needs Mamba.)")

    banner("A2 BetaPredictor — (h_α, scratch_full) -> first T_β latents of upcoming β scan")
    print(f"  what 'world' is being predicted: future β-segment Mamba latents BEFORE seeing β")
    print(f"  input  : h_α (B, d), scratch_flat (B, W={cfg.W})  — scratch IS filled at this point")
    print(f"  target : h_C[:, :T_β]  shape=(B, T_β, d)  — actual Mamba latents during β scan")
    print(f"  loss   : MSE(pred, target)")
    print(f"")
    print(f"  Conceptually: 'given α (encoded in scratch), what will my hidden")
    print(f"  state trajectory be when I scan an upcoming β scene?'")
    print(f"  Note: β scenes vary across compare blocks (β_counts differ), but")
    print(f"  predictor only sees α-side info → its prediction is the AVERAGE β")
    print(f"  trajectory or the trajectory if β==α.")

    banner("A5 ComparePredictor — (h_α, h_β, scratch) -> compare logits")
    print(f"  what 'world' is being predicted: own compare_head output (self-consistency)")
    print(f"  input  : h_α, h_β, scratch_flat  shape=(B, 2d + W)")
    print(f"  target : compare_head([h_α, h_β]).detach()  shape=(B, 1) for gap_head or (B, 3) for ce")
    print(f"  loss   : MSE(pred_logits, actual_logits.detach())")
    print(f"")
    print(f"  Conceptually: 'can I predict my own decision from scratch alone?'")
    print(f"  Self-consistency loss — pushes scratch to encode info sufficient")
    print(f"  for the same compare decision the model would make end-to-end.")
    print(f"  detach() means compare_head itself is NOT pushed — only encoder/scratch.")

    banner("Sample compare-block β scenes (what A2/A5 see at compare time)")
    for i in range(min(2, cfg.K)):
        rstart, beta_start, compare_step = cfg.compare_block_phases(i)
        print(f"\n  compare block {i}: β_count={m['beta_counts'][i]}")
        print(f"    [rstart={rstart}, beta_start={beta_start}, compare={compare_step}]")
        beta_signal = inputs[0, beta_start:compare_step, 0]
        nonzero_beta = (beta_signal.abs() > 1e-6).sum().item()
        print(f"    β signal: nonzero={nonzero_beta}/{len(beta_signal)} timesteps")
        if len(beta_signal) > 30:
            print(f"    first 30 β values: {fmt_arr(beta_signal[:30], max_elems=30)}")
        else:
            print(f"    β values: {fmt_arr(beta_signal, max_elems=len(beta_signal))}")

    banner("SUMMARY: what 'world' each predictor models")
    print("""
  Predictor       | Input                | Target                    | Layer    | Imagines?
  ----------------|----------------------|---------------------------|----------|----------
  PC TopDown      | u[t-1] (CNN out)     | u[t] (CNN out)            | pre-Mamba| no — surprise extractor
  E0 Forward      | h_t (Mamba) + 0      | h[t+1..t+Tp] (Mamba)      | post-CNN | self-state forward sim
  A2 Beta         | h_α + scratch        | h_C[:Tβ] of upcoming β    | post-CNN | mean-β / α-as-β imagination
  A5 Compare      | h_α + h_β + scratch  | compare_head(detached)    | top-level| decision self-consistency

  KEY OBSERVATIONS:
  - None of these predictors model the RAW WORLD (the input float stream).
    All targets are internal latents (or output logits), not the actual
    spatial scenes the agent observes.
  - "World" here is a misnomer — these are SELF-MODELS of the agent's own
    encoder/state pipeline.
  - PC operates earliest (CNN level); A5 latest (logits). Forward+Beta in
    middle (Mamba hidden state).
  - For genuine extrapolation, predictors would need to imagine NEW input
    contents (e.g. "α has 25 items; if β had 26, what would it look like?")
    — none of them do that. This may explain why all 4 path α variants
    fail near_extrap.
""")


if __name__ == "__main__":
    main()
