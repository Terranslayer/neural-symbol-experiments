"""LRU phase diagnostic: for V18c stage 4, compute per-dim activation curve
over N ∈ [1,90], find dominant period via FFT, compare with eigenvalue phase.

If dim k has eigenvalue phase ≈ 2π/m, expect activation curve to be period-m.
"""
import argparse, sys
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
    p.add_argument("--L", type=int, default=400)
    p.add_argument("--n-max", type=int, default=90)
    p.add_argument("--n-batches", type=int, default=60)
    p.add_argument("--lru-block", type=int, default=1, help="Which LRU block to probe (0 or 1)")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    lru_outs = []
    def hook(m, inp, out):
        lru_outs.append(out.detach())
    agent.mamba_blocks[args.lru_block].register_forward_hook(hook)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.equal_weight = 0.20
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    n_to_states = {n: [] for n in range(1, args.n_max + 1)}
    for _ in range(args.n_batches):
        lru_outs.clear()
        inp, _, ci, metas = sample_training_batch(
            batch_size=64, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            agent(inp.to(device), ci.to(device))
        if lru_outs:
            end_state = lru_outs[0][:, -1, :].cpu().numpy()  # (B, d)
            for b, m in enumerate(metas):
                n_to_states[m["N_alpha"]].append(end_state[b])

    # Per-dim mean activation per N
    d = 16
    mean_per_dim = np.zeros((d, args.n_max))
    for n in range(1, args.n_max + 1):
        if n_to_states[n]:
            arr = np.array(n_to_states[n])
            for di in range(d):
                mean_per_dim[di, n - 1] = arr[:, di].mean()

    with torch.no_grad():
        lam_r, lam_i, decay = agent.mamba_blocks[args.lru_block].get_lambda()
        phases = torch.atan2(lam_i, lam_r).abs().cpu().numpy()

    print(f"=== LRU block {args.lru_block} per-dim activation curve over N ===")
    print(f"  dim  phase(rad)  expected_period  top_period_FFT  freq_purity")
    for di in range(d):
        curve = mean_per_dim[di]
        std = curve.std()
        if std < 1e-3:
            print(f"  {di:2}   {phases[di]:.3f}     constant (std={std:.4f})")
            continue
        fft = np.fft.rfft(curve - curve.mean())
        power = np.abs(fft) ** 2
        peak_k = int(np.argmax(power[1:]) + 1)
        period_fft = args.n_max / peak_k if peak_k > 0 else float("inf")
        expected_period = (2 * np.pi / phases[di]) if phases[di] > 0 else float("inf")
        # purity = peak power / total non-DC power
        purity = power[peak_k] / (power[1:].sum() + 1e-9)
        print(f"  {di:2}   {phases[di]:.3f}    {expected_period:7.2f}        {period_fft:7.2f}        {purity:.2f}  (k={peak_k}, std={std:.3f})")

    print()
    print("Interpretation:")
    print("  If expected_period ≈ top_period_FFT for some dim → LRU learned that period")
    print("  Purity > 0.5 → curve is genuinely periodic at that frequency (not noise)")
    print("  Constant dims → LRU learned to suppress them (not active)")


if __name__ == "__main__":
    main()
