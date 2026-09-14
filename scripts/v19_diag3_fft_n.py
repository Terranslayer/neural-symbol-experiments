# DIAG_FOR: v19 | ANSWERS: fft_n | INPUTS: <ckpt>
"""V19 Diag 3: FFT-vs-N on LRU end state.

For each N ∈ [1, n_max], sample a batch, push α-segment through agent,
collect LRU end state (last unpooled time step). Per dim, compute FFT
over the N axis to find dominant period.

Note: smoke ckpt was trained only on N ∈ [1, 5]. Going beyond that is
strictly OOD; behavior may collapse or saturate. We still test N up to
the requested range and report what we see — caller decides whether the
result is interpretable.
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
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=50)
    p.add_argument("--n-batches", type=int, default=16, help="batches per N")
    p.add_argument("--block", type=int, default=1, help="LRU block index (0 or 1)")
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

    block = agent.mamba_blocks[args.block]
    with torch.no_grad():
        lam_r, lam_i, decay = block.get_lambda()
    phases = torch.atan2(lam_i, lam_r).abs().detach().cpu().numpy()  # (d_state,)
    radii  = torch.sqrt(lam_r ** 2 + lam_i ** 2).detach().cpu().numpy()
    print(f"=== LRU block {args.block}: r_min={radii.min():.3f} r_max={radii.max():.3f} ===")

    captured = []
    def hook(m, inp, out):
        captured.append(out.detach())
    handle = block.register_forward_hook(hook)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"

    n_to_states = {n: [] for n in range(args.n_min, args.n_max + 1)}
    for n in range(args.n_min, args.n_max + 1):
        # build a batch with N_alpha forced to n
        rng = np.random.default_rng(1000 + n)
        for _ in range(args.n_batches):
            captured.clear()
            inp, _, ci, metas = sample_training_batch(
                batch_size=32, n_max_stage=max(n, 5), config=cfg, rng=rng,
            )
            mask = [i for i, m in enumerate(metas) if m["N_alpha"] == n]
            if not mask:
                continue
            mask_t = torch.tensor(mask)
            x_alpha = inp[mask_t, : args.L, :].to(device)
            if getattr(agent.agent_cfg, "use_oracle_spike_gate", False):
                agent._current_oracle_centers = [metas[i]["alpha_centers"] for i in mask]
            with torch.no_grad():
                agent._encode_segment_v6(x_alpha)
            if captured:
                # captured[0] is LRU output (B, L, d). Take last timestep.
                end = captured[0][:, -1, :].cpu().numpy()
                for row in end:
                    n_to_states[n].append(row)

    handle.remove()

    d = next(iter(n_to_states.values()))[0].shape[0] if any(n_to_states.values()) else 16
    Nrange = args.n_max - args.n_min + 1
    mean_per_dim = np.full((d, Nrange), np.nan)
    coverage = []
    for k, n in enumerate(range(args.n_min, args.n_max + 1)):
        if n_to_states[n]:
            arr = np.array(n_to_states[n])
            mean_per_dim[:, k] = arr.mean(axis=0)
            coverage.append((n, arr.shape[0]))
    print(f"=== coverage (n, n_samples): {coverage[:5]} ... {coverage[-3:]} ===\n")

    print(f"  dim | phase(rad) | exp_period | top_FFT_period | freq_purity | curve_std")
    for di in range(d):
        curve = mean_per_dim[di]
        if np.any(np.isnan(curve)):
            print(f"  {di:3d}  nan dims (insufficient samples)"); continue
        std = curve.std()
        if std < 1e-3:
            print(f"  {di:3d}  {phases[di]:.3f}  constant (std={std:.4f})"); continue
        cf = np.fft.rfft(curve - curve.mean())
        power = np.abs(cf) ** 2
        peak_k = int(np.argmax(power[1:]) + 1)
        period_fft = Nrange / peak_k if peak_k > 0 else float("inf")
        expected_period = (2 * np.pi / phases[di]) if phases[di] > 0 else float("inf")
        purity = power[peak_k] / (power[1:].sum() + 1e-9)
        print(f"  {di:3d}   {phases[di]:.3f}     {expected_period:7.2f}      {period_fft:7.2f}        {purity:.2f}     std={std:.3f}  k={peak_k}")

    print()
    print("Note: V18c stage 4 had top_FFT_peak == k=1 across all 16 dims (purely")
    print("monotonic, no period detection). For V19 we want to see purity > 0.5")
    print("on at least some dims with period matching expected_period.")


if __name__ == "__main__":
    main()
