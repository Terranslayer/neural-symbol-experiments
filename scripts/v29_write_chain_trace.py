# DIAG_FOR: v29;v29.1;v29.2;v29.3;v29.4;v29.5 | ANSWERS: write_chain | INPUTS: <ckpt>
"""V29 write head chain trace: I → R&F → spike_count → normalize → clamp → quantize.

After confirming sigmoid attention recovered N-info to I_complex (N-expl 0.94),
trace through the write head sub-chain to find where info dies on the way to
the final scratch output.

Layers:
  L7  I_complex (Re, Im) — input to R&F
  L8  z_re trajectory at each t ∈ [0, T_write-1]
  L9  spike_count_raw per neuron per N (hard count)
  L10 normalized = spike_count / max_count(ω)
  L11 clamped = clamp(normalized, 0, 1)
  L12 quantize round(*2).long → scratch_int
  L13 scratch_q final output

For each layer compute:
  - per-N mean value
  - per-N variance
  - N-explained variance ratio
  - distribution (min/max/quartiles)

Report which step kills the N signal.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def per_neuron_n_expl(mat_NxC, Ns_arr):
    """Compute per-column (per-neuron) N-explained variance ratio.

    mat_NxC: (samples, n_neurons) values per sample per neuron.
    Returns array (n_neurons,) of N-expl ratios.
    """
    ratios = []
    mu_n = Ns_arr.mean()
    centered_N = Ns_arr - mu_n
    var_N = centered_N.var()
    if var_N < 1e-9:
        return np.full(mat_NxC.shape[1], np.nan)
    for c in range(mat_NxC.shape[1]):
        col = mat_NxC[:, c]
        var_total = col.var()
        if var_total < 1e-12:
            ratios.append(0.0)
            continue
        col_centered = col - col.mean()
        slope = (col_centered @ centered_N) / (centered_N @ centered_N)
        var_expl = (slope ** 2) * var_N
        ratios.append(min(var_expl / var_total, 1.0))
    return np.array(ratios)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
    )
    device = next(agent.parameters()).device
    v29 = agent.v29
    wh = v29.write_head
    n_cells = wh.n_neurons
    T_write = wh.t_write

    print(f"=" * 78)
    print(f"V29 WRITE CHAIN TRACE — I_complex → R&F → spike_count → normalize → quantize")
    print(f"=" * 78)
    print(f"\nT_write = {T_write}")
    print(f"ω_init (n={n_cells}): {wh.omega_init.cpu().numpy().tolist()}")
    print(f"ω_now: {wh.omega.detach().cpu().numpy().tolist()}")
    print(f"b_init: {wh.b_init.cpu().numpy().tolist()}")
    print(f"b_now: {wh.b.detach().cpu().numpy().tolist()}")
    decay_now = np.exp(wh.b.detach().cpu().numpy())
    print(f"decay |λ|: {decay_now.tolist()}")
    print(f"threshold = {wh.threshold}")

    max_count_init = T_write * np.abs(wh.omega_init.cpu().numpy()) / (2 * math.pi)
    max_count_now = T_write * np.abs(wh.omega.detach().cpu().numpy()) / (2 * math.pi)
    max_count_clamped = np.clip(max_count_now, 0.1, None)
    print(f"max_count(ω_init) = T_write·|ω|/(2π): {max_count_init.tolist()}")
    print(f"max_count(ω_now): {max_count_now.tolist()}")
    print(f"max_count_clamped (≥0.1, used in normalize): {max_count_clamped.tolist()}")
    print(f"  ⚠️  Formula assumes 1 spike per cycle; for large |I| with soft reset")
    print(f"     R&F can spike at every timestep, so actual max = T_write = {T_write}, not formula.")

    # ------------------------------------------------------------
    # Run batches and trace
    # ------------------------------------------------------------
    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.successor_only_positive = False
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    agent.train(False)

    all_Ns = []
    all_I_re = []
    all_I_im = []
    all_z_re_traj = []  # (samples, T_write, n_cells)
    all_spike_count = []  # (samples, n_cells)
    all_normalized = []   # (samples, n_cells)
    all_clamped = []
    all_scratch_int = []
    all_scratch_q = []

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        alpha_signal = inputs[:, : agent.scene_cfg.L, 0:1].to(device)
        with torch.no_grad():
            tokens, mask, _, _ = v29.encode_to_events(alpha_signal)
            a_pfc = v29.pfc(tokens, mask)
            u = v29.alpha_query_extract(a_pfc, mask)  # (B, 5, d)
            B, N5, d = u.shape
            # V29.5: drive_proj → LN → per-cell scale.  Older ckpts (V29.1-.4)
            # don't have drive_ln/drive_scale — fall through to raw projection.
            I_raw = wh.drive_proj(u.reshape(B * N5, d))  # (B*N5, 2)
            if hasattr(wh, "drive_ln") and wh.drive_ln is not None:
                I_normalized = wh.drive_ln(I_raw)
                I_normalized = I_normalized.reshape(B, N5, 2)
                I_re_t = I_normalized[..., 0] * wh.drive_scale
                I_im_t = I_normalized[..., 1] * wh.drive_scale
            else:
                proj_out = I_raw.reshape(B, N5, 2)
                I_re_t = proj_out[..., 0]
                I_im_t = proj_out[..., 1]
            decay = torch.exp(wh.b)
            cos_w = torch.cos(wh.omega)
            sin_w = torch.sin(wh.omega)
            z_re = torch.zeros(B, N5, device=device, dtype=u.dtype)
            z_im = torch.zeros(B, N5, device=device, dtype=u.dtype)
            z_re_traj = []
            for t in range(T_write):
                new_re = decay * (z_re * cos_w - z_im * sin_w) + I_re_t
                new_im = decay * (z_re * sin_w + z_im * cos_w) + I_im_t
                z_re_traj.append(new_re)
                spike_t = (new_re > wh.threshold).to(u.dtype)
                if wh.soft_reset:
                    z_re = new_re - spike_t * wh.threshold
                    z_im = new_im
                else:
                    z_re = new_re * (1 - spike_t)
                    z_im = new_im * (1 - spike_t)
            z_re_traj = torch.stack(z_re_traj, dim=1)  # (B, T_write, 5)
            # Hard spike count (forward semantics)
            spike_count_raw = (z_re_traj > wh.threshold).float().sum(dim=1)  # (B, 5)
            # Normalize with current formula (V29.5: max_count = T_write scalar)
            try:
                # V29.5: scalar T_write
                normalized = spike_count_raw / float(T_write)
            except Exception:
                # V29.4 and earlier: per-neuron formula
                max_c = (T_write * wh.omega.abs() / (2 * math.pi)).clamp(min=0.1)
                normalized = spike_count_raw / max_c
            clamped = normalized.clamp(0, 1)
            scratch_int = (clamped * 2).round().long().clamp(0, 2)
            scratch_q = scratch_int.float() * 0.5

        all_I_re.append(I_re_t.cpu().numpy())
        all_I_im.append(I_im_t.cpu().numpy())
        all_z_re_traj.append(z_re_traj.cpu().numpy())
        all_spike_count.append(spike_count_raw.cpu().numpy())
        all_normalized.append(normalized.cpu().numpy())
        all_clamped.append(clamped.cpu().numpy())
        all_scratch_int.append(scratch_int.cpu().numpy())
        all_scratch_q.append(scratch_q.cpu().numpy())
        for m in metas:
            all_Ns.append(m["N_alpha"])

    Ns_arr = np.array(all_Ns, dtype=float)
    I_re = np.concatenate(all_I_re)
    I_im = np.concatenate(all_I_im)
    z_re_traj = np.concatenate(all_z_re_traj)  # (samples, T_write, 5)
    spike_count = np.concatenate(all_spike_count)
    normalized = np.concatenate(all_normalized)
    clamped = np.concatenate(all_clamped)
    scratch_int = np.concatenate(all_scratch_int)
    scratch_q = np.concatenate(all_scratch_q)

    n_samples = len(Ns_arr)
    print(f"\nTotal samples = {n_samples}, N range = [{int(Ns_arr.min())}, {int(Ns_arr.max())}]")

    # ------------------------------------------------------------
    # L7. I_complex per cell
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L7] I_complex per cell — input to R&F")
    print(f"{'=' * 78}")
    print(f"  cell    DC_re      DC_im      σ_re       σ_im       N-expl(re)  N-expl(im)  ρ_spear(re,N)")
    n_expl_re = per_neuron_n_expl(I_re, Ns_arr)
    n_expl_im = per_neuron_n_expl(I_im, Ns_arr)
    for c in range(n_cells):
        rho = spearmanr(I_re[:, c], Ns_arr)[0]
        print(f"  {c}      {I_re[:,c].mean():+8.3f}  {I_im[:,c].mean():+8.3f}  "
              f"{I_re[:,c].std():.3f}     {I_im[:,c].std():.3f}     "
              f"{n_expl_re[c]:.3f}      {n_expl_im[c]:.3f}      {rho:+.3f}")

    # ------------------------------------------------------------
    # L8. z_re trajectory at each timestep
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L8] z_re at each timestep (per-cell, per-t)")
    print(f"{'=' * 78}")
    for t in range(T_write):
        z_t = z_re_traj[:, t, :]  # (samples, 5)
        n_expl_t = per_neuron_n_expl(z_t, Ns_arr)
        means = [f"{z_t[:,c].mean():+7.2f}" for c in range(n_cells)]
        stds = [f"{z_t[:,c].std():5.2f}" for c in range(n_cells)]
        n_expl = [f"{n_expl_t[c]:.3f}" for c in range(n_cells)]
        above_thr = [(z_t[:,c] > wh.threshold).mean() for c in range(n_cells)]
        print(f"  t={t}  mean:[{', '.join(means)}]  std:[{', '.join(stds)}]  "
              f"N-expl:[{', '.join(n_expl)}]  P(>θ):[{', '.join(f'{p:.2f}' for p in above_thr)}]")

    # ------------------------------------------------------------
    # L9. spike count raw (hard count over T_write)
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L9] Raw spike_count per cell (hard count over T_write={T_write})")
    print(f"{'=' * 78}")
    n_expl_sc = per_neuron_n_expl(spike_count, Ns_arr)
    print(f"  cell  DC      σ       min  max   N-expl   ρ_spear   small_N_mean  large_N_mean")
    for c in range(n_cells):
        col = spike_count[:, c]
        small_mask = Ns_arr <= 5
        large_mask = Ns_arr >= 25
        small_mean = col[small_mask].mean() if small_mask.any() else float('nan')
        large_mean = col[large_mask].mean() if large_mask.any() else float('nan')
        rho = spearmanr(col, Ns_arr)[0]
        print(f"  {c}    {col.mean():.3f}   {col.std():.3f}   {col.min():.0f}   {col.max():.0f}   "
              f"{n_expl_sc[c]:.3f}    {rho:+.3f}     {small_mean:.3f}        {large_mean:.3f}")

    # Distribution histogram
    print(f"\n  spike_count distribution per cell:")
    for c in range(n_cells):
        bins = np.bincount(spike_count[:, c].astype(int), minlength=T_write + 1)
        bin_pct = bins / bins.sum() * 100
        bin_strs = [f"{i}:{bin_pct[i]:.1f}%" for i in range(min(T_write + 1, len(bin_pct)))]
        print(f"  cell {c}: [{', '.join(bin_strs)}]")

    # ------------------------------------------------------------
    # L10. normalized = spike_count / max_count(ω) (clamped formula)
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L10] Normalized = spike_count / max_count(ω)  (formula in v29.py)")
    print(f"{'=' * 78}")
    n_expl_norm = per_neuron_n_expl(normalized, Ns_arr)
    print(f"  cell  DC      σ       min     max     >1 frac   N-expl")
    for c in range(n_cells):
        col = normalized[:, c]
        over1 = (col > 1.0).mean()
        print(f"  {c}    {col.mean():.3f}   {col.std():.3f}   {col.min():.3f}   {col.max():.3f}   "
              f"{over1:.2%}     {n_expl_norm[c]:.3f}")

    # ------------------------------------------------------------
    # L11. Clamped to [0, 1]
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L11] Clamped to [0, 1] — this is where saturation kills N variance")
    print(f"{'=' * 78}")
    n_expl_clip = per_neuron_n_expl(clamped, Ns_arr)
    print(f"  cell  DC      σ       at_zero%   at_one%   middle%   N-expl")
    for c in range(n_cells):
        col = clamped[:, c]
        at_0 = (col < 0.01).mean() * 100
        at_1 = (col > 0.99).mean() * 100
        middle = 100 - at_0 - at_1
        print(f"  {c}    {col.mean():.3f}   {col.std():.3f}   {at_0:5.1f}%   {at_1:5.1f}%   {middle:5.1f}%   "
              f"{n_expl_clip[c]:.3f}")

    # ------------------------------------------------------------
    # L12. Quantize to scratch_int {0, 1, 2}
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"[L12] Quantized scratch_int (3-level: 0, 1, 2)")
    print(f"{'=' * 78}")
    n_expl_int = per_neuron_n_expl(scratch_int.astype(float), Ns_arr)
    print(f"  cell  level_0%   level_1%   level_2%   N-expl   ρ_spear")
    for c in range(n_cells):
        col = scratch_int[:, c]
        pct = [(col == k).mean() * 100 for k in [0, 1, 2]]
        rho = spearmanr(col, Ns_arr)[0] if col.std() > 0 else float('nan')
        print(f"  {c}    {pct[0]:5.1f}%    {pct[1]:5.1f}%    {pct[2]:5.1f}%   "
              f"{n_expl_int[c]:.3f}   {rho:+.3f}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"SUMMARY — N-explained variance ratio chain")
    print(f"{'=' * 78}")
    print(f"  Layer                          Mean N-expl (over 5 cells)")
    print(f"  L7  I_complex re:              {n_expl_re.mean():.3f}")
    print(f"  L7  I_complex im:              {n_expl_im.mean():.3f}")
    print(f"  L9  spike_count_raw:           {n_expl_sc.mean():.3f}")
    print(f"  L10 normalized (pre-clamp):    {n_expl_norm.mean():.3f}")
    print(f"  L11 clamped to [0,1]:          {n_expl_clip.mean():.3f}")
    print(f"  L12 quantized scratch_int:     {n_expl_int.mean():.3f}")

    print(f"\n  Saturation check (per L11):")
    avg_at_1 = np.mean([((clamped[:, c] > 0.99).mean() * 100) for c in range(n_cells)])
    avg_at_0 = np.mean([((clamped[:, c] < 0.01).mean() * 100) for c in range(n_cells)])
    print(f"    Avg % saturated at 1.0: {avg_at_1:.1f}%")
    print(f"    Avg % at 0.0:           {avg_at_0:.1f}%")
    if avg_at_1 > 70:
        print(f"    ⚠️  >70% saturated at upper clamp — clamp is the bottleneck.")

    # Alternative normalize: max_count = T_write
    print(f"\n  Alternative normalize using max_count = T_write = {T_write}:")
    alt_norm = spike_count / T_write
    alt_clip = np.clip(alt_norm, 0, 1)
    alt_int = np.round(alt_clip * 2).astype(int).clip(0, 2)
    n_expl_alt_int = per_neuron_n_expl(alt_int.astype(float), Ns_arr)
    print(f"  L12 quantize using T_write formula: N-expl = {n_expl_alt_int.mean():.3f}")
    print(f"  Distribution after alt normalize:")
    for c in range(n_cells):
        col = alt_int[:, c]
        pct = [(col == k).mean() * 100 for k in [0, 1, 2]]
        print(f"    cell {c}: level_0={pct[0]:.1f}%, level_1={pct[1]:.1f}%, level_2={pct[2]:.1f}%")


if __name__ == "__main__":
    main()
