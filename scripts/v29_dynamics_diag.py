# DIAG_FOR: v29;v29.1;v29.2;v29.3;v29.4;v29.5 | ANSWERS: write_chain,write_head_raw | INPUTS: <ckpt>
"""V29 dynamics + phase trajectory diag.

Confirms ROOT CAUSE of V29.2 codebook collapse (all (2,2,2,2,2)).

Reports:
  1. drive_proj weight Frobenius norm + per-neuron output stats
  2. u_w (cross-attn output per query) magnitude distribution
  3. I_re, I_im drive distribution per write neuron per N
  4. Per-cell z_re/z_im trajectory over T_write timesteps (for sample N values)
  5. Spike timing histogram (which timestep does each neuron typically fire?)
  6. Normalization sanity: max_count(ω) vs actual count per neuron
  7. Steady-state magnitude prediction |z*| = |I| / |1 - exp(b+iω)| for typical I
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


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

    # ------------------------------------------------------------
    # 1. drive_proj weight stats
    # ------------------------------------------------------------
    print("=" * 72)
    print("=== 1. drive_proj weights ===")
    print("=" * 72)
    w = wh.drive_proj.weight.data  # shape depends on V29.1 (d, n_neurons) or V29.2 (2, d)
    b = wh.drive_proj.bias.data
    print(f"  drive_proj.weight shape: {tuple(w.shape)}")
    print(f"  drive_proj.bias shape:   {tuple(b.shape)}")
    print(f"  weight Frobenius norm:   {w.norm().item():.3f}")
    print(f"  weight per-row L2:       {[f'{w[i].norm().item():.2f}' for i in range(w.shape[0])]}")
    print(f"  bias values:             {b.cpu().numpy().tolist()}")
    print(f"  weight stats: mean={w.mean().item():+.4f}  std={w.std().item():.4f}  "
          f"min={w.min().item():+.3f}  max={w.max().item():+.3f}")

    # Detect V29.1 (Linear(d, n_neurons)) vs V29.2 (Linear(d, 2))
    out_dim = w.shape[0]
    is_complex_drive = (out_dim == 2)
    print(f"\n  Inferred drive mode: {'COMPLEX (V29.2 Linear(d,2))' if is_complex_drive else 'SCALAR (V29.1 Linear(d,n))'}")

    # ------------------------------------------------------------
    # 2-4. Run batches, trace drive + z trajectory
    # ------------------------------------------------------------
    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.successor_only_positive = False
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    agent.train(False)

    n_to_u_mag = defaultdict(list)        # u norm per query
    n_to_I_re = defaultdict(list)         # I_re per cell
    n_to_I_im = defaultdict(list)         # I_im per cell (if complex)
    n_to_z_traj = defaultdict(list)       # z trajectory: (T_write, N) of (re, im)
    n_to_spike_steps = defaultdict(list)  # which timesteps spike per cell

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

            # Drive
            B, N5, d = u.shape
            proj_out = wh.drive_proj(u.reshape(B * N5, d))  # (B*5, out_dim)
            if is_complex_drive:
                proj_out = proj_out.reshape(B, N5, 2)
                I_re = proj_out[..., 0]
                I_im = proj_out[..., 1]
            else:
                # V29.1: Linear(d, n_neurons) — take diagonal
                proj_out = proj_out.reshape(B, N5, N5)
                I_re = torch.diagonal(proj_out, dim1=1, dim2=2)
                I_im = torch.zeros_like(I_re)

            # Simulate R&F write head manually to capture trajectories
            decay = torch.exp(wh.b)
            cos_w = torch.cos(wh.omega)
            sin_w = torch.sin(wh.omega)
            z_re = torch.zeros(B, N5, device=device, dtype=u.dtype)
            z_im = torch.zeros(B, N5, device=device, dtype=u.dtype)
            z_traj = []  # list of (B, N5, 2)
            spike_at_t = []  # list of (B, N5) bool
            for t in range(wh.t_write):
                new_re = decay * (z_re * cos_w - z_im * sin_w) + I_re
                new_im = decay * (z_re * sin_w + z_im * cos_w) + I_im
                z_traj.append(torch.stack([new_re, new_im], dim=-1))  # (B, N5, 2)
                spike_t = (new_re > wh.threshold)
                spike_at_t.append(spike_t)
                if wh.soft_reset:
                    z_re = new_re - spike_t.float() * wh.threshold
                    z_im = new_im
                else:
                    z_re = new_re * (1.0 - spike_t.float())
                    z_im = new_im * (1.0 - spike_t.float())
            z_traj = torch.stack(z_traj, dim=1)  # (B, T_write, N5, 2)
            spikes_per_step = torch.stack(spike_at_t, dim=1).float()  # (B, T_write, N5)

        # Per-batch collection
        u_mag = u.norm(dim=-1).cpu().numpy()  # (B, 5)
        I_re_np = I_re.cpu().numpy()
        I_im_np = I_im.cpu().numpy()
        z_traj_np = z_traj.cpu().numpy()
        spikes_step_np = spikes_per_step.cpu().numpy()

        for b, m in enumerate(metas):
            n_a = m["N_alpha"]
            n_to_u_mag[n_a].append(u_mag[b])
            n_to_I_re[n_a].append(I_re_np[b])
            n_to_I_im[n_a].append(I_im_np[b])
            n_to_z_traj[n_a].append(z_traj_np[b])
            n_to_spike_steps[n_a].append(spikes_step_np[b])

    Ns = sorted(n_to_u_mag.keys())

    # ------------------------------------------------------------
    # 2. u norm distribution (cross-attn output magnitude)
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 2. u (cross-attn) magnitude per query ===")
    print("=" * 72)
    u_mat = np.zeros((len(Ns), 5))
    for i, n in enumerate(Ns):
        u_mat[i] = np.array(n_to_u_mag[n]).mean(axis=0)
    print("  Mean ||u_w||_2 per query, per N (small samples):")
    print("  N\\query   q0      q1      q2      q3      q4")
    for i in [0, 4, 9, 14, 19, 24, 29] if len(Ns) > 29 else list(range(0, len(Ns), max(1, len(Ns)//7))):
        if i >= len(Ns):
            continue
        n = Ns[i]
        print(f"  N={n:3d}    " + "  ".join(f"{u_mat[i,j]:6.3f}" for j in range(5)))
    print(f"\n  ||u||_2 overall: small-N(1-5) mean = {u_mat[:5].mean():.3f}, large-N({Ns[-5]}-{Ns[-1]}) mean = {u_mat[-5:].mean():.3f}")

    # ------------------------------------------------------------
    # 3. I_re, I_im distribution per cell per N
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 3. Drive (I_re, I_im) per cell per N ===")
    print("=" * 72)
    I_re_mat = np.zeros((len(Ns), 5))
    I_im_mat = np.zeros((len(Ns), 5))
    for i, n in enumerate(Ns):
        I_re_mat[i] = np.array(n_to_I_re[n]).mean(axis=0)
        I_im_mat[i] = np.array(n_to_I_im[n]).mean(axis=0)
    print(f"  Mean I_re per cell at sampled N (small / mid / large):")
    sample_idx = [0, 4, 9, 14, 19, 24, len(Ns)-1] if len(Ns) > 14 else list(range(0, len(Ns), max(1, len(Ns)//5)))
    for i in sample_idx:
        if i >= len(Ns):
            continue
        n = Ns[i]
        re_str = "  ".join(f"{I_re_mat[i,j]:+7.3f}" for j in range(5))
        im_str = "  ".join(f"{I_im_mat[i,j]:+7.3f}" for j in range(5)) if is_complex_drive else "—"
        print(f"  N={n:3d}  Re:{re_str}  Im:{im_str}")

    # Overall magnitudes
    print(f"\n  Overall I_re: mean={I_re_mat.mean():+.3f}  std={I_re_mat.std():.3f}  min={I_re_mat.min():+.3f}  max={I_re_mat.max():+.3f}")
    if is_complex_drive:
        print(f"  Overall I_im: mean={I_im_mat.mean():+.3f}  std={I_im_mat.std():.3f}  min={I_im_mat.min():+.3f}  max={I_im_mat.max():+.3f}")
        I_mag = np.sqrt(I_re_mat**2 + I_im_mat**2)
        print(f"  Overall |I|:  mean={I_mag.mean():.3f}  max={I_mag.max():.3f}")

    # ------------------------------------------------------------
    # 4. Per-cell z trajectory for representative N values
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 4. Per-cell z trajectory (sample episodes at N=3, N=15, N=27) ===")
    print("=" * 72)
    for sample_n in [3, 15, 27]:
        if sample_n not in n_to_z_traj:
            sample_n = Ns[min(len(Ns)-1, max(0, sample_n))]
        z_arr = np.array(n_to_z_traj[sample_n])  # (samples, T_write, 5, 2)
        z_mean = z_arr.mean(axis=0)  # (T_write, 5, 2)
        print(f"\n  N = {sample_n}: z trajectory (mean across episodes), threshold={wh.threshold}")
        print(f"  step  cell0_re  cell0_im  cell1_re  cell1_im  cell2_re  cell2_im  cell3_re  cell3_im  cell4_re  cell4_im")
        for t in range(wh.t_write):
            row = []
            for c in range(5):
                row.append(f"{z_mean[t,c,0]:+7.3f}")
                row.append(f"{z_mean[t,c,1]:+7.3f}")
            print(f"  t={t}  " + "  ".join(row))

    # ------------------------------------------------------------
    # 5. Spike timing histogram per cell
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 5. Spike timing per cell (mean spike count at each timestep) ===")
    print("=" * 72)
    # Aggregate across all N
    all_spikes_step = []
    for n in Ns:
        all_spikes_step.append(np.array(n_to_spike_steps[n]))  # (samples, T_write, 5)
    all_spikes = np.concatenate(all_spikes_step, axis=0)  # (total_samples, T_write, 5)
    spike_per_step = all_spikes.mean(axis=0)  # (T_write, 5)
    print(f"  Probability of spike at each step per cell (averaged over all samples):")
    print(f"  step    cell0  cell1  cell2  cell3  cell4")
    for t in range(wh.t_write):
        print(f"  t={t}    " + "  ".join(f"{spike_per_step[t,c]:.3f}" for c in range(5)))
    total_per_cell = all_spikes.sum(axis=1).mean(axis=0)  # (5,)
    print(f"\n  Total spikes per cell (mean over samples): {total_per_cell}")

    # ------------------------------------------------------------
    # 6. Normalization sanity (max_count vs actual)
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 6. Normalization sanity: max_count(ω) vs observed ===")
    print("=" * 72)
    omega = wh.omega.detach().cpu().numpy()
    omega_init = wh.omega_init.cpu().numpy()
    t_write = wh.t_write
    max_count_init = t_write * np.abs(omega_init) / (2 * math.pi)
    max_count_now = t_write * np.abs(omega) / (2 * math.pi)
    print(f"  T_write = {t_write}")
    print(f"  cell  ω_init   ω_now    max_count(ω_init)   max_count(ω_now)   observed_mean   observed_max")
    for c in range(5):
        obs_mean = total_per_cell[c]
        # Per-sample max
        obs_max = all_spikes.sum(axis=1)[:, c].max()
        print(f"  {c}     {omega_init[c]:.3f}    {omega[c]:.3f}    {max_count_init[c]:.3f}              {max_count_now[c]:.3f}              {obs_mean:.3f}            {obs_max:.0f}")

    # ------------------------------------------------------------
    # 7. Steady-state |z*| prediction
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 7. Steady-state |z*| prediction for typical I ===")
    print("=" * 72)
    b_wh = wh.b.detach().cpu().numpy()
    print(f"  Discrete R&F: z* = -I / (1 - exp(b+iω)). For unit I:")
    print(f"  cell  ω      b        decay     1-λ_complex_mag   |z*| if |I|=1   |z*| if |I|=mean")
    I_re_mean_per_cell = I_re_mat.mean(axis=0)
    I_im_mean_per_cell = I_im_mat.mean(axis=0) if is_complex_drive else np.zeros(5)
    for c in range(5):
        lam_re = np.exp(b_wh[c]) * np.cos(omega[c])
        lam_im = np.exp(b_wh[c]) * np.sin(omega[c])
        one_minus_lam_mag = np.sqrt((1 - lam_re)**2 + lam_im**2)
        I_mag_typical = np.sqrt(I_re_mean_per_cell[c]**2 + I_im_mean_per_cell[c]**2)
        z_star_unit = 1.0 / max(one_minus_lam_mag, 1e-6)
        z_star_typical = I_mag_typical / max(one_minus_lam_mag, 1e-6)
        gt_threshold = "YES → spikes" if z_star_typical > wh.threshold else "no"
        print(f"  {c}     {omega[c]:.3f}  {b_wh[c]:+.3f}   {np.exp(b_wh[c]):.3f}     {one_minus_lam_mag:.3f}             {z_star_unit:.3f}            {z_star_typical:.3f}    {gt_threshold}")


if __name__ == "__main__":
    main()
