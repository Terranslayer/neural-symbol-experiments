# DIAG_FOR: v29;v29.1;v29.2;v29.3;v29.4;v29.5 | ANSWERS: write_chain,codebook,cell_decouple | INPUTS: <ckpt>
"""V29 pre-quantize spike_count vs post-quantize scratch_int codebook comparison.

User question (2026-05-12): does spike_count (0..T_write integer, pre-quantize)
contain more information than scratch_int (0/1/2 post-quantize)? If so the
3-level quantize is destroying useful structure.

Analysis (per N=1..n_max):
  1. spike_count distribution per cell (raw integer from R&F, pre-normalize)
  2. normalized = spike_count / T_write
  3. scratch_int = round(normalized * 2).clamp(0,2)  - quantize bottleneck
  4. Distinct count: how many unique spike_count 5-tuples vs scratch_int 5-tuples
  5. Within-bin variance: for samples landing on same scratch_int, what is the
     spike_count spread
  6. N-explained variance per stage of the chain

Self-contained: instantiates V29Pipeline directly (no MambaAgent / no mamba_ssm
needed). Works on Windows + CPU.
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.v29 import V29Pipeline, RFWriteHead, surrogate_spike_count
from backend.core.scene import SceneConfig, sample_training_batch


def patch_write_head_for_capture(wh):
    stash = {}

    def forward_with_capture(u):
        B, Nq, d = u.shape
        N = wh.n_neurons
        I_raw = wh.drive_proj(u.reshape(B * N, d))
        I_normalized = wh.drive_ln(I_raw).reshape(B, N, 2)
        I_re = I_normalized[..., 0] * wh.drive_scale
        I_im = I_normalized[..., 1] * wh.drive_scale
        decay = torch.exp(wh.b)
        cos_w = torch.cos(wh.omega)
        sin_w = torch.sin(wh.omega)
        z_re = torch.zeros(B, N, device=u.device, dtype=u.dtype)
        z_im = torch.zeros(B, N, device=u.device, dtype=u.dtype)
        v_seq = []
        for t in range(wh.t_write):
            new_re = decay * (z_re * cos_w - z_im * sin_w) + I_re
            new_im = decay * (z_re * sin_w + z_im * cos_w) + I_im
            v_seq.append(new_re)
            spike_t = (new_re > wh.threshold).to(u.dtype)
            if wh.soft_reset:
                z_re = new_re - spike_t * wh.threshold
                z_im = new_im
            else:
                z_re = new_re * (1.0 - spike_t)
                z_im = new_im * (1.0 - spike_t)
        v_stack = torch.stack(v_seq, dim=1)
        spike_count = surrogate_spike_count(v_stack, wh.threshold, wh.surrogate_alpha)
        max_count = float(wh.t_write)
        normalized = (spike_count / max_count).clamp(0, 1)
        scratch_int = (normalized * 2).round().long().clamp(0, 2)
        scratch_hard = scratch_int.float() * 0.5
        scratch_q = normalized + (scratch_hard - normalized).detach()
        stash["I_re"] = I_re.detach().cpu().numpy()
        stash["I_im"] = I_im.detach().cpu().numpy()
        stash["spike_count"] = spike_count.detach().cpu().numpy()
        stash["normalized"] = normalized.detach().cpu().numpy()
        stash["scratch_int"] = scratch_int.detach().cpu().numpy()
        stash["scratch_q"] = scratch_q.detach().cpu().numpy()
        return scratch_q, scratch_int

    wh.forward = forward_with_capture
    return stash


def build_pipeline_from_ckpt(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ac = ckpt["agent_cfg"]
    pipeline = V29Pipeline(
        n_encoder_neurons=ac["v29_encoder_n_neurons"],
        n_write_neurons=ac["v29_write_n_neurons"],
        d_id=ac["v29_d_id"],
        d_pe=ac["v29_d_pe"],
        n_pfc_layers=ac["v29_pfc_layers"],
        n_pfc_heads=ac["v29_pfc_heads"],
        n_pfc_passes=ac["v29_pfc_passes"],
        threshold=ac["v29_threshold"],
        surrogate_alpha=ac["v29_surrogate_alpha"],
        omega_enc_min=ac["v29_omega_enc_min"],
        omega_enc_max=ac["v29_omega_enc_max"],
        omega_write_min=ac["v29_omega_write_min"],
        omega_write_max=ac["v29_omega_write_max"],
        b_init_min=ac["v29_b_init_min"],
        b_init_max=ac["v29_b_init_max"],
        time_max_init=ac["v29_time_max_init"],
        firing_rate_lambda=ac["v29_firing_rate_lambda"],
        target_firing_rate=ac["v29_target_firing_rate"],
        soft_reset=ac["v29_soft_reset"],
        t_write=ac["v29_t_write"] if ac["v29_t_write"] > 0 else None,
        max_events=ac["v29_max_events"],
        n_levels=ac["quantize_levels"],
        k_max=ac["successor_predict_k_max"] // 2,
        bidirectional=True,
    )
    sd = ckpt["agent_state_dict"]
    v29_sd = {k[len("v29."):]: v for k, v in sd.items() if k.startswith("v29.")}
    missing, unexpected = pipeline.load_state_dict(v29_sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] unexpected: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    pipeline.train(False)
    return pipeline, ckpt["scene_cfg"], ac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pipeline, scene_dict, ac = build_pipeline_from_ckpt(args.ckpt, device="cpu")
    wh = pipeline.write_head
    stash = patch_write_head_for_capture(wh)
    T_write = wh.t_write
    print(f"V29 ckpt loaded: {args.ckpt}")
    print(f"T_write = {T_write}")
    print(f"  omega_init: {wh.omega_init.cpu().numpy().tolist()}")
    print(f"  omega_now:  {wh.omega.detach().cpu().numpy().tolist()}")
    print(f"  drive_scale: {wh.drive_scale.detach().cpu().numpy().tolist()}")
    print(f"  threshold: {wh.threshold}")
    print()

    cfg = SceneConfig.successor_prediction_preset(
        L=scene_dict["L"], complex_world=True, k_max=ac["successor_predict_k_max"] // 2,
    )
    cfg.K = 1
    cfg.successor_only_positive = False
    cfg.alpha_distribution = "uniform"

    rng = np.random.default_rng(args.seed)
    n_to_spike_count = defaultdict(list)
    n_to_scratch_int = defaultdict(list)
    n_to_normalized = defaultdict(list)
    n_to_I_re = defaultdict(list)

    with torch.no_grad():
        for batch_i in range(args.n_batches):
            inputs, _, ci, metas = sample_training_batch(
                batch_size=args.batch_size, n_max_stage=args.n_max,
                config=cfg, rng=rng,
            )
            L = scene_dict["L"]
            alpha_signal = inputs[:, :L, 0:1]
            rstart, beta_start, compare_step = cfg.compare_block_phases(0)
            beta_signal = inputs[:, beta_start:compare_step, 0:1]
            _ = pipeline(alpha_signal, beta_signal)
            sc = stash["spike_count"]
            si = stash["scratch_int"]
            nz = stash["normalized"]
            ire = stash["I_re"]
            for b, meta in enumerate(metas):
                n_a = meta["N_alpha"]
                n_to_spike_count[n_a].append(sc[b])
                n_to_scratch_int[n_a].append(si[b])
                n_to_normalized[n_a].append(nz[b])
                n_to_I_re[n_a].append(ire[b])

    print("=" * 78)
    print("=== 1. spike_count per N per cell (rounded int)")
    print("=" * 78)
    print(f"{'N':>3} | n_samp | spike_count per cell (mean +- std) | scratch_int (modal)")
    print("-" * 78)
    for n in sorted(n_to_spike_count.keys()):
        sc_arr = np.array(n_to_spike_count[n])
        si_arr = np.array(n_to_scratch_int[n])
        means = sc_arr.mean(0)
        stds = sc_arr.std(0)
        sc_str = "  ".join(f"{m:.1f}+-{s:.1f}" for m, s in zip(means, stds))
        si_tuples = [tuple(row.tolist()) for row in si_arr]
        modal_si = Counter(si_tuples).most_common(1)[0][0]
        print(f"{n:3d} | {len(sc_arr):6d} | {sc_str} | {modal_si}")

    print()
    print("=" * 78)
    print("=== 2. Distinct codebook sizes (5-tuples)")
    print("=" * 78)
    all_sc = np.concatenate([np.array(v) for v in n_to_spike_count.values()])
    all_si = np.concatenate([np.array(v) for v in n_to_scratch_int.values()])
    all_sc_int = np.round(all_sc).astype(int)
    sc_tuples = set(tuple(row.tolist()) for row in all_sc_int)
    si_tuples = set(tuple(row.tolist()) for row in all_si)
    print(f"Pre-quantize spike_count (rounded): {len(sc_tuples)} distinct 5-tuples")
    print(f"Post-quantize scratch_int:           {len(si_tuples)} distinct 5-tuples")
    print(f"Per-cell range: spike_count [{all_sc_int.min()}..{all_sc_int.max()}] vs scratch_int [{all_si.min()}..{all_si.max()}]")
    print(f"Theoretical max: (T_write+1)^5 = {(T_write+1)**5} vs 3^5 = 243")
    info_ratio = math.log((T_write+1)**5) / math.log(3**5)
    print(f"Info ratio per 5-tuple: log({(T_write+1)**5})/log(243) = {info_ratio:.2f}x more")

    print()
    print("=" * 78)
    print("=== 3. Within-bin variance: same scratch_int, separable by spike_count?")
    print("=" * 78)
    bin_groups = defaultdict(list)
    all_N = np.concatenate([[n] * len(v) for n, v in n_to_spike_count.items()])
    for sc, si, n in zip(all_sc_int, all_si, all_N):
        bin_groups[tuple(si.tolist())].append((sc, n))
    print(f"{'scratch_int':<20} | n_samp | N range | spike_count means per cell | N-distinct in bin")
    print("-" * 110)
    sorted_bins = sorted(bin_groups.keys(), key=lambda x: len(bin_groups[x]), reverse=True)
    for si_tup in sorted_bins[:10]:
        group = bin_groups[si_tup]
        scs = np.array([g[0] for g in group])
        ns = np.array([g[1] for g in group])
        sc_means = scs.mean(0)
        sc_str = "  ".join(f"{m:.1f}" for m in sc_means)
        n_unique = len(np.unique(ns))
        print(f"{str(si_tup):<20} | {len(group):6d} | [{ns.min()},{ns.max()}] | {sc_str} | {n_unique} distinct N")

    print()
    print("=" * 78)
    print("=== 4. Per-cell Spearman rho(N) -- pre vs post quantize")
    print("=" * 78)
    print(f"{'cell':>4} | spike_count rho | scratch_int rho | spike_count range | scratch_int range")
    print("-" * 95)
    for c in range(all_sc.shape[1]):
        rho_sc = float("nan") if np.std(all_sc_int[:, c]) < 1e-9 else spearmanr(all_sc_int[:, c], all_N)[0]
        rho_si = float("nan") if np.std(all_si[:, c]) < 1e-9 else spearmanr(all_si[:, c], all_N)[0]
        sc_range = f"[{all_sc_int[:, c].min()},{all_sc_int[:, c].max()}]"
        si_range = f"[{all_si[:, c].min()},{all_si[:, c].max()}]"
        print(f"{c:>4} | {rho_sc:+.3f}          | {rho_si:+.3f}         | {sc_range:<17} | {si_range}")

    print()
    print("=" * 78)
    print("=== 5. N-explained variance through write chain (per cell)")
    print("=" * 78)
    all_ire = np.concatenate([np.array(v) for v in n_to_I_re.values()])

    def n_expl(mat, N):
        Nc = N - N.mean()
        if Nc.var() < 1e-9:
            return np.full(mat.shape[1], np.nan)
        out = []
        for c in range(mat.shape[1]):
            col = mat[:, c]
            if col.var() < 1e-12:
                out.append(0.0); continue
            cc = col - col.mean()
            slope = (cc @ Nc) / (Nc @ Nc)
            out.append(min((slope**2) * Nc.var() / col.var(), 1.0))
        return np.array(out)

    Ns_f = all_N.astype(float)
    e_ire = n_expl(all_ire, Ns_f)
    e_sc = n_expl(all_sc, Ns_f)
    e_si = n_expl(all_si.astype(float), Ns_f)
    print(f"{'cell':>4} | I_re N-expl | spike_count N-expl | scratch_int N-expl")
    print("-" * 70)
    for c in range(all_sc.shape[1]):
        print(f"{c:>4} | {e_ire[c]:.3f}       | {e_sc[c]:.3f}             | {e_si[c]:.3f}")
    print()
    print(f"Sum N-expl across 5 cells:")
    print(f"  I_re (drive_proj output):   {e_ire.sum():.3f}")
    print(f"  spike_count (pre-quantize): {e_sc.sum():.3f}")
    print(f"  scratch_int (post-quantize):{e_si.sum():.3f}")
    if e_sc.sum() > 1e-6:
        print(f"  Info preserved through quantize: {e_si.sum() / e_sc.sum():.1%}")


if __name__ == "__main__":
    main()
