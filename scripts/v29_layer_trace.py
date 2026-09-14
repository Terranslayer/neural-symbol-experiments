# DIAG_FOR: v29;v29.1;v29.2;v29.3;v29.4;v29.5 | ANSWERS: layer_trace | INPUTS: <ckpt>
"""V29 layer-by-layer N information tracing.

Top-down: trace where N (count of α spikes) information gets lost or compressed.
For each layer, report:
  - DC offset (per-batch mean)
  - Variance across N
  - PC1 ρ(N) (Spearman of PC1 vs N)
  - Per-dim |ρ(N)| > 0.4 count

Layers traced (V29.3 with LN):
  L1. Encoder spike rate per neuron per N — sanity (should be ρ +1.0)
  L2. Event tokens per batch (mean across events) — Time2VecPE + neuron_id_embed
  L3. PFC output per batch (mean over valid event tokens) — POST-LN
  L4. PFC LN-internal γ, β params (final LN affine)
  L5. u_w per query (cross-attn output, 5 separate)
  L6. drive_proj weights + bias (raw params)
  L7. I_complex per cell (drive_proj output)
  L8. Mean and variance of I_complex DECOMPOSED:
      - DC contribution from bias
      - Linear contribution from u_w mean (DC-ish)
      - N-dependent contribution from u_w N-axis
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def per_dim_corr_with_N(H, N_arr):
    """Per-column ρ(H[:, j], N)."""
    rs = []
    for j in range(H.shape[1]):
        if H[:, j].std() < 1e-9:
            rs.append(float("nan"))
        else:
            r, _ = spearmanr(H[:, j], N_arr)
            rs.append(r)
    return rs


def variance_decomp(H, N_arr):
    """Decompose H variance:
      total var per dim
      DC contribution (mean^2)
      N-explained variance (R^2 of linear fit on N)
    """
    mu = H.mean(axis=0)
    total_var_per_dim = H.var(axis=0)
    # Linear fit on N
    N_centered = N_arr - N_arr.mean()
    if N_centered.std() < 1e-9:
        return mu, total_var_per_dim, np.zeros_like(mu)
    slopes = (H - mu).T @ N_centered / (N_centered @ N_centered)  # (D,)
    explained_var = slopes ** 2 * N_centered.var()  # var explained per dim
    return mu, total_var_per_dim, explained_var


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

    print("=" * 78)
    print("V29 LAYER TRACE — top-down N information flow")
    print("=" * 78)

    # ---- L4. PFC LN affine params ----
    pfc_enc = v29.pfc.encoder
    if hasattr(pfc_enc, "norm") and pfc_enc.norm is not None:
        ln = pfc_enc.norm
        gamma = ln.weight.detach().cpu().numpy()  # (d,)
        beta = ln.bias.detach().cpu().numpy()
        print(f"\n[L4] PFC final LN affine params:")
        print(f"  γ (weight): mean={gamma.mean():.3f}  std={gamma.std():.3f}  range=[{gamma.min():.3f}, {gamma.max():.3f}]")
        print(f"  β (bias):   mean={beta.mean():+.3f}  std={beta.std():.3f}  range=[{beta.min():+.3f}, {beta.max():+.3f}]")
        ln_present = True
    else:
        print(f"\n[L4] PFC encoder has NO final LN (V29.1/V29.2 mode)")
        ln_present = False

    # ---- L6. drive_proj weights + bias ----
    print(f"\n[L6] drive_proj raw params:")
    w = wh.drive_proj.weight.data.cpu().numpy()  # (2, 32) for V29.2/.3 complex
    b = wh.drive_proj.bias.data.cpu().numpy()  # (2,)
    print(f"  weight shape: {w.shape}, Frobenius={np.linalg.norm(w):.3f}")
    print(f"  weight per-row L2: {[f'{np.linalg.norm(w[i]):.3f}' for i in range(w.shape[0])]}")
    print(f"  weight per-row mean: Re={w[0].mean():+.4f}  Im={w[1].mean():+.4f}")
    print(f"  bias: Re={b[0]:+.4f}  Im={b[1]:+.4f}")

    # ---- Run batches and collect data ----
    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.successor_only_positive = False
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    agent.train(False)

    all_spike_rates = []         # per-batch (16,)
    all_event_summary = []       # mean across valid events (32,)
    all_pfc_summary = []         # mean across valid PFC events (32,)
    all_u_per_query = [[] for _ in range(5)]  # per-query (32,)
    all_I_re_per_cell = []       # (5,)
    all_I_im_per_cell = []
    all_Ns = []

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        alpha_signal = inputs[:, : agent.scene_cfg.L, 0:1].to(device)
        with torch.no_grad():
            spikes, _ = v29.encoder(alpha_signal)
            tokens, mask, _, _ = v29.encode_to_events(alpha_signal)
            a_pfc = v29.pfc(tokens, mask)
            u = v29.alpha_query_extract(a_pfc, mask)
            B, N5, d = u.shape
            proj_out = wh.drive_proj(u.reshape(B * N5, d))
            if proj_out.shape[-1] == 2:
                I_re_im = proj_out.reshape(B, N5, 2)
                I_re = I_re_im[..., 0]
                I_im = I_re_im[..., 1]
            else:
                # V29.1 fallback (shouldn't hit for V29.3)
                proj_out = proj_out.reshape(B, N5, N5)
                I_re = torch.diagonal(proj_out, dim1=1, dim2=2)
                I_im = torch.zeros_like(I_re)

        # Aggregate per batch element
        spike_rates = spikes.sum(dim=1).cpu().numpy()  # (B, 16)
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        token_summary = (tokens * mask_f).sum(dim=1) / denom  # (B, 32) raw event tokens
        pfc_summary = (a_pfc * mask_f).sum(dim=1) / denom    # (B, 32) post-LN PFC
        u_np = u.cpu().numpy()  # (B, 5, 32)
        I_re_np = I_re.cpu().numpy()
        I_im_np = I_im.cpu().numpy()

        for bi, m in enumerate(metas):
            n_a = m["N_alpha"]
            all_Ns.append(n_a)
            all_spike_rates.append(spike_rates[bi])
            all_event_summary.append(token_summary[bi].cpu().numpy())
            all_pfc_summary.append(pfc_summary[bi].cpu().numpy())
            for w_idx in range(5):
                all_u_per_query[w_idx].append(u_np[bi, w_idx])
            all_I_re_per_cell.append(I_re_np[bi])
            all_I_im_per_cell.append(I_im_np[bi])

    Ns_arr = np.array(all_Ns, dtype=float)
    spike_rates_arr = np.array(all_spike_rates)  # (samples, 16)
    event_arr = np.array(all_event_summary)      # (samples, 32)
    pfc_arr = np.array(all_pfc_summary)          # (samples, 32)
    u_arrays = [np.array(all_u_per_query[i]) for i in range(5)]  # 5 × (samples, 32)
    I_re_arr = np.array(all_I_re_per_cell)        # (samples, 5)
    I_im_arr = np.array(all_I_im_per_cell)

    # ---- L1. Encoder spike rate per neuron ----
    print(f"\n[L1] Encoder spike rate per neuron (mean across batch):")
    mu_enc, var_enc, expl_enc = variance_decomp(spike_rates_arr, Ns_arr)
    rhos_enc = per_dim_corr_with_N(spike_rates_arr, Ns_arr)
    print(f"  Across 16 neurons: mean spike count = {mu_enc.mean():.2f}")
    print(f"  Avg N-explained variance fraction: {(expl_enc / (var_enc + 1e-9)).mean():.3f}")
    print(f"  Per-neuron ρ(N): min={min(rhos_enc):.2f}, max={max(rhos_enc):.2f}, mean={np.mean(rhos_enc):.2f}")

    # ---- L2. Event tokens mean (input to PFC) ----
    print(f"\n[L2] Event tokens (mean over valid events per batch):")
    mu_ev, var_ev, expl_ev = variance_decomp(event_arr, Ns_arr)
    print(f"  per-dim mean (DC): mean={mu_ev.mean():+.3f}  std={mu_ev.std():.3f}")
    print(f"  per-dim mean magnitude: mean(|μ|)={np.abs(mu_ev).mean():.3f}")
    print(f"  per-dim total var: mean={var_ev.mean():.4f}")
    print(f"  per-dim N-explained var: mean={expl_ev.mean():.4f}")
    print(f"  N-explained / total ratio: {(expl_ev.sum() / (var_ev.sum() + 1e-9)):.3f}")
    rhos_ev = per_dim_corr_with_N(event_arr, Ns_arr)
    n_strong_ev = sum(1 for r in rhos_ev if not np.isnan(r) and abs(r) > 0.4)
    print(f"  strong-N dims (|ρ|>0.4): {n_strong_ev}/{event_arr.shape[1]}")

    # ---- L3. PFC summary (after final LN) ----
    print(f"\n[L3] PFC output summary (mean over events, POST final LN):")
    mu_pfc, var_pfc, expl_pfc = variance_decomp(pfc_arr, Ns_arr)
    print(f"  per-dim DC: mean(|μ|)={np.abs(mu_pfc).mean():.3f}, max(|μ|)={np.abs(mu_pfc).max():.3f}")
    print(f"  per-dim total var: mean={var_pfc.mean():.4f}")
    print(f"  per-dim N-explained var: mean={expl_pfc.mean():.4f}")
    print(f"  N-explained / total ratio: {(expl_pfc.sum() / (var_pfc.sum() + 1e-9)):.3f}")
    rhos_pfc = per_dim_corr_with_N(pfc_arr, Ns_arr)
    n_strong_pfc = sum(1 for r in rhos_pfc if not np.isnan(r) and abs(r) > 0.4)
    print(f"  strong-N dims (|ρ|>0.4): {n_strong_pfc}/{pfc_arr.shape[1]}")
    print(f"  per-dim ρ: {['%+.2f' % r for r in rhos_pfc]}")

    # ---- L5. u_w per query (5 cross-attn outputs) ----
    print(f"\n[L5] u_w per query (5 cross-attn outputs):")
    for w_idx in range(5):
        u_arr = u_arrays[w_idx]  # (samples, 32)
        mu_u, var_u, expl_u = variance_decomp(u_arr, Ns_arr)
        n_strong = sum(1 for r in per_dim_corr_with_N(u_arr, Ns_arr) if not np.isnan(r) and abs(r) > 0.4)
        ratio = expl_u.sum() / (var_u.sum() + 1e-9)
        print(f"  query {w_idx}: DC mean(|μ|)={np.abs(mu_u).mean():.3f}  "
              f"total var/dim={var_u.mean():.4f}  N-expl/total={ratio:.3f}  "
              f"strong-N dims={n_strong}/32")

    # ---- L7. I_complex per cell ----
    print(f"\n[L7] I_complex (drive_proj output) per cell, threshold={wh.threshold}:")
    print(f"  cell    DC_re      DC_im      var_re     var_im    range_re         ρ(I_re,N)")
    for c in range(5):
        re_col = I_re_arr[:, c]
        im_col = I_im_arr[:, c]
        rho_re = spearmanr(re_col, Ns_arr)[0] if re_col.std() > 1e-9 else float("nan")
        print(f"  {c}      {re_col.mean():+8.4f}  {im_col.mean():+8.4f}  "
              f"{re_col.var():.4f}     {im_col.var():.4f}    "
              f"[{re_col.min():+.3f}, {re_col.max():+.3f}]    {rho_re:+.3f}")

    # ---- L8. I decomposition: DC vs N-dependent contributions ----
    print(f"\n[L8] I decomposition — where N signal lives:")
    # I = drive_proj.weight @ u + drive_proj.bias
    # = w @ (u_DC + u_N_dependent) + bias
    # u_DC = u.mean(axis=0), u_N_dependent = u - u_DC
    # I_DC_contribution = w @ u_DC + bias
    # I_N_contribution = w @ u_N_dependent
    # Compute for query 0 → cell 0 mapping (V29.2/.3 share drive_proj across queries; pair is via diagonal in V29.1 only; in V29.2/.3 drive_proj is shared Linear(d, 2))
    u_all = np.stack(u_arrays, axis=1)  # (samples, 5, 32)
    for c in range(5):
        u_c = u_all[:, c, :]  # (samples, 32)
        u_mean = u_c.mean(axis=0)  # (32,)
        u_dev = u_c - u_mean       # (samples, 32)
        # I = w @ u + bias. w shape (2, 32)
        I_from_dc = w @ u_mean + b   # (2,) constant across samples
        # std of I from N variation: for each sample, w @ u_dev (samples, 2)
        I_from_dev = u_dev @ w.T   # (samples, 2)
        dev_std_re = I_from_dev[:, 0].std()
        dev_std_im = I_from_dev[:, 1].std()
        print(f"  cell {c}: DC contrib I_re={I_from_dc[0]:+.3f}  I_im={I_from_dc[1]:+.3f}  |  "
              f"N-driven std: σ(I_re)={dev_std_re:.4f}  σ(I_im)={dev_std_im:.4f}  |  "
              f"DC/var ratio: I_re {abs(I_from_dc[0])/max(dev_std_re,1e-6):.1f}× | I_im {abs(I_from_dc[1])/max(dev_std_im,1e-6):.1f}×")

    print(f"\n{'='*78}")
    print("Summary: At which layer does N-explained variance ratio drop sharply?")
    print(f"{'='*78}")
    print(f"  L1 encoder spike rates:   N-expl ratio ≈ {(expl_enc.sum() / (var_enc.sum() + 1e-9)):.3f}")
    print(f"  L2 event tokens mean:     N-expl ratio ≈ {(expl_ev.sum() / (var_ev.sum() + 1e-9)):.3f}")
    print(f"  L3 PFC output mean:       N-expl ratio ≈ {(expl_pfc.sum() / (var_pfc.sum() + 1e-9)):.3f}")
    # u_w avg over queries
    u_ratios = []
    for w_idx in range(5):
        u_arr = u_arrays[w_idx]
        _, vv, ee = variance_decomp(u_arr, Ns_arr)
        u_ratios.append(ee.sum() / (vv.sum() + 1e-9))
    print(f"  L5 cross-attn u_w (avg):  N-expl ratio ≈ {np.mean(u_ratios):.3f}")
    # I_complex
    I_full = np.stack([I_re_arr.flatten(), I_im_arr.flatten()], axis=1)
    # use per-cell average for ratio
    I_ratios = []
    for c in range(5):
        for ch in [0, 1]:
            col = (I_re_arr if ch == 0 else I_im_arr)[:, c]
            mu_c = col.mean()
            v_total = col.var()
            v_expl = ((col - mu_c) @ (Ns_arr - Ns_arr.mean()) / (Ns_arr - Ns_arr.mean()).var() / len(Ns_arr)) ** 2 * (Ns_arr - Ns_arr.mean()).var()
            I_ratios.append(v_expl / max(v_total, 1e-9))
    print(f"  L7 I_complex (avg):       N-expl ratio ≈ {np.mean(I_ratios):.3f}")


if __name__ == "__main__":
    main()
