# DIAG_FOR: v29;v29.1;v29.2;v29.3;v29.4;v29.5 | ANSWERS: encoder_spikes,pca,cross_attn,write_head_raw,codebook,cell_decouple | INPUTS: <ckpt>
"""V29 full internal-representation + dynamics diag.

Reports for V29 ckpts:
  1. RFEncoder per-neuron stats — spike rate vs N, learned (b, omega) drift from init
  2. Event tokens after PFC — PCA (B, max_events, d) flatten over events → PC1 vs N
  3. CrossAttentionExtractor output u (B, 5, d) — per-query PC1 vs N + orthogonality
  4. RFWriteHead per-neuron stats — spike count vs N, learned (b, omega) drift
  5. Scratch quantize codebook — distinct codes, per-cell ρ(N), cell-cell Pearson on raw spike count

Compare to V27 / V28-LRU diag output for substrate-vs-substrate comparison.
"""
import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def per_dim_corr(H, N):
    rs = []
    for i in range(H.shape[1]):
        if H[:, i].std() < 1e-9:
            rs.append(float("nan"))
        else:
            r, _ = spearmanr(H[:, i], N)
            rs.append(r)
    return rs


def pca_brief(name, H, N):
    if H.shape[0] < 2:
        print(f"  {name}: too few samples")
        return
    Hc = H - H.mean(0)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var = S ** 2 / max(H.shape[0] - 1, 1)
    var = var / var.sum()
    PC1 = Hc @ Vt[0]
    PC2 = Hc @ Vt[1] if len(Vt) > 1 else np.zeros_like(PC1)
    rho1 = float("nan") if PC1.std() < 1e-9 else spearmanr(PC1, N)[0]
    rho2 = float("nan") if PC2.std() < 1e-9 else spearmanr(PC2, N)[0]
    print(f"  {name}: PC1 var={var[0]:.3f} rho={rho1:+.3f}  PC2 rho={rho2:+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
    )
    device = next(agent.parameters()).device

    if getattr(agent, "v29", None) is None:
        print("ERROR: ckpt is not a V29 model (agent.v29 is None)")
        return

    v29 = agent.v29

    # ------------------------------------------------------------
    # 1. Learned (b, omega) drift from init
    # ------------------------------------------------------------
    print("=" * 72)
    print("=== 1. Learned (b, omega) drift from init ===")
    print("=" * 72)

    enc = v29.encoder
    wh = v29.write_head
    b_enc = enc.b.detach().cpu().numpy()
    omega_enc = enc.omega.detach().cpu().numpy()
    b_enc_init = enc.b_init.detach().cpu().numpy()
    omega_enc_init = enc.omega_init.detach().cpu().numpy()
    decay_enc = np.exp(b_enc)
    decay_enc_init = np.exp(b_enc_init)

    print("\nRFEncoder (16 neurons):")
    print(f"  omega init: linspace[{omega_enc_init.min():.3f}, {omega_enc_init.max():.3f}]")
    print(f"  omega now:  min={omega_enc.min():.3f}  max={omega_enc.max():.3f}  mean drift={np.abs(omega_enc - omega_enc_init).mean():.4f}")
    print(f"  b init:     uniform[{b_enc_init.min():.3f}, {b_enc_init.max():.3f}]  → decay [{decay_enc_init.min():.3f}, {decay_enc_init.max():.3f}]")
    print(f"  b now:      min={b_enc.min():.3f}  max={b_enc.max():.3f}  mean drift={np.abs(b_enc - b_enc_init).mean():.4f}")
    print(f"  decay now:  min={decay_enc.min():.3f}  max={decay_enc.max():.3f}")
    print(f"  per-neuron omega (now vs init):")
    for i in range(min(16, enc.n_neurons)):
        d = omega_enc[i] - omega_enc_init[i]
        marker = "*" if abs(d) > 0.05 else " "
        print(f"    [{i:2d}] init={omega_enc_init[i]:.3f}  now={omega_enc[i]:.3f}  Δ={d:+.4f} {marker}")

    b_wh = wh.b.detach().cpu().numpy()
    omega_wh = wh.omega.detach().cpu().numpy()
    b_wh_init = wh.b_init.detach().cpu().numpy()
    omega_wh_init = wh.omega_init.detach().cpu().numpy()

    print("\nRFWriteHead (5 neurons):")
    print(f"  omega init: linspace[{omega_wh_init.min():.3f}, {omega_wh_init.max():.3f}]")
    print(f"  omega now:  min={omega_wh.min():.3f}  max={omega_wh.max():.3f}")
    print(f"  b init:     [{b_wh_init.min():.3f}, {b_wh_init.max():.3f}]")
    print(f"  b now:      [{b_wh.min():.3f}, {b_wh.max():.3f}]")
    print(f"  per-cell omega (now vs init):")
    for i in range(5):
        d_o = omega_wh[i] - omega_wh_init[i]
        d_b = b_wh[i] - b_wh_init[i]
        marker = "*" if abs(d_o) > 0.05 or abs(d_b) > 0.05 else " "
        print(f"    cell {i}: omega init={omega_wh_init[i]:.3f} → {omega_wh[i]:.3f} (Δ={d_o:+.4f})  "
              f"b init={b_wh_init[i]:.3f} → {b_wh[i]:.3f} (Δ={d_b:+.4f}) {marker}")

    print(f"\nT_write (fixed at construction): {wh.t_write}")

    # ------------------------------------------------------------
    # 2. Query orthogonality (achieved vs init)
    # ------------------------------------------------------------
    cae = v29.alpha_query_extract
    q = cae.queries.detach().cpu().numpy()
    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9)
    cos_sim = q_norm @ q_norm.T
    n = q.shape[0]
    off_diag = cos_sim[~np.eye(n, dtype=bool)]
    print(f"\nCrossAttentionExtractor query orthogonality:")
    print(f"  n_queries = {n}, d_model = {q.shape[1]}")
    print(f"  cosine similarity off-diagonal: mean abs = {np.abs(off_diag).mean():.3f}  max abs = {np.abs(off_diag).max():.3f}")
    print(f"  (small abs = orthogonal; near 1 = collapsed)")
    print(f"  full cosine sim matrix:")
    for i in range(n):
        print(f"    {' '.join(f'{cos_sim[i,j]:+.2f}' for j in range(n))}")

    # ------------------------------------------------------------
    # 3. Run batches and collect internal stats per N
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("=== 2. Spike rates + module PCA vs N (data collection) ===")
    print("=" * 72)

    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.successor_only_positive = False
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    agent.train(False)

    # Per-N stash
    n_to_enc_rate = defaultdict(list)   # per-neuron spike count per α episode
    n_to_pfc_summary = defaultdict(list)  # mean of PFC output over events
    n_to_u = defaultdict(list)          # cross-attn output (5, d)
    n_to_drive = defaultdict(list)      # I_w driving currents (5,)
    n_to_spike_count = defaultdict(list)  # write head per-neuron spike count
    n_to_scratch_int = defaultdict(list)
    n_to_scratch_q = defaultdict(list)

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))
        # Read from V29 stashes
        a_spikes = agent._v29_alpha_stash["spikes"]  # (B, L, 16)
        a_tokens = agent._v29_alpha_stash["tokens"]  # (B, max_events, d)
        a_mask = agent._v29_alpha_stash["mask"]      # (B, max_events)
        sc_q = agent._scratch_alpha_for_aux          # (B, 5)
        sc_int = agent._v29_scratch_int              # (B, 5) long

        # Re-run α-side to capture intermediate outputs (PFC + cross-attn + drive)
        alpha_signal = inputs[:, : agent.scene_cfg.L, 0:1].to(device)
        with torch.no_grad():
            tokens, mask, _, _ = v29.encode_to_events(alpha_signal)
            a_pfc = v29.pfc(tokens, mask)  # (B, max_events, d)
            u = v29.alpha_query_extract(a_pfc, mask)  # (B, 5, d)
            # Drive currents (B, 5) — extract per-query scalar
            B, N5, d = u.shape
            I_drive = v29.write_head.drive_proj(u.reshape(B * N5, d)).reshape(B, N5)

        # PFC summary: mean over valid events per batch
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        a_pfc_summary = (a_pfc * mask_f).sum(dim=1) / denom  # (B, d)

        # Per-neuron spike count over α scan
        spike_count_enc = a_spikes.sum(dim=1).cpu().numpy()  # (B, 16)

        for b, m in enumerate(metas):
            n_a = m["N_alpha"]
            n_to_enc_rate[n_a].append(spike_count_enc[b])
            n_to_pfc_summary[n_a].append(a_pfc_summary[b].cpu().numpy())
            n_to_u[n_a].append(u[b].cpu().numpy())
            n_to_drive[n_a].append(I_drive[b].cpu().numpy())
            n_to_scratch_int[n_a].append(sc_int[b].cpu().numpy())
            n_to_scratch_q[n_a].append(sc_q[b].cpu().numpy())

    # ------------------------------------------------------------
    # 4. Encoder per-neuron spike rate vs N
    # ------------------------------------------------------------
    print("\n--- Encoder per-neuron spike count vs N (mean ± std over batches) ---")
    Ns_sorted = sorted(n_to_enc_rate.keys())
    # Per-neuron ρ(N) (Spearman correlation between mean spike count and N)
    # Build matrix (Ns, 16) of mean spike counts per N per neuron
    rate_matrix = []
    for n in Ns_sorted:
        mat = np.array(n_to_enc_rate[n])  # (samples_for_this_N, 16)
        rate_matrix.append(mat.mean(axis=0))
    rate_matrix = np.array(rate_matrix)  # (n_Ns, 16)
    Ns_arr = np.array(Ns_sorted, dtype=float)
    rhos = []
    for j in range(rate_matrix.shape[1]):
        col = rate_matrix[:, j]
        if col.std() < 1e-9:
            rhos.append(float("nan"))
        else:
            r, _ = spearmanr(col, Ns_arr)
            rhos.append(r)
    print(f"  Per-neuron ρ(spike_count vs N) (Spearman):")
    for j in range(16):
        marker = "MONO" if abs(rhos[j]) > 0.7 else ("weak" if abs(rhos[j]) > 0.3 else "    ")
        print(f"    [{j:2d}] ω={omega_enc[j]:.3f}  ρ={rhos[j]:+.3f}  range=[{rate_matrix[:,j].min():.1f}, {rate_matrix[:,j].max():.1f}]  mean={rate_matrix[:,j].mean():.2f}  {marker}")

    # Spike rate at small N vs large N (averaged across all neurons)
    small_N = [n for n in Ns_sorted if n <= 5]
    large_N = [n for n in Ns_sorted if n >= args.n_max - 5]
    rate_small = rate_matrix[[Ns_sorted.index(n) for n in small_N], :].mean()
    rate_large = rate_matrix[[Ns_sorted.index(n) for n in large_N], :].mean()
    print(f"\n  Avg spike count per neuron, small N (1..5): {rate_small:.2f}")
    print(f"  Avg spike count per neuron, large N ({large_N[0]}..{large_N[-1]}): {rate_large:.2f}")

    # ------------------------------------------------------------
    # 5. PFC summary PCA vs N
    # ------------------------------------------------------------
    print("\n--- PFC output (mean-pooled over events) PCA ---")
    # Flatten all (b, n) into rows
    all_pfc = []
    all_pfc_Ns = []
    for n in Ns_sorted:
        for v in n_to_pfc_summary[n]:
            all_pfc.append(v)
            all_pfc_Ns.append(n)
    all_pfc = np.array(all_pfc)
    all_pfc_Ns = np.array(all_pfc_Ns, dtype=float)
    pca_brief("PFC summary", all_pfc, all_pfc_Ns)

    rhos_pfc = per_dim_corr(all_pfc, all_pfc_Ns)
    n_strong = sum(1 for r in rhos_pfc if not np.isnan(r) and abs(r) > 0.4)
    print(f"  PFC strong-N dims (|ρ|>0.4): {n_strong}/{len(rhos_pfc)}")
    print(f"  per-dim ρ: {['%+.2f' % r for r in rhos_pfc]}")

    # ------------------------------------------------------------
    # 6. Cross-attention output u — per-query PCA vs N
    # ------------------------------------------------------------
    print("\n--- Cross-attention output u (5 queries × d) per-query ---")
    # u[b, w, :] for each query w
    for w in range(5):
        all_u_w = []
        for n in Ns_sorted:
            for v in n_to_u[n]:
                all_u_w.append(v[w])
        all_u_w = np.array(all_u_w)
        Ns_for_u = []
        for n in Ns_sorted:
            for _ in n_to_u[n]:
                Ns_for_u.append(n)
        pca_brief(f"u[query {w}]", all_u_w, np.array(Ns_for_u, dtype=float))

    # ------------------------------------------------------------
    # 7. Drive current I_w per neuron vs N
    # ------------------------------------------------------------
    print("\n--- Drive current I_w per write neuron vs N ---")
    drive_matrix = []
    for n in Ns_sorted:
        mat = np.array(n_to_drive[n])  # (samples, 5)
        drive_matrix.append(mat.mean(axis=0))
    drive_matrix = np.array(drive_matrix)  # (n_Ns, 5)
    print(f"  Per-cell ρ(I_w vs N) (Spearman):")
    for w in range(5):
        col = drive_matrix[:, w]
        if col.std() < 1e-9:
            r = float("nan")
        else:
            r, _ = spearmanr(col, Ns_arr)
        print(f"    cell {w}: ω={omega_wh[w]:.3f}  b={b_wh[w]:.3f}  ρ(I,N)={r:+.3f}  "
              f"range=[{col.min():+.3f},{col.max():+.3f}]  mean={col.mean():+.3f}")

    # ------------------------------------------------------------
    # 8. Scratch raw spike count + quantize codebook
    # ------------------------------------------------------------
    print("\n--- Write head raw spike count per cell (sc_int 0/1/2+) ---")
    int_matrix = []
    for n in Ns_sorted:
        mat = np.array(n_to_scratch_int[n])  # (samples, 5)
        int_matrix.append(mat.mean(axis=0))
    int_matrix = np.array(int_matrix)  # (n_Ns, 5)
    print(f"  Per-cell ρ(scratch_int vs N) (Spearman):")
    for w in range(5):
        col = int_matrix[:, w]
        if col.std() < 1e-9:
            r = float("nan")
        else:
            r, _ = spearmanr(col, Ns_arr)
        small_v = int_matrix[: min(5, len(Ns_sorted)), w].mean()
        large_v = int_matrix[-min(5, len(Ns_sorted)):, w].mean()
        print(f"    cell {w}: ρ={r:+.3f}  small-N avg={small_v:.2f}  large-N avg={large_v:.2f}  range=[{col.min():.2f},{col.max():.2f}]")

    # Cell-cell Pearson on scratch_int (averaged per N)
    print(f"\n  Cell-cell Pearson on scratch_int (rows of int_matrix):")
    print(f"       c0    c1    c2    c3    c4")
    for i in range(5):
        row = []
        for j in range(5):
            ai = int_matrix[:, i]
            aj = int_matrix[:, j]
            if ai.std() < 1e-9 or aj.std() < 1e-9:
                row.append("nan")
            else:
                r, _ = pearsonr(ai, aj)
                row.append(f"{r:+.2f}")
        print(f"  c{i}  " + "  ".join(row))


if __name__ == "__main__":
    main()
