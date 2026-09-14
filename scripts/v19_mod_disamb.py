# DIAG_FOR: v19 | ANSWERS: mod_disamb,codebook | INPUTS: <ckpt>
"""V19 mod-aux disambiguation diagnostics.

Two questions:

1. Is mod 5 bal_acc real modular learning, or task confound?
   Decompose mod 5 acc by |δ| (δ = N_β - N_α). For trio_wide N>=15:
     - δ = 0  →  same-mod-5 AND main-task E-label
     - δ = ±5 → same-mod-5 AND main-task G/L-label (NOT E!)
     - δ = ±10/±15 → same-mod-5 AND main-task G/L
     - other δ → different-mod-5
   If model only succeeds on δ=0 (E pair shortcut), it's NOT learning mod 5.
   If model succeeds on δ=±5 too, it IS learning mod 5.

2. Is per-position codebook structure thermometer binning, or RNS / mixed-mod?
   - Dump modal codes per N
   - Compute, per position p (0..4), pos[p] vs N curve. If monotonic-ish (with
     plateaus), it's a thermometer bin. If period-3 or period-5 like, it's a
     residue indicator.
   - Compute pos-pos mutual information. If RNS, low; if thermometer-binning,
     high (positions co-vary along the same axis).

Usage:
  python scripts/v19_mod_disamb.py <ckpt> --L <L> --n-max <n_max>
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
    p.add_argument("--L", type=int, default=400)
    p.add_argument("--n-max", type=int, default=90)
    p.add_argument("--n-batches", type=int, default=64)
    p.add_argument("--no-oracle", action="store_true")
    p.add_argument("--task", choices=["trio_wide", "trio_wide_extended"], default="trio_wide")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
        use_oracle_spike_gate=not args.no_oracle,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    if args.task == "trio_wide_extended":
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
    else:
        cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    rng = np.random.default_rng(99)

    # --- mod 5 by δ decomposition ---
    delta_buckets: dict[int, dict[str, int]] = {}
    n_alpha_to_codes: dict[int, list[tuple]] = {}
    pos_to_n_pairs: list[tuple] = []  # (N, codes tuple)

    for _ in range(args.n_batches):
        inp, lbl, ci, metas = sample_training_batch(
            batch_size=64, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            out = agent(inp.to(device), ci.to(device), metas=metas)
        # out can be (logits, scratch, ...) tuple
        logits = out[0]
        scratch = out[1] if len(out) > 1 else None
        aux = getattr(agent, "_last_aux_mod_logits", None)

        # Per-block N_alpha (single value per episode), N_beta varies per block.
        Nas = np.array([m["N_alpha"] for m in metas])  # (B,)
        Nbs = np.array([m["beta_counts"] for m in metas])  # (B, K)

        # mod 5 head index = 2 (moduli 2,3,5)
        if aux is not None:
            mod5_logits = aux[..., 2].cpu().numpy()  # (B, K)
            mod5_pred = (mod5_logits > 0).astype(int)
            mod5_target = ((Nas[:, None] % 5) == (Nbs % 5)).astype(int)
            deltas = Nbs - Nas[:, None]  # (B, K)

            for b in range(Nas.shape[0]):
                for k in range(Nbs.shape[1]):
                    d = int(deltas[b, k])
                    tgt = int(mod5_target[b, k])
                    pr = int(mod5_pred[b, k])
                    bucket = delta_buckets.setdefault(d, {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tot": 0})
                    bucket["tot"] += 1
                    if tgt == 1 and pr == 1: bucket["tp"] += 1
                    elif tgt == 0 and pr == 0: bucket["tn"] += 1
                    elif tgt == 0 and pr == 1: bucket["fp"] += 1
                    else: bucket["fn"] += 1

        # Modal codes per N (use scratch_pad)
        if scratch is not None:
            sc = scratch.cpu().numpy()  # (B, W, signal_dim) typically W=5
            if sc.ndim == 3 and sc.shape[2] == 1:
                sc = sc[..., 0]  # (B, W)
            sc = np.round(sc, 2)  # avoid float spurious uniqueness
            # discretize to {0, 0.5, 1.0} bins (q=3 unit range)
            sc_q = np.clip(np.round(sc * 2) / 2, 0.0, 1.0)
            for b in range(Nas.shape[0]):
                code = tuple(sc_q[b].tolist())
                n_alpha_to_codes.setdefault(int(Nas[b]), []).append(code)

    # ---- Output ----
    print("=== mod 5 bal_acc by |δ| (δ = N_β - N_α) ===")
    print(f"{'δ':>4} | {'tot':>5} | {'tgt=1?':<6} | {'rec_pos':>7} | {'rec_neg':>7} | {'bal':>5} | interp")
    for d in sorted(delta_buckets.keys()):
        b = delta_buckets[d]
        tot = b["tot"]
        if tot < 50: continue
        is_pos = (d % 5 == 0)
        rec_pos = b["tp"] / max(b["tp"] + b["fn"], 1)
        rec_neg = b["tn"] / max(b["tn"] + b["fp"], 1)
        if is_pos:
            r = rec_pos  # what fraction of same-mod-5 pairs at this δ get pred=1
            interp = "E pair" if d == 0 else f"same-mod-5 (real)"
        else:
            r = rec_neg
            interp = "diff-mod-5"
        bal = "—"
        print(f"{d:>4} | {tot:>5} | {'pos' if is_pos else 'neg':<6} | {rec_pos:>7.3f} | {rec_neg:>7.3f} | {bal:>5} | {interp}")

    # Aggregate "real mod 5" recall (only δ != 0 same-mod-5 pairs)
    real_pos_tp = sum(b["tp"] for d, b in delta_buckets.items() if d % 5 == 0 and d != 0)
    real_pos_tot = sum(b["tot"] for d, b in delta_buckets.items() if d % 5 == 0 and d != 0)
    e_pair_tp = delta_buckets.get(0, {}).get("tp", 0)
    e_pair_tot = delta_buckets.get(0, {}).get("tot", 0)
    print()
    print(f"  E-pair (δ=0) recall_pos                   = {e_pair_tp}/{e_pair_tot} = {e_pair_tp/max(e_pair_tot,1):.3f}")
    print(f"  Real-mod-5 (δ=±5,±10,±15...) recall_pos  = {real_pos_tp}/{real_pos_tot} = {real_pos_tp/max(real_pos_tot,1):.3f}")
    print(f"  → if E-pair recall ≫ real-mod-5 recall, mod 5 is task confound")
    print(f"  → if both high, real modular learning")

    # ---- Codebook & per-pos analysis ----
    print()
    print("=== Modal code per N (top 1 per N) ===")
    pure_n = 0
    distinct_codes = set()
    code_share_max = 0
    n_modal: dict[int, tuple] = {}
    for n in sorted(n_alpha_to_codes.keys()):
        codes = n_alpha_to_codes[n]
        if not codes: continue
        from collections import Counter
        cnt = Counter(codes)
        modal, share = cnt.most_common(1)[0]
        n_modal[n] = modal
        share_frac = share / len(codes)
        if share_frac >= 0.95: pure_n += 1
        distinct_codes.add(modal)
        if share > code_share_max: code_share_max = share
    print(f"  distinct modal codes: {len(distinct_codes)}")
    print(f"  pure-N (>=95% same code): {pure_n}/{len(n_alpha_to_codes)}")
    print(f"  max single-code share (any one code): {code_share_max} episodes")

    # Per-position pos[p] vs N
    if n_modal:
        Ns = sorted(n_modal.keys())
        W = len(next(iter(n_modal.values())))
        print()
        print(f"=== Per-position modal value over N (W={W}) ===")
        for p in range(W):
            vals = [n_modal[n][p] for n in Ns]
            arr = np.array(vals)
            # Spearman with N
            from scipy.stats import spearmanr
            rho, _ = spearmanr(Ns, arr)
            # FFT for periodic structure
            curve = arr - arr.mean()
            fft = np.fft.rfft(curve)
            power = np.abs(fft) ** 2
            if len(power) > 1:
                peak_k = int(np.argmax(power[1:]) + 1)
                purity = power[peak_k] / (power[1:].sum() + 1e-9)
                period = len(arr) / peak_k if peak_k > 0 else float("inf")
            else:
                peak_k, purity, period = 0, 0.0, float("inf")
            unique_count = len(set(vals))
            transitions = (np.diff(arr) != 0).sum()
            print(f"  pos {p}: spearman_N={rho:+.3f}  uniq={unique_count}  transitions={transitions}  fft_period={period:.1f}  purity={purity:.2f}")

        # Pos-pos correlation (high → thermometer binning; low → independent residues)
        print()
        print("=== Pos-pos Pearson correlation across N (high = thermometer; low = independent) ===")
        mat = np.array([[n_modal[n][p] for p in range(W)] for n in Ns])
        if mat.shape[0] > 2:
            corr = np.corrcoef(mat.T)
            for r in range(W):
                row = " ".join(f"{corr[r, c]:+.2f}" for c in range(W))
                print(f"  pos{r}: {row}")


if __name__ == "__main__":
    main()
