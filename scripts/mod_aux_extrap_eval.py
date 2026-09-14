"""Acid-test for mod-aux: oscillation vs lookup-table representation.

Trained model achieves high mod-aux acc on training N range. Question:
- Did model learn TRUE periodic representation (oscillation in latent)?
  → mod-aux acc stays high on N >> training range
- Or BINARY-FEATURE LOOKUP (e.g., bit for "even", per-N-class bits)?
  → mod-aux acc drops to chance on extrapolation N (lookup unknown classes)

This eval splits N into train-range (1..n_max_train) and extrap-range
(n_max_train+1..n_max_eval). Reports per-modulus balanced accuracy on each.
Drop on extrap = lookup. Sustained = oscillation.

Also reports BALANCED accuracy (mean of recall_pos, recall_neg) to defeat
class-imbalance trivial baselines (always predict 'different' for high m).
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
    p.add_argument("--n-train", type=int, default=90, help="Training n_max (in-distribution)")
    p.add_argument("--n-extrap-min", type=int, default=120, help="Extrap range start")
    p.add_argument("--n-extrap-max", type=int, default=200, help="Extrap range end")
    p.add_argument("--moduli", type=str, default="2,3,5")
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--L", type=int, default=400)
    p.add_argument("--cnn-pfc-only", action="store_true")
    args = p.parse_args()

    moduli = tuple(int(x) for x in args.moduli.split(",") if x.strip())

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=args.cnn_pfc_only,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # Need to use scene with extrap range. Custom batch builder.
    def eval_range(n_min, n_max, label):
        # Configure scene to sample alpha and beta from [n_min, n_max]
        cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
        cfg.K = 8
        cfg.equal_weight = 0.20
        cfg.alpha_distribution = "uniform"
        rng = np.random.default_rng(42)

        # Container per modulus
        all_preds = {m: [] for m in moduli}
        all_labels = {m: [] for m in moduli}

        for _ in range(args.n_batches):
            inputs, _, ci, metas = sample_training_batch(
                batch_size=args.batch_size, n_max_stage=n_max,
                config=cfg, rng=rng,
            )
            # Filter: only keep batches where N_alpha in [n_min, n_max] and beta_counts also
            # Actually scene generates from [1, n_max]. For extrap eval we need n_min cutoff.
            # Simplest: do uniform [1, n_max] for in-dist, [n_extrap_min, n_extrap_max] for extrap.
            # The scene generator doesn't support n_min, so we filter post-hoc.
            with torch.no_grad():
                _ = agent(inputs.to(device), ci.to(device))
            aux_logits = agent._last_aux_mod_logits  # (B, K, n_mod)
            if aux_logits is None:
                print(f"WARN: no aux logits for {label}")
                return None
            preds = (aux_logits > 0).float().cpu().numpy()

            for b, meta in enumerate(metas):
                n_a = meta["N_alpha"]
                if not (n_min <= n_a <= n_max):
                    continue
                for i, n_b in enumerate(meta["beta_counts"]):
                    if not (n_min <= n_b <= n_max):
                        continue
                    for k_m, mod_v in enumerate(moduli):
                        lbl = float((n_a % mod_v) == (n_b % mod_v))
                        all_preds[mod_v].append(preds[b, i, k_m])
                        all_labels[mod_v].append(lbl)

        result = {}
        for m in moduli:
            p_arr = np.array(all_preds[m])
            l_arr = np.array(all_labels[m])
            if len(p_arr) == 0:
                result[m] = None
                continue
            acc = (p_arr == l_arr).mean()
            pos_mask = l_arr == 1.0
            neg_mask = l_arr == 0.0
            if pos_mask.any() and neg_mask.any():
                rec_pos = (p_arr[pos_mask] == 1.0).mean()
                rec_neg = (p_arr[neg_mask] == 0.0).mean()
                bal_acc = 0.5 * (rec_pos + rec_neg)
            else:
                bal_acc = float("nan")
            result[m] = {
                "acc": acc, "bal_acc": bal_acc,
                "n_samples": len(p_arr),
                "pos_rate_label": l_arr.mean(),
                "pos_rate_pred": p_arr.mean(),
            }
        return result

    print(f"=== {args.ckpt.name} mod-aux extrap eval ===")
    print(f"Moduli: {moduli}")

    print(f"\n--- IN-DIST (N ∈ [1, {args.n_train}]) ---")
    in_dist = eval_range(1, args.n_train, "in")
    if in_dist:
        for m in moduli:
            r = in_dist[m]
            if r is None:
                continue
            print(f"  mod {m}: acc={r['acc']:.3f}, bal_acc={r['bal_acc']:.3f}, "
                  f"pos_rate label={r['pos_rate_label']:.2f} pred={r['pos_rate_pred']:.2f}, n={r['n_samples']}")

    print(f"\n--- EXTRAP (N ∈ [{args.n_extrap_min}, {args.n_extrap_max}]) ---")
    extrap = eval_range(args.n_extrap_min, args.n_extrap_max, "ex")
    if extrap:
        for m in moduli:
            r = extrap[m]
            if r is None or np.isnan(r["bal_acc"]):
                print(f"  mod {m}: insufficient samples for balanced acc")
                continue
            print(f"  mod {m}: acc={r['acc']:.3f}, bal_acc={r['bal_acc']:.3f}, "
                  f"pos_rate label={r['pos_rate_label']:.2f} pred={r['pos_rate_pred']:.2f}, n={r['n_samples']}")

    print()
    print("Interpretation:")
    print("  bal_acc near 0.5 = chance / lookup table failing on novel N")
    print("  bal_acc > 0.7 with extrap = TRUE periodic representation (oscillation)")
    print("  Drop from in-dist to extrap = degree of lookup vs oscillation")


if __name__ == "__main__":
    main()
