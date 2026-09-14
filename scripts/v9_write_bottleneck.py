"""V9 write-head bottleneck probe.

Hook write_head.forward to capture its input — the post-kWTA hidden vector that
gets compressed by Linear(d_model, K_pos * Q). Compare:

  - Pre-write h_for_write (16 floats, "what PFC wants to write")
  - Quantized scratch     (5 floats q∈{0, 0.5, 1.0}, "what got written")

Metrics per representation:
  1. PCA: PC1 var, PC1↔N spearman, inversions, gap/std ratio
  2. KNN-1 leave-one-out N classification accuracy (proxy for separability)
  3. # distinct rounded codes per N — codebook utilization
"""
import argparse, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def pca(name, H, N_arr):
    if H.shape[0] < 4:
        return
    Hc = H - H.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var_ratio = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio /= var_ratio.sum()
    PC1 = Hc @ Vt[0]
    rho1, _ = spearmanr(PC1, N_arr)
    pc1_per_n = {n: PC1[N_arr == n] for n in np.unique(N_arr)}
    within_std = float(np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1]))
    means = np.array([pc1_per_n[n].mean() for n in sorted(pc1_per_n.keys())])
    adj = np.abs(np.diff(means))
    direction = 1 if means[-1] > means[0] else -1
    inv = sum(1 for i in range(len(means)-1) if direction * (means[i+1]-means[i]) < 0)
    print(f"  [{name}] PC1 var {var_ratio[0]:.3f}  PC1↔N {rho1:+.3f}  "
          f"gap/std {adj.mean()/max(within_std,1e-9):.3f}  "
          f"inv {inv}/{len(means)-1} ({inv/max(len(means)-1,1)*100:.0f}%)")


def knn_n_acc(H, N, k=1):
    """5-fold CV KNN classification accuracy on N labels."""
    if H.shape[0] < 50:
        return float('nan')
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, H, N, cv=5)
    return scores.mean()


def codebook_stats(H, N, decimals=3):
    """Count distinct rounded codes per N + overall."""
    rounded = [tuple(np.round(h, decimals).tolist()) for h in H]
    overall = len(set(rounded))
    per_n_distinct = defaultdict(set)
    for r, n in zip(rounded, N):
        per_n_distinct[n].add(r)
    avg = np.mean([len(s) for s in per_n_distinct.values()])
    # Cross-N overlap: how many N share the same code
    code_to_ns = defaultdict(set)
    for r, n in zip(rounded, N):
        code_to_ns[r].add(n)
    n_per_code = [len(s) for s in code_to_ns.values()]
    return overall, avg, np.mean(n_per_code), max(n_per_code)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=True,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    # Hook write_head forward: input[0] is h_for_write (B, d_model)
    captured = []
    def hook(module, inp, out):
        captured.append(inp[0].detach().clone())
    agent.write_head.register_forward_hook(hook)

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.equal_weight = 0.20
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    n_to_h = defaultdict(list)
    n_to_scratch = defaultdict(list)

    for _ in range(args.n_batches):
        captured.clear()
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _, scratch, _ = agent(inputs.to(device), ci.to(device))

        if not captured:
            print("WARN: write_head hook never fired"); return

        # write_head is called once per write step; first call is α-write
        h_pre = captured[0].cpu().numpy()  # (B, d_model)
        scr = scratch.cpu().numpy()  # (B, 5)

        for b, m in enumerate(metas):
            n = m["N_alpha"]
            n_to_h[n].append(h_pre[b])
            n_to_scratch[n].append(scr[b])

    H_pre = np.array([h for n in sorted(n_to_h.keys()) for h in n_to_h[n]])
    N_pre = np.array([n for n in sorted(n_to_h.keys()) for h in n_to_h[n]])
    H_scr = np.array([h for n in sorted(n_to_scratch.keys()) for h in n_to_scratch[n]])
    N_scr = N_pre.copy()

    print(f"\n=== {args.ckpt.name}: write-head bottleneck ===")
    print(f"Samples: {H_pre.shape[0]}, Ns: {len(np.unique(N_pre))}\n")

    print("PCA structure:")
    pca("h_for_write (pre-quantize, d=16)", H_pre, N_pre)
    pca("scratch     (post-quantize, d=5)", H_scr, N_scr)

    print("\nKNN-1 leave-out-one N-classification (5-fold CV):")
    knn_pre = knn_n_acc(H_pre, N_pre, k=1)
    knn_scr = knn_n_acc(H_scr, N_scr, k=1)
    print(f"  h_for_write: {knn_pre:.3f}  (KNN can recover N → measures separability)")
    print(f"  scratch    : {knn_scr:.3f}")
    print(f"  loss from quantize: {knn_pre - knn_scr:+.3f}")

    print("\nKNN-3 (smoothed):")
    knn_pre3 = knn_n_acc(H_pre, N_pre, k=3)
    knn_scr3 = knn_n_acc(H_scr, N_scr, k=3)
    print(f"  h_for_write: {knn_pre3:.3f}")
    print(f"  scratch    : {knn_scr3:.3f}")

    print("\nCodebook utilization:")
    o_pre, avg_pre, mean_share_pre, max_share_pre = codebook_stats(H_pre, N_pre, decimals=2)
    o_scr, avg_scr, mean_share_scr, max_share_scr = codebook_stats(H_scr, N_scr, decimals=3)
    print(f"  h_for_write (round 2dp): {o_pre} distinct codes; avg {avg_pre:.1f} per N; max {max_share_pre} N share 1 code")
    print(f"  scratch              : {o_scr} distinct codes; avg {avg_scr:.1f} per N; max {max_share_scr} N share 1 code")

    # Key: how often does quantize merge "neighboring N" codes
    print("\nNN-N collisions (pairs of (N, N+1) sharing dominant code):")
    for label, n_to in [("h_for_write 2dp", n_to_h), ("scratch", n_to_scratch)]:
        decimals = 2 if "h_for" in label else 3
        dominant = {}
        for n, hs in n_to.items():
            codes = [tuple(np.round(h, decimals).tolist()) for h in hs]
            dominant[n] = Counter(codes).most_common(1)[0][0]
        ns_sorted = sorted(dominant.keys())
        collisions = sum(1 for i in range(len(ns_sorted)-1)
                         if dominant[ns_sorted[i]] == dominant[ns_sorted[i+1]])
        print(f"  {label}: {collisions}/{len(ns_sorted)-1} adjacent N share dominant code")


if __name__ == "__main__":
    main()
