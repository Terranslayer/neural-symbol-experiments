# DIAG_FOR: v25;v25-v19;v25-v22;v25-v24;v26 | ANSWERS: lru_dynamics,fft_n | INPUTS: <ckpt> --task
"""V25 LRU deep diagnostic.

Probes:
  1. LRU end state (16-dim) per N — per-dim ρ(N), ρ(log N)
  2. PC2/PC3/PC4 ↔ N (where N is hiding beyond PC1)
  3. Effective dim (cum 99% var), how many active dims
  4. Per-dim FFT-vs-N: does any dim oscillate with period dividing N?
     (V19 oracle's "phase rotation" test — period-2/3/5/7 etc)
  5. α-scan vs β-scan end state shape similarity (mechanism consistency)
  6. LRU mid-scan trajectory: at t=L/4, L/2, 3L/4, L, how does PC1 grow with N?
  7. spike-gate stats (oracle-gate firing fraction per N)

For V25-V19/V22 (trio_wide_extended) and V25-V24 (successor_prediction).
"""
import argparse, sys
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
from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl


def per_dim_corr(H, N):
    """Per-dim spearman with N and log(N). Returns lists of length d."""
    d = H.shape[1]
    rs = []
    rs_log = []
    for i in range(d):
        if H[:, i].std() < 1e-9:
            rs.append(float('nan')); rs_log.append(float('nan'))
        else:
            r, _ = spearmanr(H[:, i], N)
            rl, _ = spearmanr(H[:, i], np.log(N + 1e-3))
            rs.append(r); rs_log.append(rl)
    return rs, rs_log


def fft_per_dim_vs_n(H, N, n_min, n_max):
    """For each dim, average across samples per N, then FFT the (n_max-n_min+1) sequence.
    Reports top 3 frequencies (period > 1) per dim."""
    d = H.shape[1]
    Ns = sorted(set(int(n) for n in N if n_min <= n <= n_max))
    if len(Ns) < 4:
        return []
    means_per_n = np.array([H[N == n].mean(0) for n in Ns])  # (n_unique, d)
    n_T = means_per_n.shape[0]
    # Mean-center per dim, then FFT
    centered = means_per_n - means_per_n.mean(0, keepdims=True)
    fft = np.fft.rfft(centered, axis=0)
    power = np.abs(fft) ** 2
    freqs = np.arange(power.shape[0])  # cycles per (n_max-n_min+1)
    periods = np.zeros_like(freqs, dtype=float)
    periods[1:] = n_T / freqs[1:]
    results = []
    for i in range(d):
        p = power[:, i]
        # Sort indices by power descending, skip DC (idx 0)
        order = np.argsort(p[1:])[::-1] + 1
        top = []
        for k in order[:3]:
            top.append((float(periods[k]), float(p[k])))
        results.append(top)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", choices=["trio_wide_extended", "successor_prediction"], required=True)
    p.add_argument("--k-max", type=int, default=5)
    p.add_argument("--jsonl", type=Path, default=None)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=False,
    )
    if args.jsonl is not None:
        restore_agent_cfg_from_jsonl(agent, args.jsonl)
    agent.train(False)
    device = next(agent.parameters()).device

    if args.task == "trio_wide_extended":
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
        cfg.K = scene_cfg.K
    else:
        cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
        cfg.K = 1
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    rng = np.random.default_rng(args.seed)

    # === Hooks ===
    # Capture FULL LRU output sequence per call (B, T, d) — first call = α scan
    lru_calls = []  # list of (B, T, d) numpy
    def lru_hook(module, inp, out):
        lru_calls.append(out.detach().cpu())
    agent.mamba_blocks[-1].register_forward_hook(lru_hook)

    # Capture spike gate output if event-gated
    gate_calls = []
    if hasattr(agent, "spike_detector") and agent.spike_detector is not None:
        # We'll capture the post-sigmoid gate via wrapping the LRU block forward...
        # Instead, hook spike_detector
        def gate_hook(module, inp, out):
            gate_calls.append(out.detach().cpu())
        agent.spike_detector.register_forward_hook(gate_hook)

    n_to_lru_alpha_end = defaultdict(list)
    n_to_lru_beta_end = defaultdict(list)
    n_to_lru_traj = defaultdict(list)  # (T, d) per sample for trajectory analysis
    n_to_gate_alpha = defaultdict(list)  # gate output during α
    sample_traj = []  # store a few full sequences for plotting

    for batch_i in range(args.n_batches):
        lru_calls.clear()
        gate_calls.clear()

        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))

        if not lru_calls:
            print("WARN: no LRU capture")
            return
        # First call = α scan (T = L)
        lru_a = lru_calls[0].numpy()  # (B, T, d)
        # Second call onward = β scan(s). Take first β block.
        lru_b = lru_calls[1].numpy() if len(lru_calls) > 1 else None  # (B, T+1, d)

        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            n_to_lru_alpha_end[n_a].append(lru_a[b, -1, :])
            # Mid-scan trajectory: take 5 sample timesteps
            T = lru_a.shape[1]
            ts = [T // 4, T // 2, 3 * T // 4, T - 1]
            for t in ts:
                n_to_lru_traj[(n_a, t)].append(lru_a[b, t, :])
            if lru_b is not None:
                n_b = meta["beta_counts"][0]
                n_to_lru_beta_end[n_b].append(lru_b[b, -1, :])

        # Spike gate: take per-batch summed activation (gate_calls is from
        # spike_detector, may be (B, 1, T))
        if gate_calls:
            ga = gate_calls[0].numpy()  # shape varies
            # Sigmoid to get gate in [0, 1]
            gate_sig = 1.0 / (1.0 + np.exp(-ga * 5.0))  # gate_sharpness=5 init
            for b, meta in enumerate(metas):
                if gate_sig.ndim == 3 and gate_sig.shape[1] == 1:
                    n_to_gate_alpha[meta["N_alpha"]].append(gate_sig[b, 0, :].mean())
                elif gate_sig.ndim == 2:
                    n_to_gate_alpha[meta["N_alpha"]].append(gate_sig[b, :].mean())

    def build(d):
        keys = sorted(d.keys())
        H = np.array([h for n in keys for h in d[n]])
        N = np.array([n for n in keys for h in d[n]])
        return H, N

    print(f"\n{'='*72}\n=== V25 LRU diag: {args.ckpt.name}\n=== task={args.task}\n{'='*72}")

    if n_to_lru_alpha_end:
        H_a, N_a = build(n_to_lru_alpha_end)
        d = H_a.shape[1]
        print(f"\n--- LRU α-scan end state per-dim ρ(N) (d={d}, n={H_a.shape[0]}) ---")
        rs, rs_log = per_dim_corr(H_a, N_a)
        for i in range(d):
            tag = "★" if abs(rs[i]) > 0.4 else (" " if abs(rs[i]) > 0.2 else "·")
            print(f"  dim{i:2d}: ρ(N)={rs[i]:+.3f}  ρ(logN)={rs_log[i]:+.3f}  std={H_a[:,i].std():.3f}  {tag}")

        # Effective dim
        Hc = H_a - H_a.mean(0, keepdims=True)
        _, S, _ = np.linalg.svd(Hc, full_matrices=False)
        var = (S ** 2) / max(H_a.shape[0]-1, 1); var /= var.sum()
        cum = np.cumsum(var)
        print(f"\n  Effective dim (cum var): 50%={int(np.argmax(cum>=0.5)+1)}  90%={int(np.argmax(cum>=0.9)+1)}  99%={int(np.argmax(cum>=0.99)+1)}")
        print(f"  Top 5 PC var ratios: {[f'{v:.3f}' for v in var[:5]]}")
        print(f"  Per-dim std: {[f'{H_a[:,i].std():.3f}' for i in range(d)]}")
        print(f"  Active dims (std > 0.01): {sum(1 for i in range(d) if H_a[:,i].std() > 0.01)}/{d}")

        # FFT-vs-N per dim
        print(f"\n--- LRU α-end FFT-vs-N per dim (n_min=2, n_max={args.n_max}) ---")
        print("  Top 3 (period, power) per dim. Period 1 = monotonic; period 2/3/5/7 = modular oscillation.")
        n_unique = len(set(int(n) for n in N_a))
        print(f"  N_unique = {n_unique} (FFT length)")
        fft_results = fft_per_dim_vs_n(H_a, N_a, 2, args.n_max)
        for i, top in enumerate(fft_results):
            top_str = "  ".join(f"(p={t[0]:.1f}, P={t[1]:.4f})" for t in top)
            print(f"  dim{i:2d}: {top_str}")

    if n_to_lru_beta_end:
        H_b, N_b = build(n_to_lru_beta_end)
        rs_b, _ = per_dim_corr(H_b, N_b)
        print(f"\n--- LRU β-scan end state per-dim ρ(N_β) (n={H_b.shape[0]}) ---")
        active_strong = sum(1 for r in rs_b if abs(r) > 0.4)
        active_med = sum(1 for r in rs_b if 0.2 <= abs(r) <= 0.4)
        print(f"  strong-N dims (|ρ|>0.4): {active_strong}/{len(rs_b)}")
        print(f"  med-N dims (0.2<=|ρ|<=0.4): {active_med}/{len(rs_b)}")
        print(f"  Per-dim ρ(N_β): {[f'{r:+.2f}' for r in rs_b]}")

        # α vs β: same dim signs?
        if n_to_lru_alpha_end:
            sign_match = sum(1 for i in range(len(rs)) if not (np.isnan(rs[i]) or np.isnan(rs_b[i]))
                             and np.sign(rs[i]) == np.sign(rs_b[i]) and abs(rs[i]) > 0.2 and abs(rs_b[i]) > 0.2)
            print(f"  α-β dim sign match (both |ρ|>0.2): {sign_match} dims")

    # Trajectory: PC1 of LRU at t=T/4, T/2, 3T/4, T per N
    if n_to_lru_traj:
        print(f"\n--- LRU mid-scan trajectory (PC1 grows with N over time?) ---")
        T_actual = max(t for (_, t) in n_to_lru_traj.keys())
        ts_used = sorted(set(t for (_, t) in n_to_lru_traj.keys()))
        for t in ts_used:
            data = []
            ns = []
            for (n, tt), lst in n_to_lru_traj.items():
                if tt == t:
                    for v in lst:
                        data.append(v); ns.append(n)
            data = np.array(data); ns = np.array(ns)
            if data.shape[0] < 2:
                continue
            # PC1 of this slice
            Hc = data - data.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
            PC1 = Hc @ Vt[0]
            r, _ = spearmanr(PC1, ns) if PC1.std() > 1e-9 else (float('nan'), None)
            within = float(np.mean([PC1[ns == n].std() for n in np.unique(ns) if (ns == n).sum() > 1]))
            print(f"  t={t:4d} (frac {t/T_actual:.2f}):  PC1 var={S[0]**2/(S**2).sum():.3f}  ρ(N)={r:+.3f}  within-N std={within:.4f}")

    # Spike gate stats
    if n_to_gate_alpha:
        print(f"\n--- Spike gate (post-sigmoid, sharpness=5) per N ---")
        Ns_g = sorted(n_to_gate_alpha.keys())
        for n in [Ns_g[0], Ns_g[len(Ns_g)//4], Ns_g[len(Ns_g)//2], Ns_g[3*len(Ns_g)//4], Ns_g[-1]]:
            vals = n_to_gate_alpha[n]
            print(f"  N={n:3d}:  mean gate={np.mean(vals):.3f}  std={np.std(vals):.3f}  n_samples={len(vals)}")


if __name__ == "__main__":
    main()
