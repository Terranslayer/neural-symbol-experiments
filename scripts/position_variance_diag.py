"""Position-variance diagnostic for V6 cnn_pfc_only.

Question: V6 ckpt uses 14 distinct scratch codes for N=1 (per code_n_mapping).
Is this because (a) different spike POSITIONS produce different codes, even at
the same N, or (b) just random variability?

Method:
  - Manually construct alpha signals at controlled spike positions
  - Forward through V6 cnn_pfc_only model
  - Capture: cnn_α_summary (B, d) + scratch (B, W) per sample
  - Report:
    * For each N: code distribution per spike position
    * cnn_α_summary variance across positions vs across runs
    * Are scratch codes deterministic given positions?

If (a) confirmed: scratch encoding is position-sensitive → not pure cardinality
detection. Need position-invariant aggregation in CNN→PFC pathway.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.core.scene import SceneConfig, comparison_label
from scripts.inspect_checkpoint import load_agent


# Match _scan_world_complex thresholds + intensities
FOOD_CENTER_THRESH = 0.9
FOOD_NEIGHBOR_THRESH = 0.8
BACKGROUND_LOW = 0.0
BACKGROUND_HIGH = 0.2


def make_alpha_signal_at(positions, L, rng):
    """Manually construct alpha signal with food spikes at exact positions.
    Mirrors _scan_world_complex placement (peak + neighbors)."""
    sig = rng.uniform(BACKGROUND_LOW, BACKGROUND_HIGH, size=L).astype(np.float32)
    for c in positions:
        sig[c] = rng.uniform(FOOD_CENTER_THRESH, 1.0)
        if c - 1 >= 0:
            sig[c - 1] = rng.uniform(FOOD_NEIGHBOR_THRESH, FOOD_CENTER_THRESH)
        if c + 1 < L:
            sig[c + 1] = rng.uniform(FOOD_NEIGHBOR_THRESH, FOOD_CENTER_THRESH)
    return sig


def build_input_tensor(alpha_signal, beta_signal, scene_cfg):
    """Build (T, 3) input tensor with given alpha/beta signals."""
    T = scene_cfg.total_timesteps
    L = scene_cfg.L
    inp = np.zeros((T, 3), dtype=np.float32)
    inp[0:L, 0] = alpha_signal
    inp[0:L, 1] = 1
    # write/forget zone: valid=0
    inp[L:scene_cfg.forget_end, 1] = 0
    # K compare blocks
    for i in range(scene_cfg.K):
        rstart, beta_start, compare_step = scene_cfg.compare_block_phases(i)
        inp[rstart:beta_start, 1] = 1
        inp[rstart:beta_start, 2] = 1
        inp[beta_start:compare_step, 0] = beta_signal
        inp[beta_start:compare_step, 1] = 1
        inp[beta_start:compare_step, 2] = 1
        inp[compare_step, 0] = 0
        inp[compare_step, 1] = 0
        inp[compare_step, 2] = 1
    return inp


def quantize_code(scratch, q_levels=3):
    """Convert continuous scratch to quantized levels (0, 0.5, 1)."""
    levels = np.linspace(0, 1, q_levels)
    return tuple(levels[np.abs(scratch[:, None] - levels[None, :]).argmin(axis=1)].tolist())


def run_n1_position_scan(agent, scene_cfg, device, args):
    """N=1: scan spike position across L. Each position run K times for noise."""
    L = scene_cfg.L
    print(f"\n{'='*72}\n=== N=1 position scan (L={L}, K_per_pos={args.k_per_pos}) ===\n{'='*72}")
    positions_to_test = list(range(2, L - 1, 2))  # every 2 cells, avoid edges
    results = []  # (position, code_tuple, cnn_alpha_summary)
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for pos in positions_to_test:
            for k in range(args.k_per_pos):
                # alpha = single spike at pos
                alpha_sig = make_alpha_signal_at([pos], L, rng)
                # beta: doesn't matter for scratch — use same N=1 spike
                beta_pos = (pos + 10) % (L - 2) if (pos + 10) % (L - 2) > 1 else 5
                beta_sig = make_alpha_signal_at([beta_pos], L, rng)
                inp = build_input_tensor(alpha_sig, beta_sig, scene_cfg)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                ci = torch.zeros(1, scene_cfg.K, dtype=torch.long, device=device)
                _, scratch, _ = agent(inp_t, ci)
                cnn_alpha = agent._last_cnn_alpha_summary[0].cpu().numpy()
                code = quantize_code(scratch[0].cpu().numpy())
                results.append((pos, code, cnn_alpha))

    # Aggregate
    pos_to_codes = defaultdict(list)
    for pos, code, _ in results:
        pos_to_codes[pos].append(code)
    cnn_alpha_per_pos = defaultdict(list)
    for pos, _, ca in results:
        cnn_alpha_per_pos[pos].append(ca)

    # Per-position dominant code + diversity
    print(f"\n{'pos':>4}  {'#runs':>6}  {'#distinct_codes':>16}  {'most_common_code':<35}  {'freq':>5}")
    for pos in positions_to_test:
        codes = pos_to_codes[pos]
        cnt = Counter(codes)
        mc, mc_freq = cnt.most_common(1)[0]
        print(f"  {pos:>4}  {len(codes):>6}  {len(cnt):>16}  {str(list(mc)):<35}  {mc_freq/len(codes):>5.2f}")

    # Global stats
    all_codes = [code for _, code, _ in results]
    distinct = set(all_codes)
    print(f"\n  TOTAL (N=1): {len(results)} samples, {len(distinct)} distinct codes")

    # cnn_alpha_summary variance across positions
    cnn_means = np.stack([np.mean(cnn_alpha_per_pos[p], axis=0) for p in positions_to_test])  # (P, d)
    cnn_within = np.mean([np.std(cnn_alpha_per_pos[p], axis=0).mean() for p in positions_to_test])
    cnn_between = cnn_means.std(axis=0).mean()
    print(f"\n  cnn_alpha_summary stats:")
    print(f"    within-position std (avg over pos+dim): {cnn_within:.4f}")
    print(f"    between-position std (avg over dim):    {cnn_between:.4f}")
    print(f"    ratio between/within: {cnn_between/max(cnn_within,1e-9):.2f}")
    print(f"    >1 = position dominates noise (sensitive); <1 = noise dominates (invariant)")
    return results


def run_n2_n3_position_scan(agent, scene_cfg, device, args, N):
    """N=2 or N=3: pick a few canonical position configurations."""
    L = scene_cfg.L
    print(f"\n{'='*72}\n=== N={N} position configurations (L={L}) ===\n{'='*72}")
    rng = np.random.default_rng(args.seed + N * 1000)

    # Canonical configurations
    if N == 2:
        configs = [
            ("tight-left",      [5, 10]),
            ("tight-mid",       [22, 27]),
            ("tight-right",     [40, 45]),
            ("medium-spread",   [10, 30]),
            ("wide-spread",     [5, 45]),
            ("medium-mid",      [15, 35]),
        ]
    else:  # N=3
        configs = [
            ("tight-cluster",     [5, 10, 15]),
            ("mid-cluster",       [20, 25, 30]),
            ("right-cluster",     [35, 40, 45]),
            ("uniform-spread",    [10, 25, 40]),
            ("left-pair-right",   [5, 10, 45]),
            ("left-mid-right",    [10, 25, 40]),
            ("max-spread",        [5, 25, 45]),
        ]

    results = []
    with torch.no_grad():
        for name, positions in configs:
            for k in range(args.k_per_pos):
                alpha_sig = make_alpha_signal_at(positions, L, rng)
                # beta: same N at different positions (avoid contaminating compare)
                beta_pos = [(p + 7) % (L - 2) if (p + 7) % (L - 2) > 1 else 4 for p in positions]
                beta_pos = sorted(set(beta_pos))[:N]
                if len(beta_pos) < N:
                    beta_pos = positions[:N]  # fallback
                beta_sig = make_alpha_signal_at(beta_pos, L, rng)
                inp = build_input_tensor(alpha_sig, beta_sig, scene_cfg)
                inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
                ci = torch.zeros(1, scene_cfg.K, dtype=torch.long, device=device)
                _, scratch, _ = agent(inp_t, ci)
                cnn_alpha = agent._last_cnn_alpha_summary[0].cpu().numpy()
                code = quantize_code(scratch[0].cpu().numpy())
                results.append((name, positions, code, cnn_alpha))

    # Per-config dominant code
    cfg_to_codes = defaultdict(list)
    for name, _, code, _ in results:
        cfg_to_codes[name].append(code)

    print(f"\n  {'config':<20}  {'positions':<20}  {'#runs':>6}  {'#distinct':>9}  {'dominant_code':<35}  {'freq':>5}")
    for name, positions in configs:
        codes = cfg_to_codes[name]
        cnt = Counter(codes)
        mc, mc_freq = cnt.most_common(1)[0]
        print(f"  {name:<20}  {str(positions):<20}  {len(codes):>6}  {len(cnt):>9}  {str(list(mc)):<35}  {mc_freq/len(codes):>5.2f}")

    all_codes = [code for _, _, code, _ in results]
    distinct = set(all_codes)
    print(f"\n  TOTAL (N={N}): {len(results)} samples, {len(distinct)} distinct codes")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, default=51)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--cnn-pfc-only", action="store_true", default=True)
    p.add_argument("--k-per-pos", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="/root/symbolicai/research/pos_variance_diag.json")
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=args.complex,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=args.cnn_pfc_only,
    )
    device = next(agent.parameters()).device

    n1_results = run_n1_position_scan(agent, scene_cfg, device, args)
    n2_results = run_n2_n3_position_scan(agent, scene_cfg, device, args, N=2)
    n3_results = run_n2_n3_position_scan(agent, scene_cfg, device, args, N=3)

    out = {
        "config": vars(args),
        "n1_total_distinct": len(set(c for _, c, _ in n1_results)),
        "n2_total_distinct": len(set(c for _, _, c, _ in n2_results)),
        "n3_total_distinct": len(set(c for _, _, c, _ in n3_results)),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
