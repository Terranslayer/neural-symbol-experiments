# scripts/v33_compare_layers.py
"""V33 layer-trace comparison between two checkpoints.

DIAG_FOR: v33

Captures activations at 6 forward-path stages for each ckpt and computes
N-information metrics so we can localize where N-discrimination breaks down.

Layers captured (all aggregated to (B_total, d) over N sweep):
  L0_input    : raw input signal,         mean over time            -> (B, 3)
  L1_cnn      : post dp_convs concat,     mean over time            -> (B, 12)
  L2_proj     : post dp_proj (V33 input), mean over time            -> (B, 16)
  L3_hmid     : V33 h trajectory at t=L/2                            -> (B, 16)
  L3_hend     : V33 h trajectory at t=L (feeds write_head)           -> (B, 16)
  L5_raw      : write_head raw output (pre-quantize)                 -> (B, 5)
  L5_q        : write_head quantized scratch                         -> (B, 5)

Metrics per layer:
  n_exp_R2    : fraction of activation variance linearly explained by N
                (rank-1 best-fit; high = layer carries linear N signal)
  pc1_pct     : top eigenvalue / sum(eigenvalues) (PCA collapse indicator)
  eff_rank    : (sum(λ))² / sum(λ²)  participation ratio
  rho_pc1_N   : Spearman ρ between leading PC score and N

Usage:
  python scripts/v33_compare_layers.py --ckpt-a <baseline> --ckpt-b <ablation> \\
    --L 200 --n-min 1 --n-max 30 --n-samples 32 --task successor_prediction
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import sample_training_batch
from scripts.inspect_checkpoint import load_agent


@torch.no_grad()
def capture_layers(agent, inputs, device):
    """Forward V33 pipeline and capture intermediate activations at 7 stages.

    inputs: (B, T, 3) on cpu or device.
    Returns dict layer_name -> (B, ..., d) tensor on cpu.
    """
    inputs = inputs.to(device)
    cfg = agent.scene_cfg
    L = cfg.L
    alpha_in = inputs[:, :L, :]  # (B, L, 3)

    # L0: raw input (signal channel only -> (B, L, 1))
    # We keep all 3 channels for completeness
    L0 = alpha_in

    # L1: CNN concat output, (B, L, 12) after transpose dance
    x_for_cnn = alpha_in.transpose(1, 2)  # (B, 3, L)
    conv_outs = [conv(x_for_cnn) for conv in agent.dp_convs]
    feat_cnn = torch.cat(conv_outs, dim=1).transpose(1, 2)  # (B, L, 12)
    L1 = feat_cnn

    # L2: post-projection (V33 substrate input), (B, L, 16)
    feat_proj = agent.dp_proj(feat_cnn)
    L2 = feat_proj

    # L3: V33 substrate forward, capture h at mid and end
    substrate = agent.v33_substrate
    layer = substrate.layers[0]
    B_, L_, _ = feat_proj.shape
    state = layer.initial_state(B_, device)
    h = state["h"]
    g_state = state["g"]
    dt = layer.dt()
    v_hat = layer.channels.unit_directions()
    E = layer.channels.E
    g_c = layer.channels.g()
    u_seq = layer.B_leak(feat_proj)
    VtV = v_hat @ v_hat.T

    h_mid = None
    h_end = None
    mid_t = L_ // 2
    for t in range(L_):
        V_per_gate = layer.project_V_per_gate(h)
        g_state = layer.gates.step(g_state, V_per_gate, dt)
        P = layer.channel_open_prob(g_state)
        gP = g_c.unsqueeze(0) * P
        b_eff = (gP * E.unsqueeze(0)) @ v_hat
        rhs = h + dt * (b_eff + u_seq[:, t])
        D_inv = 1.0 / gP.clamp(min=1e-3)
        M = torch.diag_embed(D_inv) + dt * VtV.unsqueeze(0)
        M_inv = torch.linalg.inv(M)
        VtRhs = rhs @ v_hat.T
        inner = (M_inv @ VtRhs.unsqueeze(-1)).squeeze(-1)
        correction = inner @ v_hat
        h = rhs - dt * correction
        if t == mid_t:
            h_mid = h.detach().clone()
        if t == L_ - 1:
            h_end = h.detach().clone()

    # L5: write_head raw + quantized
    raw, q = agent.write_head(
        h_end,
        agent.agent_cfg.quantize_levels,
        agent.agent_cfg.quantize_range,
    )

    return {
        "L0_input": L0.cpu(),
        "L1_cnn": L1.cpu(),
        "L2_proj": L2.cpu(),
        "L3_hmid": h_mid.cpu(),
        "L3_hend": h_end.cpu(),
        "L5_raw": raw.cpu(),
        "L5_q": q.cpu(),
    }


def collapse_temporal(X, mode="mean"):
    """X: (B, L, d) or (B, d). Returns (B, d) by mean-pool over time if temporal."""
    if X.ndim == 3:
        if mode == "mean":
            return X.mean(dim=1)
        elif mode == "last":
            return X[:, -1, :]
    return X


def metrics_per_layer(X_np, Ns_np):
    """X_np: (B, d) numpy. Ns_np: (B,) int. Returns dict of metrics."""
    B, d = X_np.shape
    X = X_np - X_np.mean(axis=0, keepdims=True)
    total_var = float((X ** 2).sum() / max(B - 1, 1))

    # N-explained: rank-1 linear regression Ns -> X
    Ns_c = (Ns_np.astype(np.float64) - Ns_np.mean()).reshape(-1, 1)
    denom = float((Ns_c.T @ Ns_c).item())
    if denom > 1e-12:
        beta = (Ns_c.T @ X) / denom  # (1, d)
        X_pred = Ns_c @ beta
        explained = float((X_pred ** 2).sum() / max(B - 1, 1))
        n_exp_R2 = explained / max(total_var, 1e-12)
    else:
        n_exp_R2 = 0.0
    n_exp_R2 = float(np.clip(n_exp_R2, 0.0, 1.0))

    # PCA via SVD
    if total_var < 1e-12:
        return {
            "total_var": 0.0,
            "n_exp_R2": 0.0,
            "pc1_pct": float("nan"),
            "eff_rank": float("nan"),
            "rho_pc1_N": float("nan"),
        }
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eigvals = S ** 2 / max(B - 1, 1)
    pc1_pct = float(eigvals[0] / (eigvals.sum() + 1e-12))
    eff_rank = float((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-12))
    pc1_scores = U[:, 0] * S[0]
    rho_pc1_N = float(spearmanr(pc1_scores, Ns_np).correlation)

    return {
        "total_var": total_var,
        "n_exp_R2": n_exp_R2,
        "pc1_pct": pc1_pct,
        "eff_rank": eff_rank,
        "rho_pc1_N": rho_pc1_N,
    }


def run_capture_one_ckpt(ckpt_path, N_values, n_samples_per_N, L, task, seed):
    agent, agent_cfg, scene_cfg = load_agent(
        ckpt_path, q=3, qrange="unit", L=L, complex_world=True,
    )
    device = next(agent.parameters()).device
    scene_cfg.task = task
    agent.train(False)
    rng = np.random.default_rng(seed)

    accum = {layer: [] for layer in
             ["L0_input", "L1_cnn", "L2_proj", "L3_hmid", "L3_hend", "L5_raw", "L5_q"]}
    Ns = []
    for N in N_values:
        saved_dist = scene_cfg.alpha_distribution
        scene_cfg.alpha_distribution = "uniform"
        inputs, _, _, _ = sample_training_batch(
            batch_size=n_samples_per_N, n_max_stage=N, config=scene_cfg, rng=rng,
        )
        scene_cfg.alpha_distribution = saved_dist
        caps = capture_layers(agent, inputs, device)
        for layer_name, val in caps.items():
            accum[layer_name].append(val)
        Ns.extend([N] * n_samples_per_N)

    # Concatenate per-layer
    layer_metrics = {}
    for layer_name, chunks in accum.items():
        X_full = torch.cat(chunks, dim=0)  # (B_total, ...)
        X_collapsed = collapse_temporal(X_full, mode="mean")
        X_np = X_collapsed.numpy()
        Ns_np = np.array(Ns)
        layer_metrics[layer_name] = metrics_per_layer(X_np, Ns_np)
    return layer_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, type=Path,
                        help="Baseline ckpt (e.g. lambda=0)")
    parser.add_argument("--ckpt-b", required=True, type=Path,
                        help="Ablation ckpt (e.g. lambda=0.1)")
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--task", default="successor_prediction")
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    N_values = list(range(args.n_min, args.n_max + 1))
    print(f"# v33_compare_layers")
    print(f"# ckpt_a: {args.ckpt_a}")
    print(f"# ckpt_b: {args.ckpt_b}")
    print(f"# N sweep: {args.n_min}..{args.n_max}, {args.n_samples} samples/N, L={args.L}")
    print()

    print("Running ckpt-a (baseline)...")
    m_a = run_capture_one_ckpt(
        args.ckpt_a, N_values, args.n_samples, args.L, args.task, args.seed,
    )
    print("Running ckpt-b (ablation)...")
    m_b = run_capture_one_ckpt(
        args.ckpt_b, N_values, args.n_samples, args.L, args.task, args.seed,
    )

    layer_order = ["L0_input", "L1_cnn", "L2_proj",
                   "L3_hmid", "L3_hend", "L5_raw", "L5_q"]

    print("\n" + "=" * 90)
    print("Layer-by-layer N-information comparison")
    print("=" * 90)
    print(f"{'Layer':<10} | {'metric':<11} | {'ckpt-a (λ=0)':>14} | "
          f"{'ckpt-b (λ=0.1)':>14} | {'delta':>10}")
    print("-" * 90)
    for layer in layer_order:
        a = m_a[layer]
        b = m_b[layer]
        for key in ["total_var", "n_exp_R2", "pc1_pct", "eff_rank", "rho_pc1_N"]:
            va = a[key]
            vb = b[key]
            delta = vb - va if (np.isfinite(va) and np.isfinite(vb)) else float("nan")
            print(f"{layer:<10} | {key:<11} | {va:>14.4f} | {vb:>14.4f} | {delta:>+10.4f}")
        print("-" * 90)

    # Compact one-liner per layer for n_exp_R2
    print("\n=== n_exp_R2 chain (where does N-info live?) ===")
    print(f"{'Layer':<10} | {'a (λ=0)':>10} | {'b (λ=0.1)':>10}")
    for layer in layer_order:
        print(f"{layer:<10} | {m_a[layer]['n_exp_R2']:>10.4f} | {m_b[layer]['n_exp_R2']:>10.4f}")


if __name__ == "__main__":
    main()
