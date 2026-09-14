# scripts/v33_full_diag.py
"""V33 full internal diagnostic.

DIAG_FOR: v33

Captures internal state from a single forward pass and reports:
  - v_c pairwise cos overlap (channel coupling strength)
  - E_c values per channel (with init reference for drift)
  - g_c values (which channel(s) SGD prioritized)
  - tau_i per gate (learned vs init)
  - Channel activation trace: P_c(t) for each channel across episode
  - Gate trajectory: g_i(t) for each gate
  - h trajectory PCA: top-3 PCs of h_seq, spearman with N

Usage:
  python scripts/v33_full_diag.py --ckpt <path> --task successor_prediction \\
    --n-values 5,15,30,60,90 --batch-size 8 --L <episode_length>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def capture_v33_internals(agent, batch_inputs, device):
    """Run forward and capture per-timestep internal state of the HH-SSM layer.

    Reproduces HHSSMLayer.forward step-by-step to record per-t internals.
    Math mirrors the fixed HHSSMLayer.forward exactly:
      - gP.clamp(min=1e-3) for numerical stability
      - dt() uses clamped log_dt (min=log(1e-2), max=log(10.0))

    Returns dict with keys:
      P (B, L, 4)   — channel open probabilities per timestep
      g (B, L, 4)   — gate state per timestep
      h (B, L, d)   — hidden state trajectory
      V (B, L, 4)   — per-channel voltage projection V_c = v_hat_c^T h
    """
    inputs = batch_inputs.to(device)

    substrate = agent.v33_substrate
    layer = substrate.layers[0]

    cfg = agent.scene_cfg
    L = cfg.L
    alpha_in = inputs[:, :L, :]
    x = alpha_in.transpose(1, 2)
    conv_outs = [conv(x) for conv in agent.dp_convs]
    feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
    x_seq = agent.dp_proj(feat)

    B_, L_, _ = x_seq.shape
    state = layer.initial_state(B_, x_seq.device)
    h = state["h"]
    g_state = state["g"]

    dt = layer.dt()
    v_hat = layer.channels.unit_directions()
    E = layer.channels.E
    g_c = layer.channels.g()
    u_seq = layer.B_leak(x_seq)
    VtV = v_hat @ v_hat.T

    h_out, P_out, g_out, V_out = [], [], [], []
    for t in range(L_):
        V_per_gate = layer.project_V_per_gate(h)
        g_state = layer.gates.step(g_state, V_per_gate, dt)
        P = layer.channel_open_prob(g_state)
        V_per_channel = h @ v_hat.T
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
        h_out.append(h.detach().cpu())
        P_out.append(P.detach().cpu())
        g_out.append(g_state.detach().cpu())
        V_out.append(V_per_channel.detach().cpu())

    return {
        "P": torch.stack(P_out, dim=1),
        "g": torch.stack(g_out, dim=1),
        "h": torch.stack(h_out, dim=1),
        "V": torch.stack(V_out, dim=1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--task", default="successor_prediction")
    parser.add_argument("--L", type=int, required=True,
                        help="Episode length (must match training --curriculum-L)")
    parser.add_argument("--n-values", default="5,15,30,60,90")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
    )
    device = next(agent.parameters()).device
    scene_cfg.task = args.task
    agent.train(False)

    if agent.v33_substrate is None:
        print("ERROR: ckpt is not a V33 model (agent.v33_substrate is None)")
        return

    substrate = agent.v33_substrate
    layer = substrate.layers[0]
    overlap = layer.channels.channel_overlap().detach().cpu().numpy()
    E_c = layer.channels.E.detach().cpu().numpy()
    g_c = layer.channels.g().detach().cpu().numpy()
    tau = layer.gates.tau().detach().cpu().numpy()

    print("=== V33 Static Diagnostics ===")
    print(f"\nChannel cos overlap matrix (Na, K, NMDA, leak):")
    print(np.array2string(overlap, precision=3))
    print(f"\nE_c (Na, K, NMDA, leak):     {E_c}")
    print(f"g_c (Na, K, NMDA, leak):     {g_c}")
    print(f"tau_i (m_Na, h_Na, n_K, s_NMDA): {tau}")

    N_values = [int(x) for x in args.n_values.split(",")]
    rng = np.random.default_rng(42)
    all_h_finals, all_Ns = [], []
    print("\n=== Per-N Channel Activation Pattern (mean P_c at t=L) ===")
    print(f"{'N':>4} | P_Na   P_K     P_NMDA  P_leak")
    for N in N_values:
        with torch.no_grad():
            saved_dist = scene_cfg.alpha_distribution
            scene_cfg.alpha_distribution = "uniform"
            inputs, _, ci, metas = sample_training_batch(
                batch_size=args.batch_size, n_max_stage=N, config=scene_cfg, rng=rng,
            )
            scene_cfg.alpha_distribution = saved_dist

            cap = capture_v33_internals(agent, inputs, device)
            P_final = cap["P"][:, -1, :].mean(dim=0).numpy()
            print(f"{N:>4} | {P_final[0]:.3f}  {P_final[1]:.3f}  {P_final[2]:.3f}  {P_final[3]:.3f}")
            h_final = cap["h"][:, -1, :].numpy()
            for b in range(h_final.shape[0]):
                all_h_finals.append(h_final[b])
                all_Ns.append(N)

    H = np.stack(all_h_finals, axis=0)
    H_centered = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    pc_scores = U * S
    Ns_arr = np.array(all_Ns)

    print("\n=== h PCA — Spearman rho(PC, N) for top 3 PCs ===")
    for k in range(3):
        rho, _ = spearmanr(pc_scores[:, k], Ns_arr)
        print(f"  PC{k+1}: rho(PC, N) = {rho:.3f}, S^2/sum = {S[k]**2 / (S**2).sum():.3f}")


if __name__ == "__main__":
    main()
