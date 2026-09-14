"""V35 inverted-tuning diagnostic: decompose Layer A's per-timestep drive per N
to find WHY spike count is inverted with N (N=1 fires ~38, N=5 fires ~3).

Layer A net drive per timestep (from v35.py HHSSMLayer1D.forward):
    drive = channel_term + u_seq + ip_bias
    channel_term = sum_c (g_c * P_c * E_c * ch_sign_c)      # gate-driven
    u_seq        = B_leak(feat_alpha)                       # input-driven
    ip_bias      = scalar tonic
    h_{t} = (h_{t-1} + dt*drive) / (1 + dt*a_eff);  fire if h>=theta_a, then h-=theta_a

Reuses v35_inspect.py loading. Captures via layer_A(feat, capture_for_pc=True),
then recomputes channel_term from the captured gate_seq + frozen channel params.

Usage:
  PYTHONPATH=. python scripts/v35_inverted_tuning_diag.py <ckpt> [--n-list 1,2,3,4,5] [--per-n 64]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from backend.core.scene import SceneConfig, build_episode, sample_beta_count
from backend.core.mamba_agent import MambaAgent, MambaAgentConfig


def _get_attr(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--n-list", type=str, default="1,2,3,4,5")
    ap.add_argument("--per-n", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    agent_cfg = ckpt["agent_cfg"]
    scene_cfg = ckpt["scene_cfg"]
    if isinstance(agent_cfg, dict):
        agent_cfg = MambaAgentConfig(**agent_cfg)
    if isinstance(scene_cfg, dict):
        scene_cfg = SceneConfig(**scene_cfg)

    agent = MambaAgent(agent_cfg, scene_cfg)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent = agent.to(device)
    agent.train(False)

    layer_A = agent.v35_substrate.cascade.layer_A
    L = scene_cfg.L
    theta_a = float(layer_A._theta())
    ip_bias = float(layer_A.ip_bias)
    # Frozen channel params for channel_term recomputation (match v35.py forward).
    g_c = layer_A.log_g.exp()                 # (4,)
    E = layer_A.E                             # (4,)
    ch_sign = layer_A.ch_sign                 # (4,)
    B_leak_w = layer_A.B_leak.weight.detach()  # (1, d_input)

    def channel_open_prob(g):
        m_Na, h_Na, n_K, s_NMDA = g.unbind(dim=-1)
        P_Na = m_Na * h_Na
        return torch.stack([P_Na, n_K, s_NMDA, torch.ones_like(P_Na)], dim=-1)

    print(f"=== V35 inverted-tuning diag: {args.ckpt} ===")
    print(f"theta_a = {theta_a:.4f}   ip_bias_A = {ip_bias:.4f}   L = {L}")
    print(f"B_leak_A weight: norm={float(B_leak_w.norm()):.3f}  "
          f"mean={float(B_leak_w.mean()):+.4f}  min={float(B_leak_w.min()):+.4f}  max={float(B_leak_w.max()):+.4f}")
    print(f"log_g.exp()={g_c.detach().cpu().numpy().round(4)}  E={E.detach().cpu().numpy().round(3)}  ch_sign={ch_sign.cpu().numpy()}")
    print()

    n_list = [int(x) for x in args.n_list.split(",")]
    rng = np.random.default_rng(seed=args.seed)

    hdr = (f"{'N':>3} | {'inE':>6} {'|feat|':>6} | {'u_in':>7} {'chan':>7} {'ipb':>7} {'netdr':>7} | "
           f"{'h_mn':>6} {'h_mx':>6} | {'mNa':>5} {'hNa':>5} {'nK':>5} {'sNMDA':>5} {'P_Na':>5} | {'spk':>5}")
    print(hdr)
    print("-" * len(hdr))

    traces = {}  # N -> (h_t, drive_t, spike_t, sig_t) for one representative episode
    with torch.no_grad():
        for N in n_list:
            inputs_list = []
            for _ in range(args.per_n):
                beta_counts = [sample_beta_count(N, max(n_list), scene_cfg, rng) for _ in range(scene_cfg.K)]
                inp, _l, _c, _m = build_episode(N, beta_counts, scene_cfg, rng)
                inputs_list.append(inp)
            inputs = torch.stack(inputs_list).to(device)  # (B,T,3)
            inputs[:, :, 1:3] = 0.0
            alpha_in = inputs[:, :L, :]                    # raw signal (ch0)
            x = alpha_in.transpose(1, 2)
            conv_outs = [conv(x) for conv in agent.dp_convs]
            feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
            feat_alpha = agent.dp_proj(feat)               # (B, L, d_input)

            _h, spike_seq, _res, _state, cap = layer_A(feat_alpha, capture_for_pc=True)
            gate_seq = cap["gate_seq"]            # (B, L, 4)
            u_seq = cap["u_seq"].squeeze(-1)      # (B, L)
            h_seq = cap["h_seq"].squeeze(-1)      # (B, L)

            # channel_term per timestep from captured gates
            P = channel_open_prob(gate_seq)                       # (B,L,4)
            gP = g_c.view(1, 1, -1) * P
            channel_term = (gP * E.view(1, 1, -1) * ch_sign.view(1, 1, -1)).sum(-1)  # (B,L)
            net_drive = channel_term + u_seq + ip_bias

            in_energy = alpha_in[:, :, 0].sum(dim=1).mean()        # total raw signal per episode
            feat_abs = feat_alpha.abs().mean()
            mNa, hNa, nK, sNMDA = gate_seq.unbind(-1)
            P_Na = (mNa * hNa)
            spk = spike_seq.sum(dim=1)

            print(f"{N:>3} | {float(in_energy):>6.2f} {float(feat_abs):>6.3f} | "
                  f"{float(u_seq.mean()):>+7.3f} {float(channel_term.mean()):>+7.3f} {ip_bias:>+7.2f} {float(net_drive.mean()):>+7.3f} | "
                  f"{float(h_seq.mean()):>+6.2f} {float(h_seq.max()):>+6.2f} | "
                  f"{float(mNa.mean()):>5.3f} {float(hNa.mean()):>5.3f} {float(nK.mean()):>5.3f} {float(sNMDA.mean()):>5.3f} {float(P_Na.mean()):>5.3f} | "
                  f"{float(spk.mean()):>5.2f}")

            # save trace for the representative (median-spike) episode
            spk_np = spk.cpu().numpy()
            idx = int(np.argsort(spk_np)[len(spk_np) // 2])
            traces[N] = (
                h_seq[idx].cpu().numpy(), net_drive[idx].cpu().numpy(),
                spike_seq[idx].cpu().numpy(), alpha_in[idx, :, 0].cpu().numpy(),
                u_seq[idx].cpu().numpy(), channel_term[idx].cpu().numpy(),
            )

    # Time-resolved trace for N=min and N=max (downsampled to ~25 cols)
    for N in (n_list[0], n_list[-1]):
        h_t, dr_t, sp_t, sig_t, u_t, ch_t = traces[N]
        step = max(1, L // 25)
        print(f"\n=== Time trace N={N} (every {step} steps; sig=raw signal, u=input drive, ch=channel drive, h=state, *=spike) ===")
        idxs = list(range(0, L, step))
        print("  t   : " + " ".join(f"{t:>5d}" for t in idxs))
        print("  sig : " + " ".join(f"{sig_t[t]:>5.2f}" for t in idxs))
        print("  u   : " + " ".join(f"{u_t[t]:>+5.1f}" for t in idxs))
        print("  ch  : " + " ".join(f"{ch_t[t]:>+5.1f}" for t in idxs))
        print("  h   : " + " ".join(f"{h_t[t]:>+5.1f}" for t in idxs))
        print("  spk : " + " ".join(f"{'  *  ' if sp_t[t] > 0.5 else '  .  '}" for t in idxs))


if __name__ == "__main__":
    main()
