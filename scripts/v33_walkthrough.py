# scripts/v33_walkthrough.py
"""V33 forward+backward numerical walkthrough.

Loads a V33 ckpt, runs a SINGLE-EPISODE forward+backward, and prints
intermediate tensors at every distinct layer (raw input -> CNN -> dp_proj
-> HHSSMLayer per-t snapshots -> write_head per-cell -> predict_head ->
loss -> gradients).

Designed to feed a hand-written walkthrough markdown.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import sample_training_batch
from scripts.inspect_checkpoint import load_agent


def fmt(t, max_elems=8):
    if isinstance(t, torch.Tensor):
        a = t.detach().cpu().numpy()
    else:
        a = np.asarray(t)
    flat = a.flatten()
    head = flat[:max_elems]
    s = ", ".join(f"{v:+.4f}" for v in head)
    if flat.size > max_elems:
        s += f", ...(+{flat.size - max_elems})"
    nrm = float(np.linalg.norm(flat))
    return f"[{s}] shape={tuple(a.shape)} ‖·‖₂={nrm:.4f}"


def banner(s):
    print()
    print("=" * 80)
    print(f"  {s}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--target-n-alpha", type=int, default=5)
    parser.add_argument("--target-k", type=int, default=5,
                        help="Target N_beta - N_alpha (e.g. +5)")
    parser.add_argument("--max-resample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
    )
    device = next(agent.parameters()).device
    scene_cfg.task = "successor_prediction"
    agent.train()

    # Resample until we find an episode with the target N_alpha / k
    rng = np.random.default_rng(args.seed)
    found = None
    for trial in range(args.max_resample):
        saved = scene_cfg.alpha_distribution
        scene_cfg.alpha_distribution = "uniform"
        inputs, labels, ci, metas = sample_training_batch(
            batch_size=1, n_max_stage=30, config=scene_cfg, rng=rng,
        )
        scene_cfg.alpha_distribution = saved
        n_a = metas[0]["N_alpha"]
        n_b = metas[0]["beta_counts"][0]
        if n_a == args.target_n_alpha and (n_b - n_a) == args.target_k:
            found = (inputs, labels, ci, metas)
            break
    if found is None:
        print(f"WARN: could not find N_a={args.target_n_alpha}, k={args.target_k} "
              f"after {args.max_resample} tries. Using last sample "
              f"N_a={n_a}, k={n_b - n_a}.")
        found = (inputs, labels, ci, metas)

    inputs, labels, ci, metas = found
    meta = metas[0]
    n_a = meta["N_alpha"]
    n_b = meta["beta_counts"][0]
    print(f"Episode: N_alpha={n_a}, N_beta={n_b}, k_target={n_b - n_a}")

    inputs = inputs.to(device)

    banner("L0 — Raw input signal")
    L = scene_cfg.L
    print(f"scene_cfg: L={L}, W={scene_cfg.W}, T_gap={scene_cfg.T_gap}, "
          f"K={scene_cfg.K}, input_dim={inputs.shape[-1]}")
    print(f"Total episode length T = {inputs.shape[1]}")
    alpha_in = inputs[:, :L, :]
    print(f"alpha_in (B=1, L={L}, 3):")
    print(f"  channel-0 (signal):  {fmt(alpha_in[0, :, 0], max_elems=10)}")
    print(f"  channel-1 (phase):   {fmt(alpha_in[0, :, 1], max_elems=10)}")
    print(f"  channel-2 (mask):    {fmt(alpha_in[0, :, 2], max_elems=10)}")
    print(f"  alpha_centers: {meta.get('alpha_centers', '(none)')}")
    # Compute summary: peak count via thresholding
    sig = alpha_in[0, :, 0].cpu().numpy()
    peaks = int(np.sum(sig > 0.5))
    print(f"  > 0.5 count (rough peak count): {peaks}")

    banner("L1 — dual_pathway CNN (3 kernels: 5, 21, 51)")
    print("Param: agent.dp_convs[0..2] = Conv1d(in=3, out=4, k={5,21,51}, padding=k//2)")
    for ki, conv in enumerate(agent.dp_convs):
        w = conv.weight  # (4, 3, k)
        b = conv.bias
        print(f"  dp_convs[{ki}] kernel={w.shape[-1]}: W shape={tuple(w.shape)}  "
              f"W ‖·‖={float(w.norm()):.4f}  b={fmt(b) if b is not None else 'None'}")
    x_for_cnn = alpha_in.transpose(1, 2)
    conv_outs = [conv(x_for_cnn) for conv in agent.dp_convs]
    for i, out in enumerate(conv_outs):
        print(f"  conv_outs[{i}] shape={tuple(out.shape)}  ‖·‖={float(out.norm()):.4f}")
    feat_cnn = torch.cat(conv_outs, dim=1).transpose(1, 2)  # (1, L, 12)
    print(f"feat_cnn (concat) shape={tuple(feat_cnn.shape)}  "
          f"‖·‖={float(feat_cnn.norm()):.4f}")
    print(f"  last timestep (12 ch): {fmt(feat_cnn[0, -1, :])}")

    banner("L2 — dp_proj Linear(12, 16)")
    proj = agent.dp_proj
    print(f"W shape={tuple(proj.weight.shape)} ‖·‖={float(proj.weight.norm()):.4f}")
    if proj.bias is not None:
        print(f"b shape={tuple(proj.bias.shape)} {fmt(proj.bias)}")
    feat_proj = proj(feat_cnn)  # (1, L, 16)
    print(f"feat_alpha (V33 substrate input) shape={tuple(feat_proj.shape)}  "
          f"‖·‖={float(feat_proj.norm()):.4f}")
    print(f"  last timestep: {fmt(feat_proj[0, -1, :])}")

    banner("L3 — V33 HH-SSM substrate (single layer)")
    substrate = agent.v33_substrate
    layer = substrate.layers[0]
    v_hat = layer.channels.unit_directions()
    E = layer.channels.E
    log_g = layer.channels.log_g
    g_c = layer.channels.g()
    w_g = layer.gates.w
    b_g = layer.gates.b
    log_tau = layer.gates.log_tau
    tau = layer.gates.tau()
    polarity = layer.gates.polarity
    gate_ch = layer.gates.gate_channel_idx
    dt = layer.dt()
    B_leak = layer.B_leak
    log_dt = layer.log_dt

    print("Channel params (Na, K, NMDA, leak):")
    print(f"  log_g:       {fmt(log_g)}")
    print(f"  g_c=exp(log_g): {fmt(g_c)}")
    print(f"  E_c:         {fmt(E)}")
    print(f"  v_hat (4x16):")
    for c, name in enumerate(["Na", "K", "NMDA", "leak"]):
        print(f"    v̂_{name}: {fmt(v_hat[c])}")
    cos_overlap = (v_hat @ v_hat.T).detach().cpu().numpy()
    print(f"  cos(v_c, v_c') matrix:")
    for c in range(4):
        print(f"    {cos_overlap[c]}")

    print("Gate params (m_Na, h_Na, n_K, s_NMDA):")
    print(f"  polarity (frozen): {fmt(polarity)}")
    print(f"  w_g:               {fmt(w_g)}")
    print(f"  b_g:               {fmt(b_g)}")
    print(f"  log_tau:           {fmt(log_tau)}")
    print(f"  tau_i=exp(log_tau): {fmt(tau)}")
    print(f"  gate_channel_idx:  {gate_ch.cpu().numpy()}")

    print(f"log_dt={float(log_dt.item()):+.4f}  dt=exp(log_dt)={float(dt.item()):.4f}")
    print(f"B_leak: Linear(d_in=16, d_out=16, bias=False)  "
          f"W shape={tuple(B_leak.weight.shape)}  ‖·‖={float(B_leak.weight.norm()):.4f}")

    # Replay HHSSMLayer.forward and capture snapshots
    x_seq = feat_proj
    B_, L_, _ = x_seq.shape
    state = layer.initial_state(B_, device)
    h = state["h"]
    g_state = state["g"]
    u_seq = B_leak(x_seq) if not layer.input_to_gate else torch.zeros_like(B_leak(x_seq))
    VtV = v_hat @ v_hat.T
    snap_ts = [0, L_ // 2, L_ - 1]
    snapshots = {}

    for t in range(L_):
        V_per_channel = h @ v_hat.T
        V_per_gate = V_per_channel[:, gate_ch]
        g_inf = torch.sigmoid(polarity * (w_g * V_per_gate + b_g))
        g_state_new = g_state + dt * (g_inf - g_state) / tau
        P = layer.channel_open_prob(g_state_new)
        gP = g_c.unsqueeze(0) * P
        b_eff = (gP * E.unsqueeze(0)) @ v_hat
        rhs = h + dt * (b_eff + u_seq[:, t])
        D_inv = 1.0 / gP.clamp(min=1e-3)
        M = torch.diag_embed(D_inv) + dt * VtV.unsqueeze(0)
        M_inv = torch.linalg.inv(M)
        VtRhs = rhs @ v_hat.T
        inner = (M_inv @ VtRhs.unsqueeze(-1)).squeeze(-1)
        correction = inner @ v_hat
        h_new = rhs - dt * correction

        if t in snap_ts:
            snapshots[t] = dict(
                V_per_channel=V_per_channel.detach().clone(),
                V_per_gate=V_per_gate.detach().clone(),
                g_inf=g_inf.detach().clone(),
                g_prev=g_state.detach().clone(),
                g_new=g_state_new.detach().clone(),
                P=P.detach().clone(),
                gP=gP.detach().clone(),
                b_eff=b_eff.detach().clone(),
                u_t=u_seq[:, t].detach().clone(),
                rhs=rhs.detach().clone(),
                M_inv=M_inv.detach().clone(),
                correction=correction.detach().clone(),
                h_new=h_new.detach().clone(),
            )

        g_state = g_state_new
        h = h_new

    h_alpha_end = h
    print(f"\n--- Snapshots at t ∈ {snap_ts} ---")
    for t in snap_ts:
        s = snapshots[t]
        print(f"\n  [t={t}]")
        print(f"    V_per_channel (B, 4):           {fmt(s['V_per_channel'])}")
        print(f"    V_per_gate (B, 4):              {fmt(s['V_per_gate'])}")
        print(f"    g_prev (B, 4):                  {fmt(s['g_prev'])}")
        print(f"    g_inf = sigmoid(...) (B, 4):    {fmt(s['g_inf'])}")
        print(f"    g_new (B, 4):                   {fmt(s['g_new'])}")
        print(f"    P (B, 4):                       {fmt(s['P'])}")
        print(f"    gP = g_c * P (B, 4):            {fmt(s['gP'])}")
        print(f"    b_eff = Σ_c gP_c E_c v̂_c (B,16): {fmt(s['b_eff'])}")
        print(f"    u_t = B_leak(x_t) (B, 16):       {fmt(s['u_t'])}")
        print(f"    rhs = h + dt*(b_eff+u_t) (B,16): {fmt(s['rhs'])}")
        print(f"    M_inv (4, 4):                    {fmt(s['M_inv'][0])}")
        print(f"    correction (B, 16):              {fmt(s['correction'])}")
        print(f"    h_new (B, 16):                   {fmt(s['h_new'])}")
        print(f"    ‖h_new‖ = {float(s['h_new'].norm()):.4f}")

    print(f"\nh_alpha_end ≡ h[:, t=L-1, :] (B, 16): {fmt(h_alpha_end)}")
    print(f"‖h_alpha_end‖ = {float(h_alpha_end.norm()):.4f}")

    banner("L4 — write_head (Sequential, W=5)")
    wh = agent.write_head
    print(f"cell_mlp = Sequential(")
    print(f"  Linear(d_pfc+W={wh.cell_mlp[0].in_features}, hidden={wh.cell_mlp[0].out_features}),")
    print(f"  LayerNorm({wh.cell_mlp[1].normalized_shape}),")
    print(f"  GELU(),")
    print(f"  Linear({wh.cell_mlp[3].in_features}, 1)")
    print(f")")
    print(f"  L0.W: shape={tuple(wh.cell_mlp[0].weight.shape)}  "
          f"‖·‖={float(wh.cell_mlp[0].weight.norm()):.4f}")
    print(f"  L3.W: shape={tuple(wh.cell_mlp[3].weight.shape)}  "
          f"‖·‖={float(wh.cell_mlp[3].weight.norm()):.4f}")
    print(f"  L3.b: {fmt(wh.cell_mlp[3].bias)}")

    h_pfc = h_alpha_end
    W = wh.W
    written = torch.zeros(B_, W, device=device, dtype=h_pfc.dtype)
    cells_raw, cells_q = [], []
    for w in range(W):
        input_w = torch.cat([h_pfc, written], dim=-1)
        cell_w_raw = wh.cell_mlp(input_w).squeeze(-1)
        # quantize STE forward
        soft = torch.sigmoid(cell_w_raw)
        scaled = soft * (agent_cfg.quantize_levels - 1)
        rounded = torch.round(scaled)
        hard = rounded / (agent_cfg.quantize_levels - 1)
        cell_w_q = soft + (hard - soft).detach()
        cells_raw.append(cell_w_raw)
        cells_q.append(cell_w_q)
        print(f"  cell {w}: input concat (16+5={16+W} dim), "
              f"raw={float(cell_w_raw.item()):+.4f}, "
              f"sigmoid={float(soft.item()):.4f}, "
              f"q={float(cell_w_q.item()):.4f}")
        written = written.clone()
        written[:, w] = cell_w_q
    scratch_alpha_raw = torch.stack(cells_raw, dim=-1)
    scratch_alpha_q = torch.stack(cells_q, dim=-1)
    print(f"scratch_alpha_raw (B, 5): {fmt(scratch_alpha_raw)}")
    print(f"scratch_alpha_q   (B, 5): {fmt(scratch_alpha_q)}")

    banner("L5 — β scan (re-run substrate carrying alpha_state)")
    rstart, beta_start, compare_step = scene_cfg.compare_block_phases(0)
    beta_in = inputs[:, beta_start:compare_step, :]
    print(f"beta_in: t=[{beta_start}, {compare_step}) shape={tuple(beta_in.shape)}")
    # Re-encode + substrate again from alpha_state
    x_for_cnn_b = beta_in.transpose(1, 2)
    conv_outs_b = [conv(x_for_cnn_b) for conv in agent.dp_convs]
    feat_cnn_b = torch.cat(conv_outs_b, dim=1).transpose(1, 2)
    feat_beta = agent.dp_proj(feat_cnn_b)
    # alpha_state is the final state from the alpha scan
    alpha_state = {"h": h_alpha_end, "g": g_state}
    h_beta_seq, _ = substrate(feat_beta, state=[alpha_state])
    h_beta_end = h_beta_seq[:, -1, :]
    print(f"h_beta_end (B, 16): {fmt(h_beta_end)}  ‖·‖={float(h_beta_end.norm()):.4f}")

    # write_head for beta = scratch placeholder = scratch_alpha (V33 uses alpha repeat)
    # But run real write_head on h_beta_end for completeness
    written_b = torch.zeros(B_, W, device=device, dtype=h_beta_end.dtype)
    cells_raw_b, cells_q_b = [], []
    for w in range(W):
        input_w = torch.cat([h_beta_end, written_b], dim=-1)
        cell_w_raw = wh.cell_mlp(input_w).squeeze(-1)
        soft = torch.sigmoid(cell_w_raw)
        scaled = soft * (agent_cfg.quantize_levels - 1)
        rounded = torch.round(scaled)
        hard = rounded / (agent_cfg.quantize_levels - 1)
        cell_w_q = soft + (hard - soft).detach()
        cells_raw_b.append(cell_w_raw)
        cells_q_b.append(cell_w_q)
        written_b = written_b.clone()
        written_b[:, w] = cell_w_q
    scratch_beta_raw = torch.stack(cells_raw_b, dim=-1)
    scratch_beta_q = torch.stack(cells_q_b, dim=-1)
    print(f"scratch_beta_q (B, 5):  {fmt(scratch_beta_q)}")

    # NOTE: V33 _v33_forward overrides sc_beta to be sc_alpha repeated.
    # But predict_head reads from agent.scratch fields. For walkthrough authenticity
    # we'll do exactly what _v33_forward does (sc_alpha is placeholder).
    print("(_v33_forward uses sc_beta_K = sc_alpha placeholder — for predict_head input)")

    banner("L6 — successor_predict_head Linear(2W=10, k_max=10)")
    ph = agent.successor_predict_head
    print(f"successor_predict_head: Linear(in={ph.in_features}, out={ph.out_features})")
    print(f"  W shape={tuple(ph.weight.shape)} ‖·‖={float(ph.weight.norm()):.4f}")
    print(f"  b shape={tuple(ph.bias.shape)} {fmt(ph.bias)}")
    # V33: predict_head input = concat(sc_alpha_q, sc_beta_q_first), where sc_beta_q_first = sc_alpha_q
    v24_input = torch.cat([scratch_alpha_q, scratch_alpha_q], dim=-1)
    print(f"v24_input = concat(sc_α_q, sc_β_q[=sc_α_q]) shape={tuple(v24_input.shape)}")
    k_logits = ph(v24_input)
    print(f"k_logits (B, 10): {fmt(k_logits)}")
    k_probs = F.softmax(k_logits, dim=-1)
    print(f"k_probs (softmax): {fmt(k_probs)}")
    pred_k_class = int(k_probs.argmax(dim=-1).item())
    print(f"argmax class = {pred_k_class}")

    banner("L7 — target k → class mapping & CE loss")
    k_max_val = ph.out_features // 2  # 5 for bidirectional 10-class
    k_raw = n_b - n_a
    if k_raw < 0:
        k_class = max(0, k_raw + k_max_val)
    elif k_raw > 0:
        k_class = min(ph.out_features - 1, k_raw + k_max_val - 1)
    else:
        k_class = 0
    print(f"k_raw = N_β - N_α = {n_b} - {n_a} = {k_raw}")
    print(f"k_target class (bidir mapping): {k_class}")
    loss = F.cross_entropy(k_logits, torch.tensor([k_class], device=device))
    print(f"loss = -log P[class={k_class}] = {float(loss.item()):.6f}")
    print(f"  (chance loss for 10-class = ln(10) = {np.log(10):.4f})")

    banner("L8 — Backward: ‖∂L/∂W‖ for each param group")
    agent.zero_grad()
    loss.backward()

    groups = []
    for name, p in agent.named_parameters():
        if p.grad is None:
            continue
        groups.append((name, float(p.norm()), float(p.grad.norm()),
                       float(p.grad.abs().max())))
    # Sort by grad norm descending
    groups.sort(key=lambda r: r[2], reverse=True)
    print(f"{'param':<60} {'‖p‖':>10} {'‖∂L/∂p‖':>12} {'max|grad|':>12}")
    print("-" * 96)
    for name, pnrm, gnrm, gmax in groups:
        print(f"{name:<60} {pnrm:>10.4f} {gnrm:>12.6f} {gmax:>12.6f}")

    banner("L8.5 — V33-substrate gradient highlights")
    keys = ["v33_substrate.layers.0.channels.v",
            "v33_substrate.layers.0.channels.E",
            "v33_substrate.layers.0.channels.log_g",
            "v33_substrate.layers.0.gates.w",
            "v33_substrate.layers.0.gates.b",
            "v33_substrate.layers.0.gates.log_tau",
            "v33_substrate.layers.0.B_leak.weight",
            "v33_substrate.layers.0.log_dt",
            "dp_proj.weight",
            "write_head.cell_mlp.0.weight",
            "write_head.cell_mlp.3.weight",
            "successor_predict_head.weight"]
    for k in keys:
        for n, p in agent.named_parameters():
            if n == k and p.grad is not None:
                print(f"  {k}:")
                print(f"    p:    {fmt(p)}")
                print(f"    grad: {fmt(p.grad)}")
                break

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
