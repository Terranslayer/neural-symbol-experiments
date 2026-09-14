"""V32 forward + backward trace.

DIAG_FOR: v32_multigate_ec

Loads a stage 0 ckpt, runs a single signal through the full pipeline with hooks
to capture every intermediate tensor, prints them in walkthrough order, then
runs a paired EC+aux backward and prints gradient norms per param group.

Usage:
    PYTHONPATH=. python scripts/v32_trace_forward.py \\
        --ckpt checkpoints/v32_f2_aux_nodetach/stage0_agent0.pt \\
        --partner-ckpt checkpoints/v32_f2_aux_nodetach/stage0_agent1.pt \\
        --n-a 2 --n-b 3 --L 200 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.core.ec_agent import ECAgent, ECAgentConfig
from backend.core.scene import _scan_world_complex


def load_agent(ckpt_path: Path, device: str) -> ECAgent:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ECAgentConfig(**state["agent_cfg"])
    agent = ECAgent(cfg).to(device)
    agent.load_state_dict(state["agent_state_dict"])
    return agent, cfg


def _stats(t: torch.Tensor) -> str:
    """Compact tensor stats string."""
    if t.numel() == 0:
        return "(empty)"
    t = t.detach()
    flat = t.flatten()
    return (
        f"shape={tuple(t.shape)}  "
        f"min={flat.min().item():+.4f}  "
        f"max={flat.max().item():+.4f}  "
        f"mean={flat.mean().item():+.4f}  "
        f"std={flat.std().item():.4f}"
    )


def trace_encode(agent: ECAgent, signal: torch.Tensor, label: str):
    """Forward through encoder with stash, print every intermediate.

    Returns (pfc_state, raw_scratch, q_scratch, aux_logits).
    """
    print(f"\n{'='*70}")
    print(f"=== {label} ===")
    print('='*70)

    feat = agent._signal_to_input_feature(signal)
    print(f"\n[input feat] {_stats(feat)}")
    print(f"  signal[0, :20] preview: "
          f"{[round(v, 3) for v in signal[0, :20].tolist()]}")

    # Use encoder's built-in stash to capture cnn_chunks + pfc_state per chunk
    stash = {}
    pfc_state = agent.encoder(
        feat, n_chunks=agent.cfg.n_chunks_phase1,
        token_type_id=0, stash=stash,
    )

    cnn_chunks = stash["cnn_chunks"]  # (1, n_chunks, d_model)
    print(f"\n[CNN chunked output] {_stats(cnn_chunks)}")
    for c in range(cnn_chunks.shape[1]):
        vals = [round(v, 4) for v in cnn_chunks[0, c, :].tolist()]
        print(f"  chunk {c}: {vals}")

    print(f"\n[LRU token per chunk]")
    for c, lru_tok in enumerate(stash["lru_h_per_chunk"]):
        vals = [round(v, 4) for v in lru_tok[0].tolist()]
        print(f"  chunk {c}: {vals}")

    print(f"\n[PFC state per chunk (after PFC step)]")
    for c, p_state in enumerate(stash["pfc_states_per_chunk"]):
        vals = [round(v, 4) for v in p_state[0].tolist()]
        print(f"  chunk {c}: {vals}  "
              f"(norm={p_state.norm().item():.4f})")

    print(f"\n[final pfc_state] {_stats(pfc_state)}")
    print(f"  values: {[round(v, 4) for v in pfc_state[0].tolist()]}")

    # Write head: instrument by manually walking through cells
    raw, q = _trace_write_head(agent, pfc_state)

    # Aux head
    aux_in = pfc_state.detach() if agent.cfg.aux_detach else pfc_state
    aux_logits = agent.aux_n_head(aux_in)
    print(f"\n[aux_logits] {_stats(aux_logits)}")
    pred_n = aux_logits.argmax(dim=-1).item()
    top5_vals, top5_idx = aux_logits[0].topk(5)
    print(f"  predicted N = {pred_n}")
    print(f"  top 5 classes: {list(zip(top5_idx.tolist(), [round(v,3) for v in top5_vals.tolist()]))}")

    return pfc_state, raw, q, aux_logits


def _trace_write_head(agent: ECAgent, pfc_state: torch.Tensor):
    """Manually unroll write_head, printing per-cell raw + quantized."""
    wh = agent.write_head
    B = pfc_state.shape[0]
    device = pfc_state.device
    written = torch.zeros(B, wh.W, device=device, dtype=pfc_state.dtype)
    raws, qs = [], []

    print(f"\n[Write head rollout] output_scale={wh.output_scale.item():.4f}")
    for w in range(wh.W):
        input_w = torch.cat([pfc_state, written], dim=-1)
        tau_w = wh.role_embeds[w].unsqueeze(0).expand(B, -1)

        # MultiGateLinear internals
        g_role = torch.sigmoid(wh.gate_block.W_role(tau_w))
        g_cont = torch.sigmoid(wh.gate_block.W_cont(input_w))
        v_val = wh.gate_block.W_v(input_w)
        gated = g_role * g_cont * v_val
        hidden_pre = wh.act(gated)
        hidden = wh.norm(hidden_pre)
        raw_w_pre_scale = wh.out_proj(hidden).squeeze(-1)
        raw_w = raw_w_pre_scale * wh.output_scale

        from backend.core.multigate import ste_quantize
        q_w = ste_quantize(raw_w, agent.cfg.quantize_levels)

        print(f"\n  cell w={w}:")
        print(f"    input_w (= [pfc; written]): {_stats(input_w)}")
        print(f"      pfc part: {[round(v,3) for v in pfc_state[0].tolist()]}")
        print(f"      written part: {[round(v,3) for v in written[0].tolist()]}")
        print(f"    g_role (sigmoid(W_role·tau)): "
              f"mean={g_role.mean().item():.4f}  min={g_role.min().item():.4f}  max={g_role.max().item():.4f}")
        print(f"    g_cont (sigmoid(W_cont·x)):  "
              f"mean={g_cont.mean().item():.4f}  min={g_cont.min().item():.4f}  max={g_cont.max().item():.4f}")
        print(f"    v = W_v·x: {_stats(v_val)}")
        print(f"    g_role ⊙ g_cont ⊙ v: {_stats(gated)}")
        print(f"    after GELU: {_stats(hidden_pre)}")
        print(f"    after LN:   {_stats(hidden)}")
        print(f"    raw_pre_scale (out_proj): {raw_w_pre_scale.item():+.4f}")
        print(f"    raw_w = raw_pre × output_scale = {raw_w.item():+.4f}")
        print(f"    sigmoid(raw_w) = {torch.sigmoid(raw_w).item():.4f}")
        print(f"    q_w (after STE bin to {{0, 0.25, 0.5, 0.75, 1.0}}): {q_w.item():.4f}")

        raws.append(raw_w)
        qs.append(q_w)
        written = written.clone()
        written[:, w] = q_w

    raw_t = torch.stack(raws, dim=-1)
    q_t = torch.stack(qs, dim=-1)
    print(f"\n[Final scratch] raw={[round(v,4) for v in raw_t[0].tolist()]}  "
          f"q={[round(v,3) for v in q_t[0].tolist()]}")
    return raw_t, q_t


def trace_phase3(agent: ECAgent, own_scratch: torch.Tensor,
                 partner_scratch: torch.Tensor, label: str):
    """Phase 3 trace: scratch cross-read → δ logits."""
    print(f"\n{'='*70}")
    print(f"=== Phase 3: {label} ===")
    print('='*70)
    combined = torch.cat([own_scratch, partner_scratch], dim=-1)
    print(f"\n[combined scratch input]")
    print(f"  own:     {[round(v,3) for v in own_scratch[0].tolist()]}")
    print(f"  partner: {[round(v,3) for v in partner_scratch[0].tolist()]}")
    print(f"  combined: {[round(v,3) for v in combined[0].tolist()]}")

    feat = agent._scratch_to_input_feature(combined)
    stash = {}
    pfc_state = agent.encoder(
        feat, n_chunks=agent.cfg.n_chunks_phase3,
        token_type_id=1, stash=stash,
    )
    print(f"\n[cross-read CNN chunks] {_stats(stash['cnn_chunks'])}")
    for c in range(stash["cnn_chunks"].shape[1]):
        vals = [round(v, 4) for v in stash["cnn_chunks"][0, c, :].tolist()]
        print(f"  chunk {c}: {vals}")
    print(f"\n[cross-read PFC state per chunk]")
    for c, p in enumerate(stash["pfc_states_per_chunk"]):
        print(f"  chunk {c}: {[round(v,4) for v in p[0].tolist()]}")
    print(f"\n[final phase-3 pfc_state] {_stats(pfc_state)}")

    logits = agent.predict_head(pfc_state)
    softmax = F.softmax(logits, dim=-1)
    print(f"\n[δ logits] {[round(v,4) for v in logits[0].tolist()]}")
    print(f"[δ softmax] {[round(v,4) for v in softmax[0].tolist()]}")
    return logits


def grad_norms_by_group(agent: ECAgent) -> dict:
    """After .backward(), summarize gradient L2 norm per major param group."""
    groups = {
        "encoder.cnn.*": [],
        "encoder.lru_input_proj": [],
        "encoder.lru_type_embed": [],
        "encoder.lru_block.B_real": [],
        "encoder.lru_block.B_imag": [],
        "encoder.lru_block.C_real": [],
        "encoder.lru_block.C_imag": [],
        "encoder.lru_block.D": [],
        "encoder.lru_block.nu/theta_log": [],
        "encoder.pfc_step.Q/K/V": [],
        "encoder.pfc_step.out_proj/norm": [],
        "encoder.pfc_step.ffn": [],
        "encoder.init_state": [],
        "write_head.gate_block": [],
        "write_head.out_proj": [],
        "write_head.norm": [],
        "write_head.role_embeds": [],
        "write_head.output_scale": [],
        "predict_head": [],
        "aux_n_head": [],
    }
    for name, p in agent.named_parameters():
        if p.grad is None:
            continue
        g_norm = p.grad.norm().item()
        # Classify
        if name.startswith("encoder.convs_per_type") or name.startswith("encoder.cnn_proj_per_type"):
            groups["encoder.cnn.*"].append((name, g_norm))
        elif name.startswith("encoder.lru_input_proj"):
            groups["encoder.lru_input_proj"].append((name, g_norm))
        elif name.startswith("encoder.lru_type_embed"):
            groups["encoder.lru_type_embed"].append((name, g_norm))
        elif name.startswith("lru_block.B_real"):
            groups["encoder.lru_block.B_real"].append((name, g_norm))
        elif name.startswith("lru_block.B_imag"):
            groups["encoder.lru_block.B_imag"].append((name, g_norm))
        elif name.startswith("lru_block.C_real"):
            groups["encoder.lru_block.C_real"].append((name, g_norm))
        elif name.startswith("lru_block.C_imag"):
            groups["encoder.lru_block.C_imag"].append((name, g_norm))
        elif name.startswith("lru_block.D"):
            groups["encoder.lru_block.D"].append((name, g_norm))
        elif "lru_block.nu_log" in name or "lru_block.theta_log" in name:
            groups["encoder.lru_block.nu/theta_log"].append((name, g_norm))
        elif "pfc_step.Q" in name or "pfc_step.K" in name or "pfc_step.V" in name:
            groups["encoder.pfc_step.Q/K/V"].append((name, g_norm))
        elif "pfc_step.out_proj" in name or "pfc_step.norm" in name:
            groups["encoder.pfc_step.out_proj/norm"].append((name, g_norm))
        elif "pfc_step.ffn" in name:
            groups["encoder.pfc_step.ffn"].append((name, g_norm))
        elif "encoder.init_state" in name:
            groups["encoder.init_state"].append((name, g_norm))
        elif name.startswith("write_head.gate_block"):
            groups["write_head.gate_block"].append((name, g_norm))
        elif name.startswith("write_head.out_proj"):
            groups["write_head.out_proj"].append((name, g_norm))
        elif name.startswith("write_head.norm"):
            groups["write_head.norm"].append((name, g_norm))
        elif name.startswith("write_head.role_embeds"):
            groups["write_head.role_embeds"].append((name, g_norm))
        elif name == "write_head.output_scale":
            groups["write_head.output_scale"].append((name, g_norm))
        elif name.startswith("predict_head"):
            groups["predict_head"].append((name, g_norm))
        elif name.startswith("aux_n_head"):
            groups["aux_n_head"].append((name, g_norm))
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--partner-ckpt", default=None,
                   help="If given, use this for partner; else use same as own")
    p.add_argument("--n-a", type=int, default=2)
    p.add_argument("--n-b", type=int, default=3)
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--scan-n", type=str, default="",
                   help="Optional: comma-separated N values to scan (e.g. '1,2,3,4,5'). "
                        "If set, only does encode scan; no backward.")
    args = p.parse_args()

    agent_A, cfg_A = load_agent(Path(args.ckpt), args.device)
    if args.partner_ckpt:
        agent_B, cfg_B = load_agent(Path(args.partner_ckpt), args.device)
    else:
        agent_B, cfg_B = agent_A, cfg_A
    agent_A.train(True)  # need grad
    agent_B.train(True)

    rng = np.random.default_rng(args.seed)

    if args.scan_n:
        # Just scan N values, no backward
        ns = [int(x) for x in args.scan_n.split(",")]
        print(f"\n### Scanning N values: {ns} ###\n")
        for N in ns:
            sig = _scan_world_complex(N, args.L, rng)
            sig_t = torch.from_numpy(sig).unsqueeze(0).to(args.device)
            trace_encode(agent_A, sig_t, f"N={N}")
        return

    # Full trace: encode A, encode B, cross-read, backward
    sig_A_np = _scan_world_complex(args.n_a, args.L, rng)
    sig_B_np = _scan_world_complex(args.n_b, args.L, rng)
    sig_A = torch.from_numpy(sig_A_np).unsqueeze(0).to(args.device)
    sig_B = torch.from_numpy(sig_B_np).unsqueeze(0).to(args.device)

    pfc_A, raw_A, q_A, aux_A = trace_encode(agent_A, sig_A, f"agent A, N_A={args.n_a}")
    pfc_B, raw_B, q_B, aux_B = trace_encode(agent_B, sig_B, f"agent B, N_B={args.n_b}")

    logits_dA = trace_phase3(agent_A, q_A, q_B, f"agent A reads own + B")
    logits_dB = trace_phase3(agent_B, q_B, q_A, f"agent B reads own + A")

    # Labels (POV)
    delta_A = args.n_a - args.n_b  # own - partner
    delta_B = args.n_b - args.n_a
    target_A = torch.tensor([delta_A + 2], device=args.device)  # class offset
    target_B = torch.tensor([delta_B + 2], device=args.device)
    n_A_t = torch.tensor([args.n_a], device=args.device)
    n_B_t = torch.tensor([args.n_b], device=args.device)

    ec_A = F.cross_entropy(logits_dA, target_A)
    ec_B = F.cross_entropy(logits_dB, target_B)
    aux_loss_A = F.cross_entropy(aux_A, n_A_t)
    aux_loss_B = F.cross_entropy(aux_B, n_B_t)

    print(f"\n{'='*70}")
    print("=== Loss components ===")
    print('='*70)
    print(f"  ec_A   = {ec_A.item():.4f}  (target class {target_A.item()} for δ={delta_A})")
    print(f"  ec_B   = {ec_B.item():.4f}  (target class {target_B.item()} for δ={delta_B})")
    print(f"  aux_A  = {aux_loss_A.item():.4f}  (target N={args.n_a})")
    print(f"  aux_B  = {aux_loss_B.item():.4f}  (target N={args.n_b})")
    total = ec_A + ec_B + 1.0 * (aux_loss_A + aux_loss_B)
    print(f"  total  = {total.item():.4f}")
    print(f"  ec_chance = 2·ln(5) = {2*np.log(5):.4f}")
    print(f"  aux_chance = 2·ln(31) = {2*np.log(31):.4f}")

    # Backward separately for ec vs aux to compare grad magnitudes
    print(f"\n{'='*70}")
    print("=== Backward: EC loss only ===")
    print('='*70)
    for ag in [agent_A, agent_B]:
        for pp in ag.parameters():
            if pp.grad is not None:
                pp.grad = None
    (ec_A + ec_B).backward(retain_graph=True)
    ec_grads_A = grad_norms_by_group(agent_A)
    print("\n[Agent A param-group L2 norms — from EC loss only]")
    for grp, lst in ec_grads_A.items():
        total_g = sum(g for _, g in lst)
        print(f"  {grp:40s}  total_L2={total_g:.6e}  ({len(lst)} params)")

    print(f"\n{'='*70}")
    print("=== Backward: AUX loss only ===")
    print('='*70)
    for ag in [agent_A, agent_B]:
        for pp in ag.parameters():
            if pp.grad is not None:
                pp.grad = None
    (aux_loss_A + aux_loss_B).backward()
    aux_grads_A = grad_norms_by_group(agent_A)
    print("\n[Agent A param-group L2 norms — from AUX loss only]")
    for grp, lst in aux_grads_A.items():
        total_g = sum(g for _, g in lst)
        print(f"  {grp:40s}  total_L2={total_g:.6e}  ({len(lst)} params)")

    # Side-by-side ratio
    print(f"\n{'='*70}")
    print("=== EC vs AUX gradient ratio (Agent A) ===")
    print('='*70)
    print(f"  {'group':40s} {'EC':>12s} {'AUX':>12s} {'EC/AUX':>10s}")
    for grp in ec_grads_A:
        ec_g = sum(g for _, g in ec_grads_A[grp])
        aux_g = sum(g for _, g in aux_grads_A[grp])
        ratio = (ec_g / aux_g) if aux_g > 0 else float('inf') if ec_g > 0 else 0.0
        print(f"  {grp:40s} {ec_g:12.4e} {aux_g:12.4e} {ratio:10.4f}")


if __name__ == "__main__":
    main()
