# DIAG_FOR: v27 | ANSWERS: codebook | INPUTS: <ckpt> --L --n-max
"""V27 per-N modal scratch codes.

V27 has no refiner — scratch is written directly by agent forward.
Just dump per-N modal codes for sc_α (indexed by N_α) and sc_β (by N_β).
"""
import argparse, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k-max", type=int, default=5)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
    )
    device = next(agent.parameters()).device

    # Build successor_predict_head if present in ckpt
    raw_state = torch.load(args.ckpt, map_location=device)
    if "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    if "successor_predict_head.weight" in raw_state:
        succ_w = raw_state["successor_predict_head.weight"]
        in_dim = succ_w.shape[1]
        n_classes = succ_w.shape[0]
        head = nn.Linear(in_dim, n_classes).to(device)
        head.load_state_dict({"weight": succ_w, "bias": raw_state["successor_predict_head.bias"]})
        agent.successor_predict_head = head
        print(f"Built successor_predict_head: Linear({in_dim}, {n_classes})")

    # V27: build CoTPFC if cot_pfc_n_chunks > 0
    if agent.agent_cfg.cot_pfc_n_chunks > 0 and "cot_pfc.init_state" in raw_state:
        from backend.core.mamba_agent import CoTPFC
        cot = CoTPFC(
            d_model=agent.agent_cfg.d_model,
            n_chunks=agent.agent_cfg.cot_pfc_n_chunks,
            kernels=tuple(agent.agent_cfg.cot_pfc_cnn_kernels),
            cnn_channels_per_scale=agent.agent_cfg.dp_channels_per_scale,
            lru_block=agent.mamba_blocks[0],
            input_dim=agent.agent_cfg.input_dim,
        ).to(device)
        cot_state = {k.replace("cot_pfc.", ""): v for k, v in raw_state.items()
                     if k.startswith("cot_pfc.") and "lru_block" not in k}
        cot.load_state_dict(cot_state, strict=False)
        agent.cot_pfc = cot
        print(f"Built CoTPFC: n_chunks={agent.agent_cfg.cot_pfc_n_chunks}")

    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.alpha_distribution = "uniform"
    cfg.successor_only_positive = False  # V27 bidirectional
    agent.scene_cfg = cfg
    agent.train(False)

    n_to_a = defaultdict(list)
    nb_to_b = defaultdict(list)

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))
            sc_a = agent._scratch_alpha_for_aux  # (B, W)
            sc_b = agent._last_scratch_beta_K_q[:, 0, :]  # (B, W)
        for b, m in enumerate(metas):
            n_a = m["N_alpha"]
            n_b = m["beta_counts"][0]
            n_to_a[n_a].append(tuple(int(round(x*2)) for x in sc_a[b].cpu().numpy()))
            nb_to_b[n_b].append(tuple(int(round(x*2)) for x in sc_b[b].cpu().numpy()))

    # === sc_α per N_α modal table ===
    print(f"\n=== sc_α modal per N_α (V27 forward direct, no refiner) ===")
    print(f" N_α | count | modal tuple   | freq | distinct | top-2 if any")
    print("-" * 75)
    for n in sorted(n_to_a.keys()):
        c = Counter(n_to_a[n])
        modal, mc = c.most_common(1)[0]
        total = len(n_to_a[n])
        top2 = c.most_common(2)
        top2_str = f", 2nd={top2[1][0]} ({top2[1][1]/total:.2f})" if len(top2) > 1 else ""
        print(f"  {n:3d} | {total:5d} | {modal} | {mc/total:.2f} | {len(c)}{top2_str}")

    print(f"\n=== sc_β modal per N_β ===")
    print(f" N_β | count | modal tuple   | freq | distinct")
    print("-" * 60)
    for n in sorted(nb_to_b.keys()):
        c = Counter(nb_to_b[n])
        modal, mc = c.most_common(1)[0]
        total = len(nb_to_b[n])
        print(f"  {n:3d} | {total:5d} | {modal} | {mc/total:.2f} | {len(c)}")

    # === Code → N_α mapping ===
    print(f"\n=== Code → N_α mapping for sc_α ===")
    code_to_ns = defaultdict(list)
    for n, tuples in n_to_a.items():
        for t in tuples:
            code_to_ns[t].append(n)
    for code in sorted(code_to_ns.keys()):
        ns = code_to_ns[code]
        n_unique = sorted(set(ns))
        print(f"  {code} → N_α: count={len(ns)}, range=[{min(ns)},{max(ns)}], mean={np.mean(ns):.1f}, unique={n_unique[:8]}{'...' if len(n_unique)>8 else ''}")

    # Code → N_β
    print(f"\n=== Code → N_β mapping for sc_β ===")
    code_to_nbs = defaultdict(list)
    for n, tuples in nb_to_b.items():
        for t in tuples:
            code_to_nbs[t].append(n)
    for code in sorted(code_to_nbs.keys()):
        ns = code_to_nbs[code]
        n_unique = sorted(set(ns))
        print(f"  {code} → N_β: count={len(ns)}, range=[{min(ns)},{max(ns)}], mean={np.mean(ns):.1f}, unique={n_unique[:8]}{'...' if len(n_unique)>8 else ''}")


if __name__ == "__main__":
    main()
