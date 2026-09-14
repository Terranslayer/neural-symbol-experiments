# DIAG_FOR: v26 | ANSWERS: codebook | INPUTS: <ckpt>
"""V26-correct REFINED scratch per-N modal code.

Shows for each N_α ∈ [1, 30], what the modal refined scratch tuple is.
This is the KEY check user requested — "what specifically is written for each N".
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
from backend.core.mamba_agent import RecurrentScratchRefiner
from scripts.inspect_checkpoint import load_agent
from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("jsonl", type=Path)
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
    restore_agent_cfg_from_jsonl(agent, args.jsonl)

    raw_state = torch.load(args.ckpt, map_location="cuda")
    if "agent_state_dict" in raw_state:
        raw_state = raw_state["agent_state_dict"]
    if "recurrent_refiner.cnn_scratch.weight" in raw_state:
        cnn_w = raw_state["recurrent_refiner.cnn_scratch.weight"]
        iter_proj_w = raw_state["recurrent_refiner.iter_proj.weight"]
        cps = cnn_w.shape[0]
        kk = cnn_w.shape[2]
        d_pfc = iter_proj_w.shape[0]
        W = agent.scene_cfg.W
        d_lru = iter_proj_w.shape[1] - d_pfc - cps * W
        refiner = RecurrentScratchRefiner(W=W, d_pfc=d_pfc, d_lru=d_lru,
                                           scratch_cnn_kernel=kk, scratch_cnn_channels=cps,
                                           n_iter=3).cuda()
        refiner.load_state_dict({k.replace("recurrent_refiner.", ""): v
                                 for k, v in raw_state.items()
                                 if k.startswith("recurrent_refiner.")})
        agent.recurrent_refiner = refiner

    cfg = SceneConfig.successor_prediction_preset(L=args.L, complex_world=True, k_max=args.k_max)
    cfg.K = 1
    cfg.alpha_distribution = "uniform"
    agent.scene_cfg = cfg
    agent.train(False)

    qlevels = agent.agent_cfg.quantize_levels
    qrange = agent.agent_cfg.quantize_range

    n_to_a_init, n_to_a_ref = defaultdict(list), defaultdict(list)
    n_to_b_init, n_to_b_ref = defaultdict(list), defaultdict(list)
    nb_to_b_init, nb_to_b_ref = defaultdict(list), defaultdict(list)

    rng = np.random.default_rng(42)
    for _ in range(args.n_batches):
        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.cuda(), ci.cuda())
            sc_a_init = agent._scratch_alpha_for_aux
            sc_b_init = agent._last_scratch_beta_K_q[:, 0, :]
            sc_a_ref = refiner(sc_a_init, agent._last_h_alpha_pfc_for_refine,
                               agent._last_lru_h_alpha_for_refine,
                               agent.write_head, agent.scratch_pos_emb, qlevels, qrange)
            sc_b_ref = refiner(sc_b_init, agent._last_h_beta_pfc_for_refine,
                               agent._last_lru_h_beta_for_refine,
                               agent.write_head, agent.scratch_pos_emb, qlevels, qrange)
        for b, m in enumerate(metas):
            n_a = m["N_alpha"]
            n_b = m["beta_counts"][0]
            n_to_a_init[n_a].append(tuple(int(round(x*2)) for x in sc_a_init[b].cpu().numpy()))
            n_to_a_ref[n_a].append(tuple(int(round(x*2)) for x in sc_a_ref[b].cpu().numpy()))
            nb_to_b_init[n_b].append(tuple(int(round(x*2)) for x in sc_b_init[b].cpu().numpy()))
            nb_to_b_ref[n_b].append(tuple(int(round(x*2)) for x in sc_b_ref[b].cpu().numpy()))

    print(f"\n=== sc_α REFINED modal per N_α ===")
    print(f" N_α | count | modal tuple   | freq | distinct | top-2 if any")
    print("-" * 75)
    for n in sorted(n_to_a_ref.keys()):
        c = Counter(n_to_a_ref[n])
        modal, mcount = c.most_common(1)[0]
        total = len(n_to_a_ref[n])
        top2 = c.most_common(2)
        top2_str = f", 2nd={top2[1][0]} ({top2[1][1]/total:.2f})" if len(top2) > 1 else ""
        print(f"  {n:3d} | {total:5d} | {modal} | {mcount/total:.2f} | {len(c)}{top2_str}")

    print(f"\n=== sc_β REFINED modal per N_β ===")
    print(f" N_β | count | modal tuple   | freq | distinct")
    print("-" * 60)
    for n in sorted(nb_to_b_ref.keys()):
        c = Counter(nb_to_b_ref[n])
        modal, mcount = c.most_common(1)[0]
        total = len(nb_to_b_ref[n])
        print(f"  {n:3d} | {total:5d} | {modal} | {mcount/total:.2f} | {len(c)}")

    # Code → set-of-N: which N values map to each unique code?
    print(f"\n=== Code → N_α mapping for sc_α REFINED ===")
    code_to_ns = defaultdict(list)
    for n, tuples in n_to_a_ref.items():
        for t in tuples:
            code_to_ns[t].append(n)
    for code in sorted(code_to_ns.keys()):
        ns = code_to_ns[code]
        n_unique = sorted(set(ns))
        n_min, n_max = min(ns), max(ns)
        n_mean = float(np.mean(ns))
        print(f"  {code} → N_α: count={len(ns)}, range=[{n_min},{n_max}], mean={n_mean:.1f}, unique={n_unique[:8]}{'...' if len(n_unique)>8 else ''}")

    print(f"\n=== Code → N_β mapping for sc_β REFINED ===")
    code_to_nbs = defaultdict(list)
    for n, tuples in nb_to_b_ref.items():
        for t in tuples:
            code_to_nbs[t].append(n)
    for code in sorted(code_to_nbs.keys()):
        ns = code_to_nbs[code]
        n_unique = sorted(set(ns))
        n_min, n_max = min(ns), max(ns)
        n_mean = float(np.mean(ns))
        print(f"  {code} → N_β: count={len(ns)}, range=[{n_min},{n_max}], mean={n_mean:.1f}, unique={n_unique[:8]}{'...' if len(n_unique)>8 else ''}")


if __name__ == "__main__":
    main()
