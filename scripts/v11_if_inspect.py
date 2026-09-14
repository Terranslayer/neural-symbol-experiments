"""V11 IF write head inspection: cell values, fire scores, top-W timing."""
import argparse, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=8)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True, cnn_pfc_only=True,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    print(f"IF threshold = {agent.write_head.threshold.item():.3f}")
    print(f"IF reset_strength = {torch.sigmoid(agent.write_head.reset_logit).item():.3f}")
    print(f"IF write_proj weight norm = {agent.write_head.write_proj.weight.norm().item():.3f}")

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.equal_weight = 0.20
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)
    S, Ns, FIRES, TIDX = [], [], [], []
    for _ in range(args.n_batches):
        inp, _, ci, m = sample_training_batch(
            batch_size=64, n_max_stage=args.n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            _, scr, _ = agent(inp.to(device), ci.to(device))
        S.append(scr.cpu().numpy())
        Ns += [x["N_alpha"] for x in m]
        FIRES.append(agent._last_if_fire_scores.cpu().numpy())
        TIDX.append(agent._last_if_top_idx.cpu().numpy())
    S = np.concatenate(S, 0)
    Ns = np.array(Ns)
    F_arr = np.concatenate(FIRES, 0)
    T_arr = np.concatenate(TIDX, 0)

    print()
    print("Per-cell value frequencies:")
    for i in range(5):
        vals, cts = np.unique(S[:, i].round(2), return_counts=True)
        freqs = (cts / cts.sum()).round(3)
        print(f"  c{i+1}: " + "  ".join(f"{v}:{f}" for v, f in zip(vals.tolist(), freqs.tolist())))

    print()
    print("Per-N top code:")
    for n in sorted(np.unique(Ns)):
        mask = Ns == n
        codes = [tuple(np.round(s, 2).tolist()) for s in S[mask]]
        cnt = Counter(codes)
        top = cnt.most_common(2)
        line = f"  N={n} ({mask.sum()}): {top[0][1]/mask.sum():.2f} {list(top[0][0])}"
        if len(top) > 1:
            line += f" | {top[1][1]/mask.sum():.2f} {list(top[1][0])}"
        print(line)

    distinct = len(set(tuple(np.round(s, 2).tolist()) for s in S))
    print(f"Distinct codes: {distinct}")

    print()
    print("Fire score stats:")
    print(f"  mean per t: {F_arr.mean(axis=0).round(3).tolist()}")
    print(f"  std per t:  {F_arr.std(axis=0).round(3).tolist()}")
    print(f"  threshold = {agent.write_head.threshold.item():.3f}")
    print(f"  fraction of (b,t) with score > threshold: {(F_arr > agent.write_head.threshold.item()).mean():.3f}")

    print()
    print("Top-W timing distribution per N:")
    for n in sorted(np.unique(Ns))[:10]:
        mask = Ns == n
        if mask.sum() < 3:
            continue
        avg_idx = T_arr[mask].mean(axis=0)
        print(f"  N={n}: avg fire idx = {avg_idx.round(2).tolist()}")

    print()
    print("Cell-cell Pearson:")
    print(np.corrcoef(S.T).round(3))


if __name__ == "__main__":
    main()
