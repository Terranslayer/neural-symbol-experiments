"""V32 codebook diag.

DIAG_FOR: v32_multigate_ec

Usage:
    python scripts/v32_codebook_diag.py --ckpt-dir checkpoints/v32_ec_default \
        --n-min 1 --n-max 30 --n-eval 200

For each agent_k.pt in ckpt-dir:
1. Load agent
2. For N in [n_min, n_max], generate n_eval signals, encode → scratch
3. Compute per-N modal scratch tuple, distinct codes total, cell-cell Pearson
4. Dump report to stdout (or --out json)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from backend.core.ec_agent import ECAgent, ECAgentConfig
from backend.core.scene import _scan_world_complex


def load_agent(ckpt_path: Path, device: str) -> ECAgent:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["agent_cfg"]
    cfg = ECAgentConfig(**cfg_dict)
    agent = ECAgent(cfg).to(device)
    agent.load_state_dict(state["agent_state_dict"])
    agent.train(False)  # inference mode
    return agent


def collect_scratches(agent: ECAgent, n_min: int, n_max: int,
                       n_eval: int, L: int, device: str,
                       rng: np.random.Generator):
    """For each N in [n_min, n_max], generate n_eval signals and collect
    quantized scratch tuples.
    """
    results = {}
    with torch.no_grad():
        for N in range(n_min, n_max + 1):
            scratches = []
            for _ in range(n_eval):
                sig = _scan_world_complex(N, L, rng)
                sig_t = torch.from_numpy(sig).unsqueeze(0).to(device)
                _, q, _ = agent.encode_signal(sig_t)
                scratches.append(tuple(round(v.item(), 4) for v in q.squeeze(0)))
            results[N] = scratches
    return results


def analyze_codebook(scratches_per_N: dict) -> dict:
    """Compute distinct codes, per-N modal, cell-cell Pearson."""
    all_codes = []
    per_N_modal = {}
    for N, codes in scratches_per_N.items():
        all_codes.extend(codes)
        modal, freq = Counter(codes).most_common(1)[0]
        per_N_modal[N] = {
            "modal": list(modal), "modal_freq": freq / len(codes),
            "distinct_in_N": len(set(codes)),
        }
    distinct_overall = len(set(all_codes))
    # Cell-cell Pearson across all (N, sample)
    arr = np.array([list(c) for c in all_codes])  # (n_total, W=5)
    pearson = np.corrcoef(arr.T)  # (W, W)
    return {
        "distinct_codes_total": distinct_overall,
        "n_total_samples": len(all_codes),
        "per_N_modal": per_N_modal,
        "cell_cell_pearson": pearson.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--stage", type=int, default=-1,
                        help="Stage index (default -1 = highest)")
    parser.add_argument("--n-min", type=int, default=1)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--L", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None,
                        help="Write JSON report to this path; else stdout")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    stage_files = sorted(ckpt_dir.glob("stage*_agent*.pt"))
    if not stage_files:
        raise SystemExit(f"No ckpts in {ckpt_dir}")
    if args.stage == -1:
        max_stage = max(int(p.name.split("_")[0][5:]) for p in stage_files)
    else:
        max_stage = args.stage
    agent_files = sorted(ckpt_dir.glob(f"stage{max_stage}_agent*.pt"))

    rng = np.random.default_rng(0)
    report = {"stage": max_stage, "agents": {}}
    for ap in agent_files:
        agent_id = int(ap.stem.split("agent")[1])
        agent = load_agent(ap, args.device)
        scratches = collect_scratches(
            agent, args.n_min, args.n_max, args.n_eval, args.L, args.device, rng,
        )
        analysis = analyze_codebook(scratches)
        report["agents"][agent_id] = analysis
        print(f"=== Agent {agent_id} ===")
        print(f"  distinct codes: {analysis['distinct_codes_total']}")
        cp = np.array(analysis["cell_cell_pearson"])
        off_diag = cp[np.triu_indices(cp.shape[0], k=1)]
        print(f"  cell-cell Pearson (off-diag): "
              f"min={off_diag.min():.3f}, max={off_diag.max():.3f}, "
              f"mean={off_diag.mean():.3f}")
        for N in sorted(analysis["per_N_modal"]):
            m = analysis["per_N_modal"][N]
            print(f"  N={N:2d}: modal={m['modal']} freq={m['modal_freq']:.2f}, "
                  f"distinct_in_N={m['distinct_in_N']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
