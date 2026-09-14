# -*- coding: utf-8 -*-
"""
Parameter-scaling U-curve experiment.

Sweeps d_model over a list of values, runs full 4-stage curriculum for each,
and aggregates the final metrics into a single comparison figure + JSONL.

Usage:
    python scripts/sweep_d_model.py --device cuda

Writes:
    logs/sweep_d_model_<timestamp>/
      run_d<N>/run.jsonl         # per-config full log
      summary.jsonl              # final metrics per config
      u_curve.png                # extrap acc & comp_idx vs param count
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from backend.core.agent import AgentConfig, GRUAgent
from backend.core.scene import SceneConfig
from backend.training.train_phase1 import TrainConfig, train


def _num_parameters(d_model: int) -> int:
    scene_cfg = SceneConfig()
    tmp = GRUAgent(AgentConfig(d_model=d_model), scene_cfg)
    return tmp.num_parameters()


def run_one(
    d_model: int,
    device: torch.device,
    out_root: Path,
    quantize_levels: int = None,
) -> Dict:
    """Run one full training with given d_model, return final metrics."""
    agent_cfg = AgentConfig(d_model=d_model, quantize_levels=quantize_levels)
    scene_cfg = SceneConfig()
    train_cfg = TrainConfig()
    curriculum = TrainConfig.default_curriculum()

    tag = f"d{d_model}"
    if quantize_levels is not None:
        tag += f"_q{quantize_levels}"
    log_dir = out_root / f"run_{tag}"
    ckpt_dir = out_root / "checkpoints" / tag

    print(f"\n{'=' * 60}")
    print(f"Sweep: d_model={d_model}  params={_num_parameters(d_model)}")
    print('=' * 60)

    t0 = time.time()
    train(scene_cfg, agent_cfg, train_cfg, curriculum, device, log_dir, ckpt_dir)
    elapsed = time.time() - t0

    # Extract final metrics from the log
    jsonl = log_dir / "run.jsonl"
    final_rec = None
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") == "final_metrics":
            final_rec = rec
            break

    return {
        "d_model": d_model,
        "num_parameters": _num_parameters(d_model),
        "quantize_levels": quantize_levels,
        "elapsed_seconds": elapsed,
        "rollout_accuracy": final_rec["rollout_accuracy"] if final_rec else None,
        "composition_index": final_rec["composition_index"] if final_rec else None,
        "topo_sim": final_rec["topo_sim"] if final_rec else None,
        "total_r2": final_rec["total_r2"] if final_rec else None,
        "per_position_r2": final_rec["per_position_r2"] if final_rec else None,
        "best_base": final_rec["best_base"] if final_rec else None,
        "best_diagonal_score": final_rec["best_diagonal_score"] if final_rec else None,
        "extrapolation_full": final_rec.get("extrapolation_full", {}) if final_rec else {},
    }


def plot_u_curve(rows: List[Dict], out_path: Path):
    rows_sorted = sorted(rows, key=lambda r: r["num_parameters"])
    params = [r["num_parameters"] for r in rows_sorted]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(params, [r["rollout_accuracy"] for r in rows_sorted], "o-", color="green")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel("rollout accuracy (N ≤ 45)")
    ax.set_title("In-distribution accuracy vs model size")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    extrap_values = []
    for r in rows_sorted:
        vals = list(r["extrapolation_full"].values())
        extrap_values.append(float(np.mean(vals)) if vals else float("nan"))
    ax.plot(params, extrap_values, "s-", color="orange")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel("mean extrapolation accuracy")
    ax.set_title("Out-of-range accuracy vs model size (U-curve target)")
    ax.grid(alpha=0.3)
    ax.axhline(0.333, color="red", linestyle=":", alpha=0.5, label="chance")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(params, [r["composition_index"] for r in rows_sorted], "o-", color="purple")
    ax.set_xscale("log")
    ax.set_xlabel("model parameters")
    ax.set_ylabel("composition_index")
    ax.set_title("Compositional encoding signal vs model size")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bases = [r["best_base"] for r in rows_sorted]
    ax.bar(range(len(params)), bases, color="steelblue")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([str(p) for p in params], rotation=30)
    ax.set_xlabel("model parameters")
    ax.set_ylabel("best emergent base")
    ax.set_title("Emergent base-scanning winner")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Parameter Scaling U-Curve (run_v3 Stage 1 4-stage curriculum)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--d-models", type=str, default="4,8,16,32,64,128",
                        help="Comma-separated d_model values")
    parser.add_argument("--quantize-levels", type=int, default=None,
                        help="Apply same quantization to every run")
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    d_values = [int(x) for x in args.d_models.split(",")]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root or f"logs/sweep_d_model_{timestamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    summary_path = out_root / "summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as fh:
        for d in d_values:
            row = run_one(d, device, out_root, quantize_levels=args.quantize_levels)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            print(f"  d_model={d}: rollout={row['rollout_accuracy']:.3f} "
                  f"comp_idx={row['composition_index']:.3f} "
                  f"best_base={row['best_base']}")

    plot_u_curve(results, out_root / "u_curve.png")
    print(f"\nSweep complete. Results: {out_root}")


if __name__ == "__main__":
    main()
