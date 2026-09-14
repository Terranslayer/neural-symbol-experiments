# -*- coding: utf-8 -*-
"""
Render training-log plots from a Stage 1 JSONL run.

Usage:
    python scripts/render_report.py logs/run_v3_v2/

Reads run.jsonl in the given log directory and writes PNG figures to
<log_dir>/figures/:
  - training_curves.png    Train loss & validation accuracy per stage
  - metric_trajectory.png  comp_idx / topo_sim / extrap across stages
  - per_position_r2.png    Per-position probe R² with marginal gains
  - base_scores.png        Diagonal score per candidate base, per stage
  - summary_table.txt      Plain-text summary of key numbers
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_events(log_dir: Path) -> List[dict]:
    jsonl = log_dir / "run.jsonl"
    if not jsonl.exists():
        print(f"No run.jsonl in {log_dir}", file=sys.stderr)
        sys.exit(1)
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]


def by_event(events: List[dict], name: str) -> List[dict]:
    return [e for e in events if e.get("event") == name]


def plot_training_curves(events, out_path: Path):
    vals = by_event(events, "validation")
    if not vals:
        return
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    stages = sorted({v["stage"] for v in vals}, key=lambda s: next(v["stage_idx"] for v in vals if v["stage"] == s))
    colors = plt.get_cmap("tab10")(range(len(stages)))

    for i, stage in enumerate(stages):
        stage_vals = [v for v in vals if v["stage"] == stage]
        steps = [v["global_step"] for v in stage_vals]
        ax_loss.plot(steps, [v["train_loss"] for v in stage_vals], color=colors[i], marker=".", label=stage)
        ax_acc.plot(steps, [v["validation_accuracy"] for v in stage_vals], color=colors[i], marker=".", label=stage)

    ax_loss.set_xlabel("global step")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_acc.set_xlabel("global step")
    ax_acc.set_ylabel("validation accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.axhline(0.333, color="red", linestyle=":", alpha=0.5, label="chance (1/3)")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_metric_trajectory(events, out_path: Path):
    stage_metrics = by_event(events, "stage_metrics")
    final = by_event(events, "final_metrics")
    if not stage_metrics:
        return

    stages = [m["stage"] for m in stage_metrics]
    x = list(range(len(stages)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # comp_idx + topo_sim
    ax = axes[0, 0]
    ax.plot(x, [m["composition_index"] for m in stage_metrics], "o-", label="composition_index")
    ax.plot(x, [m["topo_sim"] for m in stage_metrics], "s-", label="topographic_similarity")
    if final:
        ax.plot(len(stages) - 0.5, final[0]["composition_index"], "o", markersize=15,
                markeredgecolor="black", markerfacecolor="none", label="final (larger sample)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, fontsize=8)
    ax.set_ylabel("metric")
    ax.set_title("Compositional signal across curriculum")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # rollout accuracy
    ax = axes[0, 1]
    ax.plot(x, [m["rollout_accuracy"] for m in stage_metrics], "o-", color="green", label="rollout_acc")
    # extrap_near
    extrap_values = []
    for m in stage_metrics:
        if m.get("extrapolation") and "near_extrap" in m["extrapolation"]:
            extrap_values.append(m["extrapolation"]["near_extrap"])
        else:
            extrap_values.append(np.nan)
    ax.plot(x, extrap_values, "s-", color="orange", label="extrap near")
    ax.axhline(0.333, color="red", linestyle=":", alpha=0.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_title("Rollout & extrapolation accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # best base
    ax = axes[1, 0]
    best_bases = [m["best_base"] for m in stage_metrics]
    ax.bar(x, best_bases, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, fontsize=8)
    ax.set_ylabel("best base (max diagonal MI)")
    ax.set_title("Emergent base (base-scanning winner)")
    ax.grid(alpha=0.3, axis="y")

    # position 0 R² vs total R²
    ax = axes[1, 1]
    pos0_r2 = [m["per_position_r2"][0] for m in stage_metrics]
    total_r2 = [m["total_r2"] for m in stage_metrics]
    ax.plot(x, pos0_r2, "o-", label="position 0 alone", color="C0")
    ax.plot(x, total_r2, "s-", label="all 5 positions", color="C1")
    ax.fill_between(x, pos0_r2, total_r2, alpha=0.2, color="C1", label="composition gain")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=30, fontsize=8)
    ax.set_ylabel("R² predicting N")
    ax.set_title("Probe R²: position 0 vs all positions")
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_position_r2(events, out_path: Path):
    stage_metrics = by_event(events, "stage_metrics")
    final = by_event(events, "final_metrics")
    if not stage_metrics:
        return

    stages = [m["stage"] for m in stage_metrics]
    W = len(stage_metrics[0]["per_position_r2"])
    positions = list(range(W))

    fig, (ax_r2, ax_gain) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.9, len(stages)))
    for i, m in enumerate(stage_metrics):
        ax_r2.plot(positions, m["per_position_r2"], "o-", color=colors[i], label=m["stage"])
        ax_gain.plot(positions, m["marginal_gains"], "o-", color=colors[i], label=m["stage"])
    if final:
        f = final[0]
        ax_r2.plot(positions, f["per_position_r2"], "*-", color="red",
                   markersize=12, linewidth=2, label="FINAL (512 ep)")
        ax_gain.plot(positions, f["marginal_gains"], "*-", color="red",
                     markersize=12, linewidth=2, label="FINAL (512 ep)")

    ax_r2.set_xlabel("notes positions used [0..j]")
    ax_r2.set_ylabel("cumulative R² predicting N")
    ax_r2.set_title("Cumulative probe R² as positions added")
    ax_r2.set_xticks(positions)
    ax_r2.set_ylim(0.6, 1.0)
    ax_r2.grid(alpha=0.3)
    ax_r2.legend(fontsize=9)

    ax_gain.set_xlabel("notes position j")
    ax_gain.set_ylabel("marginal R² gain at position j")
    ax_gain.set_title("Marginal R² added by each position")
    ax_gain.set_xticks(positions)
    ax_gain.axhline(0, color="gray", linewidth=0.5)
    ax_gain.set_yscale("symlog", linthresh=0.001)
    ax_gain.grid(alpha=0.3)
    ax_gain.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_base_scores(events, out_path: Path):
    stage_metrics = by_event(events, "stage_metrics")
    final = by_event(events, "final_metrics")
    if not stage_metrics:
        return

    stages = [m["stage"] for m in stage_metrics]
    if final:
        stages = stages + ["final"]

    all_bases_set = set()
    for m in stage_metrics:
        all_bases_set.update(int(b) for b in m["all_alignments"].keys())
    bases = sorted(all_bases_set)

    # Matrix: rows = stages, cols = bases
    matrix = np.zeros((len(stages), len(bases)))
    for i, m in enumerate(stage_metrics):
        for j, b in enumerate(bases):
            matrix[i, j] = m["all_alignments"].get(str(b), 0.0)
    if final:
        for j, b in enumerate(bases):
            matrix[-1, j] = final[0]["all_alignments"].get(str(b), 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels(bases)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=9)
    ax.set_xlabel("candidate base")
    ax.set_title("Base-scanning MI diagonal score (brighter = more place-value-like at this base)")

    # Annotate cells
    for i in range(len(stages)):
        for j in range(len(bases)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.mean() else "black",
                    fontsize=8)

    # Highlight best base per stage
    for i in range(len(stages)):
        best = matrix[i].argmax()
        ax.add_patch(plt.Rectangle((best - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="red", linewidth=2))

    fig.colorbar(im, ax=ax, label="diagonal_score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_weber_curves(events, out_path: Path):
    """Plot accuracy(N_alpha) curves per stage — the Weber's-law diagnostic."""
    stage_metrics = by_event(events, "stage_metrics")
    final = by_event(events, "final_metrics")
    has_weber = any(m.get("weber_n_values") for m in stage_metrics + final)
    if not has_weber:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.9, max(1, len(stage_metrics))))
    for i, m in enumerate(stage_metrics):
        ns = m.get("weber_n_values", [])
        accs = m.get("weber_accuracy", [])
        if not ns or not accs:
            continue
        label = f"{m['stage']} (slope={m.get('weber_slope', 0):+.2f})"
        ax.plot(ns, accs, "o-", color=colors[i], label=label, alpha=0.8)
    for m in final:
        ns = m.get("weber_n_values", [])
        accs = m.get("weber_accuracy", [])
        if ns and accs:
            ax.plot(ns, accs, "*-", color="red", markersize=11, linewidth=2,
                    label=f"FINAL (slope={m.get('weber_slope', 0):+.2f})")

    ax.axhline(1 / 3, color="gray", linestyle=":", alpha=0.6, label="chance (1/3)")
    ax.axhline(0.85, color="green", linestyle="--", alpha=0.4,
               label="counting threshold (~0.85)")
    ax.set_xlabel(r"$N_\alpha$")
    ax.set_ylabel(r"3-class accuracy on $N_\beta \in \{N_\alpha \pm 1, N_\alpha\}$")
    ax.set_title("Weber curve — flat = counting, decreasing = magnitude/log encoding")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary(events, out_path: Path):
    stage_metrics = by_event(events, "stage_metrics")
    final = by_event(events, "final_metrics")
    lines = []

    lines.append("=" * 70)
    lines.append(f"Stage 1 run summary — {out_path.parent.parent.name}")
    lines.append("=" * 70)

    run_start = next((e for e in events if e.get("event") == "run_start"), None)
    if run_start:
        lines.append(f"Device:        {run_start.get('device', '?')}")
        lines.append(f"Params:        {run_start.get('num_parameters', '?')}")
        lines.append(f"Scene L / W / T_gap / K: "
                     f"{run_start['scene_cfg']['L']} / {run_start['scene_cfg']['W']} / "
                     f"{run_start['scene_cfg']['T_gap']} / {run_start['scene_cfg']['K']}")
    lines.append("")

    lines.append(f"{'Stage':<20} {'rollout':>8} {'topo':>7} {'comp_idx':>9} "
                 f"{'base':>5} {'diag':>6} {'extrap':>7}")
    lines.append("-" * 70)
    for m in stage_metrics:
        extrap = m.get("extrapolation", {}).get("near_extrap", float("nan"))
        lines.append(
            f"{m['stage']:<20} {m['rollout_accuracy']:>8.3f} {m['topo_sim']:>7.3f} "
            f"{m['composition_index']:>9.3f} {m['best_base']:>5d} "
            f"{m['best_alignment']:>6.3f} {extrap:>7.3f}"
        )
    if final:
        f = final[0]
        extrap_values = list(f.get("extrapolation_full", {}).values())
        extrap_disp = f"{np.mean(extrap_values):.3f}" if extrap_values else "-"
        lines.append(
            f"{'FINAL (512 ep)':<20} {f['rollout_accuracy']:>8.3f} {f['topo_sim']:>7.3f} "
            f"{f['composition_index']:>9.3f} {f['best_base']:>5d} "
            f"{f['best_alignment']:>6.3f} {extrap_disp:>7}"
        )

    lines.append("")
    lines.append("Per-position R² and marginal gains (final metrics):")
    if final:
        f = final[0]
        for j, (r2, gain) in enumerate(zip(f["per_position_r2"], f["marginal_gains"])):
            lines.append(f"  position {j}: R²={r2:.3f}  gain={gain:+.3f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()

    events = load_events(args.log_dir)
    out_dir = args.log_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(events, out_dir / "training_curves.png")
    plot_metric_trajectory(events, out_dir / "metric_trajectory.png")
    plot_per_position_r2(events, out_dir / "per_position_r2.png")
    plot_base_scores(events, out_dir / "base_scores.png")
    plot_weber_curves(events, out_dir / "weber_curves.png")
    write_summary(events, out_dir / "summary_table.txt")

    print(f"Wrote figures to: {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
