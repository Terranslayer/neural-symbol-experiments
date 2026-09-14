# -*- coding: utf-8 -*-
"""Extract key training metrics from a run.jsonl into a single tidy table."""
import json, sys, argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=str)
    args = p.parse_args()

    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ev = o.get("event", "")
            if ev not in ("stage_metrics", "final_metrics"):
                continue
            si = o.get("stage_idx", "final")
            rollout = o.get("rollout_accuracy", {})
            acc = rollout.get("acc") if isinstance(rollout, dict) else rollout
            extrap = o.get("extrapolation", {})
            comp = o.get("composition_index")
            topo = o.get("topo_sim")
            r2 = o.get("total_r2")
            near = extrap.get("near_extrap") if isinstance(extrap, dict) else None
            far = extrap.get("far_extrap") if isinstance(extrap, dict) else None
            best_base = o.get("best_base")
            best_diag = o.get("best_diagonal_score")
            best_align = o.get("best_alignment")
            wf = o.get("weber_flatness")
            ws = o.get("weber_slope")
            row = {
                "stage": si,
                "rollout_acc": round(acc, 3) if isinstance(acc, (int, float)) else acc,
                "comp_idx": round(comp, 3) if isinstance(comp, (int, float)) else comp,
                "topo": round(topo, 3) if isinstance(topo, (int, float)) else topo,
                "r2": round(r2, 3) if isinstance(r2, (int, float)) else r2,
                "near_extrap": round(near, 3) if isinstance(near, (int, float)) else near,
                "far_extrap": round(far, 3) if isinstance(far, (int, float)) else far,
                "best_base": best_base,
                "old_diag": round(best_diag, 3) if isinstance(best_diag, (int, float)) else best_diag,
                "fair_align": round(best_align, 3) if isinstance(best_align, (int, float)) else best_align,
                "weber_flat": round(wf, 3) if isinstance(wf, (int, float)) else wf,
                "weber_slope": round(ws, 3) if isinstance(ws, (int, float)) else ws,
            }
            print(json.dumps(row))


if __name__ == "__main__":
    main()
