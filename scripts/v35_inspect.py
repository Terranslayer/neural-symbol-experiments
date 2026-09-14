"""V35 inspect: layer A spike count + per-recursion residuals + scratch codebook
per N, on a trained V35 ckpt. Designed to diagnose the cell-cascade-collapse
hypothesis (layer A under-firing -> cascade dead chain -> cells 2-4 dead).

Captures intermediates by re-running CNN encoder + V35Substrate manually on
alpha segment after the standard agent.forward (which populates stashes for
scratch and successor logits). The substrate has no mutable cross-call state,
so re-running is safe.

SUCCESS CRITERIA (user correction 6/03 — read before judging a ckpt):
  The two REAL targets are (1) Layer A COUNTS: spkA(N) ≈ N (slope≈1, monotone +),
  and (2) PROGRESSIVE position-opening: high recursions stay SILENT at low N
  (under base θ, residual_k opens at N≥θ^(k-1); recursion k fires at N≥θ^k — so
  res4 silent until N≥27, res5 until N≥81 for θ=3).
  ANTI-PATTERN: "more distinct codes" / "all res1-5 active across N=1..30" /
  "high-position res revived" is NOT success — it is the FLOODING signature
  (substrate saturated, high-N codes are noise). The `_counting_and_schedule_check`
  block below turns these into explicit verdicts so this is not mis-read again.

Usage:
  python scripts/v35_inspect.py <ckpt_path> [--n-min 1] [--n-max 30] [--per-n 32]

CRITICAL: matches training-time _v35_forward exactly:
  - same dp_convs/dp_proj weights
  - same input channel zeroing (inputs[:, :, 1:3] = 0)
  - same v35_substrate call (fresh state each invocation)
  - bidirectional class mapping per train_phase1.py V35 branch
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np
import torch

# Make the project root importable when run as `python scripts/v35_inspect.py`
# (Python puts the script's dir on sys.path, not the cwd).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.scene import SceneConfig, build_episode, sample_beta_count
from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
from backend.core.label_mapping import eval_k_to_class


def _get_attr(cfg, key, default=None):
    """Read field from agent_cfg whether it's a dataclass instance or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _counting_and_schedule_check(per_n_stats, n_min, n_max, n_recursions, base, theta):
    """Turn the two REAL success criteria (user 6/03) into explicit verdicts so no
    future session re-chases 'all residuals firing' (flooding) as if it were progress.

      (1) Layer A counting: spkA(N) must track N (slope≈1, monotone +). spkA ≫ N =
          over-firing/flooding; slope<0 = inverted tuning (the recurring V19/V25/V28/V35
          attractor — readout still decodes monotone but the mechanism is wrong).
      (2) Progressive position-opening: high recursions must be SILENT at low N. Under
          base θ, residual_k (the digit cell k reads) only varies at N≥θ^(k-1), and
          recursion k only FIRES at N≥θ^k. res_k active below that = flooding, NOT a
          richer codebook.

    Flooding test is base-free where it matters: at the smallest N, NO recursion k≥2
    can legitimately have fired (needs ≥ base^k objects), so spkB_k(N_min)>0 for k≥2 is
    unambiguous flooding regardless of how θ_b actually calibrated.
    """
    Ns = list(range(n_min, n_max + 1))
    Narr = np.array(Ns, dtype=float)
    spkA = np.array([per_n_stats[N]["spike_A_mean"] for N in Ns])

    print("\n=== (1) Layer A counting check: spkA should TRACK N (ideal slope ~ +1) ===")
    a = float(np.polyfit(Narr, spkA, 1)[0]) if len(Ns) >= 2 else float("nan")
    if len(Ns) >= 2 and spkA.std() > 0:
        ra = np.argsort(np.argsort(spkA)).astype(float)
        rn = np.argsort(np.argsort(Narr)).astype(float)
        rho = float(np.corrcoef(ra, rn)[0, 1])
    else:
        rho = float("nan")
    ratio_lo = spkA[0] / max(Narr[0], 1e-9)
    ratio_hi = spkA[-1] / max(Narr[-1], 1e-9)
    print(f"  spkA/N:  N={n_min} -> {ratio_lo:.1f}x ,  N={n_max} -> {ratio_hi:.1f}x      (ideal ~1x at both)")
    print(f"  linear slope d(spkA)/dN = {a:+.2f}      (ideal ~+1 ; < 0 = INVERTED ; >> 1 = OVER-FIRING)")
    print(f"  Spearman(spkA, N) = {rho:+.3f}            (ideal +1)")
    over_firing = (ratio_lo > 2.0 or ratio_hi > 2.0)
    if rho < 0:
        verdict_a = ("INVERTED tuning (more N -> fewer spikes)"
                     + (" + OVER-FIRING" if over_firing else "")
                     + " -- Layer A is not counting objects")
        ok_count = False
    elif over_firing:
        verdict_a = "OVER-FIRING / FLOODING (spkA >> N) -- Layer A is not counting objects"
        ok_count = False
    elif a < 0.3:
        verdict_a = "UNDER-FIRING / FLAT (spkA barely tracks N)"
        ok_count = False
    else:
        verdict_a = "COUNTING OK (spkA tracks N)"
        ok_count = True
    print(f"  VERDICT: {verdict_a}")

    print("\n=== (2) Position-opening schedule: high recursions SILENT at low N ===")
    print(f"  base theta_b ~ {base} (from theta={theta:.2f}).  res_k digit opens at N>=theta^(k-1);  recursion k fires at N>=theta^k.")
    print(f"  {'rec':>3} | {'res opens@N':>11} | {'fires@N':>8} | {'spkB@Nmin':>9} | {'emp.open@N':>11} | verdict")
    engage = 0.5
    n_flood_lo = 0  # recursions k>=2 wrongly firing at the smallest N (base-free flooding count)
    for k in range(1, n_recursions + 1):
        res_open = base ** (k - 1)
        fire_open = base ** k
        spkB_k = np.array([per_n_stats[N]["spike_B_mean"][k - 1] for N in Ns])
        spkB_at_min = float(spkB_k[0])
        emp_idx = np.where(spkB_k > engage)[0]
        emp_open = str(Ns[int(emp_idx[0])]) if emp_idx.size else "--(silent)"
        below = spkB_k[Narr < fire_open]               # N where recursion k should NOT yet fire
        flooding_k = (below.size > 0 and float(below.mean()) > engage)
        if k >= 2 and n_min < fire_open and spkB_at_min > engage:
            n_flood_lo += 1
        if fire_open > n_max:
            verdict_k = "FLOODING (should be silent in range)" if flooding_k else "OK (silent; opens beyond range)"
        else:
            verdict_k = "FLOODING (fires below open@N)" if flooding_k else "OK (silent below, engages above)"
        print(f"  {k:>3} | {res_open:>11} | {fire_open:>8} | {spkB_at_min:>9.2f} | {emp_open:>11} | {verdict_k}")

    print()
    if n_flood_lo == 0 and ok_count:
        print("  GLOBAL: PROGRESSIVE OPENING [OK]  (only low recursions at low N, and spkA counts) -- read the codebook.")
    elif n_flood_lo >= 1:
        print(f"  GLOBAL: FLOODING [FAIL]  ({n_flood_lo} high recursion(s) already firing at N={n_min}; substrate saturated,")
        print("          NOT positional. high-N codes are NOISE -- do NOT read 'more codes' as progress. Fix spkA~N upstream.")
    else:
        print("  GLOBAL: upstream counting issue (see check 1) -- fix spkA~N before trusting the codebook.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--n-min", type=int, default=1)
    ap.add_argument("--n-max", type=int, default=30)
    ap.add_argument("--per-n", type=int, default=32, help="Episodes per N value")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    agent_cfg = ckpt["agent_cfg"]
    scene_cfg = ckpt["scene_cfg"]

    use_v35 = _get_attr(agent_cfg, "use_v35", False)
    assert use_v35, f"ckpt is not a V35 run (use_v35 != True). agent_cfg type: {type(agent_cfg)}"

    # If agent_cfg was saved as dict (older save), rebuild dataclass for MambaAgent ctor.
    if isinstance(agent_cfg, dict):
        agent_cfg = MambaAgentConfig(**agent_cfg)
    # Same for scene_cfg
    if isinstance(scene_cfg, dict):
        scene_cfg = SceneConfig(**scene_cfg)

    # V35 _v35_forward assumes K=1 (asserted at forward time); reject K!=1 ckpts here too.
    assert scene_cfg.K == 1, (
        f"V35 inspect requires scene_cfg.K == 1 (got K={scene_cfg.K}). "
        "Non-canonical ckpt — V35 was trained with K=1 (successor_prediction_preset)."
    )

    agent = MambaAgent(agent_cfg, scene_cfg)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent = agent.to(device)
    agent.train(False)  # inference mode (disables dropout etc.; equivalent to .eval())

    theta = agent.v35_substrate.cascade.theta().item()
    n_recursions = agent_cfg.v35_n_recursions
    W = scene_cfg.W
    L = scene_cfg.L
    scene_bidir = not scene_cfg.successor_only_positive
    # 6/03: detector mode. For threshold-layer-a / food_pattern ckpts the manual cascade
    # re-run below MUST use the same spike path as training (detector ->
    # forward_with_external_layer_a_spike), NOT the HH Layer-A path -- else spkA/spkB/res are
    # HH-path artifacts (spkA=0, res=baseline; the old documented caveat).
    threshold_la = getattr(agent_cfg, "v35_threshold_layer_a", False)
    la_detect = getattr(agent_cfg, "v35_layer_a_detect", "level")
    la_thresh = getattr(agent_cfg, "v35_layer_a_threshold", 0.5)
    la_center = getattr(agent_cfg, "v35_layer_a_center_thresh", 0.9)
    la_neighbor = getattr(agent_cfg, "v35_layer_a_neighbor_thresh", 0.8)
    # Derive n_classes the same way mamba_agent.py builds V35CompareAttention:
    #   n_classes = max(agent_cfg.successor_predict_k_max, 2)
    # We'll also assert after the first forward that actual logit width matches.
    n_classes = max(agent_cfg.successor_predict_k_max, 2)

    print(f"=== V35 inspect: ckpt = {args.ckpt} ===")
    print(f"theta = {theta:.4f}    (init was 3.0)")
    _la = agent.v35_substrate.cascade.layer_A
    print(f"log_g_leak_A = {float(_la.log_g.data[3]):.3f}   (idx3=leak; -4 = no leak)")
    print(f"warm_b_leak_A = ||B_leak_A|| = {float(_la.B_leak.weight.norm()):.4f}")
    print(f"n_recursions = {n_recursions},  W (scratch cells) = {W},  L (signal) = {L}")
    print(f"substrate path (matches training) = {'threshold-layer-a / detect=' + la_detect if threshold_la else 'HH Layer A'}")
    print(f"scene_cfg.successor_only_positive = {scene_cfg.successor_only_positive}  (-> bidir = {scene_bidir})")
    print(f"agent_cfg.successor_predict_k_max = {n_classes}  (V35 compare n_classes)")
    print()

    rng = np.random.default_rng(seed=args.seed)
    per_n_stats = {}
    successor_correct = 0
    successor_total = 0
    all_scratch_codes_int = []  # list of W-tuples in {0,1,2}

    with torch.no_grad():
        for N in range(args.n_min, args.n_max + 1):
            # Build per_n episodes with N_alpha = N
            inputs_list, metas_list = [], []
            for _ in range(args.per_n):
                beta_counts = [
                    sample_beta_count(N, args.n_max, scene_cfg, rng)
                    for _ in range(scene_cfg.K)
                ]
                inp, _lbl, _cidx, meta = build_episode(N, beta_counts, scene_cfg, rng)
                inputs_list.append(inp)
                metas_list.append(meta)
            inputs = torch.stack(inputs_list).to(device)  # (B, T, 3)
            compare_indices = torch.zeros(args.per_n, scene_cfg.K, dtype=torch.long, device=device)

            # Standard forward populates _scratch_alpha_for_aux + _v35_successor_logits
            agent(inputs, compare_indices)
            scratch_q = agent._scratch_alpha_for_aux  # (B, W) in {0, 0.5, 1.0}
            v35_logits = agent._v35_successor_logits  # (B, n_classes)
            assert scratch_q is not None and v35_logits is not None
            # Verify our pre-derived n_classes matches actual logit width.
            assert v35_logits.shape[-1] == n_classes, (
                f"n_classes mismatch: derived {n_classes} vs actual {v35_logits.shape[-1]}"
            )

            # Re-run cascade manually to capture intermediates (substrate has no
            # mutable cross-call state, so this is safe & matches forward).
            inputs_diag = inputs.clone()
            inputs_diag[:, :, 1:3] = 0.0
            if threshold_la:
                # Same spike path as training (food_pattern/level detector -> external-spike
                # cascade), NOT the HH Layer-A path -> spkA/spkB/res reflect real training.
                if la_detect == "food_pattern":
                    from backend.core.v35 import food_pattern_spike_gate
                    spike_a = food_pattern_spike_gate(inputs_diag[:, :L, 0], la_center, la_neighbor)
                else:
                    spike_a = (inputs_diag[:, :L, 0] > la_thresh).float()
                residuals, aux = agent.v35_substrate.forward_with_external_layer_a_spike(spike_a)
            else:
                alpha_in = inputs_diag[:, :L, :]
                x = alpha_in.transpose(1, 2)
                conv_outs = [conv(x) for conv in agent.dp_convs]
                feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
                feat_alpha = agent.dp_proj(feat)
                residuals, aux = agent.v35_substrate(feat_alpha)  # HH Layer A path

            # Per-sample diagnostics
            spike_A_count = aux["spike_train_A"].sum(dim=1).cpu().numpy()  # (B,)
            # spike_trains_B is list of (B, L) tensors; sum over L per recursion
            spike_B_counts = torch.stack(
                [s.sum(dim=1) for s in aux["spike_trains_B"]], dim=-1
            ).cpu().numpy()  # (B, n_recursions)
            residuals_np = residuals.cpu().numpy()  # (B, n_recursions)

            # Scratch codes in alphabet {0, 0.5, 1.0} -> integer {0, 1, 2}
            scratch_int = (scratch_q.detach().cpu().numpy() * 2).round().astype(int)  # (B, W)
            assert ((scratch_int >= 0) & (scratch_int <= 2)).all(), \
                f"scratch out of alphabet at N={N}: {scratch_int[~((scratch_int>=0)&(scratch_int<=2))]}"

            # Successor preds vs labels (bidirectional class mapping)
            preds = v35_logits.argmax(dim=-1).cpu().numpy()
            targets = np.array([
                eval_k_to_class(m["beta_counts"][0] - m["N_alpha"], scene_cfg, n_classes)
                for m in metas_list
            ])
            correct = int((preds == targets).sum())
            successor_correct += correct
            successor_total += args.per_n

            per_n_stats[N] = {
                "spike_A_mean": float(spike_A_count.mean()),
                "spike_A_std": float(spike_A_count.std()),
                "spike_A_min": int(spike_A_count.min()),
                "spike_A_max": int(spike_A_count.max()),
                "spike_B_mean": spike_B_counts.mean(axis=0),  # (n_recursions,)
                "residual_mean": residuals_np.mean(axis=0),
                "residual_std": residuals_np.std(axis=0),
                "scratch_codes": [tuple(row.tolist()) for row in scratch_int],
                "acc_at_n": correct / args.per_n,
            }
            all_scratch_codes_int.extend(per_n_stats[N]["scratch_codes"])

    # Print per-N table
    print("=== Per-N statistics ===")
    print("Layer A spike count (over L timesteps, ideal ~N) + Layer B per-recursion spike count + residuals + modal code")
    print()
    header = f"{'N':>3} | {'spkA mn':>7} {'std':>5} {'mn-mx':>7} | " + \
             " ".join(f"{'spkB'+str(k+1):>5}" for k in range(n_recursions)) + " | " + \
             " ".join(f"{'res'+str(k+1):>6}" for k in range(n_recursions)) + " | " + \
             f"{'modal code':>13} | {'acc':>5}"
    print(header)
    print("-" * len(header))
    for N in range(args.n_min, args.n_max + 1):
        s = per_n_stats[N]
        sb = s["spike_B_mean"]
        rs = s["residual_mean"]
        cnt = Counter(s["scratch_codes"])
        modal_code, modal_count = cnt.most_common(1)[0]
        modal_str = str(modal_code) + f"x{modal_count}"
        print(f"{N:>3} | {s['spike_A_mean']:>7.2f} {s['spike_A_std']:>5.2f} {str(s['spike_A_min'])+'-'+str(s['spike_A_max']):>7} | "
              + " ".join(f"{sb[k]:>5.2f}" for k in range(n_recursions)) + " | "
              + " ".join(f"{rs[k]:>+6.3f}" for k in range(n_recursions)) + " | "
              + f"{modal_str:>13} | {s['acc_at_n']:>5.3f}")

    # Counting + position-opening schedule verdicts (the REAL success criteria).
    # Placed before the codebook summary on purpose: judge the substrate FIRST, so
    # "distinct codes" below is read in context (a fat codebook over a flooded
    # substrate is noise, not progress — user correction 6/03).
    base = max(2, int(round(theta)))
    _counting_and_schedule_check(per_n_stats, args.n_min, args.n_max, n_recursions, base, theta)

    # Distinct codes + cell utilization
    counter = Counter(all_scratch_codes_int)
    total_samples = sum(counter.values())
    print(f"\n=== Codebook summary ===")
    print(f"Distinct scratch codes seen across N={args.n_min}..{args.n_max}: {len(counter)}  / capacity 3^{W} = {3**W}")
    print(f"Top 10 most frequent codes:")
    for code, cnt in counter.most_common(10):
        print(f"  {code} : {cnt} ({cnt/total_samples:.1%})")

    scratch_arr = np.array(all_scratch_codes_int)  # (total_samples, W)
    print(f"\n=== Per-cell utilization (W={W}) ===")
    print(f"{'cell':>4} | {'%val=0':>7} {'%val=1':>7} {'%val=2':>7} | {'unique':>7}")
    for k in range(W):
        col = scratch_arr[:, k]
        p0 = (col == 0).mean()
        p1 = (col == 1).mean()
        p2 = (col == 2).mean()
        uniq = sorted(set(col.tolist()))
        print(f"{k:>4} | {p0:>7.1%} {p1:>7.1%} {p2:>7.1%} | {str(uniq):>7}")

    # --- (4) Detection: base-theta place-value vs lookup cipher vs thermometer (spec 2026-06-05 §6) ---
    try:
        from backend.evaluation.v35_detection import classify_codebook
        qm1 = max(1, 3 - 1)  # levels {0..q-1} -> {0,0.5,1.0}
        modal_by_n = {}
        for N in sorted(per_n_stats):
            modal_int = Counter(per_n_stats[N]["scratch_codes"]).most_common(1)[0][0]
            modal_by_n[N] = tuple(c / qm1 for c in modal_int)
        det = classify_codebook(modal_by_n, n_recursions, theta=int(round(theta)))
        c3 = det["carry"]
        print(f"\n=== (4) Detection: base-theta vs cipher vs thermometer (spec 2026-06-05 §6) ===")
        print(f"  VERDICT: {det['verdict']}   (n_dead_cells={det['n_dead_cells']})")
        print(f"  D3 carry-consistency (PRIMARY base-3 vs cipher): "
              f"mean_inc_hamming={c3['mean_increment_hamming']:.2f}  "
              f"low_cell_only={c3['low_cell_only_frac']:.2f}  boundary_hit={c3['boundary_hit_frac']:.2f}")
        print(f"     (place-value: small hamming ~1, high low_cell_only, boundary_hit ~1 ; cipher: dense, boundary_hit ~chance)")
        print(f"  D2 high-cell fineness (band distinct levels; >=2 = fine, 1 = thermometer/dead): {det['high_cell_fineness']}")
        print(f"  NOTE: read AFTER the counting gate above — a fat codebook over a flooded substrate is noise (6/03).")
    except Exception as _e:
        print(f"\n[detection] skipped: {_e}")

    print(f"\n=== Overall successor acc ===")
    print(f"{successor_correct}/{successor_total} = {successor_correct/successor_total:.3f}    chance ({n_classes}-class) = {1.0/n_classes:.3f}")


if __name__ == "__main__":
    main()
