"""Inter-trial symbol consistency probe for the active-agent.

For each fixed N in a test set, run many independent trials with:
  - fresh world sampled with EXACTLY that N,
  - target_count fixed = N (label = equal; removes target-driven write variance)
Let the agent complete the session. At emit (or timeout), snapshot the
active_page (the 4 content slots — the agent's notebook for this session).

Reports per N:
  - per-slot mean / std across trials
  - distinct quantized (level_0, level_1, level_2, level_3) tuples
  - mode-tuple match rate  (higher = more stable "symbol" for this N)

Across N:
  - per-slot correlation with N (distinguishability)
  - Total tuple diversity (how many distinct tuples used for the set of N)

Usage:
  python scripts/consistency_probe.py path/to/final.pt
      [--n-trials 50] [--n-values "1,3,5,8,10,15,25,33"]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.active_agent import (
    ACTION_KINDS, ActivePolicy, ActivePolicyConfig,
    action_to_record, sample_action,
)
from backend.core.hier_active_agent import HierActivePolicy, HierActivePolicyConfig
from backend.core.world_event import (
    NBucket, WorldConfig, WorldEngine, generate_world_pool,
)
from backend.core.scene import ALPHA_EQUAL, _scan_world_complex


LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)


def quantize_tuple(notes):
    return tuple(int(np.abs(LEVELS - v).argmin()) for v in notes)


def run_one_trial(pol, pol_cfg, world_cfg, N, is_hier, rng):
    """Run a single session with alpha_count=N, target=N (equal case),
    return (final_notes, correct, session_ticks)."""
    # Build a world with exactly N foods
    world_signal = _scan_world_complex(N, world_cfg.L, rng).astype(np.float32)

    # Manually construct engine with a 1-world pool for deterministic sample
    engine = WorldEngine(
        world_cfg, n_max=N + 1, t_budget=80,
        notebook_capacity=pol_cfg.n_review_pages,
        rng=rng, pool_worlds=[world_signal], pool_Ns=[N],
    )
    # Override _cue_norm so target/_cue_norm matches training-time scale
    engine._cue_norm = 120.0

    # Force session to be equal (target = N)
    from backend.core.world_event import Session
    engine.state.current_session = Session(
        world_signal=world_signal.copy(),
        alpha_count=N, target_count=N, label=ALPHA_EQUAL,
        t_budget=(world_cfg.t_observe + world_cfg.t_gap + world_cfg.t_answer)
                  if world_cfg.phased else 80,
    )
    engine.state.t = 0
    engine.state.attn_pos = 0.0 if world_cfg.auto_scan else float(world_cfg.L // 2)
    engine.state.active_page = np.zeros(4, dtype=np.float32)

    if is_hier:
        latent = torch.zeros(1, pol_cfg.n_latent_tokens, pol_cfg.d_model)
    else:
        latent = torch.zeros(1, pol_cfg.d_model)

    session_ticks = 0
    correct = False
    while True:
        obs = engine.observe()
        window = torch.tensor(obs["window"], dtype=torch.float32).unsqueeze(0)
        active = torch.tensor(obs["active_page"], dtype=torch.float32).unsqueeze(0)
        target_cue = torch.tensor([obs["target_cue"]], dtype=torch.float32)
        past = [engine.read_page(i) for i in range(pol_cfg.n_review_pages)]
        past_pages = torch.tensor(np.stack(past), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = pol(latent, window, active, target_cue, past_pages)
        latent = out["new_latent"]
        action, _ = sample_action(out, deterministic=False)
        rec = action_to_record(action, 0)
        notes_snapshot = engine.state.active_page.copy()
        r, done = engine.step(rec)
        session_ticks += 1
        if done:
            correct = (r > 0.5)
            break
        if session_ticks > 200:  # safety
            break
    return notes_snapshot, correct, session_ticks


def probe(ckpt_path, n_values, n_trials):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["pol_cfg"]
    is_hier = "n_latent_tokens" in raw and raw["n_latent_tokens"] > 1
    if is_hier:
        pol_cfg = HierActivePolicyConfig(
            **{k: raw[k] for k in HierActivePolicyConfig.__dataclass_fields__ if k in raw}
        )
        pol = HierActivePolicy(pol_cfg)
    else:
        pol_cfg = ActivePolicyConfig(
            **{k: raw[k] for k in ActivePolicyConfig.__dataclass_fields__ if k in raw}
        )
        pol = ActivePolicy(pol_cfg)
    pol.load_state_dict(ckpt["policy_state"])
    pol.train(False)

    wcfg_raw = ckpt["world_cfg"]
    world_cfg = WorldConfig(
        **{k: wcfg_raw[k] for k in WorldConfig.__dataclass_fields__ if k in wcfg_raw}
    )
    print(f"  hier={is_hier}, phased={world_cfg.phased}, L={world_cfg.L}, "
          f"t_obs={world_cfg.t_observe}, t_gap={world_cfg.t_gap}, t_ans={world_cfg.t_answer}")

    # Aggregate per-N data
    per_N = {n: [] for n in n_values}
    per_N_correct = {n: [] for n in n_values}
    per_N_ticks = {n: [] for n in n_values}

    for N in n_values:
        rng = np.random.default_rng(1000 + N)
        for trial in range(n_trials):
            notes, ok, ticks = run_one_trial(pol, pol_cfg, world_cfg, N, is_hier, rng)
            per_N[N].append(notes)
            per_N_correct[N].append(ok)
            per_N_ticks[N].append(ticks)
        arr = np.stack(per_N[N], axis=0)
        correct_rate = np.mean(per_N_correct[N])
        tuples = [quantize_tuple(a) for a in arr]
        tc = Counter(tuples)
        mode_tuple, mode_count = tc.most_common(1)[0]
        mode_rate = mode_count / len(tuples)
        print(f"\nN = {N:3d}  n={n_trials}  equal_acc={correct_rate:.2f}  "
              f"avg_ticks={np.mean(per_N_ticks[N]):.0f}")
        print(f"  per-slot mean: {arr.mean(axis=0).round(3)}")
        print(f"  per-slot std:  {arr.std(axis=0).round(3)}")
        print(f"  distinct quantized tuples: {len(tc)}   (out of {6**4} possible)")
        print(f"  mode tuple: {mode_tuple}  seen {mode_count}/{n_trials} "
              f"({mode_rate:.0%})")
        top3 = tc.most_common(3)
        for i, (t, c) in enumerate(top3):
            print(f"    #{i+1}: {t}  x{c}")

    # Cross-N distinguishability
    print("\n=== Cross-N encoding distinguishability ===")
    all_N = np.concatenate([[n] * n_trials for n in n_values])
    all_notes = np.concatenate([np.stack(per_N[n]) for n in n_values], axis=0)
    for slot in range(4):
        if all_notes[:, slot].std() > 1e-6:
            r = np.corrcoef(all_notes[:, slot], all_N.astype(float))[0, 1]
            print(f"  pos{slot} corr with N: {r:+.3f}")
        else:
            print(f"  pos{slot}: constant (no correlation info)")

    # Is the mode tuple for each N unique?
    mode_tuples = {}
    for N in n_values:
        tuples = [quantize_tuple(a) for a in per_N[N]]
        mode_tuples[N] = Counter(tuples).most_common(1)[0][0]
    distinct_modes = len(set(mode_tuples.values()))
    print(f"\n  distinct mode-tuples across {len(n_values)} N values: {distinct_modes}")
    print(f"  (distinct_modes == N count means agent uses separate symbol per N)")
    for N, t in mode_tuples.items():
        print(f"    N={N:3d} -> {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-values", default="1,3,5,8,10,15,25,33")
    args = ap.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    print(f"ckpt: {args.ckpt}")
    probe(args.ckpt, n_values, args.n_trials)


if __name__ == "__main__":
    main()
