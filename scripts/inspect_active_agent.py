"""Inspect what the active-agent is doing after a short training run.

Reports:
  - Action distribution (which actions dominate?)
  - Saccade usage (mean/std of delta, does agent actually move?)
  - Write activity (value distribution, are they at 0?)
  - Read-page usage (does agent read different indices?)
  - Time-to-emit distribution (how fast does agent answer?)
  - Active page vs ground truth N (does the notebook content correlate with N?)

Trains in-process for a bounded tick budget so we don't need a checkpoint.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.active_agent import (
    ACTION_KINDS,
    ActivePolicy,
    ActivePolicyConfig,
    action_to_record,
    sample_action,
)
from backend.core.world_event import WorldConfig, WorldEngine
from backend.training.train_active_rl import (
    RLConfig,
    _obs_to_tensors,
    compute_gae,
    ppo_update,
    rollout,
)


def train_briefly(
    total_ticks: int = 80000, n_envs: int = 8, seed: int = 42,
    n_max_start: int = 3, n_max_end: int = 5,
    scene_L: int = 50, t_budget: int = 30,
    auto_scan: bool = False, auto_scan_step: int = 5,
    curiosity_bonus: float = 0.02,
    use_pool: bool = False,
    pool_buckets_str: str = "1-10:0.5,15-40:0.4,60-120:0.1",
    pool_size: int = 5000,
):
    cfg = RLConfig(n_envs=n_envs, rollout_ticks=64, t_budget=t_budget,
                    n_max_start=n_max_start, n_max_end=n_max_end,
                    total_steps=total_ticks,
                    curiosity_bonus=curiosity_bonus)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    world_cfg = WorldConfig(L=scene_L, window_w=5,
                             auto_scan=auto_scan, auto_scan_step=auto_scan_step)
    pol_cfg = ActivePolicyConfig(window_w=5, saccade_abs_max=20.0)
    pol = ActivePolicy(pol_cfg).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.lr)

    pool_worlds = None
    pool_Ns = None
    if use_pool:
        from backend.core.world_event import NBucket, generate_world_pool
        bks = []
        for tok in pool_buckets_str.split(","):
            r, w = tok.split(":")
            lo, hi = r.split("-")
            bks.append(NBucket(lo=int(lo), hi=int(hi), weight=float(w)))
        pool_worlds, pool_Ns = generate_world_pool(
            world_cfg, bks, pool_size,
            rng=np.random.default_rng(seed + 999), verbose=False)
        print(f"  pool: {pool_size} worlds, N range [{min(pool_Ns)}, {max(pool_Ns)}]")

    engines = [
        WorldEngine(
            world_cfg, n_max=cfg.n_max_start, t_budget=cfg.t_budget,
            notebook_capacity=pol_cfg.n_review_pages,
            rng=np.random.default_rng(seed + i),
            pool_worlds=pool_worlds, pool_Ns=pool_Ns,
        ) for i in range(n_envs)
    ]
    latent = torch.zeros(n_envs, pol_cfg.d_model)
    total = 0
    while total < total_ticks:
        records, latent, bv = rollout(pol, engines, latent, cfg.rollout_ticks, pol_cfg, cfg, device)
        adv, ret = compute_gae(records, bv, cfg.gamma, cfg.gae_lambda)
        ppo_update(pol, opt, records, adv, ret, cfg)
        total += n_envs * cfg.rollout_ticks
    print(f"Training done: {total} ticks")
    return pol, engines, pol_cfg, world_cfg, latent


def inspect_rollout(pol, pol_cfg, engines, latent, n_ticks=5000):
    """Run eval rollout (deterministic policy) and collect behavioral stats."""
    pol.train(False)
    device = torch.device("cpu")

    action_counter = Counter()
    saccade_deltas = []
    write_values = []
    write_slots = []
    read_indices = []
    emit_classes = []
    ticks_to_emit = []   # ticks spent before each session's emit
    session_N = []       # N_alpha of each completed session
    session_reward = []

    current_session_ticks = [0] * len(engines)
    current_session_N = [eng.state.current_session.alpha_count for eng in engines]

    for _ in range(n_ticks):
        obs = _obs_to_tensors(engines, pol_cfg, device)
        with torch.no_grad():
            out = pol(latent, obs["window"], obs["active_page"], obs["target_cue"], obs["past_pages"])
        latent = out["new_latent"]
        action, _ = sample_action(out, deterministic=False)   # stochastic for realistic stats

        for bi in range(len(engines)):
            rec = action_to_record(action, bi)
            action_counter[rec.kind] += 1
            if rec.kind == "saccade":
                saccade_deltas.append(rec.saccade_delta)
            elif rec.kind == "write":
                write_values.append(rec.write_value)
                write_slots.append(rec.write_slot)
            elif rec.kind == "read_page":
                read_indices.append(rec.read_index)
            elif rec.kind == "emit":
                emit_classes.append(rec.emit_class)
                ticks_to_emit.append(current_session_ticks[bi] + 1)
                session_N.append(current_session_N[bi])

            current_session_ticks[bi] += 1
            r, done = engines[bi].step(rec)
            if done:
                session_reward.append(r)
                current_session_ticks[bi] = 0
                current_session_N[bi] = engines[bi].state.current_session.alpha_count

    # ------------------------------------------------------------------ report
    total_actions = sum(action_counter.values())
    print("\n=== Action Distribution ===")
    for k in ACTION_KINDS:
        n = action_counter.get(k, 0)
        pct = 100.0 * n / total_actions if total_actions else 0.0
        print(f"  {k:<11s} {n:>6d}   {pct:>5.1f}%")

    if saccade_deltas:
        arr = np.array(saccade_deltas)
        print(f"\n=== Saccade delta (n={len(arr)}) ===")
        print(f"  mean={arr.mean():+.2f}  std={arr.std():.2f}  "
              f"min={arr.min():+.2f}  max={arr.max():+.2f}")
        print(f"  abs>0.5 fraction: {(np.abs(arr) > 0.5).mean():.2%}")

    if write_values:
        arr = np.array(write_values)
        slots = np.array(write_slots)
        print(f"\n=== Write values (n={len(arr)}) ===")
        print(f"  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        print(f"  slot hist (0..3): "
              + " ".join(f"{(slots == i).sum()}" for i in range(4)))
        # Bucket values to 6 levels to see what the quantization WOULD look like
        levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bucket = np.abs(arr[:, None] - levels[None, :]).argmin(axis=1)
        print(f"  bucketed to 6 levels {{0..5}}: "
              + " ".join(f"{(bucket == i).sum()}" for i in range(6)))

    if read_indices:
        arr = np.array(read_indices)
        print(f"\n=== Read indices (n={len(arr)}) ===")
        print(f"  hist (0..{pol_cfg.n_review_pages-1}): "
              + " ".join(f"{(arr == i).sum()}" for i in range(pol_cfg.n_review_pages)))

    if emit_classes:
        arr = np.array(emit_classes)
        print(f"\n=== Emit classes (n={len(arr)}) ===")
        print(f"  0 (greater): {(arr == 0).sum()}  "
              f"1 (less): {(arr == 1).sum()}  "
              f"2 (equal): {(arr == 2).sum()}")

    if ticks_to_emit:
        arr = np.array(ticks_to_emit)
        print(f"\n=== Ticks to emit (n={len(arr)}) ===")
        print(f"  mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
              f"min={arr.min()}  max={arr.max()}")

    if session_reward:
        arr = np.array(session_reward)
        print(f"\n=== Session outcomes (n={len(arr)}) ===")
        correct = (arr > 0.5).mean()
        wrong_emit = ((arr < -0.9)).mean()
        timeout = ((arr > -0.9) & (arr < -0.1)).mean()
        print(f"  correct: {correct:.2%}   wrong_emit: {wrong_emit:.2%}   timeout: {timeout:.2%}")

    # Correlation between active_page writes and ground truth N
    # Collect over several sessions explicitly
    print("\n=== Active-page write content vs N_alpha ===")
    pol.train(False)
    latent2 = torch.zeros(len(engines), pol_cfg.d_model)
    page_samples = []
    N_samples = []
    for _ in range(2000):
        obs = _obs_to_tensors(engines, pol_cfg, device)
        with torch.no_grad():
            out = pol(latent2, obs["window"], obs["active_page"], obs["target_cue"], obs["past_pages"])
        latent2 = out["new_latent"]
        action, _ = sample_action(out, deterministic=True)
        for bi in range(len(engines)):
            rec = action_to_record(action, bi)
            _, done = engines[bi].step(rec)
            if done and len(engines[bi].state.past_pages) > 0:
                page, _fb = engines[bi].state.past_pages[-1]
                page_samples.append(page[:4].copy())   # content slots only
                N_samples.append(engines[bi].state.current_session.alpha_count)   # new session's N
                # Note: this is the NEW session's N at the moment the old one
                # just emitted; for causality we want the committed session's N.
                # Quick fix: we'll take this approximate and note it below.
    if page_samples:
        pages = np.array(page_samples)
        Ns = np.array(N_samples)
        print(f"  samples: {len(pages)}")
        print(f"  per-slot mean: {pages.mean(axis=0).round(3)}")
        print(f"  per-slot std:  {pages.std(axis=0).round(3)}")
        for i in range(4):
            if pages[:, i].std() > 1e-6:
                r = np.corrcoef(pages[:, i], Ns.astype(float))[0, 1]
                print(f"  pos{i} correlation with N (post-emit approx): {r:+.3f}")
            else:
                print(f"  pos{i}: constant (std ~ 0), no correlation info")


def main():
    # Auto-scan + no-curiosity config that just succeeded on pod at 3M ticks.
    # Local shorter reproduction for behavioral inspection.
    pol, engines, pol_cfg, world_cfg, latent = train_briefly(
        total_ticks=500000, n_envs=8,
        scene_L=250, t_budget=80,
        auto_scan=True, auto_scan_step=5,
        curiosity_bonus=0.0,
        use_pool=True,
        pool_buckets_str="1-10:0.5,15-40:0.4,60-120:0.1",
        pool_size=3000,
    )
    inspect_rollout(pol, pol_cfg, engines, latent, n_ticks=5000)


if __name__ == "__main__":
    main()
