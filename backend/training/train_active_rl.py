"""Minimal recurrent PPO for the active-agent MVP (Stage 2 Phase 5).

Design:
  - Parallel envs (B engines); each env runs a life independently
  - Each rollout: run all envs for T ticks, collect (obs, action, logprob, reward,
    value, done) tuples
  - GAE-λ advantage on per-env basis; hidden latent carried across ticks but
    reset on life-end (defined as max_life_ticks, NOT per-session — agent
    keeps notebook across many sessions within a life)
  - PPO clip objective with per-head log-prob summed (PPO over joint action)
  - Entropy bonus on action-kind distribution only (avoid pathological
    parameter-head entropy blowups)

Usage (CPU smoke):
  python backend/training/train_active_rl.py --total-steps 5000 --n-envs 4

(Autoscale in background to something meaningful after first run validates.)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.active_agent import (
    ACTION_KINDS,
    ActivePolicy,
    ActivePolicyConfig,
    action_to_record,
    sample_action,
)  # noqa: F401 (ACTION_KINDS used by rollout)
from backend.core.hier_active_agent import (
    HierActivePolicy,
    HierActivePolicyConfig,
)
from backend.core.world_event import (
    NBucket,
    WorldConfig,
    WorldEngine,
    generate_world_pool,
    load_world_pool,
)


@dataclass
class RLConfig:
    n_envs: int = 4
    rollout_ticks: int = 64
    total_steps: int = 20000
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    ppo_epochs: int = 4
    minibatch_envs: int = 4
    max_grad_norm: float = 0.5
    # Curiosity: only when (saccade action) AND (prediction accurate) AND
    # (actual window has non-trivial content — variance above threshold). The
    # content-variance filter defeats "saccade to background, predict zeros"
    # gaming. Also capped per session to keep task reward dominant.
    curiosity_threshold: float = 0.12
    curiosity_bonus: float = 0.02
    curiosity_window_var_min: float = 0.03   # actual window must have this std to count
    curiosity_session_cap: float = 0.20       # max curiosity total per session
    predictor_coef: float = 0.5          # MSE auxiliary loss weight
    life_ticks: int = 2000              # hidden+notebook reset every this many ticks
    t_budget: int = 40                  # per-session ticks before timeout
    n_max_start: int = 5
    n_max_end: int = 10                 # curriculum end (MVP stays small)
    # Accuracy-based curriculum: bump n_max by 1 when rolling accuracy over the
    # last accuracy_window sessions exceeds accuracy_promote. Never decrement.
    accuracy_promote: float = 0.70
    accuracy_window: int = 200


# --------------------------------------------------------------------- rollout


def _obs_to_tensors(
    engines: List[WorldEngine], pol_cfg: ActivePolicyConfig, device: torch.device
) -> Dict[str, torch.Tensor]:
    B = len(engines)
    windows = np.stack([eng.observe()["window"] for eng in engines])
    active_pages = np.stack([eng.state.active_page for eng in engines])
    target_cues = np.array([eng.observe()["target_cue"] for eng in engines], dtype=np.float32)

    past_pages = np.zeros((B, pol_cfg.n_review_pages, pol_cfg.past_page_len), dtype=np.float32)
    for bi, eng in enumerate(engines):
        for pi in range(pol_cfg.n_review_pages):
            past_pages[bi, pi] = eng.read_page(pi)

    return {
        "window": torch.as_tensor(windows, dtype=torch.float32, device=device),
        "active_page": torch.as_tensor(active_pages, dtype=torch.float32, device=device),
        "target_cue": torch.as_tensor(target_cues, dtype=torch.float32, device=device),
        "past_pages": torch.as_tensor(past_pages, dtype=torch.float32, device=device),
    }


@dataclass
class StepRecord:
    window: torch.Tensor
    active_page: torch.Tensor
    target_cue: torch.Tensor
    past_pages: torch.Tensor
    latent_in: torch.Tensor
    action: Dict[str, torch.Tensor]
    logprob: Dict[str, torch.Tensor]
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    window_prediction: torch.Tensor           # policy's prediction of NEXT window
    next_window: torch.Tensor                 # actual NEXT window (for supervised MSE)


def rollout(
    pol: ActivePolicy,
    engines: List[WorldEngine],
    latent: torch.Tensor,
    n_ticks: int,
    pol_cfg: ActivePolicyConfig,
    cfg: RLConfig,
    device: torch.device,
) -> tuple:
    records: List[StepRecord] = []
    B = len(engines)
    # Per-env accumulated curiosity in current session (reset on done). Caps
    # total curiosity bonus so task reward cannot be drowned.
    session_curiosity = np.zeros(B, dtype=np.float32)

    for _ in range(n_ticks):
        obs = _obs_to_tensors(engines, pol_cfg, device)
        with torch.no_grad():
            out = pol(latent, obs["window"], obs["active_page"], obs["target_cue"], obs["past_pages"])
        new_latent = out["new_latent"]
        action, logprob = sample_action(out)
        predicted_next_window = out["window_prediction"].detach()

        rewards = np.zeros(B, dtype=np.float32)
        dones = np.zeros(B, dtype=np.float32)
        for bi in range(B):
            rec = action_to_record(action, bi)
            r, d = engines[bi].step(rec)
            rewards[bi] = r
            dones[bi] = 1.0 if d else 0.0

        # Curiosity: grant bonus only when ALL conditions hold --
        # (1) action was a saccade, (2) prediction was accurate, (3) the actual
        # next window has non-trivial content variance (defeats
        # "saccade-to-background-and-predict-zero" gaming we observed in the
        # first GPU run where entropy collapsed to 0.15), and (4) the session's
        # accumulated curiosity has not exhausted its cap.
        next_obs = _obs_to_tensors(engines, pol_cfg, device)
        next_window = next_obs["window"].detach()
        pred_err = (predicted_next_window - next_window).pow(2).mean(dim=-1).sqrt()
        window_std = next_window.std(dim=-1)
        saccade_idx = ACTION_KINDS.index("saccade") if "saccade" in ACTION_KINDS else -1
        for bi in range(B):
            kind_idx = int(action["kind"][bi].item())
            is_saccade = (kind_idx == saccade_idx)
            content_ok = float(window_std[bi].item()) > cfg.curiosity_window_var_min
            pred_ok = float(pred_err[bi].item()) < cfg.curiosity_threshold
            cap_room = cfg.curiosity_session_cap - float(session_curiosity[bi])
            if is_saccade and content_ok and pred_ok and cap_room > 0:
                bonus = min(cfg.curiosity_bonus, cap_room)
                rewards[bi] += bonus
                session_curiosity[bi] += bonus

        # Reset per-env curiosity budget at session boundaries
        for bi in range(B):
            if dones[bi] > 0.5:
                session_curiosity[bi] = 0.0

        records.append(StepRecord(
            window=obs["window"],
            active_page=obs["active_page"],
            target_cue=obs["target_cue"],
            past_pages=obs["past_pages"],
            latent_in=latent,
            action={k: v.detach() for k, v in action.items()},
            logprob={k: v.detach() for k, v in logprob.items()},
            value=out["value"].detach(),
            reward=torch.as_tensor(rewards, device=device),
            done=torch.as_tensor(dones, device=device),
            window_prediction=predicted_next_window,
            next_window=next_window,
        ))
        latent = new_latent.detach()

    # Last-state value for bootstrap
    obs = _obs_to_tensors(engines, pol_cfg, device)
    with torch.no_grad():
        out = pol(latent, obs["window"], obs["active_page"], obs["target_cue"], obs["past_pages"])
    bootstrap_value = out["value"].detach()

    return records, latent.detach(), bootstrap_value


def compute_gae(
    records: List[StepRecord], bootstrap_value: torch.Tensor, gamma: float, lam: float
) -> tuple:
    T = len(records)
    B = records[0].reward.shape[0]
    advantages = torch.zeros(T, B, device=records[0].reward.device)
    returns = torch.zeros(T, B, device=records[0].reward.device)
    gae = torch.zeros(B, device=records[0].reward.device)

    next_value = bootstrap_value
    for t in reversed(range(T)):
        r = records[t].reward
        d = records[t].done
        v = records[t].value
        non_terminal = 1.0 - d
        delta = r + gamma * next_value * non_terminal - v
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + v
        next_value = v
    return advantages, returns


# ----------------------------------------------------------------------- PPO


def ppo_update(
    pol: ActivePolicy,
    opt: torch.optim.Optimizer,
    records: List[StepRecord],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: RLConfig,
) -> Dict[str, float]:
    T = len(records)
    B = advantages.shape[1]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten everything to (T*B, ...)
    flat = {
        "window": torch.cat([r.window for r in records], dim=0),
        "active_page": torch.cat([r.active_page for r in records], dim=0),
        "target_cue": torch.cat([r.target_cue for r in records], dim=0),
        "past_pages": torch.cat([r.past_pages for r in records], dim=0),
        "latent_in": torch.cat([r.latent_in for r in records], dim=0),
        "adv": advantages.reshape(-1),
        "ret": returns.reshape(-1),
        "next_window": torch.cat([r.next_window for r in records], dim=0),
    }
    flat_actions = {k: torch.cat([r.action[k] for r in records], dim=0) for k in records[0].action}
    flat_oldlogp = {k: torch.cat([r.logprob[k] for r in records], dim=0) for k in records[0].logprob}

    losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "predictor": 0.0}
    N = T * B

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(N, device=advantages.device)
        mb_size = max(cfg.minibatch_envs * T, 1)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]

            out = pol(
                flat["latent_in"][idx],
                flat["window"][idx],
                flat["active_page"][idx],
                flat["target_cue"][idx],
                flat["past_pages"][idx],
            )
            kind_dist = torch.distributions.Categorical(logits=out["kind_logits"])
            saccade_dist = torch.distributions.Normal(out["saccade_mu"], out["saccade_logstd"].exp())
            ws_dist = torch.distributions.Categorical(logits=out["write_slot_logits"])
            wv_dist = torch.distributions.Normal(out["write_value_mu"], out["write_value_logstd"].exp())
            ri_dist = torch.distributions.Categorical(logits=out["read_index_logits"])
            em_dist = torch.distributions.Categorical(logits=out["emit_class_logits"])

            new_logp = (
                kind_dist.log_prob(flat_actions["kind"][idx])
                + saccade_dist.log_prob(flat_actions["saccade_delta"][idx])
                + ws_dist.log_prob(flat_actions["write_slot"][idx])
                + wv_dist.log_prob(flat_actions["write_value"][idx])
                + ri_dist.log_prob(flat_actions["read_index"][idx])
                + em_dist.log_prob(flat_actions["emit_class"][idx])
            )
            old_logp = (
                flat_oldlogp["kind"][idx]
                + flat_oldlogp["saccade_delta"][idx]
                + flat_oldlogp["write_slot"][idx]
                + flat_oldlogp["write_value"][idx]
                + flat_oldlogp["read_index"][idx]
                + flat_oldlogp["emit_class"][idx]
            )

            ratio = (new_logp - old_logp).exp()
            adv = flat["adv"][idx]
            unclipped = ratio * adv
            clipped = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (out["value"] - flat["ret"][idx]).pow(2).mean()
            entropy = kind_dist.entropy().mean()   # only kind head, avoid Gaussian-blowup
            predictor_loss = (out["window_prediction"] - flat["next_window"][idx]).pow(2).mean()
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                + cfg.predictor_coef * predictor_loss
                - cfg.entropy_coef * entropy
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg.max_grad_norm)
            opt.step()

            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean().item()

            losses["policy"] += policy_loss.item()
            losses["value"] += value_loss.item()
            losses["entropy"] += entropy.item()
            losses["kl"] += approx_kl
            losses["predictor"] += predictor_loss.item()
    return {k: v / (cfg.ppo_epochs * max(1, N // mb_size)) for k, v in losses.items()}


# ------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rollout-ticks", type=int, default=64)
    parser.add_argument("--life-ticks", type=int, default=2000)
    parser.add_argument("--t-budget", type=int, default=40)
    parser.add_argument("--n-max-start", type=int, default=5)
    parser.add_argument("--n-max-end", type=int, default=10)
    parser.add_argument("--scene-L", type=int, default=50)
    parser.add_argument("--saccade-abs-max", type=float, default=10.0)
    parser.add_argument("--auto-scan", action="store_true",
                        help="Env advances attn_pos by auto_scan_step each tick; "
                             "saccade actions are ignored. Removes the scanning "
                             "strategy-learning burden from the policy.")
    parser.add_argument("--auto-scan-step", type=int, default=5)
    parser.add_argument("--timeout-penalty", type=float, default=-0.5)
    parser.add_argument("--phased", action="store_true",
                        help="Enable 3-phase session: observe -> gap -> answer. "
                             "Emit only honored during answer; window blank in gap/answer.")
    parser.add_argument("--t-observe", type=int, default=50)
    parser.add_argument("--t-gap", type=int, default=10)
    parser.add_argument("--t-answer", type=int, default=20)
    parser.add_argument("--hier-latent", type=int, default=1,
                        help="If > 1, use HierActivePolicy with this many latent "
                             "tokens (hierarchical predictive-coding levels).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Where to save final policy state_dict (and args)")
    parser.add_argument("--seed", type=int, default=42)
    # Pool mode: stratified N buckets replace the accuracy-based curriculum.
    parser.add_argument("--use-pool", action="store_true",
                        help="Pre-generate a stratified pool of worlds and sample "
                             "from it, skipping the curriculum.")
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--pool-buckets", default="1-50:0.3,150-300:0.6,300-600:0.1",
                        help="Comma-sep list of lo-hi:weight defining pool stratification")
    parser.add_argument("--load-pool", default=None,
                        help="Path to a pre-generated .npz pool; skip in-process generation")
    # Optional warmup phase: first warmup_ticks use a softer bucket spec, then
    # pool is swapped to the main buckets. Helps cold-start on hard tasks.
    parser.add_argument("--warmup-ticks", type=int, default=0,
                        help="If > 0, use --warmup-buckets for the first N ticks "
                             "then switch to --pool-buckets.")
    parser.add_argument("--warmup-buckets", default="1-20:0.6,30-60:0.4")
    parser.add_argument("--warmup-pool-size", type=int, default=10000)
    parser.add_argument("--curiosity-bonus", type=float, default=0.02)
    parser.add_argument("--curiosity-window-var-min", type=float, default=0.03)
    parser.add_argument("--curiosity-threshold", type=float, default=0.12)
    parser.add_argument("--curiosity-session-cap", type=float, default=0.20)
    args = parser.parse_args()

    cfg = RLConfig(
        n_envs=args.n_envs,
        rollout_ticks=args.rollout_ticks,
        total_steps=args.total_steps,
        t_budget=args.t_budget,
        n_max_start=args.n_max_start,
        n_max_end=args.n_max_end,
        curiosity_bonus=args.curiosity_bonus,
        curiosity_window_var_min=args.curiosity_window_var_min,
        curiosity_threshold=args.curiosity_threshold,
        curiosity_session_cap=args.curiosity_session_cap,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)

    world_cfg = WorldConfig(
        L=args.scene_L, window_w=5,
        auto_scan=args.auto_scan, auto_scan_step=args.auto_scan_step,
        timeout_penalty=args.timeout_penalty,
        phased=args.phased,
        t_observe=args.t_observe, t_gap=args.t_gap, t_answer=args.t_answer,
    )
    use_hier = args.hier_latent > 1
    if use_hier:
        pol_cfg = HierActivePolicyConfig(
            window_w=5, saccade_abs_max=args.saccade_abs_max,
            n_latent_tokens=args.hier_latent,
        )
        pol = HierActivePolicy(pol_cfg).to(device)
        print(f"using HierActivePolicy with {args.hier_latent} latent tokens")
    else:
        pol_cfg = ActivePolicyConfig(window_w=5, saccade_abs_max=args.saccade_abs_max)
        pol = ActivePolicy(pol_cfg).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.lr)

    def _parse_buckets(spec: str) -> list:
        out = []
        for tok in spec.split(","):
            rng_part, w_part = tok.split(":")
            lo, hi = rng_part.split("-")
            out.append(NBucket(lo=int(lo), hi=int(hi), weight=float(w_part)))
        return out

    # -------- optional pre-generated pool (load from disk or build in-process) --------
    pool_worlds: Optional[list] = None
    pool_Ns: Optional[list] = None
    warmup_pool_worlds: Optional[list] = None
    warmup_pool_Ns: Optional[list] = None
    if args.load_pool:
        t_pool = time.time()
        print(f"Loading pool from {args.load_pool}...")
        pool_worlds, pool_Ns, pool_meta = load_world_pool(args.load_pool)
        print(f"pool loaded in {time.time() - t_pool:.1f}s, size={len(pool_worlds)}, "
              f"N stats: min={min(pool_Ns)}, max={max(pool_Ns)}, mean={np.mean(pool_Ns):.1f}")
        if pool_meta:
            print(f"  metadata: {pool_meta.get('generated_at')}, "
                  f"seed={pool_meta.get('seed')}, buckets={pool_meta.get('buckets')}")
    elif args.use_pool:
        buckets = _parse_buckets(args.pool_buckets)
        print(f"main pool buckets: {buckets}, generating {args.pool_size} worlds...")
        t_pool = time.time()
        pool_worlds, pool_Ns = generate_world_pool(
            world_cfg, buckets, args.pool_size,
            rng=np.random.default_rng(args.seed + 999999),
            verbose=True,
        )
        print(f"main pool ready in {time.time() - t_pool:.1f}s, "
              f"N stats: min={min(pool_Ns)}, max={max(pool_Ns)}, mean={np.mean(pool_Ns):.1f}")

    if args.warmup_ticks > 0 and pool_worlds is not None:
        warm_buckets = _parse_buckets(args.warmup_buckets)
        print(f"warmup pool buckets: {warm_buckets}, generating {args.warmup_pool_size}...")
        t_wp = time.time()
        warmup_pool_worlds, warmup_pool_Ns = generate_world_pool(
            world_cfg, warm_buckets, args.warmup_pool_size,
            rng=np.random.default_rng(args.seed + 888888),
            verbose=True,
        )
        print(f"warmup pool ready in {time.time() - t_wp:.1f}s, "
              f"N stats: min={min(warmup_pool_Ns)}, max={max(warmup_pool_Ns)}")

    log_dir = Path(args.log_dir or f"backend/training/logs/s2_p5_active_{time.strftime('%Y%m%d_%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = (log_dir / "run.jsonl").open("w", encoding="utf-8")
    logger.write(json.dumps({
        "event": "run_start", "cfg": asdict(cfg),
        "pol_cfg": asdict(pol_cfg), "params": pol.num_parameters()
    }) + "\n")
    print(f"params={pol.num_parameters()}, device={device}, log_dir={log_dir}")

    rng = np.random.default_rng(args.seed)
    # If warmup is active, engines start on warmup pool; swap in main pool later.
    initial_pool_worlds = warmup_pool_worlds if warmup_pool_worlds else pool_worlds
    initial_pool_Ns = warmup_pool_Ns if warmup_pool_Ns else pool_Ns
    engines = [
        WorldEngine(
            world_cfg, n_max=cfg.n_max_start, t_budget=cfg.t_budget,
            notebook_capacity=pol_cfg.n_review_pages,
            rng=np.random.default_rng(args.seed + i),
            pool_worlds=initial_pool_worlds, pool_Ns=initial_pool_Ns,
        ) for i in range(cfg.n_envs)
    ]
    on_warmup = warmup_pool_worlds is not None
    if use_hier:
        latent = torch.zeros(cfg.n_envs, pol_cfg.n_latent_tokens, pol_cfg.d_model, device=device)
    else:
        latent = torch.zeros(cfg.n_envs, pol_cfg.d_model, device=device)
    life_tick_counter = 0

    total_ticks = 0
    recent_rewards: List[float] = []
    recent_accuracies: List[float] = []
    t0 = time.time()

    while total_ticks < cfg.total_steps:
        records, latent, bootstrap_v = rollout(
            pol, engines, latent, cfg.rollout_ticks, pol_cfg, cfg, device
        )
        adv, ret = compute_gae(records, bootstrap_v, cfg.gamma, cfg.gae_lambda)
        losses = ppo_update(pol, opt, records, adv, ret, cfg)

        rewards = torch.stack([r.reward for r in records])    # (T, B)
        dones = torch.stack([r.done for r in records])
        step_rewards = rewards.flatten().tolist()
        recent_rewards.extend(step_rewards)
        recent_rewards = recent_rewards[-500:]

        # Accuracy = fraction of DONE transitions with reward > 0
        done_rewards = rewards[dones > 0.5]
        if done_rewards.numel() > 0:
            recent_accuracies.extend((done_rewards > 0.5).float().tolist())
            recent_accuracies = recent_accuracies[-200:]

        total_ticks += cfg.n_envs * cfg.rollout_ticks
        life_tick_counter += cfg.rollout_ticks

        if life_tick_counter >= cfg.life_ticks:
            # reset each env for next life
            for i, eng in enumerate(engines):
                eng.state.past_pages.clear()
                eng.state.attn_pos = float(world_cfg.L // 2)
                eng._start_new_session()
            if use_hier:
                latent = torch.zeros(cfg.n_envs, pol_cfg.n_latent_tokens, pol_cfg.d_model, device=device)
            else:
                latent = torch.zeros(cfg.n_envs, pol_cfg.d_model, device=device)
            life_tick_counter = 0

        # Warmup-to-main pool swap when we cross warmup_ticks boundary.
        if on_warmup and total_ticks >= args.warmup_ticks:
            for eng in engines:
                eng.pool_worlds = pool_worlds
                eng.pool_Ns = pool_Ns
                eng._cue_norm = float(max(pool_Ns)) if pool_Ns else eng._cue_norm
                eng._start_new_session()  # refresh current session to use new pool
            on_warmup = False
            print(f"  -> warmup done at {total_ticks} ticks, swapped to main pool "
                  f"(max N: {max(pool_Ns)})")
            recent_accuracies = []
            recent_rewards = []

        # Curriculum (only when NOT using pool): accuracy-based n_max promotion.
        current_n_max = engines[0].n_max
        if pool_worlds is None:
            if (
                len(recent_accuracies) >= cfg.accuracy_window
                and float(np.mean(recent_accuracies[-cfg.accuracy_window:])) >= cfg.accuracy_promote
                and current_n_max < cfg.n_max_end
            ):
                new_n_max = current_n_max + 1
                for eng in engines:
                    eng.n_max = new_n_max
                recent_accuracies = []
                print(f"  -> promoted n_max {current_n_max} -> {new_n_max}")
        n_max_now = engines[0].n_max

        mean_r = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        acc = float(np.mean(recent_accuracies)) if recent_accuracies else 0.0
        entry = {
            "event": "update", "total_ticks": total_ticks, "elapsed_s": time.time() - t0,
            "mean_reward": mean_r, "accuracy": acc, "n_max": n_max_now, **losses,
        }
        logger.write(json.dumps(entry) + "\n")
        logger.flush()
        if (total_ticks // (cfg.n_envs * cfg.rollout_ticks)) % 5 == 0:
            print(
                f"ticks={total_ticks} n_max={n_max_now} mean_r={mean_r:+.3f} "
                f"acc={acc:.3f} pol_loss={losses['policy']:+.4f} "
                f"val_loss={losses['value']:.4f} entropy={losses['entropy']:.3f}"
            )

    logger.close()
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state": pol.state_dict(),
            "pol_cfg": asdict(pol_cfg),
            "world_cfg": asdict(world_cfg),
            "rl_cfg": asdict(cfg),
            "args": vars(args),
            "total_ticks": total_ticks,
        }, ckpt_dir / "final.pt")
        print(f"saved checkpoint to {ckpt_dir / 'final.pt'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
