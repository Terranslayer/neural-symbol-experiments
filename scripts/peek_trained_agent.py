"""Peek at a trained active-agent: runs a handful of sessions and prints the
per-session trace — N, written notes, emit, correctness. Much faster and
more interpretable than the full inspect_from_checkpoint behavioral dump.

Usage:
  python scripts/peek_trained_agent.py path/to/final.pt [--n-sessions 20]
"""
import argparse
import sys
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
from backend.core.hier_active_agent import HierActivePolicy, HierActivePolicyConfig
from backend.core.world_event import NBucket, WorldConfig, WorldEngine, generate_world_pool


LABEL_NAMES = ["greater", "less", "equal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol_cfg_raw = ckpt["pol_cfg"]
    is_hier = "n_latent_tokens" in pol_cfg_raw and pol_cfg_raw["n_latent_tokens"] > 1
    if is_hier:
        pol_cfg = HierActivePolicyConfig(
            **{k: pol_cfg_raw[k] for k in HierActivePolicyConfig.__dataclass_fields__ if k in pol_cfg_raw}
        )
    else:
        pol_cfg = ActivePolicyConfig(
            **{k: pol_cfg_raw[k] for k in ActivePolicyConfig.__dataclass_fields__ if k in pol_cfg_raw}
        )
    world_cfg = WorldConfig(
        **{k: ckpt["world_cfg"][k] for k in WorldConfig.__dataclass_fields__ if k in ckpt["world_cfg"]}
    )
    training_args = ckpt.get("args", {})
    print(f"ckpt: {args.ckpt}")
    print(f"  trained {ckpt.get('total_ticks', '?')} ticks")
    print(f"  L={world_cfg.L}, w={world_cfg.window_w}, auto_scan={world_cfg.auto_scan}, "
          f"t_budget={training_args.get('t_budget', '?')}")

    pol = HierActivePolicy(pol_cfg) if is_hier else ActivePolicy(pol_cfg)
    pol.load_state_dict(ckpt["policy_state"])
    pol.train(False)
    print(f"  hier={is_hier}, K={getattr(pol_cfg, 'n_latent_tokens', 1)}")

    buckets_str = training_args.get("pool_buckets", "1-10:0.5,15-40:0.4,60-120:0.1")
    bks = []
    for tok in buckets_str.split(","):
        r, w = tok.split(":")
        lo, hi = r.split("-")
        bks.append(NBucket(lo=int(lo), hi=int(hi), weight=float(w)))
    pool_worlds, pool_Ns = generate_world_pool(
        world_cfg, bks, args.pool_size,
        rng=np.random.default_rng(7777), verbose=False,
    )

    engine = WorldEngine(
        world_cfg, n_max=max(pool_Ns),
        t_budget=training_args.get("t_budget", 80),
        notebook_capacity=pol_cfg.n_review_pages,
        rng=np.random.default_rng(9999),
        pool_worlds=pool_worlds, pool_Ns=pool_Ns,
    )
    if is_hier:
        latent = torch.zeros(1, pol_cfg.n_latent_tokens, pol_cfg.d_model)
    else:
        latent = torch.zeros(1, pol_cfg.d_model)

    print(f"\n{'#':>3} {'N_alpha':>8} {'target':>7} {'label':>8} {'emit':>8} {'ok':>3}"
          f"  {'notes_final':>26}  {'ticks':>5}  {'writes':>6}")
    print("-" * 90)

    completed = 0
    while completed < args.n_sessions:
        sess = engine.state.current_session
        N_alpha = sess.alpha_count
        target = sess.target_count
        label = sess.label

        # Roll one session until done
        session_ticks = 0
        n_writes = 0
        emit_class = None
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
            action, _ = sample_action(out, deterministic=args.deterministic)
            rec = action_to_record(action, 0)
            if rec.kind == "write":
                n_writes += 1
            if rec.kind == "emit":
                emit_class = rec.emit_class
            notes_snapshot = engine.state.active_page.copy()
            r, done = engine.step(rec)
            session_ticks += 1
            if done:
                break

        correct = (emit_class == label) if emit_class is not None else False
        emit_str = LABEL_NAMES[emit_class] if emit_class is not None else "TIMEOUT"
        notes_str = " ".join(f"{v:.1f}" for v in notes_snapshot)
        ok_str = "Y" if correct else "N"
        completed += 1
        print(f"{completed:>3} {N_alpha:>8} {target:>7} {LABEL_NAMES[label]:>8} "
              f"{emit_str:>8} {ok_str:>3}  [{notes_str:>22}]  {session_ticks:>5}  {n_writes:>6}")


if __name__ == "__main__":
    main()
