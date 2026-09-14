"""Load a saved active-agent checkpoint and dump behavioral statistics.

Unlike inspect_active_agent.py (which retrains a small agent in-process), this
runs inference-only on a fully-trained checkpoint. Fast; useful for analyzing
long pod runs locally.

Usage:
  python scripts/inspect_from_checkpoint.py path/to/final.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.active_agent import (
    ActionRecord,
    ActivePolicy,
    ActivePolicyConfig,
    action_to_record,
    sample_action,
)
from backend.core.world_event import NBucket, WorldConfig, WorldEngine, generate_world_pool
from scripts.inspect_active_agent import inspect_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-ticks", type=int, default=5000)
    ap.add_argument("--pool-size", type=int, default=3000, help="Fresh pool for inference")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol_cfg_kwargs = ckpt["pol_cfg"]
    world_cfg_kwargs = ckpt["world_cfg"]
    training_args = ckpt.get("args", {})

    # Reconstruct configs
    pol_cfg = ActivePolicyConfig(**{k: pol_cfg_kwargs[k] for k in ActivePolicyConfig.__dataclass_fields__ if k in pol_cfg_kwargs})
    world_cfg_keys = WorldConfig.__dataclass_fields__.keys()
    world_cfg = WorldConfig(**{k: world_cfg_kwargs[k] for k in world_cfg_keys if k in world_cfg_kwargs})
    print(f"Loaded {args.ckpt}")
    print(f"  total_ticks trained: {ckpt.get('total_ticks', '?')}")
    print(f"  pol_cfg: d_model={pol_cfg.d_model}, n_heads={pol_cfg.n_heads}, "
          f"ff_dim={pol_cfg.ff_dim}, n_review={pol_cfg.n_review_pages}")
    print(f"  world_cfg: L={world_cfg.L}, w={world_cfg.window_w}, "
          f"auto_scan={world_cfg.auto_scan}")

    # Rebuild and load
    pol = ActivePolicy(pol_cfg)
    pol.load_state_dict(ckpt["policy_state"])
    pol.train(False)

    # Build pool matching training buckets
    buckets_str = training_args.get("pool_buckets", "1-10:0.5,15-40:0.4,60-120:0.1")
    bks = []
    for tok in buckets_str.split(","):
        r, w = tok.split(":")
        lo, hi = r.split("-")
        bks.append(NBucket(lo=int(lo), hi=int(hi), weight=float(w)))
    print(f"  re-generating {args.pool_size} worlds with buckets {buckets_str}...")
    pool_worlds, pool_Ns = generate_world_pool(
        world_cfg, bks, args.pool_size,
        rng=np.random.default_rng(12345), verbose=False,
    )

    engines = [
        WorldEngine(
            world_cfg, n_max=max(pool_Ns),
            t_budget=training_args.get("t_budget", 80),
            notebook_capacity=pol_cfg.n_review_pages,
            rng=np.random.default_rng(1000 + i),
            pool_worlds=pool_worlds, pool_Ns=pool_Ns,
        ) for i in range(args.n_envs)
    ]
    latent = torch.zeros(args.n_envs, pol_cfg.d_model)

    print(f"Running {args.n_ticks} inspection ticks...\n")
    inspect_rollout(pol, pol_cfg, engines, latent, n_ticks=args.n_ticks)


if __name__ == "__main__":
    main()
