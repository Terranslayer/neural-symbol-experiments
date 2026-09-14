"""One-shot generator: builds a stratified world pool and persists it to
disk so future RL runs can reuse the exact same data without regenerating.

Usage:
  python scripts/gen_pool_dataset.py \
    --out backend/data/pools/pool_L1500_50k_v1.npz \
    --L 1500 --size 50000 \
    --buckets '1-50:0.3,150-300:0.6,300-600:0.1' \
    --seed 42
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.world_event import (
    NBucket,
    WorldConfig,
    generate_world_pool,
    save_world_pool,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Destination .npz path")
    ap.add_argument("--L", type=int, default=1500)
    ap.add_argument("--size", type=int, default=50000)
    ap.add_argument("--buckets", default="1-50:0.3,150-300:0.6,300-600:0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} already exists; refuse to overwrite. Delete it first.")

    buckets = []
    for tok in args.buckets.split(","):
        rng_part, w_part = tok.split(":")
        lo, hi = rng_part.split("-")
        buckets.append(NBucket(lo=int(lo), hi=int(hi), weight=float(w_part)))

    world_cfg = WorldConfig(L=args.L, window_w=5)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print(f"Generating {args.size} worlds at L={args.L}, buckets={buckets}...")
    worlds, Ns = generate_world_pool(world_cfg, buckets, args.size, rng, verbose=True)
    dt = time.time() - t0

    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "L": args.L,
        "size": args.size,
        "buckets": [{"lo": b.lo, "hi": b.hi, "weight": b.weight} for b in buckets],
        "generation_seconds": round(dt, 1),
        "N_min": int(min(Ns)),
        "N_max": int(max(Ns)),
        "N_mean": float(np.mean(Ns)),
    }

    print(f"Saving to {out}...")
    save_world_pool(out, worlds, Ns, metadata)
    print(f"Done in {dt:.1f}s")
    print(f"  {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
    print(f"  {str(out).replace('.npz', '.meta.json')}")


if __name__ == "__main__":
    main()
