# -*- coding: utf-8 -*-
"""Dump actual VQ scratch (z_q vectors + indices) vs baseline scratch (scalars).
Shows the same-medium violation: VQ writes 8-D vectors, baseline writes scalars."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch  # noqa: E402
from scripts.inspect_checkpoint import load_agent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--range", dest="qrange", default="unit")
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--complex", action="store_true", default=True)
    p.add_argument("--n-max", type=int, default=81)
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--pfc-at-write", action="store_true", default=True)
    p.add_argument("--pfc-compare-drop-raw", action="store_true", default=True)
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=args.q, qrange=args.qrange, L=args.L,
        complex_world=args.complex, pfc_at_write=args.pfc_at_write,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    agent.train(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=args.complex)
    cfg.K = 8
    rng = np.random.default_rng(0)
    inputs, _, ci, metas = sample_training_batch(
        batch_size=64, n_max_stage=args.n_max, config=cfg, rng=rng,
    )
    inputs_dev = inputs.to(device)
    ci_dev = ci.to(device)

    is_vq = getattr(agent, "vq", None) is not None
    print(f"  ckpt: {args.ckpt}")
    print(f"  has VQ module: {is_vq}")
    if is_vq:
        cb = agent.vq.codebook.detach().cpu().numpy()
        print(f"  codebook shape: {cb.shape}  (K codes × d_z dims)")
        print(f"  codebook value range: [{cb.min():.3f}, {cb.max():.3f}]")
        print(f"  codebook example (first 4 codes):")
        for k in range(min(4, cb.shape[0])):
            print(f"    code[{k}] = {np.round(cb[k], 3).tolist()}")

    # Get input signal range (the 'world' that scratch supposedly encodes)
    sig = inputs[:, :cfg.L, 0].numpy()
    print(f"\n  RAW INPUT SIGNAL channel:")
    print(f"    shape per episode: {sig.shape[1:]}")
    print(f"    value range: [{sig.min():.3f}, {sig.max():.3f}]")
    print(f"    sample first 5 values: {np.round(sig[0, :5], 3).tolist()}")

    # Forward to get scratch
    with torch.no_grad():
        if is_vq:
            # Capture full vector slots manually by replicating write logic
            # but easier: just look at what scratch_pad returns + indices
            _, scratch_pad, _ = agent(inputs_dev, ci_dev)
            indices = scratch_pad.cpu().numpy().astype(int)  # (B, W)
            print(f"\n  VQ SCRATCH (indices into codebook):")
            print(f"    shape: {indices.shape}")
            print(f"    sample episodes (showing N -> 5 indices):")
            order = np.argsort([m["N_alpha"] for m in metas])
            for i in order[:args.n_show]:
                n_a = metas[i]["N_alpha"]
                print(f"      N={n_a:>3}: {indices[i].tolist()}")
            print(f"\n  VQ SCRATCH (actual codebook vectors written):")
            print(f"    each slot is an 8-D vector (NOT a scalar in [0,1])")
            for i in order[:4]:
                n_a = metas[i]["N_alpha"]
                vecs = cb[indices[i]]  # (W, d_z)
                print(f"    N={n_a:>3} slot vectors:")
                for j in range(cfg.W):
                    print(f"      slot[{j}] code={indices[i,j]:>2}: {np.round(vecs[j], 3).tolist()}")
        else:
            _, scratch_pad, _ = agent(inputs_dev, ci_dev)
            notes = scratch_pad.cpu().numpy()  # (B, W) scalars
            print(f"\n  BASELINE SCRATCH (scalar values per slot):")
            print(f"    shape: {notes.shape}")
            print(f"    value range: [{notes.min():.3f}, {notes.max():.3f}]")
            print(f"    distinct values: {sorted(set(notes.flatten().tolist()))[:10]}")
            order = np.argsort([m["N_alpha"] for m in metas])
            for i in order[:args.n_show]:
                n_a = metas[i]["N_alpha"]
                print(f"      N={n_a:>3}: {np.round(notes[i], 3).tolist()}")

if __name__ == "__main__":
    main()
