# DIAG_FOR: v19 | ANSWERS: lesion | INPUTS: <ckpt>
"""V19 Diag 1: LRU pathway lesion.

Load V19+det smoke ckpt; monkey-patch _encode_segment_v6 so the LRU output h
is forced to zeros (PFC bypass + gate stay live). Compare val_acc against
the unlesioned baseline.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def evaluate_val_acc(agent, cfg, n_max, n_batches, batch_size, device, rng):
    correct = 0
    total = 0
    use_oracle = getattr(agent.agent_cfg, "use_oracle_spike_gate", False)
    for _ in range(n_batches):
        inp, lbl, ci, metas = sample_training_batch(
            batch_size=batch_size, n_max_stage=n_max, config=cfg, rng=rng,
        )
        with torch.no_grad():
            if use_oracle:
                alpha_c = [m["alpha_centers"] for m in metas]
                K = len(metas[0]["beta_centers"])
                beta_cl = [[m["beta_centers"][i] for m in metas] for i in range(K)]
                out = agent(inp.to(device), ci.to(device),
                            alpha_centers=alpha_c, beta_centers_list=beta_cl)
            else:
                out = agent(inp.to(device), ci.to(device))
            logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == lbl).sum().item()
        total += lbl.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--oracle", action="store_true", help="Enable use_oracle_spike_gate at load time")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
        use_oracle_spike_gate=args.oracle,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8
    cfg.alpha_distribution = "uniform"

    # Baseline
    rng = np.random.default_rng(42)
    base_acc = evaluate_val_acc(agent, cfg, args.n_max, args.n_batches, args.batch_size, device, rng)
    print(f"=== UN-lesioned val_acc (n_max={args.n_max}) = {base_acc:.4f} ===")

    # Lesion: wrap _encode_segment_v6 to zero out the first element of return
    orig_encode = agent._encode_segment_v6

    def lesioned_encode(x_seg, apply_pool=True):
        h, cnn_pooled = orig_encode(x_seg, apply_pool=apply_pool)
        return torch.zeros_like(h), cnn_pooled

    agent._encode_segment_v6 = lesioned_encode

    rng = np.random.default_rng(42)
    lesion_acc = evaluate_val_acc(agent, cfg, args.n_max, args.n_batches, args.batch_size, device, rng)
    print(f"=== LESIONED val_acc  (LRU output → 0) = {lesion_acc:.4f} ===")
    print(f"=== drop = {base_acc - lesion_acc:+.4f} ===")
    if base_acc - lesion_acc < 0.02:
        print("  → LRU contributes < 2%. PFC bypass carries almost all the signal.")
    elif base_acc - lesion_acc < 0.10:
        print("  → LRU contributes 2-10%. Modest contribution.")
    else:
        print("  → LRU contributes > 10%. Real role in this ckpt.")


if __name__ == "__main__":
    main()
