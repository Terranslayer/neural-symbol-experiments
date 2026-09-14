# -*- coding: utf-8 -*-
"""
Extrapolation accuracy: agent's accuracy on N_alpha values OUTSIDE its
training range. This is the gold standard test for whether the agent
learned a generalizable algorithm vs memorized within-range lookups.

Strong extrapolation (train [1, 100], test [101, 500] still accurate)
is the hallmark of algorithmic compression (e.g., place-value).
"""
from typing import Dict, Optional

import numpy as np
import torch

from backend.core.agent import GRUAgent
from backend.core.scene import SceneConfig, build_episode
from backend.evaluation.collect import logits_to_preds, compute_successor_preds


def extrapolation_accuracy(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    n_train_max: int,
    n_test_ranges: Dict[str, range],
    device: torch.device,
    num_episodes_per_range: int = 256,
    seed: int = 0,
    loss_type: str = "ce",
    task: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate agent on out-of-range N values, grouped by test range.

    Beta sampling MATCHES training distribution (uses scene_cfg's
    sample_beta_count: equal/near/far mixture relative to alpha). This is
    important — extrap eval should test the SAME task as training, not a
    different uniform-beta task.

    Capacity check: N values exceeding (L-1)//2 (scene generator constraint
    L>=2N+1) are silently filtered. We log how many were skipped so the
    reported accuracy denominator is honest.

    Returns:
        {range_name: accuracy_float} plus {range_name + "_n_used": int}
    """
    from backend.core.scene import sample_beta_count  # local import to avoid cycle

    rng = np.random.default_rng(seed)
    results: Dict[str, float] = {}

    # Hard cap on N values samplable given scene_L (matches scene generator
    # constraint L >= 2N+1).
    n_cap = (scene_cfg.L - 1) // 2

    with torch.no_grad():
        for range_name, n_range in n_test_ranges.items():
            n_values = [n for n in list(n_range) if n <= n_cap]
            if not n_values:
                results[range_name] = float("nan")
                results[range_name + "_n_used"] = 0
                continue

            correct = 0
            total = 0
            n_used_alpha = 0
            for _ in range(num_episodes_per_range):
                N_alpha = int(rng.choice(n_values))
                # FIXED: beta sampled from training distribution (centered on
                # alpha via sample_beta_count). This was the bug — old code
                # uniform-sampled beta from [1, n_range.stop], turning extrap
                # eval into a different task than training.
                # Use the EXTRAP n_max as the upper clip so beta can extend
                # into extrap range (still bounded by n_cap for feasibility).
                beta_n_max = min(max(n_range.stop, n_train_max), n_cap)
                beta_counts = [
                    sample_beta_count(N_alpha, beta_n_max, scene_cfg, rng)
                    for _ in range(scene_cfg.K)
                ]

                try:
                    inp, labels, cidx, meta = build_episode(
                        N_alpha, beta_counts, scene_cfg, rng
                    )
                except ValueError:
                    continue

                inp = inp.unsqueeze(0).to(device)
                cidx = cidx.unsqueeze(0).to(device)
                if getattr(getattr(agent, "agent_cfg", None), "use_oracle_spike_gate", False):
                    logits, _, _ = agent(inp, cidx, metas=[meta])
                else:
                    logits, _, _ = agent(inp, cidx)

                # V24/V33 successor metric fix (5/22): same override as
                # collect_notes_and_predictions — when training task is
                # successor_prediction, the real metric is 10-class via
                # predict_head, not 3-class compare logits.
                succ_out = None
                if task == "successor_prediction":
                    succ_out = compute_successor_preds(agent, [meta], device)
                if succ_out is not None:
                    k_preds, k_targets, _n_cls = succ_out
                    preds_np = k_preds.cpu().numpy()       # (B=1,)
                    targets_np = k_targets.cpu().numpy()   # (B=1,)
                    correct += int((preds_np == targets_np).sum())
                    total += int(targets_np.size)
                else:
                    preds = logits_to_preds(logits, loss_type=loss_type).cpu().numpy()[0]
                    correct += int((preds == labels.numpy()).sum())
                    total += len(labels)
                n_used_alpha += 1

            results[range_name] = float(correct) / max(total, 1)
            results[range_name + "_n_used"] = n_used_alpha

    return results
