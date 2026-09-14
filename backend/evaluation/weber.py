# -*- coding: utf-8 -*-
"""
Weber curve — diagnostic for distinguishing true counting from magnitude/log
encoding.

For each N_alpha in a probe range, we run trials where N_beta is chosen
uniformly from {N_alpha - 1, N_alpha, N_alpha + 1} (all hard cases: requires
distinguishing adjacent integers). We measure the agent's 3-class
(>, <, ==) accuracy as a function of N_alpha.

Interpretation:
    flat curve (~const accuracy across N_alpha)  -> true counting (cardinality)
    decreasing curve (hi at small N, low at large) -> Weber's law / log encoding
    noisy/floor (chance=1/3)                     -> task not learned

References:
    Dehaene 1997, *The Number Sense*
    Feigenson, Dehaene & Spelke 2004, "Core systems of number"
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from backend.core.agent import GRUAgent
from backend.core.scene import SceneConfig, build_episode
from backend.evaluation.collect import logits_to_preds


@dataclass
class WeberCurve:
    n_values: np.ndarray      # (M,) int, the N_alpha values probed
    accuracy: np.ndarray      # (M,) float, accuracy at each N_alpha
    per_class_accuracy: Dict[str, np.ndarray]  # breakdown by ground-truth class
    num_trials_per_n: int

    def weber_slope(self) -> float:
        """
        Regression slope of accuracy vs log(N_alpha). A clearly negative
        slope is the Weber's-law signature. Returns 0 if curve is flat.
        """
        if len(self.n_values) < 2:
            return 0.0
        x = np.log(self.n_values.astype(np.float64))
        y = self.accuracy.astype(np.float64)
        # Simple linear regression slope
        slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        return float(slope)

    def flatness_score(self) -> float:
        """
        1.0 - (std / mean) of accuracy across N_alpha.
        ~1.0 = perfectly flat (counting). Lower = more Weber-shaped.
        """
        mean = self.accuracy.mean()
        if mean < 1e-6:
            return 0.0
        return float(1.0 - self.accuracy.std() / mean)


def compute_weber_curve(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    n_range: range,
    num_trials_per_n: int,
    device: torch.device,
    seed: int = 0,
    loss_type: str = "ce",
) -> WeberCurve:
    """
    Run the agent over trials at each N_alpha in n_range, with N_beta drawn
    from {N_alpha - 1, N_alpha, N_alpha + 1} uniformly. Returns a WeberCurve.

    Args:
        n_range: range of N_alpha to probe. Should be interior (skip
            boundaries 1 and L-1) to ensure N_alpha +- 1 is valid.
        num_trials_per_n: how many (alpha, beta) samples per N_alpha.
    """
    rng = np.random.default_rng(seed)
    n_values = np.array(list(n_range), dtype=np.int64)
    accuracies = np.zeros(len(n_values), dtype=np.float64)
    # Per-class breakdown: does the agent handle (=, >, <) cases equally?
    class_correct = {"greater": [0, 0, 0], "less": [0, 0, 0], "equal": [0, 0, 0]}
    # class_correct[key] = [correct, total, _]; we don't store the last

    with torch.no_grad():
        for i, N_alpha in enumerate(n_values):
            correct = 0
            total = 0
            for _ in range(num_trials_per_n):
                # Sample N_beta uniformly from {N_a - 1, N_a, N_a + 1}
                delta = int(rng.choice([-1, 0, 1]))
                N_beta = int(N_alpha) + delta
                if N_beta < 1 or N_beta > scene_cfg.L - 1:
                    continue  # skip boundary-invalid samples
                beta_counts = [N_beta] * scene_cfg.K
                try:
                    inp, labels, cidx, meta = build_episode(
                        int(N_alpha), beta_counts, scene_cfg, rng
                    )
                except ValueError:
                    continue
                inp = inp.unsqueeze(0).to(device)
                cidx = cidx.unsqueeze(0).to(device)
                if getattr(getattr(agent, "agent_cfg", None), "use_oracle_spike_gate", False):
                    logits, _, _ = agent(inp, cidx, metas=[meta])
                else:
                    logits, _, _ = agent(inp, cidx)
                preds = logits_to_preds(logits, loss_type=loss_type).cpu().numpy()[0]

                label_value = labels.numpy()
                correct_count = int((preds == label_value).sum())
                trial_total = len(label_value)
                correct += correct_count
                total += trial_total

                # Per-class tally. All K sub-trials in one episode share N_beta
                # so they share ground-truth class.
                cls_key = "equal" if delta == 0 else (
                    "greater" if delta < 0 else "less"
                )
                class_correct[cls_key][0] += correct_count
                class_correct[cls_key][1] += trial_total

            accuracies[i] = correct / max(total, 1)

    per_class_acc = {
        k: float(v[0]) / max(v[1], 1) for k, v in class_correct.items()
    }

    return WeberCurve(
        n_values=n_values,
        accuracy=accuracies,
        per_class_accuracy={k: np.array([v]) for k, v in per_class_acc.items()},
        num_trials_per_n=num_trials_per_n,
    )
