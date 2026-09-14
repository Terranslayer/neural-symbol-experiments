# -*- coding: utf-8 -*-
"""
Shared data-collection step: run agent over many episodes and gather the
notes it writes, the N_alpha of each episode, and its predictions.

All downstream metrics (topo_sim, compositional probe, MI matrix) operate
on the same (N_alpha, notes) pairs, so it's efficient to collect once.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from backend.core.agent import GRUAgent
from backend.core.label_mapping import eval_k_to_class
from backend.core.scene import SceneConfig, sample_training_batch


# Signed values for ordinal/regression loss types (matches train_phase1._SIGNED_LABEL_VALUES).
_SIGNED_VALUES = torch.tensor([+1.0, -1.0, 0.0])


def _k_target_from_metas(
    metas: list,
    scene_cfg,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Eval-time successor labels matching the TRAINING loss for this scene.

    Delegates to the single source of truth (`backend.core.label_mapping.eval_k_to_class`)
    which branches on the scene preset: balanced -> COARSE `k_to_class` (EQUAL->center,
    dedicated FAR classes); legacy bidir/unidir -> legacy mapping. Previously this held
    a hand-copied legacy bidirectional mapping that diverged from the balanced training
    label (fbdd5e5), corrupting V35 acc; see label_mapping.py docstring.
    """
    k_raw = [m["beta_counts"][0] - m["N_alpha"] for m in metas]
    return torch.tensor(
        [eval_k_to_class(k, scene_cfg, n_classes) for k in k_raw],
        device=device, dtype=torch.long,
    )


def compute_successor_preds(
    agent,
    metas: list,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Compute real 10-class successor predictions + targets from agent stashes.

    Assumes agent.forward(inputs, compare_idx) has just been called and the
    following stashes are populated:
      - `_scratch_alpha_for_aux` (B, W)        [scratch input level]
      - `_last_scratch_beta_K_q` (B, K, W)
      OR
      - `_h_alpha_pfc_blocks[0]` (B, d)        [pfc_hidden input level]
      - `_h_beta_pfc_blocks[0]`  (B, d)

    Returns (preds, targets, n_classes) or None if agent lacks the predict head
    or the required stashes (e.g., wrong forward path was taken).

    The fix purpose: V33 (and V29) `_*_forward` returns `fake_compare_logits=zeros`,
    so any metric using `agent(...)` return values for V24-task successor acc
    is broken. Use this helper to bypass via the predict head + stashes.
    """
    # V35 fast-path: V35 agents store successor logits directly on
    # `_v35_successor_logits` (computed by V35CompareAttention in _v35_forward).
    # Bypass the successor_predict_head check which doesn't apply to V35.
    v35_logits = getattr(agent, "_v35_successor_logits", None)
    if v35_logits is not None:
        n_classes = v35_logits.shape[-1]
        k_targets = _k_target_from_metas(metas, agent.scene_cfg, n_classes, device)
        preds = v35_logits.argmax(dim=-1)
        return preds, k_targets, n_classes

    pred_head = getattr(agent, "successor_predict_head", None)
    if pred_head is None:
        return None
    # MultiHotMLPSuccessorHead (V28-Sin / V28-LRU) takes int inputs, not
    # cat([sc_a, sc_b]). Fall back to None for those — caller keeps the
    # original (possibly broken) metric instead of silently miscomputing.
    if not isinstance(pred_head, torch.nn.Linear):
        return None

    input_level = getattr(agent.agent_cfg, "successor_predict_input_level", "scratch")
    if input_level == "pfc_hidden":
        h_a_blocks = getattr(agent, "_h_alpha_pfc_blocks", None)
        h_b_blocks = getattr(agent, "_h_beta_pfc_blocks", None)
        if not h_a_blocks or not h_b_blocks:
            return None
        v24_input = torch.cat([h_a_blocks[0], h_b_blocks[0]], dim=-1)
    else:
        sc_a = getattr(agent, "_scratch_alpha_for_aux", None)
        sc_b_K = getattr(agent, "_last_scratch_beta_K_q", None)
        if sc_a is None or sc_b_K is None:
            return None
        sc_b = sc_b_K[:, 0, :]
        v24_input = torch.cat([sc_a, sc_b], dim=-1)

    k_logits = pred_head(v24_input)
    n_classes = k_logits.shape[-1]
    k_targets = _k_target_from_metas(metas, agent.scene_cfg, n_classes, device)
    preds = k_logits.argmax(dim=-1)
    return preds, k_targets, n_classes


def logits_to_preds(logits: torch.Tensor, loss_type: str = "ce") -> torch.Tensor:
    """Convert logits to class predictions.

    Scalar-output heads ('regression', 'gap_mse4'): threshold channel 0 to
    {GREATER=0 if pred>0.5, LESS=1 if pred<-0.5, EQUAL=2 otherwise}.

    'rl_top1': output is per-block preference score, not a 3-class label —
    return all-EQUAL placeholder (block-level accuracy is undefined; consumers
    should use scratch-pad metrics or mean-reward instead).
    """
    if loss_type in ("regression", "gap_mse4"):
        pred = logits[..., 0]
        out = torch.full(pred.shape, 2, dtype=torch.long, device=pred.device)
        out = torch.where(pred > 0.5, torch.zeros_like(out), out)
        out = torch.where(pred < -0.5, torch.ones_like(out), out)
        return out
    if loss_type == "rl_top1":
        scores = logits[..., 0]
        return torch.full(scores.shape, 2, dtype=torch.long, device=scores.device)
    return logits.argmax(dim=-1)


@dataclass
class CollectedRollout:
    """Notes and predictions from a batch of episodes."""
    n_alpha: np.ndarray         # (num_episodes,) int
    notes: np.ndarray           # (num_episodes, W) float
    predictions: np.ndarray     # (num_episodes, K) int — predicted class
    labels: np.ndarray          # (num_episodes, K) int — true class
    beta_counts: np.ndarray     # (num_episodes, K) int

    @property
    def num_episodes(self) -> int:
        return self.n_alpha.shape[0]

    @property
    def accuracy(self) -> float:
        return float((self.predictions == self.labels).mean())


def collect_notes_and_predictions(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    n_max: int,
    num_episodes: int,
    device: torch.device,
    seed: int = 0,
    batch_size: int = 64,
    loss_type: str = "ce",
    task: Optional[str] = None,
) -> CollectedRollout:
    """
    Run the agent over `num_episodes` scenes with N_alpha sampled uniformly
    in [1, n_max], and collect notes + predictions without any training.

    Uses sample_training_batch under the hood for consistency with training.
    """
    rng = np.random.default_rng(seed)
    n_alpha_all = []
    notes_all = []
    preds_all = []
    labels_all = []
    beta_counts_all = []

    num_batches = (num_episodes + batch_size - 1) // batch_size
    with torch.no_grad():
        collected = 0
        for _ in range(num_batches):
            take = min(batch_size, num_episodes - collected)
            if take <= 0:
                break
            batch = sample_training_batch(
                batch_size=take,
                n_max_stage=n_max,
                config=scene_cfg,
                rng=rng,
            )
            inputs, labels, compare_idx, metas = batch
            inputs = inputs.to(device)
            compare_idx_dev = compare_idx.to(device)
            # V19 oracle-gate: forward signature on MambaAgent accepts an
            # optional `metas` kwarg. Pass it through so oracle-gate ckpts
            # build their hard-binary gate from scene meta. Other agents
            # ignore the kwarg via getattr-checked branch.
            if getattr(getattr(agent, "agent_cfg", None), "use_oracle_spike_gate", False):
                logits, scratch, _ = agent(inputs, compare_idx_dev, metas=metas)
            else:
                logits, scratch, _ = agent(inputs, compare_idx_dev)

            # V24/V33 successor metric fix (5/22): when training task is
            # successor_prediction and a successor_predict_head exists, the
            # agent's returned compare logits are either real 3-class (Mamba
            # non-V33) or fake zeros (V33/V29). For task=successor, the real
            # task accuracy is 10-class via predict_head. Override preds/labels
            # so rollout_accuracy reflects the true task metric.
            succ_out = None
            if task == "successor_prediction":
                succ_out = compute_successor_preds(agent, metas, device)
            if succ_out is not None:
                k_preds, k_targets, _n_cls = succ_out
                # Reshape to (B, K=1) to match CollectedRollout (num_episodes, K) layout
                preds = k_preds.unsqueeze(1).cpu().numpy()
                batch_labels = k_targets.unsqueeze(1).cpu().numpy()
            else:
                preds = logits_to_preds(logits, loss_type=loss_type).cpu().numpy()
                batch_labels = labels.numpy()

            n_alpha_all.append(np.array([m["N_alpha"] for m in metas]))
            notes_all.append(scratch.cpu().numpy())
            preds_all.append(preds)
            labels_all.append(batch_labels)
            beta_counts_all.append(np.array([m["beta_counts"] for m in metas]))
            collected += take

    return CollectedRollout(
        n_alpha=np.concatenate(n_alpha_all),
        notes=np.concatenate(notes_all),
        predictions=np.concatenate(preds_all),
        labels=np.concatenate(labels_all),
        beta_counts=np.concatenate(beta_counts_all),
    )
