# -*- coding: utf-8 -*-
"""
Phase 1 training: GRU agent on multi-comparison task with 4-stage curriculum.

Usage:
    python backend/training/train_phase1.py [--smoke] [--device cuda|cpu]

--smoke runs a short training (Stage 1 only, few epochs) to validate the pipeline.
Without --smoke, runs the full 4-stage curriculum.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.agent import AgentConfig, GRUAgent
from backend.core.attention_agent import AttentionAgent, AttentionAgentConfig
from backend.core.rglru_agent import RGLRUAgent, RGLRUAgentConfig
from backend.core.scene import SceneConfig, sample_training_batch
from backend.core.transformer_agent import TransformerAgent, TransformerAgentConfig
try:
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    _HAS_MAMBA_AGENT = True
except ImportError:
    _HAS_MAMBA_AGENT = False
from backend.evaluation.collect import collect_notes_and_predictions
from backend.evaluation.compositional import compositional_probe_score
from backend.evaluation.extrapolation import extrapolation_accuracy
from backend.evaluation.topo_sim import topographic_similarity
from backend.evaluation.weber import compute_weber_curve

import json as _json
from collections import Counter as _Counter


def _emit_emergence_snapshot(agent, stage_idx, stage_name, epoch, global_step, log_dir):
    """Dump V29 codebook + key params per validation. Writes to logs/<run>/emergence.jsonl.

    Only runs if agent has V29Pipeline (use_v29=True).
    """
    if not getattr(agent.agent_cfg, "use_v29", False):
        return
    if not hasattr(agent, "_v29_scratch_int") or agent._v29_scratch_int is None:
        return
    try:
        scratch_int = agent._v29_scratch_int.detach().cpu().numpy()
        # Codebook distribution
        codes = [tuple(int(x) for x in row) for row in scratch_int]
        code_counter = _Counter(codes)
        n_distinct = len(code_counter)
        top_code, top_count = code_counter.most_common(1)[0] if codes else (None, 0)
        # Key learned params (V29.5+)
        wh = agent.v29.write_head
        snap = {
            "event": "emergence_snapshot",
            "stage_idx": stage_idx,
            "stage": stage_name,
            "epoch": epoch,
            "global_step": global_step,
            "n_distinct_codes": n_distinct,
            "top_code": list(top_code) if top_code else None,
            "top_code_freq": top_count / max(1, len(codes)),
            "wh_omega": wh.omega.detach().cpu().numpy().tolist(),
            "wh_b": wh.b.detach().cpu().numpy().tolist(),
        }
        if hasattr(wh, "drive_scale"):
            snap["wh_drive_scale"] = wh.drive_scale.detach().cpu().numpy().tolist()
        snap["enc_omega_drift_mean"] = float(
            (agent.v29.encoder.omega - agent.v29.encoder.omega_init).abs().mean().item()
        )
        # enc_alive_neurons proxy: count encoder neurons that are NOT in dead state.
        # We don't have per-batch spike rates stashed, so use a proxy on learned params:
        # a neuron is "alive" if its input weight magnitude is non-trivial.
        try:
            enc_w_in = agent.v29.encoder.w_in.detach().cpu().numpy()
            snap["enc_alive_neurons"] = int((abs(enc_w_in) > 0.01).sum())
        except Exception:
            snap["enc_alive_neurons"] = -1  # unavailable
        # Append to logs/<run>/emergence.jsonl
        emer_path = Path(log_dir) / "emergence.jsonl"
        with open(emer_path, "a") as f:
            f.write(_json.dumps(snap) + "\n")
    except Exception:
        # Don't crash training over a diag failure
        import traceback
        traceback.print_exc()


@dataclass
class CurriculumStage:
    name: str
    n_max: int
    target_accuracy: float = 0.85
    max_epochs: int = 100
    L: Optional[int] = None  # per-stage scene_cfg.L override; None keeps current


@dataclass
class TrainConfig:
    # Optimizer
    learning_rate: float = 1e-3
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    # Batch / sampling
    batch_size: int = 64
    steps_per_epoch: int = 50
    # Validation
    validation_every_epochs: int = 5
    validation_batches: int = 4
    # Stage advancement
    accuracy_window: int = 3           # consecutive validations above target → advance
    # Full metric evaluation at stage boundaries
    stage_end_episodes: int = 128      # rollout size for per-stage metrics
    final_episodes: int = 512          # rollout size for end-of-run metrics
    extrapolation_episodes: int = 128  # per-range sample size
    # Reproducibility
    agent_seed: int = 42
    data_seed: int = 0

    @staticmethod
    def default_curriculum() -> List[CurriculumStage]:
        # NOTE: n_max must be <= SceneConfig.L - 1 (objects at distinct positions,
        # no stacking). Default SceneConfig.L = 50, so stage4 capped at 45 to
        # leave buffer. To push to N=100, increase SceneConfig.L to >= 150.
        return [
            CurriculumStage("stage1_unary", n_max=5, target_accuracy=0.85),
            CurriculumStage("stage2_grouping", n_max=15, target_accuracy=0.80),
            CurriculumStage("stage3_placevalue", n_max=40, target_accuracy=0.75),
            CurriculumStage("stage4_full", n_max=45, target_accuracy=0.70),
        ]

    @staticmethod
    def smoke_curriculum() -> List[CurriculumStage]:
        return [
            CurriculumStage("smoke_stage1", n_max=5, target_accuracy=0.70, max_epochs=20),
        ]


def linear_warmup(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup)


# Signed encoding: GREATER=+1, LESS=-1, EQUAL=0 (matches scene's label indices 0/1/2).
# Used for ordinal CE loss (graded by label-space distance) and regression loss
# (continuous prediction of α-β sign/magnitude).
_SIGNED_LABEL_VALUES = torch.tensor([+1.0, -1.0, 0.0])

# Loss types that consume continuous gap targets (N_α - N_β) instead of class labels.
_GAP_LOSS_TYPES = {"gap_mse4", "rl_top1", "rl_perblock"}


def _build_gap_target(metas: List[dict], device: torch.device) -> torch.Tensor:
    """(B, K) signed gap target: gap[b, k] = N_α[b] - N_β[b, k]."""
    return torch.tensor(
        [[m["N_alpha"] - nb for nb in m["beta_counts"]] for m in metas],
        dtype=torch.float32, device=device,
    )


# Label mapping moved to backend.core.label_mapping (single source of truth shared
# with eval; see that module's docstring for the divergence bug it fixes). Re-exported
# here so existing `from backend.training.train_phase1 import k_to_class, ...` keeps working.
from backend.core.label_mapping import balanced_n_classes, k_to_class  # noqa: E402,F401


def inverse_freq_class_weights(k_targets: "torch.Tensor", n_classes: int) -> "torch.Tensor":
    """Per-class inverse-frequency CE weights, mean-normalized over present classes
    (spec 2026-06-05 §3.5). Removes the constant-predict-majority class-prior shortcut
    that the COARSE balanced label otherwise allows (the FAR classes dominate the mass).
    Classes absent from the batch get weight 0 (unused — no sample targets them).
    """
    counts = torch.bincount(k_targets, minlength=n_classes).float()
    present = counts > 0
    w = torch.zeros(n_classes, device=k_targets.device, dtype=torch.float32)
    w[present] = 1.0 / counts[present]
    w[present] = w[present] / w[present].mean()
    return w


def _compute_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute training loss given logits (B,K,C) and labels (B,K) per loss type.

    loss_type == 'ce': standard 3-class cross-entropy. Supports class_weights
        for imbalanced classes (e.g., trio_wide where E class is rare).
    loss_type == 'ordinal': MSE on softmax-weighted expectation of signed-class
        values {+1, -1, 0}. Penalty grows with label-space distance.
    loss_type == 'regression': logits has 1 output channel; MSE on predicted
        continuous value vs signed target.
    """
    if loss_type == "ce":
        weight = class_weights.to(logits.device) if class_weights is not None else None
        return F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1), weight=weight,
        )
    signed = _SIGNED_LABEL_VALUES.to(logits.device)
    target = signed[labels]  # (B, K)
    if loss_type == "ordinal":
        probs = F.softmax(logits, dim=-1)  # (B, K, 3)
        expected = (probs * signed).sum(dim=-1)  # (B, K)
        return F.mse_loss(expected, target)
    if loss_type == "regression":
        # compare_head outputs C channels; regression uses channel 0 as scalar.
        pred = logits[..., 0]  # (B, K)
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _compute_gap_mse4(logits: torch.Tensor, gap_target: torch.Tensor) -> torch.Tensor:
    """Supervised gap-head loss: (pred - gap)^4. Sharper than MSE (cubic gradient
    in error magnitude) but bounded vs exp loss. Targets signed gap ∈ {-4..+4}
    when paired with trio_wide preset."""
    pred = logits[..., 0]  # (B, K) scalar gap prediction
    return ((pred - gap_target) ** 4).mean()


def _compute_rl_top1(
    logits: torch.Tensor,
    gap_target: torch.Tensor,
    baseline: float,
    ent_coef: float,
    reward_shape: str = "sharp",
    adv_normalize: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE with EMA baseline for top-1 categorical selection.

    logits: (B, K, 1) — preference score per option.
    gap_target: (B, K) — signed gap N_α - N_β per option.

    reward_shape:
        'sharp' = 100·10^-|gap| (spec default, peak 100 at gap=0, 0.01 at gap=4).
                  Aggressive baseline tracking can stall cold-start (advantage→0).
        'soft'  = exp(-|gap|/2) ∈ [0,1] (gap=0→1, gap=4→0.135). Denser gradient
                  across all gaps; mitigates baseline-tracking pathology.
    adv_normalize: z-score (reward - baseline) within batch before policy gradient.
                   Stabilizes step size when reward variance is high.

    Returns (loss, info_dict) where info_dict reports mean reward & entropy
    so the train loop can update the EMA baseline.
    """
    scores = logits[..., 0]  # (B, K)
    dist = torch.distributions.Categorical(logits=scores)
    action = dist.sample()  # (B,)
    log_prob = dist.log_prob(action)  # (B,)
    gap_chosen = gap_target.gather(1, action.unsqueeze(1)).squeeze(1).abs()  # (B,)

    if reward_shape == "sharp":
        reward = 100.0 * (10.0 ** (-gap_chosen))  # (B,)
    else:  # soft
        reward = torch.exp(-gap_chosen / 2.0)  # (B,) ∈ [0, 1]

    advantage = (reward - baseline).detach()
    if adv_normalize:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    entropy = dist.entropy()  # (B,)
    pg_loss = -(log_prob * advantage).mean()
    ent_loss = -ent_coef * entropy.mean()
    loss = pg_loss + ent_loss
    info = {
        "reward_mean": float(reward.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
        "gap0_frac": float((gap_chosen == 0).float().mean().item()),
    }
    return loss, info


# Asymmetric reward (P2-strong) constants for rl_perblock:
#   correct equal (TP):       +12
#   missed equal (FN):         −6
#   false positive equal (FP): −C(|gap|), C = [_, 4, 10, 14, 20]  (heavier for far gaps)
#   correct unequal (TN):      +6
#
# Designed for binary_balanced_preset (50% equal, 50% non-equal uniform over |gap|=1..4).
# Math: A = mean_C = 12, B = D = 6 → E[always_equal] = E[always_unequal] = 0,
# E[perfect] = 9, decision threshold p* = 0.5. Asymmetry A/D = 2 preserves
# "correct-equal is more valuable" intent while removing prior shortcut.
_ASYM_REWARD_TP = 12.0
_ASYM_REWARD_FN = -6.0
_ASYM_REWARD_TN = 6.0
_ASYM_REWARD_FP_BY_GAP = [0.0, -4.0, -10.0, -14.0, -20.0]  # indexed by |gap|, 0 unused


def _compute_rl_perblock(
    logits: torch.Tensor,
    gap_target: torch.Tensor,
    baseline: float,
    ent_coef: float,
    reward_type: str = "simple",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-block independent binary REINFORCE.

    Each of K compare blocks gets its OWN action (Bernoulli "is alpha == beta?")
    and its OWN reward. Per-block log_prob × advantage, averaged over (B*K) —
    yielding K independent gradient signals per episode instead of rl_top1's
    single K-coupled categorical gradient.

    reward_type:
        'simple'     : +1 correct, -1 wrong. Range [-1, +1]. Symmetric, small magnitude.
        'asymmetric' : P2-strong. +12 TP, -6 FN, -C(|gap|) FP (4/10/14/20), +6 TN.
                       Range [-20, +12]. Larger gradient signal; rewards correct equal
                       2x more than correct unequal; gap-graded false-positive penalty
                       discourages "always say equal" degeneration.

    Designed for --task binary (50/50 equal/non-equal balanced) so prior strategies
    'always equal' and 'always unequal' both yield 0 expected reward — forces use
    of alpha info to beat random.

    logits: (B, K, 1) — scalar score per block; sigmoid → P(equal).
    gap_target: (B, K) — signed gap N_alpha - N_beta. is_equal := (gap == 0).
    baseline: EMA of mean reward.
    """
    scores = logits[..., 0]  # (B, K)
    is_equal = (gap_target == 0).float()  # (B, K) ground truth

    probs_equal = torch.sigmoid(scores).clamp(1e-6, 1 - 1e-6)
    dist = torch.distributions.Bernoulli(probs=probs_equal)
    action = dist.sample()  # (B, K) ∈ {0, 1}; 1 = predict equal
    log_prob = dist.log_prob(action)  # (B, K)

    correct = (action == is_equal).float()

    if reward_type == "asymmetric":
        # P2-strong asymmetric reward. Use index lookup for FP penalty by |gap|.
        abs_gap = gap_target.abs().long().clamp(0, 4)  # (B, K)
        fp_penalty_table = torch.tensor(
            _ASYM_REWARD_FP_BY_GAP, device=scores.device, dtype=scores.dtype
        )
        fp_penalty = fp_penalty_table[abs_gap]  # (B, K), zero for is_equal=1 cases
        reward = (
            is_equal * action * _ASYM_REWARD_TP                      # TP
            + is_equal * (1.0 - action) * _ASYM_REWARD_FN            # FN
            + (1.0 - is_equal) * action * fp_penalty                 # FP (gap-graded)
            + (1.0 - is_equal) * (1.0 - action) * _ASYM_REWARD_TN    # TN
        )
    else:  # simple
        reward = 2.0 * correct - 1.0  # (B, K) ∈ {-1, +1}

    advantage = (reward - baseline).detach()
    pg_loss = -(log_prob * advantage).mean()
    entropy = dist.entropy()  # (B, K)
    ent_loss = -ent_coef * entropy.mean()
    loss = pg_loss + ent_loss

    info = {
        "reward_mean": float(reward.mean().item()),
        "accuracy": float(correct.mean().item()),
        "entropy_mean": float(entropy.mean().item()),
        "predict_equal_rate": float(action.mean().item()),
    }
    return loss, info


def _logits_to_preds(logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Convert logits to class predictions for accuracy computation.

    Threshold logic for scalar-output heads (regression, gap_mse4): pred>0.5=GREATER,
    pred<-0.5=LESS, else EQUAL. With gap-head trained on integer targets ±4, the
    ±0.5 threshold is the natural rounding boundary.

    rl_top1 has no per-block class semantics (output is preference score), so we
    return EQUAL placeholders; validate_stage_rl computes a different accuracy.
    """
    if loss_type in ("regression", "gap_mse4"):
        pred = logits[..., 0]
        out = torch.full_like(pred, fill_value=2, dtype=torch.long)
        out = torch.where(pred > 0.5, torch.zeros_like(out), out)
        out = torch.where(pred < -0.5, torch.ones_like(out), out)
        return out
    if loss_type == "rl_top1":
        # Block-level 3-class label is undefined for top-1 selection.
        # Returning all-EQUAL keeps shape compatible; validation overrides
        # this with its own gap-0 hit-rate metric.
        scores = logits[..., 0]
        return torch.full_like(scores, fill_value=2, dtype=torch.long)
    return logits.argmax(dim=-1)


def run_batch(
    agent: GRUAgent,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]],
    device: torch.device,
    loss_type: str = "ce",
    rl_baseline: float = 0.0,
    rl_entropy_coef: float = 0.05,
    rl_reward_shape: str = "sharp",
    rl_adv_normalize: bool = False,
    rl_perblock_reward: str = "simple",
    world_pred_lambda: float = 0.0,
    beta_pred_lambda: float = 0.0,
    compare_pred_lambda: float = 0.0,
    rs_pred_lambda: float = 0.0,
    scratch_self_pred_lambda: float = 0.0,
    cf_beta_pred_lambda: float = 0.0,
    vq_commit_lambda: float = 0.0,
    dual_world_lambda: float = 0.0,
    dual_sp_lambda: float = 0.0,
    vicreg_lambda: float = 0.0,
    pc_explicit_lambda: float = 0.0,
    cycle_loss_lambda: float = 0.0,
    ar_entropy_bonus: float = 0.0,
    multi_modular_aux_lambda: float = 0.0,
    multi_modular_aux_moduli: Tuple[int, ...] = (),
    multi_modular_aux_pool: Tuple[int, ...] = (),
    detection_aux_weight: float = 0.0,
    scratch_mod_aux_weight: float = 0.0,
    scratch_mod_aux_moduli: Tuple[int, ...] = (),
    e_pair_consistency_weight: float = 0.0,
    successor_loss_weight: float = 0.0,
    successor_detached: bool = False,
    h_pair_consistency_weight: float = 0.0,
    n_class_ce_weight: float = 0.0,
    ce_class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Forward + loss. Returns (loss, logits, labels, info).
    info contains rl-specific stats (reward, entropy, gap0_frac) when applicable;
    empty dict for non-RL paths.
    """
    inputs, labels, compare_idx, metas = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    compare_idx = compare_idx.to(device)

    # Oracle-gate plumbing: when enabled on the agent, supply per-batch spike
    # centers from scene meta. Cheap when disabled (we still build the lists
    # but agent ignores them).
    use_oracle = getattr(getattr(agent, "agent_cfg", None), "use_oracle_spike_gate", False)
    if use_oracle:
        alpha_centers = [m.get("alpha_centers", []) for m in metas]
        # beta_centers in meta is shape List[K] of List[int]; we want
        # K-major: list of K entries, each list-of-B lists.
        K = len(metas[0].get("beta_centers", []))
        beta_centers_list = [
            [m.get("beta_centers", [[]] * K)[i] for m in metas]
            for i in range(K)
        ]
        logits, _scratch, _ = agent(
            inputs, compare_idx,
            alpha_centers=alpha_centers,
            beta_centers_list=beta_centers_list,
        )
    else:
        logits, _scratch, _ = agent(inputs, compare_idx)
    info: Dict[str, float] = {}

    # V34 unsupervised dispatch (5/21): pure self-supervised symbol emergence.
    # No task CE, no β/N_β loss — only 4 sg-protected aux losses on α-side scratch
    # and the closed-loop readback. Must be checked BEFORE V29 / V24 paths.
    if getattr(agent.agent_cfg, "unsupervised_mode", False):
        sc_alpha_raw = getattr(agent, "_scratch_alpha_raw", None)
        h_rb_actual = getattr(agent, "_h_readback_alpha_end", None)
        h_predicted = getattr(agent, "_h_predicted_alpha", None)
        if sc_alpha_raw is None or h_rb_actual is None or h_predicted is None:
            raise RuntimeError(
                "V34 unsupervised_mode requires v33_same_medium=True AND "
                "predict_coding_enable=True (stashes _scratch_alpha_raw, "
                "_h_readback_alpha_end, _h_predicted_alpha must be present)."
            )
        # L_commit nearest-level formula is hardcoded for q=3 unit alphabet
        # {0, 0.5, 1.0}. Reject other configs explicitly rather than silently
        # computing wrong targets. CLI guard fires earlier, but defense in depth.
        q_levels = getattr(agent.agent_cfg, "quantize_levels", 3)
        q_range = getattr(agent.agent_cfg, "quantize_range", "unit")
        if q_levels != 3 or q_range != "unit":
            raise RuntimeError(
                f"V34 unsupervised_mode L_commit is hardcoded for q=3 unit range; "
                f"got quantize_levels={q_levels}, quantize_range={q_range!r}."
            )
        # Lambdas live on agent_cfg
        lam_pred = getattr(agent.agent_cfg, "predict_coding_lambda", 1.0)
        lam_commit = getattr(agent.agent_cfg, "commit_lambda", 0.25)
        lam_div = getattr(agent.agent_cfg, "diversity_lambda", 0.05)
        lam_l2 = getattr(agent.agent_cfg, "raw_l2_reg_lambda", 0.001)

        # L_predict: predictor learns to match actual h_readback_end.
        # Stop-gradient on actual so predictor (and write_head via raw) is updated;
        # actual h_readback is the FROZEN target (the "ground truth" we want to match).
        l_predict = (h_predicted - h_rb_actual.detach()).pow(2).mean()

        # L_commit: σ(raw) commits to nearest alphabet level (VQ-VAE commitment).
        # Stop-gradient on nearest so commit drives raw, not the discretization.
        soft = torch.sigmoid(sc_alpha_raw)
        nearest = torch.round(soft * 2) / 2  # ∈ {0, 0.5, 1.0} for q=3 unit
        l_commit = (soft - nearest.detach()).pow(2).mean()

        # L_diversity: pairwise spread of σ(raw) across W cells. Negated so
        # minimizing the loss MAXIMIZES the spread (anti-collapse).
        diff = soft.unsqueeze(2) - soft.unsqueeze(1)  # (B, W, W)
        l_diversity = -(diff.pow(2).mean())

        # L_raw_l2: weak L2 on raw to prevent commit pushing it to ±∞.
        l_raw_l2 = sc_alpha_raw.pow(2).mean()

        loss = (lam_pred * l_predict + lam_commit * l_commit
                + lam_div * l_diversity + lam_l2 * l_raw_l2)

        info["l_predict"] = float(l_predict.item())
        info["l_commit"] = float(l_commit.item())
        info["l_diversity"] = float(l_diversity.item())
        info["l_raw_l2"] = float(l_raw_l2.item())
        # Diagnostic: mean σ(raw); cross-cell std measures per-episode cell spread
        # (cascade-break monitor — high std = cells differ, low std = cascade collapse).
        # Naming: "cross-cell" because std is taken across the W cell axis per sample.
        info["soft_mean"] = float(soft.mean().item())
        info["soft_cross_cell_std"] = float(soft.std(dim=-1).mean().item())

        # Return dummy logits/labels matching framework signature.
        # NOTE: dummy_labels=ones (not zeros) intentionally — argmax(zeros)=0 always,
        # so labels=0 would give val_acc=1.0 and trigger curriculum auto-advancement
        # (target_accuracy=0.7 default). With labels=1, preds(=0) never match, val_acc=0,
        # and the curriculum never advances. val_acc has no meaningful interpretation
        # under unsupervised_mode anyway — watch info["l_predict"] etc. instead.
        # Width must match the framework's expected class count. AgentConfig field
        # successor_predict_k_max already accounts for bidirectional doubling at
        # build time (= 2*k_max for bidir, k_max for unidir). Fallback to 10 covers
        # the canonical --successor-task-k-max 5 --successor-bidirectional config.
        n_dummy_classes = max(getattr(agent.agent_cfg, "successor_predict_k_max", 10), 1)
        B = inputs.shape[0]
        dummy_logits = torch.zeros(B, n_dummy_classes, device=device, dtype=loss.dtype)
        dummy_labels = torch.ones(B, dtype=torch.long, device=device)
        return loss, dummy_logits, dummy_labels, info

    # V29 task path: V29Pipeline produces logits internally (its own predict_head).
    # Use the same bidirectional successor k-class indexing as V27.
    v29_module = getattr(agent, "v29", None)
    if v29_module is not None:
        v29_logits = getattr(agent, "_v29_logits", None)
        if v29_logits is None:
            raise RuntimeError("V29 agent did not stash _v29_logits during forward.")
        n_classes = v29_logits.shape[-1]
        scene_bidir = not getattr(agent.scene_cfg, "successor_only_positive", True)
        k_max_val = n_classes // 2 if scene_bidir else n_classes
        k_raw = [m["beta_counts"][0] - m["N_alpha"] for m in metas]
        if scene_bidir:
            def _k_to_class_v29(k):
                if k < 0:
                    return max(0, k + k_max_val)
                elif k > 0:
                    return min(n_classes - 1, k + k_max_val - 1)
                else:
                    return 0
            k_targets = torch.tensor([_k_to_class_v29(k) for k in k_raw],
                                     device=device, dtype=torch.long)
        else:
            k_targets = torch.tensor([k - 1 for k in k_raw],
                                     device=device, dtype=torch.long)
            k_targets = k_targets.clamp(min=0, max=n_classes - 1)
        ce_loss = F.cross_entropy(v29_logits, k_targets)
        # Aux losses (orthogonality + omega/b regularization)
        orth_loss = getattr(agent, "_v29_orth_loss", None)
        reg_loss = getattr(agent, "_v29_reg_loss", None)
        orth_lambda = agent.agent_cfg.v29_query_orth_lambda
        reg_lambda = agent.agent_cfg.v29_omega_clamp_decay
        loss = ce_loss
        if orth_loss is not None:
            loss = loss + orth_lambda * orth_loss
        if reg_loss is not None:
            loss = loss + reg_lambda * reg_loss
        firing_reg = getattr(agent, "_v29_firing_reg", None)
        firing_lambda = agent.agent_cfg.v29_firing_rate_lambda
        if firing_reg is not None:
            loss = loss + firing_lambda * firing_reg
            info["v29_firing_reg"] = float(firing_reg.item())
        info["v29_ce"] = float(ce_loss.item())
        if orth_loss is not None:
            info["v29_orth"] = float(orth_loss.item())
        if reg_loss is not None:
            info["v29_reg"] = float(reg_loss.item())
        info["v29_k_acc"] = float((v29_logits.argmax(dim=-1) == k_targets).float().mean().item())
        return loss, v29_logits, k_targets, info

    # V35 task path: V35CompareAttention produces logits internally and stashes
    # them at `_v35_successor_logits` (set by _v35_forward). Skip the standard
    # successor_predict_head path which doesn't apply to V35.
    v35_logits = getattr(agent, "_v35_successor_logits", None)
    if v35_logits is not None:
        n_classes = v35_logits.shape[-1]
        scene_bidir = not getattr(agent.scene_cfg, "successor_only_positive", True)
        balanced = getattr(agent.scene_cfg, "beta_range_preset", "successor") == "balanced"
        k_raw = [m["beta_counts"][0] - m["N_alpha"] for m in metas]
        if balanced:
            # V35 PSS leak-kill (spec 2026-06-05 §3.4): COARSE relational label via the
            # module-level k_to_class. Recover k_max from n_classes = 2*k_max+1+2*far_bands.
            far_bands = getattr(agent.scene_cfg, "successor_far_bands", 1)
            k_max_val = (n_classes - 1 - 2 * far_bands) // 2
            k_targets = torch.tensor([k_to_class(k, k_max_val, far_bands) for k in k_raw],
                                     device=device, dtype=torch.long)
        elif scene_bidir:
            k_max_val = n_classes // 2
            def _k_to_class_legacy(k):
                if k < 0:
                    return max(0, k + k_max_val)
                elif k > 0:
                    return min(n_classes - 1, k + k_max_val - 1)
                else:
                    return 0
            k_targets = torch.tensor([_k_to_class_legacy(k) for k in k_raw],
                                     device=device, dtype=torch.long)
        else:
            k_targets = torch.tensor([k - 1 for k in k_raw],
                                     device=device, dtype=torch.long)
            k_targets = k_targets.clamp(min=0, max=n_classes - 1)
        # V35 PSS leak-kill (spec 2026-06-05 §3.5): inverse-frequency CE weighting for the
        # balanced preset removes the constant-predict-FAR class-prior bypass. None for the
        # legacy successor preset (keeps existing behavior unchanged).
        ce_weight = inverse_freq_class_weights(k_targets, n_classes) if balanced else None
        ce_loss = F.cross_entropy(v35_logits, k_targets, weight=ce_weight)
        loss = ce_loss
        info["v35_ce"] = float(ce_loss.item())
        # Codebook-fidelity recon aux (spec 2026-06-04): MSE(D(scratch_alpha), residuals_alpha).
        # Forces scratch to losslessly carry the substrate residuals so the high digits aren't
        # dropped under the weak |k|<=5 task pressure. Target detached (substrate = reference).
        recon_lambda = getattr(agent.agent_cfg, "v35_recon_aux_lambda", 0.0) or 0.0
        recon_decoder = getattr(agent, "v35_recon_decoder", None)
        res_alpha = getattr(agent, "_v35_residuals_alpha", None)
        sc_alpha_q = getattr(agent, "_scratch_alpha_for_aux", None)
        if recon_lambda > 0.0 and recon_decoder is not None and res_alpha is not None and sc_alpha_q is not None:
            res_recon = recon_decoder(sc_alpha_q)
            recon_loss = F.mse_loss(res_recon, res_alpha.detach())
            loss = loss + recon_lambda * recon_loss
            info["v35_recon"] = float(recon_loss.item())
        # Cheap-arithmetic aux (INSIGHTS §9A): a CHEAP (single Linear) successor head on
        # the quantized alpha scratch code, trained with the SAME k_targets as the main
        # task. Its CE can only drop if the scratch supports linear arithmetic readout,
        # which pressures the codebook toward the positional class (an expressive head
        # exerts no such pressure). Quantize-STE in the write head carries this gradient
        # back into the scratch code. lambda <= 0 -> exact no-op.
        cheap_lambda = getattr(agent.agent_cfg, "v35_cheap_arith_aux_lambda", 0.0) or 0.0
        cheap_head = getattr(agent, "v35_cheap_arith_head", None)
        if cheap_lambda > 0.0 and cheap_head is not None and sc_alpha_q is not None:
            cheap_logits = cheap_head(sc_alpha_q)
            cheap_ce = F.cross_entropy(cheap_logits, k_targets, weight=ce_weight)
            loss = loss + cheap_lambda * cheap_ce
            info["v35_cheap_arith_ce"] = float(cheap_ce.item())
            info["v35_cheap_arith_acc"] = float(
                (cheap_logits.argmax(dim=-1) == k_targets).float().mean().item()
            )
        # Counting-pressure aux (direction-1, 2026-06-16): port count+carry's PROVEN objective so the
        # LEARNED Layer-A counter has a gradient toward a CONSISTENT, INJECTIVE per-N code (V35's
        # successor-CE alone gives no counting pressure -> the learned detector over-fires inconsistently
        # -> codebook collapses to a constant code). WITHIN-N consistency = penalize scratch variance
        # within each N_alpha group (same N, different object positions -> same code). ACROSS-N injectivity
        # = hinge that pushes different-N mean codes apart (>= margin), which PREVENTS the trivial collapse
        # (all-same code zeroes consistency but is killed by injectivity). Admissible: unsupervised, rewards
        # a distinct-stable code per N; does NOT hand N or the code. lambda <= 0 -> exact no-op.
        consist_lambda = getattr(agent.agent_cfg, "v35_count_consist_lambda", 0.0) or 0.0
        if consist_lambda > 0.0 and sc_alpha_q is not None:
            N_a = torch.tensor([m["N_alpha"] for m in metas], device=sc_alpha_q.device)
            uniqN = N_a.unique()
            cons = sc_alpha_q.new_zeros(())
            means = []
            for n in uniqN:
                g = sc_alpha_q[N_a == n]                       # (g, W) codes for this N
                if g.shape[0] > 1:
                    cons = cons + g.var(dim=0, unbiased=False).mean()
                means.append(g.mean(dim=0))
            cons = cons / max(len(uniqN), 1)
            inj = sc_alpha_q.new_zeros(())
            if len(means) > 1:
                M = torch.stack(means)                         # (U, W) per-N mean codes
                D = torch.cdist(M, M)                           # (U, U)
                margin = 0.5
                off = D + torch.eye(len(M), device=D.device) * 1e9
                inj = torch.relu(margin - off).mean()          # push different-N codes >= margin apart
            consist_aux = cons + inj
            loss = loss + consist_lambda * consist_aux
            info["v35_count_consist"] = float(cons.item())
            info["v35_count_inj"] = float(inj.item())
        info["v35_k_acc"] = float((v35_logits.argmax(dim=-1) == k_targets).float().mean().item())
        return loss, v35_logits, k_targets, info

    # V24 task pivot: when successor_predict_head is built, run successor
    # prediction loss path INSTEAD of standard 3-class CE / mod_aux / etc.
    succ_predict_head = getattr(agent, "successor_predict_head", None)
    if succ_predict_head is not None:
        input_level = agent.agent_cfg.successor_predict_input_level
        if input_level == "pfc_hidden":
            # V25-V24: read PFC hidden state (h_α_pfc, h_β_pfc) instead of scratch.
            h_a_blocks = getattr(agent, "_h_alpha_pfc_blocks", None)
            h_b_blocks = getattr(agent, "_h_beta_pfc_blocks", None)
            if not h_a_blocks or not h_b_blocks:
                raise RuntimeError("V25-V24 (pfc_hidden) requires _h_alpha_pfc_blocks + "
                                   "_h_beta_pfc_blocks (need pfc_at_write).")
            # K=1: take first compare block.
            h_a_pfc = h_a_blocks[0]
            h_b_pfc = h_b_blocks[0]
            v24_input = torch.cat([h_a_pfc, h_b_pfc], dim=-1)  # (B, 2*d)
            k_logits = succ_predict_head(v24_input)
        else:
            sc_a = getattr(agent, "_scratch_alpha_for_aux", None)
            sc_b_q = getattr(agent, "_last_scratch_beta_K_q", None)
            if sc_a is None or sc_b_q is None:
                raise RuntimeError("V24 task requires _scratch_alpha_for_aux + "
                                   "_last_scratch_beta_K_q (need pfc_at_write + virtual β path).")
            # K=1 for V24: take first compare block's β scratch.
            sc_b_first = sc_b_q[:, 0, :]  # (B, W)
            # V24R: optional recurrent refinement on α and β scratch.
            refiner = getattr(agent, "recurrent_refiner", None)
            if refiner is not None:
                h_a_pfc = getattr(agent, "_last_h_alpha_pfc_for_refine", None)
                h_b_pfc = getattr(agent, "_last_h_beta_pfc_for_refine", None)
                lru_a = getattr(agent, "_last_lru_h_alpha_for_refine", None)
                lru_b = getattr(agent, "_last_lru_h_beta_for_refine", None)
                if all(x is not None for x in (h_a_pfc, h_b_pfc, lru_a, lru_b)):
                    qlevels = agent.agent_cfg.quantize_levels
                    qrange = agent.agent_cfg.quantize_range
                    sc_a = refiner(sc_a, h_a_pfc, lru_a,
                                   agent.write_head, agent.scratch_pos_emb,
                                   qlevels, qrange)
                    sc_b_first = refiner(sc_b_first, h_b_pfc, lru_b,
                                         agent.write_head, agent.scratch_pos_emb,
                                         qlevels, qrange)
            # V28-Sin: predict_head reads multi-hot encoded categorical levels,
            # not scalar quantized values. Convert sc_a / sc_b_first → int via
            # quantize-to-int helper, then call MultiHotMLPSuccessorHead.
            from backend.core.mamba_agent import (
                MultiHotMLPSuccessorHead, SinusoidalWriteHead, LRUDecoder,
            )
            if isinstance(succ_predict_head, MultiHotMLPSuccessorHead):
                # V28-Sin / V28-LRU enforce same-medium scratch ∈ [0, 1] regardless
                # of CLI quantize_range (they pre-map sin/tanh output).
                # Other write heads honor agent.agent_cfg.quantize_range.
                qlevels = agent.agent_cfg.quantize_levels
                if isinstance(agent.write_head, (SinusoidalWriteHead, LRUDecoder)):
                    # output guaranteed in [0, 1] (modules force linear_unit internally)
                    sc_a_int = (sc_a * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                    sc_b_int = (sc_b_first * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                elif agent.agent_cfg.quantize_range == "symmetric":
                    sc_a_int = ((sc_a + 1.0) / 2.0 * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                    sc_b_int = ((sc_b_first + 1.0) / 2.0 * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                else:
                    sc_a_int = (sc_a * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                    sc_b_int = (sc_b_first * (qlevels - 1)).round().long().clamp(0, qlevels - 1)
                k_logits = succ_predict_head(sc_a_int, sc_b_int)
            else:
                v24_input = torch.cat([sc_a, sc_b_first], dim=-1)  # (B, 2W)
                k_logits = succ_predict_head(v24_input)
        n_classes = k_logits.shape[-1]
        # V27 bidirectional detection: when scene allows ±k, n_classes = 2 * k_max
        # Map k = N_β - N_α to class index:
        #   positive-only (n_classes = k_max):  k ∈ [1, k_max]    → class = k - 1
        #   bidirectional (n_classes = 2*k_max): k ∈ [-k_max..-1, 1..k_max] (no 0)
        #     → class = k + k_max  (k=-k_max → 0, k=-1 → k_max-1)
        #     → class = k + k_max - 1 for k > 0  (k=1 → k_max, k=k_max → 2*k_max-1)
        scene_bidir = not getattr(agent.scene_cfg, "successor_only_positive", True)
        k_max_val = n_classes // 2 if scene_bidir else n_classes
        k_raw = [m["beta_counts"][0] - m["N_alpha"] for m in metas]   # k = N_β - N_α
        if scene_bidir:
            # 10-class: k=-5→0, k=-1→4, k=+1→5, k=+5→9
            def _k_to_class_v24(k):
                if k < 0:
                    return max(0, k + k_max_val)        # k=-k_max→0
                elif k > 0:
                    return min(n_classes - 1, k + k_max_val - 1)  # k=+1→k_max
                else:
                    return 0  # shouldn't happen, but fallback
            k_targets = torch.tensor([_k_to_class_v24(k) for k in k_raw],
                                     device=device, dtype=torch.long)
        else:
            # k - 1 indexing (V24 default)
            k_targets = torch.tensor([k - 1 for k in k_raw],
                                     device=device, dtype=torch.long)
            k_targets = k_targets.clamp(min=0, max=n_classes - 1)
        loss = F.cross_entropy(k_logits, k_targets)
        info["v24_k_acc"] = float((k_logits.argmax(dim=-1) == k_targets).float().mean().item())

        # V33-A (5/20): channel orthogonality regularizer. Penalizes v_c geometric
        # collapse — baseline (λ=0) saw pairwise cos > 0.91 across all 6 channel pairs,
        # collapsing the substrate to effective rank-1. Lambda > 0 pushes mean cos² → 0.
        v33_ortho_lambda = getattr(agent.agent_cfg, "v33_ortho_lambda", 0.0)
        v33_sub = getattr(agent, "v33_substrate", None)
        if v33_ortho_lambda > 0.0 and v33_sub is not None:
            ortho_loss_v33 = v33_sub.channel_ortho_loss()
            loss = loss + v33_ortho_lambda * ortho_loss_v33
            info["v33_ortho_loss"] = float(ortho_loss_v33.item())
            info["v33_max_cos_overlap"] = v33_sub.max_pairwise_overlap()

        # V34-supervised-commit (5/21): VQ-VAE commit + raw L2 aux losses on the
        # supervised V24 path. Designed for SM+PM ckpts to fix the STE bottleneck:
        # raw values cluster in one bin (e.g., -1.20 for SM+PM 50ep, lv0 side)
        # and don't get gradient pressure to commit firmly. L_commit pushes σ(raw)
        # toward nearest alphabet level (bypassing STE entirely); L_raw_l2 prevents
        # commit from driving raw → ±∞. Task CE decides which bin per N; these
        # aux losses just sharpen the commitment. Math hardcoded for q=3 unit
        # (alphabet {0, 0.5, 1.0}) — CLI guard enforces. Reuses agent stash
        # _scratch_alpha_raw populated by write_head in agent forward.
        commit_lambda = getattr(agent.agent_cfg, "commit_lambda", 0.0)
        raw_l2_lambda = getattr(agent.agent_cfg, "raw_l2_reg_lambda", 0.0)
        diversity_lambda = getattr(agent.agent_cfg, "diversity_lambda", 0.0)
        sc_a_raw = getattr(agent, "_scratch_alpha_raw", None)
        if (commit_lambda > 0.0 or raw_l2_lambda > 0.0 or diversity_lambda > 0.0) and sc_a_raw is not None:
            soft = torch.sigmoid(sc_a_raw)
            if commit_lambda > 0.0:
                nearest = torch.round(soft * 2) / 2  # ∈ {0, 0.5, 1.0}
                l_commit = (soft - nearest.detach()).pow(2).mean()
                loss = loss + commit_lambda * l_commit
                info["l_commit"] = float(l_commit.item())
                info["soft_mean"] = float(soft.mean().item())
                info["soft_cross_cell_std"] = float(soft.std(dim=-1).mean().item())
            if diversity_lambda > 0.0:
                # Anti-cascade-collapse: maximize pairwise squared diff of σ(raw)
                # across W cells. Minimize -mean(diff²) → maximize cell spread.
                diff = soft.unsqueeze(2) - soft.unsqueeze(1)  # (B, W, W)
                l_diversity = -(diff.pow(2).mean())
                loss = loss + diversity_lambda * l_diversity
                info["l_diversity"] = float(l_diversity.item())
            if raw_l2_lambda > 0.0:
                l_raw_l2 = sc_a_raw.pow(2).mean()
                loss = loss + raw_l2_lambda * l_raw_l2
                info["l_raw_l2"] = float(l_raw_l2.item())

        # V33-VICReg (5/22): substrate-side anti-collapse aux. Stashes computed
        # in _v33_forward when v33_vicreg_lambda_* > 0. Decoupled from commit/
        # diversity (those act on scratch raw), this acts on substrate h directly.
        v33_vlam_var = getattr(agent.agent_cfg, "v33_vicreg_lambda_var", 0.0)
        v33_vlam_cov = getattr(agent.agent_cfg, "v33_vicreg_lambda_cov", 0.0)
        v33_lv = getattr(agent, "_v33_vicreg_var", None)
        v33_lc = getattr(agent, "_v33_vicreg_cov", None)
        if (v33_vlam_var > 0.0 or v33_vlam_cov > 0.0) and v33_lv is not None and v33_lc is not None:
            loss = loss + v33_vlam_var * v33_lv + v33_vlam_cov * v33_lc
            info["v33_vicreg_var"] = float(v33_lv.item())
            info["v33_vicreg_cov"] = float(v33_lc.item())

        return loss, k_logits, k_targets, info

    if loss_type in _GAP_LOSS_TYPES:
        gap_target = _build_gap_target(metas, device)
        if loss_type == "gap_mse4":
            loss = _compute_gap_mse4(logits, gap_target)
        elif loss_type == "rl_perblock":
            loss, info = _compute_rl_perblock(
                logits, gap_target, rl_baseline, rl_entropy_coef,
                reward_type=rl_perblock_reward,
            )
        else:  # rl_top1
            loss, info = _compute_rl_top1(
                logits, gap_target, rl_baseline, rl_entropy_coef,
                reward_shape=rl_reward_shape,
                adv_normalize=rl_adv_normalize,
            )
    else:
        loss = _compute_loss(logits, labels, loss_type, class_weights=ce_class_weights)

    # World-prediction auxiliary loss (E0-prime). Mamba agent stashes its
    # per-batch world_loss on _last_world_loss; we add it to total here.
    world_loss = getattr(agent, "_last_world_loss", None)
    if world_loss is not None and world_pred_lambda > 0.0:
        loss = loss + world_pred_lambda * world_loss
        info["world_loss"] = float(world_loss.item())

    # A2: β-scan prediction auxiliary loss.
    beta_loss = getattr(agent, "_last_beta_pred_loss", None)
    if beta_loss is not None and beta_pred_lambda > 0.0:
        loss = loss + beta_pred_lambda * beta_loss
        info["beta_pred_loss"] = float(beta_loss.item())

    # A5: ComparePredictor self-consistency auxiliary loss.
    cmp_loss = getattr(agent, "_last_compare_pred_loss", None)
    if cmp_loss is not None and compare_pred_lambda > 0.0:
        loss = loss + compare_pred_lambda * cmp_loss
        info["compare_pred_loss"] = float(cmp_loss.item())

    # RS-1: Raw signal prediction auxiliary loss.
    rs_loss = getattr(agent, "_last_rs_pred_loss", None)
    if rs_loss is not None and rs_pred_lambda > 0.0:
        loss = loss + rs_pred_lambda * rs_loss
        info["rs_pred_loss"] = float(rs_loss.item())

    # A3: Scratch self-prediction auxiliary loss.
    ssp_loss = getattr(agent, "_last_scratch_self_pred_loss", None)
    if ssp_loss is not None and scratch_self_pred_lambda > 0.0:
        loss = loss + scratch_self_pred_lambda * ssp_loss
        info["scratch_self_pred_loss"] = float(ssp_loss.item())

    # A4: Counterfactual β prediction auxiliary loss.
    cfb_loss = getattr(agent, "_last_cf_beta_pred_loss", None)
    if cfb_loss is not None and cf_beta_pred_lambda > 0.0:
        loss = loss + cf_beta_pred_lambda * cfb_loss
        info["cf_beta_pred_loss"] = float(cfb_loss.item())

    # β-1: VQ commitment loss + codebook utilization tracking.
    vq_commit = getattr(agent, "_last_vq_commit_loss", None)
    if vq_commit is not None and vq_commit_lambda > 0.0:
        loss = loss + vq_commit_lambda * vq_commit
        info["vq_commit_loss"] = float(vq_commit.item())
    vq_idx = getattr(agent, "_last_vq_indices", None)
    if vq_idx is not None:
        # Distinct codes used in this batch (utilization proxy)
        info["vq_distinct_codes"] = int(vq_idx.unique().numel())

    # P1 (L1579): dual-loss world + sp.
    dual_world = getattr(agent, "_last_dual_world_loss", None)
    if dual_world is not None and dual_world_lambda > 0.0:
        loss = loss + dual_world_lambda * dual_world
        info["dual_world_loss"] = float(dual_world.item())
    dual_sp = getattr(agent, "_last_dual_sp_loss", None)
    if dual_sp is not None and dual_sp_lambda > 0.0:
        loss = loss + dual_sp_lambda * dual_sp
        info["dual_sp_loss"] = float(dual_sp.item())

    # COLLAPSE FIXES: VICReg variance regularizers + PC explicit loss
    if vicreg_lambda > 0.0:
        for name in ("_last_world_vicreg", "_last_beta_vicreg",
                     "_last_compare_vicreg", "_last_pc_vicreg"):
            v = getattr(agent, name, None)
            if v is not None:
                loss = loss + vicreg_lambda * v
                info[name.lstrip("_")] = float(v.item())
    pc_explicit = getattr(agent, "_last_pc_explicit_loss", None)
    if pc_explicit is not None and pc_explicit_lambda > 0.0:
        loss = loss + pc_explicit_lambda * pc_explicit
        info["pc_explicit_loss"] = float(pc_explicit.item())

    # 51_M5: cycle loss
    cycle_loss = getattr(agent, "_last_cycle_loss", None)
    if cycle_loss is not None and cycle_loss_lambda > 0.0:
        loss = loss + cycle_loss_lambda * cycle_loss
        info["cycle_loss"] = float(cycle_loss.item())

    # V9-AR: anti-collapse entropy bonus. We MAXIMIZE entropy → SUBTRACT λ·H from loss.
    ar_entropy = getattr(agent, "_last_ar_entropy", None)
    if ar_entropy is not None and ar_entropy_bonus > 0.0:
        loss = loss - ar_entropy_bonus * ar_entropy
        info["ar_entropy"] = float(ar_entropy.item())

    # V17: Multi-modular aux loss. For each co-prime modulus m, BCE on
    # (α mod m == β mod m) per-block prediction. Trio_wide compare optimum
    # is 1D thermometer; RNS pressure forces ≥k independent oscillations.
    aux_logits = getattr(agent, "_last_aux_mod_logits", None)
    if (aux_logits is not None and multi_modular_aux_lambda > 0.0
            and multi_modular_aux_moduli):
        # V19: if pool is set, sample n_mod moduli FRESH per batch from pool
        # (random per batch reduces "specific frequency" prior). Sort ascending
        # so head index has consistent positional meaning across batches.
        n_mod = len(multi_modular_aux_moduli)
        if multi_modular_aux_pool and len(multi_modular_aux_pool) >= n_mod:
            # Random sample n_mod from pool, sorted ascending
            import random
            sampled = sorted(random.sample(list(multi_modular_aux_pool), n_mod))
            active_moduli = tuple(sampled)
            info["aux_mod_active"] = ",".join(str(m) for m in active_moduli)
        else:
            active_moduli = multi_modular_aux_moduli
        # Build labels (B, K, n_moduli) on device
        B = len(metas)
        K = aux_logits.shape[1]
        labels_aux = torch.empty(B, K, n_mod, device=device)
        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            for i, n_b in enumerate(meta["beta_counts"]):
                for k_m, mod_v in enumerate(active_moduli):
                    labels_aux[b, i, k_m] = float((n_a % mod_v) == (n_b % mod_v))
        # V19 fix: per-modulus pos_weight to counteract trio_wide's class
        # imbalance. mod 5 same-rate ~0.20 → trivial baseline gets 0.80
        # acc by always predicting "different". pos_weight = neg/pos rebalances
        # BCE so model can't exploit bias to "win".
        n_per = float(B * K)
        pos_count = labels_aux.sum(dim=(0, 1)).clamp(min=1.0)  # (n_mod,)
        pos_weight = ((n_per - pos_count) / pos_count).clamp(min=0.5, max=20.0)
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logits, labels_aux, pos_weight=pos_weight,
        )
        loss = loss + multi_modular_aux_lambda * aux_loss
        info["aux_mod_loss"] = float(aux_loss.item())
        # Per-modulus accuracy (diagnostic) — also report balanced acc
        with torch.no_grad():
            preds = (aux_logits > 0).float()
            for k_m, mod_v in enumerate(active_moduli):
                acc_m = (preds[..., k_m] == labels_aux[..., k_m]).float().mean()
                # Balanced acc: average of recall_pos and recall_neg
                pos_mask = labels_aux[..., k_m] > 0.5
                neg_mask = ~pos_mask
                if pos_mask.any() and neg_mask.any():
                    rec_pos = (preds[..., k_m][pos_mask] == 1.0).float().mean()
                    rec_neg = (preds[..., k_m][neg_mask] == 0.0).float().mean()
                    bal_acc = 0.5 * (rec_pos + rec_neg)
                    info[f"aux_mod_{mod_v}_bal_acc"] = float(bal_acc.item())
                info[f"aux_mod_{mod_v}_acc"] = float(acc_m.item())

    # V20 scratch_mod_aux: BCE supervision on scratch-domain modular head.
    # Pushes write_head + scratch cells toward modular structure.
    scratch_aux_logits = getattr(agent, "_last_scratch_aux_mod_logits", None)
    if (scratch_aux_logits is not None and scratch_mod_aux_weight > 0.0
            and scratch_mod_aux_moduli):
        n_mod_s = len(scratch_mod_aux_moduli)
        B_s = len(metas)
        K_s = scratch_aux_logits.shape[1]
        labels_scr = torch.empty(B_s, K_s, n_mod_s, device=device)
        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            for i, n_b in enumerate(meta["beta_counts"]):
                for k_m, mod_v in enumerate(scratch_mod_aux_moduli):
                    labels_scr[b, i, k_m] = float((n_a % mod_v) == (n_b % mod_v))
        n_per_s = float(B_s * K_s)
        pos_count_s = labels_scr.sum(dim=(0, 1)).clamp(min=1.0)
        pos_weight_s = ((n_per_s - pos_count_s) / pos_count_s).clamp(min=0.5, max=20.0)
        scratch_aux_loss = F.binary_cross_entropy_with_logits(
            scratch_aux_logits, labels_scr, pos_weight=pos_weight_s,
        )
        loss = loss + scratch_mod_aux_weight * scratch_aux_loss
        info["scratch_mod_aux_loss"] = float(scratch_aux_loss.item())
        with torch.no_grad():
            preds_s = (scratch_aux_logits > 0).float()
            for k_m, mod_v in enumerate(scratch_mod_aux_moduli):
                pos_mask = labels_scr[..., k_m] > 0.5
                neg_mask = ~pos_mask
                if pos_mask.any() and neg_mask.any():
                    rec_pos = (preds_s[..., k_m][pos_mask] == 1.0).float().mean()
                    rec_neg = (preds_s[..., k_m][neg_mask] == 0.0).float().mean()
                    bal_acc = 0.5 * (rec_pos + rec_neg)
                    info[f"scratch_mod_{mod_v}_bal_acc"] = float(bal_acc.item())

    # V22 E-pair consistency: when N_α == N_β (δ=0), virtual β scratch should
    # match α scratch over W cells. Forces code-N 1-to-1 mapping. Both inputs
    # are post-quantize (gradient via STE to write_head + upstream).
    sc_alpha_q = getattr(agent, "_scratch_alpha_for_aux", None)
    sc_beta_K_q = getattr(agent, "_last_scratch_beta_K_q", None)
    if (e_pair_consistency_weight > 0.0
            and sc_alpha_q is not None and sc_beta_K_q is not None):
        # Build δ=0 mask per (b, k) from metas.
        B_e = len(metas)
        K_e = sc_beta_K_q.shape[1]
        e_mask = torch.zeros(B_e, K_e, device=sc_alpha_q.device)
        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            for i, n_b in enumerate(meta["beta_counts"]):
                if n_b == n_a:
                    e_mask[b, i] = 1.0
        n_e_pairs = e_mask.sum().clamp(min=1.0)
        diff_sq = (sc_alpha_q.unsqueeze(1) - sc_beta_K_q).pow(2).mean(dim=-1)  # (B, K)
        e_pair_loss = (diff_sq * e_mask).sum() / n_e_pairs
        loss = loss + e_pair_consistency_weight * e_pair_loss
        info["e_pair_consistency_loss"] = float(e_pair_loss.item())
        info["e_pair_n_in_batch"] = float(e_mask.sum().item())

    # V22 successor loss: S learns +1 mapping on raw scratch space, supervised
    # by within-episode |δ|=1 G/L pairs. δ = N_α − N_β.
    #   G pair (δ=+1, N_α=N_β+1):  S(sc_β_raw) ≈ sc_α_q (target detached)
    #   L pair (δ=-1, N_β=N_α+1):  S(sc_α_raw) ≈ sc_β_q (target detached)
    sc_alpha_raw = getattr(agent, "_scratch_alpha_raw", None)
    sc_beta_K_raw = getattr(agent, "_last_scratch_beta_K_raw", None)
    sc_beta_K_q = getattr(agent, "_last_scratch_beta_K_q", None)
    sc_alpha_q = getattr(agent, "_scratch_alpha_for_aux", None)
    successor = getattr(agent, "successor", None)
    if (successor_loss_weight > 0.0 and successor is not None
            and sc_alpha_raw is not None and sc_beta_K_raw is not None):
        # Build signed δ (N_α − N_β) (B, K)
        K_s = sc_beta_K_raw.shape[1]
        deltas = torch.tensor(
            [[m["N_alpha"] - m["beta_counts"][i] for i in range(K_s)] for m in metas],
            device=sc_alpha_raw.device, dtype=torch.long,
        )
        g_mask = deltas == 1   # S(sc_β) ≈ sc_α  (β + 1 = α)
        l_mask = deltas == -1  # S(sc_α) ≈ sc_β  (α + 1 = β)
        succ_terms = []
        if g_mask.any():
            b_idx, k_idx = torch.where(g_mask)
            sc_b_g_raw = sc_beta_K_raw[b_idx, k_idx]      # (Ng, W)
            # V25-V22: detach scratch input so gradient flows only through S
            if successor_detached:
                sc_b_g_raw = sc_b_g_raw.detach()
            target_g_q = sc_alpha_q[b_idx].detach()        # (Ng, W) detach: no grad to α
            pred_g_raw = successor(sc_b_g_raw)
            from backend.core.agent import _quantize_ste as _qste
            if agent.agent_cfg.quantize_levels is not None:
                pred_g_q = _qste(pred_g_raw, agent.agent_cfg.quantize_levels,
                                 agent.agent_cfg.quantize_range)
            else:
                pred_g_q = pred_g_raw
            succ_terms.append((pred_g_q - target_g_q).pow(2).mean())
        if l_mask.any():
            b_idx, k_idx = torch.where(l_mask)
            sc_a_l_raw = sc_alpha_raw[b_idx]               # (Nl, W)
            if successor_detached:
                sc_a_l_raw = sc_a_l_raw.detach()
            target_l_q = sc_beta_K_q[b_idx, k_idx].detach()  # (Nl, W)
            pred_l_raw = successor(sc_a_l_raw)
            from backend.core.agent import _quantize_ste as _qste
            if agent.agent_cfg.quantize_levels is not None:
                pred_l_q = _qste(pred_l_raw, agent.agent_cfg.quantize_levels,
                                 agent.agent_cfg.quantize_range)
            else:
                pred_l_q = pred_l_raw
            succ_terms.append((pred_l_q - target_l_q).pow(2).mean())
        if succ_terms:
            successor_loss = torch.stack(succ_terms).mean()
            loss = loss + successor_loss_weight * successor_loss
            info["successor_loss"] = float(successor_loss.item())
            info["successor_n_pairs"] = float((g_mask.sum() + l_mask.sum()).item())

    # V25-V22 h_pair_consistency: PFC-level E pair MSE. Push h_α_pfc ≈ h_β_pfc
    # when N_α == N_β (δ=0). Operates on PFC representation, not scratch —
    # avoids the V22 scratch-level supervisory pathology.
    h_a_blocks = getattr(agent, "_h_alpha_pfc_blocks", None)
    h_b_blocks = getattr(agent, "_h_beta_pfc_blocks", None)
    if h_pair_consistency_weight > 0.0 and h_a_blocks and h_b_blocks:
        # h_α is same per episode, but stashed K times. Stack β across K.
        h_b_K = torch.stack(h_b_blocks, dim=1)  # (B, K, d_pfc)
        h_a = h_a_blocks[0]  # (B, d_pfc) — same per block
        K_h = h_b_K.shape[1]
        e_mask = torch.zeros(len(metas), K_h, device=h_a.device)
        for b, meta in enumerate(metas):
            n_a = meta["N_alpha"]
            for i, n_b in enumerate(meta["beta_counts"]):
                if n_b == n_a:
                    e_mask[b, i] = 1.0
        n_e_pairs = e_mask.sum().clamp(min=1.0)
        h_a_exp = h_a.unsqueeze(1).expand(-1, K_h, -1)
        diff_sq = (h_a_exp - h_b_K).pow(2).mean(dim=-1)  # (B, K)
        h_pair_loss = (diff_sq * e_mask).sum() / n_e_pairs
        loss = loss + h_pair_consistency_weight * h_pair_loss
        info["h_pair_consistency_loss"] = float(h_pair_loss.item())
        info["h_pair_n_in_batch"] = float(e_mask.sum().item())

    # V23b N-class CE: supervise scratch_α → N_α (max_n classes). Strong
    # supervisory pressure for 1-to-1 readability. Target: N_α-1 (0-indexed).
    n_class_head = getattr(agent, "n_class_head", None)
    sc_alpha_for_n = getattr(agent, "_scratch_alpha_for_aux", None)
    if (n_class_ce_weight > 0.0 and n_class_head is not None
            and sc_alpha_for_n is not None):
        n_logits = n_class_head(sc_alpha_for_n)  # (B, max_n)
        targets = torch.tensor(
            [m["N_alpha"] - 1 for m in metas],
            device=sc_alpha_for_n.device, dtype=torch.long,
        )
        # Clip targets to valid range (just in case stage n_max exceeds head's max_n)
        targets = targets.clamp(min=0, max=n_logits.shape[-1] - 1)
        n_class_loss = F.cross_entropy(n_logits, targets)
        loss = loss + n_class_ce_weight * n_class_loss
        info["n_class_ce_loss"] = float(n_class_loss.item())
        with torch.no_grad():
            n_pred = n_logits.argmax(dim=-1)
            n_top1_acc = (n_pred == targets).float().mean()
            info["n_class_top1_acc"] = float(n_top1_acc.item())

    # V19 detection aux: BCE supervision for spike_detector. Target = signal > 0.5.
    # Without this, the spike_detector learns to suppress LRU updates everywhere
    # (gate≈0 minimizes LRU noise) — exactly the wrong direction.
    if detection_aux_weight > 0.0 and hasattr(agent, "_spike_score_history"):
        history = getattr(agent, "_spike_score_history", []) or []
        if history:
            det_losses = []
            for score, sig in history:
                target = (sig > 0.5).float()  # (B, L)
                # pos_weight: rebalance because background dominates (~80%)
                n_pos = target.sum().clamp(min=1.0)
                n_neg = (target.numel() - n_pos).clamp(min=1.0)
                pw = (n_neg / n_pos).clamp(min=0.5, max=20.0)
                det_losses.append(F.binary_cross_entropy_with_logits(
                    score, target, pos_weight=pw,
                ))
            det_loss = torch.stack(det_losses).mean()
            loss = loss + detection_aux_weight * det_loss
            info["detection_aux_loss"] = float(det_loss.item())
            with torch.no_grad():
                # diagnostic: sigmoid(score * sharpness) at signal-spike vs background
                if hasattr(agent, "gate_sharpness") and agent.gate_sharpness is not None:
                    sharp = agent.gate_sharpness.detach()
                    s0, sig0 = history[0]
                    g0 = torch.sigmoid(s0 * sharp)
                    sp = (sig0 > 0.5)
                    bg = (sig0 <= 0.1)
                    if sp.any():
                        info["det_gate_at_spike"] = float(g0[sp].mean().item())
                    if bg.any():
                        info["det_gate_at_bg"] = float(g0[bg].mean().item())

    return loss, logits, labels, info


def validate_stage(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    train_cfg: TrainConfig,
    stage: CurriculumStage,
    device: torch.device,
    validation_rng: np.random.Generator,
    loss_type: str = "ce",
    rl_baseline: float = 0.0,
    rl_entropy_coef: float = 0.05,
    rl_reward_shape: str = "sharp",
    rl_adv_normalize: bool = False,
    rl_perblock_reward: str = "simple",
) -> Dict[str, float]:
    correct, total, loss_sum = 0, 0, 0.0
    # rl_top1 stats (deterministic argmax pick): mean reward, gap=0 hit rate.
    rl_reward_sum = 0.0
    rl_gap0_sum = 0
    rl_episodes = 0
    with torch.no_grad():
        for _ in range(train_cfg.validation_batches):
            batch = sample_training_batch(
                batch_size=train_cfg.batch_size,
                n_max_stage=stage.n_max,
                config=scene_cfg,
                rng=validation_rng,
            )
            loss, logits, labels, _info = run_batch(
                agent, batch, device, loss_type=loss_type,
                rl_baseline=rl_baseline, rl_entropy_coef=rl_entropy_coef,
                rl_reward_shape=rl_reward_shape, rl_adv_normalize=rl_adv_normalize,
                rl_perblock_reward=rl_perblock_reward,
            )
            if loss_type == "rl_top1":
                # Deterministic eval: argmax over K options; reward at the picked option.
                # Reward shape MUST match training so mean_reward is meaningful for
                # the success criteria thresholds (sharp ∈ [0.01, 100], soft ∈ [0.135, 1]).
                metas = batch[3]
                gap_target = _build_gap_target(metas, device)
                scores = logits[..., 0]
                action = scores.argmax(dim=1)
                gap_chosen = gap_target.gather(1, action.unsqueeze(1)).squeeze(1).abs()
                if rl_reward_shape == "sharp":
                    reward = 100.0 * (10.0 ** (-gap_chosen))
                else:  # soft
                    reward = torch.exp(-gap_chosen / 2.0)
                rl_reward_sum += float(reward.sum().item())
                rl_gap0_sum += int((gap_chosen == 0).sum().item())
                rl_episodes += gap_chosen.numel()
            elif loss_type == "rl_perblock":
                # Deterministic eval: per-block threshold (score > 0 → predict equal).
                # Reward shape MUST match training (simple ±1 vs asymmetric P2-strong)
                # so reported mean_reward is comparable to training-time signal.
                metas = batch[3]
                gap_target = _build_gap_target(metas, device)
                scores = logits[..., 0]  # (B, K)
                action = (scores > 0.0).float()
                is_equal = (gap_target == 0).float()
                correct = (action == is_equal).float()
                if rl_perblock_reward == "asymmetric":
                    abs_gap = gap_target.abs().long().clamp(0, 4)
                    fp_table = torch.tensor(
                        _ASYM_REWARD_FP_BY_GAP, device=scores.device, dtype=scores.dtype
                    )
                    fp_penalty = fp_table[abs_gap]
                    reward = (
                        is_equal * action * _ASYM_REWARD_TP
                        + is_equal * (1.0 - action) * _ASYM_REWARD_FN
                        + (1.0 - is_equal) * action * fp_penalty
                        + (1.0 - is_equal) * (1.0 - action) * _ASYM_REWARD_TN
                    )
                else:
                    reward = 2.0 * correct - 1.0
                rl_reward_sum += float(reward.sum().item())
                rl_gap0_sum += int(correct.sum().item())  # repurpose: correct count
                rl_episodes += correct.numel()  # total per-block decisions
            else:
                preds = _logits_to_preds(logits, loss_type)
                correct += (preds == labels).sum().item()
                total += labels.numel()
            loss_sum += loss.item()
    metrics: Dict[str, float] = {"validation_loss": loss_sum / train_cfg.validation_batches}
    if loss_type == "rl_top1":
        # validation_accuracy = fraction of episodes where argmax pick had gap=0.
        # Acts as the curriculum-advancement metric (target_accuracy threshold).
        metrics["validation_accuracy"] = rl_gap0_sum / max(rl_episodes, 1)
        metrics["mean_reward"] = rl_reward_sum / max(rl_episodes, 1)
    elif loss_type == "rl_perblock":
        # validation_accuracy = per-block correct rate (binary equal/not-equal decisions).
        # mean_reward = average of {-1, +1} = 2 * accuracy - 1.
        metrics["validation_accuracy"] = rl_gap0_sum / max(rl_episodes, 1)
        metrics["mean_reward"] = rl_reward_sum / max(rl_episodes, 1)
    else:
        metrics["validation_accuracy"] = correct / max(total, 1)
    return metrics


def run_delta_h_diagnostic(
    agent, scene_cfg: SceneConfig, stage, device, rng,
    batch_size: int = 32,
) -> Dict[str, float]:
    """51_M5: |h[t]-h[t-1]| diagnostic to test event-driven structure.

    Forward 1 batch, capture alpha-scan Mamba latents h_A (B, L+W+T_gap, d).
    Compute delta_h_t = norm(h_A[:, t, :] - h_A[:, t-1, :]) per t.
    Compare delta_h at peak positions (signal > 0.5) vs gap positions.
    Returns peak_corr, peak_mean_dh, gap_mean_dh, ratio.
    """
    agent.train(False)
    L = scene_cfg.L
    batch = sample_training_batch(
        batch_size=batch_size, n_max_stage=stage.n_max,
        config=scene_cfg, rng=rng,
    )
    inputs, _, ci, _ = batch
    inputs = inputs.to(device)
    ci = ci.to(device)
    seg_a_end = L + scene_cfg.W + scene_cfg.T_gap
    seg_A = inputs[:, 0:seg_a_end, :]
    with torch.no_grad():
        if hasattr(agent, "_encode_segment"):
            h_A = agent._encode_segment(seg_A)  # (B, seg_a_end, d)
        else:
            agent.train(True)
            return {}
    h_alpha = h_A[:, :L, :]  # (B, L, d)
    dh = (h_alpha[:, 1:, :] - h_alpha[:, :-1, :]).norm(dim=-1)  # (B, L-1)
    sig = inputs[:, 1:L, 0]  # (B, L-1) align with dh
    peak_mask = (sig > 0.5).float()
    gap_mask = 1.0 - peak_mask
    peak_count = peak_mask.sum().item()
    gap_count = gap_mask.sum().item()
    peak_mean = (dh * peak_mask).sum().item() / max(peak_count, 1)
    gap_mean = (dh * gap_mask).sum().item() / max(gap_count, 1)
    dh_flat = dh.reshape(-1).cpu().numpy()
    pk_flat = peak_mask.reshape(-1).cpu().numpy()
    if dh_flat.std() > 1e-6 and pk_flat.std() > 1e-6:
        corr = float(np.corrcoef(dh_flat, pk_flat)[0, 1])
    else:
        corr = float("nan")
    agent.train(True)
    return {
        "peak_corr": corr,
        "peak_mean_dh": peak_mean,
        "gap_mean_dh": gap_mean,
        "peak_to_gap_ratio": peak_mean / max(gap_mean, 1e-9),
    }


_WEBER_COMPATIBLE_TASKS = frozenset({
    "compare", "discrim", "trio", "trio_wide", "trio_wide_extended",
})


def run_stage_metrics(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    stage: CurriculumStage,
    device: torch.device,
    num_episodes: int,
    near_extrap_episodes: int,
    seed: int,
    loss_type: str = "ce",
    task: Optional[str] = None,
) -> Dict[str, object]:
    """
    Run the full v3 metric suite at a stage boundary.

    Collects one rollout and computes:
      - rollout_accuracy (comparison accuracy)
      - topographic similarity (notes-distance vs N-distance)
      - compositional probe (position-by-position marginal gain)
      - base-scanning MI matrix diagnostic
      - near-extrapolation accuracy
    """
    rollout = collect_notes_and_predictions(
        agent, scene_cfg, n_max=stage.n_max,
        num_episodes=num_episodes, device=device, seed=seed,
        loss_type=loss_type,
        task=task,
    )

    topo = topographic_similarity(rollout.n_alpha, rollout.notes)
    comp = compositional_probe_score(
        rollout.n_alpha, rollout.notes, cv_folds=3
    )
    # NOTE (2026-05-08): base_scanning_mi (best_base / best_alignment /
    # all_alignments) removed from training-time metrics. Codebooks are
    # currently all thermometer; base-alignment scores are noise. The
    # function still lives in mi_matrix.py for post-hoc inspect scripts.

    # Near-extrapolation test (N just above stage max, bounded by scene.L).
    # Scene generator constraint: L >= 2N+1, so N_max = (L-1)//2 (NOT L-1).
    extrap: Dict[str, float] = {}
    n_cap = (scene_cfg.L - 1) // 2
    upper = min(stage.n_max * 2, n_cap)
    if upper > stage.n_max:
        extrap = extrapolation_accuracy(
            agent=agent,
            scene_cfg=scene_cfg,
            n_train_max=stage.n_max,
            n_test_ranges={"near_extrap": range(stage.n_max + 1, upper + 1)},
            device=device,
            num_episodes_per_range=near_extrap_episodes,
            seed=seed + 1,
            loss_type=loss_type,
            task=task,
        )

    # Weber curve: N_alpha vs N_alpha+-1 discrimination across N range.
    # Skip boundaries (N=1 and N=n_max) where +-1 is invalid.
    # NOTE: weber semantics assume 3-class (>,<,==) labels via compute_weber_curve
    # in backend/evaluation/weber.py. Other tasks (successor_prediction, binary)
    # have different label semantics and the metric is nonsense for them — gate
    # here so the jsonl doesn't carry misleading numbers.
    weber_low = max(2, 2)
    weber_high = max(weber_low + 1, stage.n_max - 1)
    weber_n_values = []
    weber_accuracy = []
    weber_flatness = float("nan")
    weber_slope = float("nan")
    weber_per_class: Dict[str, float] = {}
    weber_enabled = (task is None) or (task in _WEBER_COMPATIBLE_TASKS)
    if weber_enabled and weber_high > weber_low:
        curve = compute_weber_curve(
            agent=agent,
            scene_cfg=scene_cfg,
            n_range=range(weber_low, weber_high + 1),
            num_trials_per_n=32,
            device=device,
            seed=seed + 2,
            loss_type=loss_type,
        )
        weber_n_values = curve.n_values.tolist()
        weber_accuracy = curve.accuracy.tolist()
        weber_flatness = curve.flatness_score()
        weber_slope = curve.weber_slope()
        weber_per_class = {k: float(v[0]) for k, v in curve.per_class_accuracy.items()}

    return {
        "rollout_accuracy": rollout.accuracy,
        "topo_sim": topo["topo_sim"],
        "topo_p_value": topo["p_value"],
        "composition_index": comp["composition_index"],
        "total_r2": comp["total_r2"],
        "per_position_r2": comp["per_position_r2"],
        "marginal_gains": comp["marginal_gains"],
        "extrapolation": extrap,
        "weber_n_values": weber_n_values,
        "weber_accuracy": weber_accuracy,
        "weber_flatness": weber_flatness,
        "weber_slope": weber_slope,
        "weber_per_class": weber_per_class,
    }


def run_final_metrics(
    agent: GRUAgent,
    scene_cfg: SceneConfig,
    train_cfg: TrainConfig,
    final_stage: CurriculumStage,
    device: torch.device,
    seed: int,
    loss_type: str = "ce",
    task: Optional[str] = None,
) -> Dict[str, object]:
    """
    End-of-training evaluation with larger sample + full extrapolation ranges.
    """
    metrics = run_stage_metrics(
        agent, scene_cfg, final_stage, device,
        num_episodes=train_cfg.final_episodes,
        near_extrap_episodes=train_cfg.extrapolation_episodes,
        seed=seed,
        loss_type=loss_type,
        task=task,
    )

    # Fuller extrapolation: near + far (bounded by (L-1)//2, scene generator
    # constraint L >= 2N+1).
    n_cap_final = (scene_cfg.L - 1) // 2
    far_lower = min(int(final_stage.n_max * 1.5) + 1, n_cap_final)
    far_upper = n_cap_final
    test_ranges: Dict[str, range] = {}
    if final_stage.n_max + 1 <= far_lower:
        test_ranges["near_extrap"] = range(final_stage.n_max + 1, far_lower + 1)
    if far_lower < far_upper:
        test_ranges["far_extrap"] = range(far_lower + 1, far_upper + 1)

    if test_ranges:
        metrics["extrapolation_full"] = extrapolation_accuracy(
            agent=agent,
            scene_cfg=scene_cfg,
            n_train_max=final_stage.n_max,
            n_test_ranges=test_ranges,
            device=device,
            num_episodes_per_range=train_cfg.extrapolation_episodes,
            seed=seed + 2,
            loss_type=loss_type,
            task=task,
        )

    return metrics


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def log(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")


def train(
    scene_cfg: SceneConfig,
    agent_cfg,  # AgentConfig or TransformerAgentConfig
    train_cfg: TrainConfig,
    curriculum: List[CurriculumStage],
    device: torch.device,
    log_dir: Path,
    checkpoint_dir: Path,
    loss_type: str = "ce",
    rl_entropy_coef: float = 0.05,
    rl_baseline_ema: float = 0.99,
    init_from_ckpt: Optional[str] = None,
    init_from_ckpt_nonstrict: bool = False,
    rl_reward_shape: str = "sharp",
    rl_adv_normalize: bool = False,
    rl_perblock_reward: str = "simple",
    per_stage_K: Optional[List[int]] = None,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    write_head_wd: float = 0.0,
    sgd_momentum: float = 0.9,
    world_pred_lambda: float = 0.0,
    beta_pred_lambda: float = 0.0,
    compare_pred_lambda: float = 0.0,
    rs_pred_lambda: float = 0.0,
    scratch_self_pred_lambda: float = 0.0,
    cf_beta_pred_lambda: float = 0.0,
    vq_commit_lambda: float = 0.0,
    dual_world_lambda: float = 0.0,
    dual_sp_lambda: float = 0.0,
    vicreg_lambda: float = 0.0,
    pc_explicit_lambda: float = 0.0,
    cycle_loss_lambda: float = 0.0,
    ar_entropy_bonus: float = 0.0,
    multi_modular_aux_lambda: float = 0.0,
    multi_modular_aux_moduli: Tuple[int, ...] = (),
    multi_modular_aux_pool: Tuple[int, ...] = (),
    detection_aux_weight: float = 0.0,
    scratch_mod_aux_weight: float = 0.0,
    scratch_mod_aux_moduli: Tuple[int, ...] = (),
    e_pair_consistency_weight: float = 0.0,
    successor_loss_weight: float = 0.0,
    successor_detached: bool = False,
    h_pair_consistency_weight: float = 0.0,
    n_class_ce_weight: float = 0.0,
    delta_h_diagnostic_every: int = 0,
    ce_class_weights: Optional[torch.Tensor] = None,
    write_noise_anneal: Optional[Tuple[float, float, float, float]] = None,
    task: Optional[str] = None,
) -> None:
    # Fail fast on curriculum/scene mismatch. Scene generator (complex_world
    # with min spacing 2) requires L >= 2N+1, so N_max = (L-1)//2 NOT L-1.
    for stage in curriculum:
        L_stage = stage.L if stage.L is not None else scene_cfg.L
        n_cap_stage = (L_stage - 1) // 2
        if stage.n_max > n_cap_stage:
            raise ValueError(
                f"Curriculum stage '{stage.name}' has n_max={stage.n_max} but "
                f"L={L_stage} only fits N <= {n_cap_stage} (need "
                f"L >= 2N+1). Either cap n_max or increase L."
            )

    torch.manual_seed(train_cfg.agent_seed)
    np.random.seed(train_cfg.agent_seed)

    if isinstance(agent_cfg, TransformerAgentConfig):
        agent = TransformerAgent(agent_cfg, scene_cfg).to(device)
        arch_name = "transformer_gated" if agent_cfg.use_event_gate else "transformer"
    elif isinstance(agent_cfg, AttentionAgentConfig):
        agent = AttentionAgent(agent_cfg, scene_cfg).to(device)
        arch_name = "attention"
    elif isinstance(agent_cfg, RGLRUAgentConfig):
        agent = RGLRUAgent(agent_cfg, scene_cfg).to(device)
        arch_name = "rglru"
    elif _HAS_MAMBA_AGENT and isinstance(agent_cfg, MambaAgentConfig):
        agent = MambaAgent(agent_cfg, scene_cfg).to(device)
        arch_name = "mamba"
    else:
        agent = GRUAgent(agent_cfg, scene_cfg).to(device)
        arch_name = "gru"

    if init_from_ckpt is not None:
        state = torch.load(init_from_ckpt, map_location=device)
        sd = state.get("agent_state_dict", state) if isinstance(state, dict) else state
        _strict = not init_from_ckpt_nonstrict
        _load_result = agent.load_state_dict(sd, strict=_strict)
        if not _strict:
            # Partial warm-start (e.g. swapping one module like the write head): report
            # skipped keys so a silent architecture mismatch can't masquerade as a clean load.
            _missing = list(getattr(_load_result, "missing_keys", []))
            _unexpected = list(getattr(_load_result, "unexpected_keys", []))
            print(
                f"[init] Loaded agent weights from {init_from_ckpt} (strict=False): "
                f"{len(_missing)} missing (left at init), {len(_unexpected)} unexpected (skipped).",
                file=sys.stderr,
            )
            if _missing:
                print(f"[init]   missing (fresh init): {_missing}", file=sys.stderr)
            if _unexpected:
                print(f"[init]   unexpected (ckpt-only, skipped): {_unexpected}", file=sys.stderr)
        else:
            print(f"[init] Loaded agent weights from {init_from_ckpt}", file=sys.stderr)

    # V12-fix2 (2026-05-06): per-group weight_decay so write_head can have higher
    # wd than rest of net. Targets V10/V11/V12 saturation pathology (write_head
    # learns large weights → raw output ±9-15 → sigmoid saturated → cells frozen).
    # If write_head_wd > 0, write_head params get this wd; else they get default
    # wd (same as rest).
    # V35-PC: exclude substrate params from SGD optimizer (they get PC updates instead).
    v35_pc_mode = getattr(agent.agent_cfg, "v35_local_pc", False) and getattr(agent, "v35_substrate", None) is not None
    if v35_pc_mode:
        substrate_param_ids = {id(p) for p in agent.v35_substrate.parameters()}
        if getattr(agent.agent_cfg, "v35_theta_in_sgd", False):
            _tr = getattr(agent.v35_substrate.cascade, "theta_raw", None)
            if _tr is not None:
                substrate_param_ids.discard(id(_tr))  # Rung B1': keep the LEARNABLE base in SGD
                print("[v35-pc] theta_raw KEPT in SGD optimizer (--v35-theta-in-sgd): base is "
                      "genuinely gradient-learnable.", file=sys.stderr)
        excluded_params = [p for p in agent.parameters() if id(p) not in substrate_param_ids]
        print(
            f"[v35-pc] Excluding {sum(1 for p in agent.v35_substrate.parameters())} substrate params "
            f"from SGD optimizer; local PC update will be applied instead.",
            file=sys.stderr,
        )
    else:
        excluded_params = None  # use defaults below

    if write_head_wd > 0.0:
        wh_params = list(agent.write_head.parameters())
        wh_ids = {id(p) for p in wh_params}
        # When v35_pc_mode, also exclude substrate from "other_params"
        if v35_pc_mode:
            substrate_param_ids = {id(p) for p in agent.v35_substrate.parameters()}
            if getattr(agent.agent_cfg, "v35_theta_in_sgd", False):
                _tr = getattr(agent.v35_substrate.cascade, "theta_raw", None)
                if _tr is not None:
                    substrate_param_ids.discard(id(_tr))  # Rung B1': learnable base stays in SGD
            other_params = [
                p for p in agent.parameters()
                if id(p) not in wh_ids and id(p) not in substrate_param_ids
            ]
        else:
            other_params = [p for p in agent.parameters() if id(p) not in wh_ids]
        param_groups = [
            {"params": other_params, "weight_decay": weight_decay},
            {"params": wh_params, "weight_decay": write_head_wd},
        ]
        print(
            f"[wd] write_head: {len(wh_params)} param tensors with wd={write_head_wd}; "
            f"rest: {len(other_params)} with wd={weight_decay}",
            file=sys.stderr,
        )
    elif v35_pc_mode:
        param_groups = excluded_params
    else:
        param_groups = agent.parameters()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups, lr=train_cfg.learning_rate,
            momentum=sgd_momentum, weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        # 51_M5: AdamW decouples weight decay from grad update — required for
        # large wd (e.g., 0.1) to be stable. Adam couples wd with momentum.
        optimizer = torch.optim.AdamW(
            param_groups, lr=train_cfg.learning_rate, weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            param_groups, lr=train_cfg.learning_rate, weight_decay=weight_decay,
        )

    train_rng = np.random.default_rng(train_cfg.data_seed)
    validation_rng = np.random.default_rng(train_cfg.data_seed + 10000)

    # Capture all runtime args (NOT in TrainConfig dataclass) so future
    # post-mortem can reconstruct exact training config without bash history.
    runtime_cfg = {
        "loss_type": loss_type,
        "optimizer_name": optimizer_name,
        "weight_decay": weight_decay,
        "sgd_momentum": sgd_momentum,
        "init_from_ckpt": init_from_ckpt,
        "init_from_ckpt_nonstrict": init_from_ckpt_nonstrict,
        # RL specific
        "rl_entropy_coef": rl_entropy_coef,
        "rl_baseline_ema": rl_baseline_ema,
        "rl_reward_shape": rl_reward_shape,
        "rl_adv_normalize": rl_adv_normalize,
        "rl_perblock_reward": rl_perblock_reward,
        "per_stage_K": per_stage_K,
        # Aux loss lambdas
        "world_pred_lambda": world_pred_lambda,
        "beta_pred_lambda": beta_pred_lambda,
        "compare_pred_lambda": compare_pred_lambda,
        "rs_pred_lambda": rs_pred_lambda,
        "scratch_self_pred_lambda": scratch_self_pred_lambda,
        "cf_beta_pred_lambda": cf_beta_pred_lambda,
        "vq_commit_lambda": vq_commit_lambda,
        "dual_world_lambda": dual_world_lambda,
        "dual_sp_lambda": dual_sp_lambda,
        "vicreg_lambda": vicreg_lambda,
        "pc_explicit_lambda": pc_explicit_lambda,
        "cycle_loss_lambda": cycle_loss_lambda,
        "ar_entropy_bonus": ar_entropy_bonus,
    }
    logger = JSONLLogger(log_dir / "run.jsonl")
    logger.log({
        "event": "run_start",
        "arch": arch_name,
        "scene_cfg": asdict(scene_cfg),
        "agent_cfg": asdict(agent_cfg),
        "train_cfg": asdict(train_cfg),
        "runtime_cfg": runtime_cfg,
        "curriculum": [asdict(s) for s in curriculum],
        "num_parameters": agent.num_parameters(),
        "device": str(device),
    })

    global_step = 0
    # 51_M6: write-noise anneal — total epochs across all stages
    total_epochs_planned = sum(s.max_epochs for s in curriculum)
    epochs_done_so_far = 0
    initial_write_noise = (
        getattr(agent_cfg, "write_noise_std", 0.0)
        if hasattr(agent_cfg, "write_noise_std") else 0.0
    )
    # RL EMA baseline: initialized lazily on first batch's mean reward, then
    # exponentially smoothed. Removes mean from the advantage to reduce variance
    # without biasing the gradient.
    rl_baseline: Optional[float] = None
    # Fix-3 three-factor advantage baseline: EMA of batch-mean p_target (compare head).
    # None until seeded by the first batch. Spec: 2026-05-29-v35-three-factor-modulation.md
    tf_baseline: Optional[float] = None
    try:
        for stage_idx, stage in enumerate(curriculum):
            # A3: per-stage K curriculum (e.g. K=2 → 6 → 12 alongside n_max=9 → 27 → 81).
            # Updates scene_cfg.K so sample_training_batch + validate_stage both see new K.
            if per_stage_K is not None:
                scene_cfg.K = per_stage_K[stage_idx]
            # V7 (2026-05-04): per-stage L override. Mutates scene_cfg.L so
            # signal length scales with n_max — keeps spike density roughly
            # constant across stages (CNN k=51 sees similar #spikes per window).
            if stage.L is not None and stage.L != scene_cfg.L:
                old_L = scene_cfg.L
                scene_cfg.L = stage.L
                print(f"  [stage L override] scene_cfg.L: {old_L} → {stage.L}")
            logger.log({"event": "stage_start", "stage_idx": stage_idx,
                        "stage": asdict(stage), "scene_K": scene_cfg.K,
                        "scene_L": scene_cfg.L})
            print(f"\n=== Stage {stage_idx}: {stage.name} (N_max={stage.n_max}, L={scene_cfg.L}, K={scene_cfg.K}) ===")

            recent_validation_accs: List[float] = []
            stage_start = time.time()

            for epoch in range(stage.max_epochs):
                # 51_M6: write_noise anneal (start, end, start_pct, end_pct of TOTAL epochs)
                if write_noise_anneal is not None and hasattr(agent.agent_cfg, "write_noise_std"):
                    s, e, sp, ep = write_noise_anneal
                    progress = (epochs_done_so_far + epoch) / max(total_epochs_planned, 1)
                    if progress < sp:
                        cur_noise = s
                    elif progress > ep:
                        cur_noise = e
                    else:
                        frac = (progress - sp) / max(ep - sp, 1e-9)
                        cur_noise = s + (e - s) * frac
                    agent.agent_cfg.write_noise_std = cur_noise
                epoch_loss = 0.0
                for _ in range(train_cfg.steps_per_epoch):
                    lr = linear_warmup(global_step, train_cfg.warmup_steps, train_cfg.learning_rate)
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                    batch = sample_training_batch(
                        batch_size=train_cfg.batch_size,
                        n_max_stage=stage.n_max,
                        config=scene_cfg,
                        rng=train_rng,
                    )
                    # V35-PC: clear capture history at start of each step
                    if v35_pc_mode:
                        agent.v35_substrate.clear_pc_history()
                    baseline_for_step = rl_baseline if rl_baseline is not None else 0.0
                    loss, _step_logits, _step_targets, info = run_batch(
                        agent, batch, device, loss_type=loss_type,
                        rl_baseline=baseline_for_step,
                        rl_entropy_coef=rl_entropy_coef,
                        rl_reward_shape=rl_reward_shape,
                        rl_adv_normalize=rl_adv_normalize,
                        rl_perblock_reward=rl_perblock_reward,
                        world_pred_lambda=world_pred_lambda,
                        beta_pred_lambda=beta_pred_lambda,
                        compare_pred_lambda=compare_pred_lambda,
                        rs_pred_lambda=rs_pred_lambda,
                        scratch_self_pred_lambda=scratch_self_pred_lambda,
                        cf_beta_pred_lambda=cf_beta_pred_lambda,
                        vq_commit_lambda=vq_commit_lambda,
                        dual_world_lambda=dual_world_lambda,
                        dual_sp_lambda=dual_sp_lambda,
                        vicreg_lambda=vicreg_lambda,
                        pc_explicit_lambda=pc_explicit_lambda,
                        cycle_loss_lambda=cycle_loss_lambda,
                        ar_entropy_bonus=ar_entropy_bonus,
                        multi_modular_aux_lambda=multi_modular_aux_lambda,
                        multi_modular_aux_moduli=multi_modular_aux_moduli,
                        multi_modular_aux_pool=multi_modular_aux_pool,
                        detection_aux_weight=detection_aux_weight,
                        scratch_mod_aux_weight=scratch_mod_aux_weight,
                        scratch_mod_aux_moduli=scratch_mod_aux_moduli,
                        e_pair_consistency_weight=e_pair_consistency_weight,
                        successor_loss_weight=successor_loss_weight,
                        successor_detached=successor_detached,
                        h_pair_consistency_weight=h_pair_consistency_weight,
                        n_class_ce_weight=n_class_ce_weight,
                        ce_class_weights=ce_class_weights,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), train_cfg.grad_clip)
                    optimizer.step()
                    # V35-PC: local update on substrate B_leak using captured fast/slow gate error.
                    # Then clear unused SGD grads from substrate (they accumulated but were discarded).
                    if v35_pc_mode:
                        _ip_target = agent.agent_cfg.v35_ip_target
                        if _ip_target is None:
                            _ip_target = (stage.n_max + 1) / 2.0  # auto: stage average N
                        # Fix-3 three-factor modulator: per-sample signed advantage
                        # M = p_target - EMA-baseline from the compare head (spec 2026-05-29).
                        _modulator = None
                        if (getattr(agent.agent_cfg, "v35_three_factor", False)
                                and _step_logits is not None and _step_targets is not None):
                            with torch.no_grad():
                                p_t = F.softmax(_step_logits, dim=-1).gather(
                                    1, _step_targets.view(-1, 1)).squeeze(1)  # (B,) prob of correct class
                                _bm = float(p_t.mean())
                                if tf_baseline is None:
                                    tf_baseline = _bm
                                else:
                                    _ema = agent.agent_cfg.v35_tf_baseline_ema
                                    tf_baseline = _ema * tf_baseline + (1.0 - _ema) * _bm
                                _modulator = p_t - tf_baseline
                        agent.v35_substrate.pc_step(
                            lr=agent.agent_cfg.v35_pc_lr,
                            use_h_upstream=agent.agent_cfg.v35_pc_h_upstream,
                            ip_enable=agent.agent_cfg.v35_ip_enable,
                            ip_lr=agent.agent_cfg.v35_ip_lr,
                            r_target=(_ip_target if agent.agent_cfg.v35_ip_enable else None),
                            ip_bias_clamp=agent.agent_cfg.v35_ip_bias_clamp,
                            weight_decay=agent.agent_cfg.v35_pc_weight_decay,
                            ip_leak=agent.agent_cfg.v35_ip_leak,
                            modulator=_modulator,
                            weight_clip=agent.agent_cfg.v35_pc_weight_clip,
                            freeze_layer_b=agent.agent_cfg.v35_freeze_layer_b_bleak,
                            freeze_layer_a=agent.agent_cfg.v35_freeze_layer_a_bleak,
                        )
                        agent.v35_substrate.zero_grad()

                    if loss_type in ("rl_top1", "rl_perblock"):
                        # EMA update on observed mean reward. First batch seeds the baseline;
                        # subsequent batches smooth at rate 1 - rl_baseline_ema.
                        r = info["reward_mean"]
                        if rl_baseline is None:
                            rl_baseline = r
                        else:
                            rl_baseline = rl_baseline_ema * rl_baseline + (1.0 - rl_baseline_ema) * r

                    epoch_loss += loss.item()
                    global_step += 1

                epoch_loss /= train_cfg.steps_per_epoch

                if (epoch + 1) % train_cfg.validation_every_epochs == 0 or epoch == stage.max_epochs - 1:
                    metrics = validate_stage(
                        agent, scene_cfg, train_cfg, stage, device, validation_rng,
                        loss_type=loss_type,
                        rl_baseline=rl_baseline if rl_baseline is not None else 0.0,
                        rl_entropy_coef=rl_entropy_coef,
                        rl_reward_shape=rl_reward_shape,
                        rl_adv_normalize=rl_adv_normalize,
                        rl_perblock_reward=rl_perblock_reward,
                    )
                    record = {
                        "event": "validation",
                        "stage": stage.name,
                        "stage_idx": stage_idx,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": epoch_loss,
                        **metrics,
                    }
                    if loss_type in ("rl_top1", "rl_perblock") and rl_baseline is not None:
                        record["rl_baseline"] = rl_baseline
                    logger.log(record)
                    # V29 emergence snapshot (codebook + learned params)
                    _emit_emergence_snapshot(
                        agent, stage_idx=stage_idx, stage_name=stage.name,
                        epoch=epoch, global_step=global_step, log_dir=log_dir,
                    )
                    extra = ""
                    if "mean_reward" in metrics:
                        extra = f" reward={metrics['mean_reward']:.2f}"
                    print(
                        f"[{stage.name} ep{epoch + 1}/{stage.max_epochs}] "
                        f"loss={epoch_loss:.4f} val_acc={metrics['validation_accuracy']:.3f}{extra}"
                    )
                    # Periodic intermediate checkpoint (for inspecting the codebook trend
                    # during long single-stage runs). Same inspect-compatible payload as the
                    # stage-end save; skips the final epoch (saved unconditionally below).
                    if (epoch + 1) < stage.max_epochs:
                        _acd = {k: v for k, v in vars(agent.agent_cfg).items() if not k.startswith("_")}
                        _scd = {k: v for k, v in vars(agent.scene_cfg).items() if not k.startswith("_")}
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        _ipath = checkpoint_dir / f"{stage.name}_ep{epoch + 1}.pt"
                        torch.save(
                            {
                                "agent_state_dict": agent.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "stage_idx": stage_idx,
                                "global_step": global_step,
                                "agent_cfg": _acd,
                                "scene_cfg": _scd,
                            },
                            _ipath,
                        )
                        logger.log({"event": "checkpoint_saved", "path": str(_ipath), "intermediate": True})

                    # 51_M5: Δh diagnostic — every N epochs, compute |h[t]-h[t-1]|
                    # over α scan and correlate with peak positions in raw signal.
                    if (delta_h_diagnostic_every > 0
                            and (epoch + 1) % delta_h_diagnostic_every == 0
                            and _HAS_MAMBA_AGENT and isinstance(agent_cfg, MambaAgentConfig)):
                        try:
                            dh_stats = run_delta_h_diagnostic(
                                agent, scene_cfg, stage, device, validation_rng,
                            )
                            logger.log({
                                "event": "delta_h_diagnostic",
                                "stage_idx": stage_idx, "epoch": epoch,
                                **dh_stats,
                            })
                        except Exception as e:
                            print(f"  (delta_h diagnostic skipped: {e})", file=sys.stderr)

                    recent_validation_accs.append(metrics["validation_accuracy"])
                    if (
                        len(recent_validation_accs) >= train_cfg.accuracy_window
                        and all(
                            a >= stage.target_accuracy
                            for a in recent_validation_accs[-train_cfg.accuracy_window:]
                        )
                    ):
                        print(f"  → stage target reached after {epoch + 1} epochs, advancing")
                        break

            logger.log({
                "event": "stage_end",
                "stage_idx": stage_idx,
                "stage": stage.name,
                "epochs_used": epoch + 1,
                "elapsed_seconds": time.time() - stage_start,
            })
            # 51_M6: track total epochs for write_noise anneal
            epochs_done_so_far += epoch + 1

            # Full metric suite at stage boundary
            metric_seed = train_cfg.data_seed + 100000 + stage_idx
            stage_metrics = run_stage_metrics(
                agent, scene_cfg, stage, device,
                num_episodes=train_cfg.stage_end_episodes,
                near_extrap_episodes=train_cfg.extrapolation_episodes,
                seed=metric_seed,
                loss_type=loss_type,
                task=task,
            )
            logger.log({
                "event": "stage_metrics",
                "stage_idx": stage_idx,
                "stage": stage.name,
                **stage_metrics,
            })
            print(
                f"  metrics: topo_sim={stage_metrics['topo_sim']:.3f} "
                f"comp_idx={stage_metrics['composition_index']:.3f} "
                f"total_r2={stage_metrics['total_r2']:.3f}"
            )
            weber_acc = stage_metrics.get("weber_accuracy", [])
            if weber_acc:
                print(
                    f"  weber: acc[{stage_metrics['weber_n_values'][0]}]={weber_acc[0]:.2f} "
                    f"acc[{stage_metrics['weber_n_values'][-1]}]={weber_acc[-1]:.2f} "
                    f"slope={stage_metrics['weber_slope']:+.3f} "
                    f"flat={stage_metrics['weber_flatness']:.3f}"
                )
            if stage_metrics["extrapolation"]:
                for rng_name, acc in stage_metrics["extrapolation"].items():
                    print(f"  extrap {rng_name}: {acc:.3f}")

            ckpt_path = checkpoint_dir / f"{stage.name}.pt"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # Save agent_cfg + scene_cfg as dicts so inspect_checkpoint can restore
            # the EXACT forward path. Without this, hyperparams like kwta_k,
            # use_oracle_spike_gate (no learnable params, only affect forward)
            # default to wrong values and forward output diverges from training.
            agent_cfg_dict = {k: v for k, v in vars(agent.agent_cfg).items()
                              if not k.startswith("_")}
            scene_cfg_dict = {k: v for k, v in vars(agent.scene_cfg).items()
                              if not k.startswith("_")}
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "stage_idx": stage_idx,
                    "global_step": global_step,
                    "agent_cfg": agent_cfg_dict,
                    "scene_cfg": scene_cfg_dict,
                },
                ckpt_path,
            )
            logger.log({"event": "checkpoint_saved", "path": str(ckpt_path)})

        # End-of-run comprehensive evaluation on the last stage's N range
        final_metrics = run_final_metrics(
            agent, scene_cfg, train_cfg, curriculum[-1], device,
            seed=train_cfg.data_seed + 999999,
            loss_type=loss_type,
            task=task,
        )
        logger.log({
            "event": "final_metrics",
            "final_stage": curriculum[-1].name,
            **final_metrics,
        })
        print(
            f"\n=== Final metrics ({curriculum[-1].name}) ===\n"
            f"  rollout_acc={final_metrics['rollout_accuracy']:.3f}\n"
            f"  topo_sim={final_metrics['topo_sim']:.3f} "
            f"comp_idx={final_metrics['composition_index']:.3f} "
            f"total_r2={final_metrics['total_r2']:.3f}"
        )
        for rng_name, acc in final_metrics.get("extrapolation_full", {}).items():
            print(f"  extrap {rng_name}: {acc:.3f}")

        logger.log({"event": "run_end", "global_step": global_step})
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Short smoke training run")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--d-model", type=int, default=16,
                        help="GRU hidden dimension (for parameter-scaling sweeps)")
    parser.add_argument("--quantize-levels", type=int, default=None,
                        help="If set, quantize write_head output to N evenly-spaced levels")
    parser.add_argument("--quantize-range", choices=["symmetric", "unit"], default="symmetric",
                        help="'symmetric': [-1, 1] via tanh; 'unit': [0, 1] via sigmoid "
                             "(aligns with complex-world value range)")
    parser.add_argument("--complex-world", action="store_true",
                        help="Use complex-world scan (peak pattern + distractors + mid-noise). "
                             "Requires --scene-L >= ~3x stage4 n_max.")
    parser.add_argument("--scene-L", type=int, default=50,
                        help="World scan length (bound on max N)")
    parser.add_argument("--stage4-n-max", type=int, default=None,
                        help="Override stage4 n_max (for pushing N range). Must be <= scene-L - 1.")
    parser.add_argument("--task", choices=["compare", "discrim", "trio", "trio_wide", "trio_wide_extended", "successor_prediction", "binary"],
                        default="compare",
                        help="'compare': default beta sampling (ordinal-friendly). "
                             "'discrim': tightened to +-1 hard-negatives, breaks magnitude/log. "
                             "'trio': 1/3 equal + 1/3 alpha+1 + 1/3 alpha-1, no far. Kills "
                             "beta-prior shortcut; emergence floor = 0.333 (random). "
                             "'trio_wide': beta = alpha + uniform({-gap_max..+gap_max}), "
                             "gap_max=4 by default (9-level gap distribution; for gap_mse4/rl_top1).")
    parser.add_argument("--trio-wide-gap-max", type=int, default=4,
                        help="(--task trio_wide) Maximum |N_β - N_α| offset; "
                             "gap distribution is uniform over {-gap_max..+gap_max}.")
    parser.add_argument("--alpha-dist", choices=["uniform", "log_uniform"], default="uniform",
                        help="N_alpha sampling distribution. 'log_uniform' over-samples small N "
                             "(log(N) ~ U[0, log(n_max+1)]); breaks the prior assumption that "
                             "alpha is uniform across [1, n_max].")
    parser.add_argument("--mamba-layers", type=int, default=2,
                        help="(--arch mamba) number of stacked Mamba blocks.")
    parser.add_argument("--mamba-d-state", type=int, default=16,
                        help="(--arch mamba) Mamba SSM state dimension d_state.")
    parser.add_argument("--loss-type",
                        choices=["ce", "ordinal", "regression", "gap_mse4", "rl_top1", "rl_perblock"],
                        default="ce",
                        help="Training loss. 'ce': standard 3-class CE. "
                             "'ordinal': MSE on softmax-expectation of {+1,-1,0}. "
                             "'regression': scalar prediction, MSE vs signed target ±1. "
                             "'gap_mse4': scalar gap_head, (pred - (N_α-N_β))^4. Requires --use-gap-head; "
                             "pair with --task trio_wide for 9-level gap distribution. "
                             "'rl_top1': REINFORCE; agent picks 1 of K options by softmax sample; "
                             "reward = 100·10^-|gap|. Requires --use-gap-head + --task trio_wide. "
                             "'rl_perblock': K independent binary REINFORCE; per block, sigmoid score → "
                             "Bernoulli(equal/not-equal), reward ±1 per block. K times more learning "
                             "signal per episode than rl_top1. Requires --use-gap-head + --task binary "
                             "(50/50 equal vs non-equal balanced, no β-prior shortcut).")
    parser.add_argument("--use-gap-head", action="store_true",
                        help="Replace 3-class compare_head with scalar gap_head (Linear -> 1). "
                             "Required for gap_mse4 and rl_top1. Output shape becomes (B, K, 1).")
    parser.add_argument("--rl-entropy-coef", type=float, default=0.05,
                        help="(--loss-type rl_top1) Entropy bonus coefficient. Higher = more "
                             "exploration. 0.05 default per design spec.")
    parser.add_argument("--rl-baseline-ema", type=float, default=0.99,
                        help="(--loss-type rl_top1) EMA decay for the running mean-reward baseline. "
                             "Closer to 1 = slower-moving baseline (lower variance, more lag).")
    parser.add_argument("--K", type=str, default=None,
                        help="Override SceneConfig.K (number of compare blocks per episode). "
                             "Single int (all stages, e.g. '--K 12') OR comma-sep per-stage list "
                             "matching curriculum length (e.g. '--K 2,6,12' for 3-stage K curriculum). "
                             "Spec recommends K=12 for rl_top1 (76%% chance of having a gap=0 β); "
                             "K=8 (default) is fine for supervised gap_mse4.")
    parser.add_argument("--write-noise-std", type=float, default=0.0,
                        help="If >0, add Gaussian noise (std=this) to write_head logit before "
                             "quantize-STE during training. Sharpens scratch encoding (off-by-one defense).")
    parser.add_argument("--init-from-ckpt", type=str, default=None,
                        help="Load agent state_dict from this checkpoint file before training. "
                             "Architecture must match exactly (strict=True). Optimizer state is "
                             "NOT loaded so LR warmup restarts. Used by RL warm-start "
                             "from a supervised checkpoint.")
    parser.add_argument("--init-from-ckpt-nonstrict", action="store_true",
                        help="Load --init-from-ckpt with strict=False: matching keys load, "
                             "mismatched keys (e.g. a new/different write head) are left at init "
                             "and reported to stderr. For warm-starting a known-good substrate/CNN "
                             "while swapping one module (e.g. attention -> per-cell write head).")
    parser.add_argument("--rl-reward-shape", choices=["sharp", "soft"], default="sharp",
                        help="(--loss-type rl_top1) Reward shape: 'sharp' = 100·10^-|gap| "
                             "(spec default, 100x peak at gap=0); "
                             "'soft' = exp(-|gap|/2) ∈ [0,1] (denser gradient across all gaps, "
                             "mitigates baseline-tracking pathology in cold-start).")
    parser.add_argument("--rl-adv-normalize", action="store_true",
                        help="(--loss-type rl_top1) Z-score advantage within each batch before "
                             "policy gradient. Stabilizes step size when reward variance is high.")
    parser.add_argument("--rl-perblock-reward", choices=["simple", "asymmetric"], default="simple",
                        help="(--loss-type rl_perblock) reward shape. "
                             "'simple' (default): +1 correct / -1 wrong, range [-1,+1]. "
                             "'asymmetric' (P2-strong): +12 TP / -6 FN / -C(|gap|) FP (4/10/14/20) / "
                             "+6 TN. No prior shortcut, decision threshold p*=0.5, A>D rewards "
                             "correct-equal 2x more than correct-unequal. Range [-20, +12].")
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam",
                        help="Optimizer choice. 'adam' (default) is fast but sparse RL reward "
                             "can pollute its v (2nd-moment) estimate, causing effective-LR "
                             "blow-ups and gradient explosion. 'sgd' (with --weight-decay) is "
                             "more stable for noisy RL gradients at the cost of slower convergence.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="L2 weight decay (passed to optimizer). Recommended ~0.01 for SGD "
                             "in RL setting to prevent slow weight drift.")
    parser.add_argument("--sgd-momentum", type=float, default=0.9,
                        help="(--optimizer sgd) Momentum for SGD.")
    parser.add_argument("--world-pred-t-pred", type=int, default=0,
                        help="(MambaAgent only) ForwardPredictor horizon: predict next "
                             "T_pred CNN latents from each alpha-scan timestep. "
                             "0 (default) disables. Recommended 10/30/80 for ablation.")
    parser.add_argument("--world-pred-lambda", type=float, default=0.3,
                        help="Weight λ_w for world prediction loss in total: "
                             "L_total = L_task + λ_w * L_world. Used only when "
                             "--world-pred-t-pred > 0.")
    parser.add_argument("--beta-pred-t-pred", type=int, default=0,
                        help="(MambaAgent only, A2) BetaPredictor horizon: predict "
                             "first T_β CNN latents of each compare-block β scan "
                             "from (h_α, scratch). 0 disables.")
    parser.add_argument("--beta-pred-lambda", type=float, default=0.3,
                        help="Weight λ_β for β-prediction loss. Used only when "
                             "--beta-pred-t-pred > 0.")
    parser.add_argument("--compare-pred", action="store_true",
                        help="(MambaAgent only, A5) Enable ComparePredictor: predict "
                             "compare_logits from (h_α, h_β, scratch). Self-consistency "
                             "MSE vs detached actual logits.")
    parser.add_argument("--compare-pred-lambda", type=float, default=0.1,
                        help="Weight λ_c for ComparePredictor self-consistency loss.")
    parser.add_argument("--predictive-coding", action="store_true",
                        help="(MambaAgent only, PC) Enable PC-lite: during α scan, "
                             "mamba processes (CNN(input) − top_down_predicted) "
                             "instead of raw CNN. Top-down predictor learns u[t] from "
                             "u[t-1].")
    # RS-1: Raw signal predictor
    parser.add_argument("--rs-pred-t-pred", type=int, default=0,
                        help="(MambaAgent only, RS-1) RawSignalPredictor horizon: "
                             "predict next T raw input signal scalars from h_t at each "
                             "alpha-scan timestep. 0 disables. Recommended 30.")
    parser.add_argument("--rs-pred-lambda", type=float, default=0.3,
                        help="Weight λ_rs for raw signal prediction loss.")
    # A3: Scratch self-predictor
    parser.add_argument("--scratch-self-pred", action="store_true",
                        help="(MambaAgent only, A3) Enable ScratchSelfPredictor: predict "
                             "scratch slot t from prior slots + h_t at each write step.")
    parser.add_argument("--scratch-self-pred-lambda", type=float, default=0.3,
                        help="Weight λ_ssp for scratch self-prediction loss.")
    # A4: Counterfactual β predictor
    parser.add_argument("--cf-beta-pred-t-pred", type=int, default=0,
                        help="(MambaAgent only, A4) CounterfactualBetaPredictor horizon: "
                             "predict T raw β signal scalars from (scratch, β_count). "
                             "0 disables. Recommended 30.")
    parser.add_argument("--cf-beta-pred-lambda", type=float, default=0.3,
                        help="Weight λ_cfb for counterfactual β prediction loss.")
    # idea2: Self-review iterative refinement
    parser.add_argument("--review-iter", type=int, default=0,
                        help="(MambaAgent only, idea2) Self-review iterative refinement: "
                             "after write, refine full scratch via ReviewModule for K iter. "
                             "0 disables.")
    parser.add_argument("--review-layers", type=int, default=2,
                        help="(idea2) Layers in ReviewModule transformer.")
    parser.add_argument("--review-heads", type=int, default=2,
                        help="(idea2) Heads in ReviewModule transformer.")
    # β-1: VQ bottleneck
    parser.add_argument("--vq-bottleneck", action="store_true",
                        help="(MambaAgent only, β-1) Enable shared VQ codebook scratch. "
                             "Replaces per-slot scalar quantize with K-code × d_z codebook.")
    parser.add_argument("--vq-codebook-size", type=int, default=32,
                        help="(β-1) Codebook size K. Default 32.")
    parser.add_argument("--vq-d-z", type=int, default=8,
                        help="(β-1) Codebook code dimension d_z. Default 8.")
    parser.add_argument("--vq-commitment", type=float, default=0.25,
                        help="(β-1) Commitment loss weight β in VQ-VAE loss formula.")
    parser.add_argument("--vq-commit-lambda", type=float, default=1.0,
                        help="(β-1) Outer weight on commit loss in total: "
                             "L_total = L_task + λ_commit * commit_loss.")
    # P1 (L1579): dual-loss imagination architecture
    parser.add_argument("--dual-loss", action="store_true",
                        help="(MambaAgent only, P1/L1579) Enable dual-loss: predict "
                             "raw signal of imagined future + scratch under that "
                             "imagination. Two new loss terms: L_world + L_sp.")
    parser.add_argument("--dual-pred-t-p", type=int, default=30,
                        help="(P1) Prediction horizon T_p — how many future raw "
                             "signal scalars to predict from end-of-α latent.")
    parser.add_argument("--dual-world-lambda", type=float, default=0.3,
                        help="(P1) Weight on L_world (signal prediction MSE).")
    parser.add_argument("--dual-sp-lambda", type=float, default=0.3,
                        help="(P1) Weight on L_sp (imagined scratch vs actual β scratch MSE).")
    # COLLAPSE FIXES (GPT recipe: L_z + L_h(sg) + L_reg)
    parser.add_argument("--aux-detach", action="store_true",
                        help="(MambaAgent) BYOL-style stop-gradient on aux loss "
                             "targets (A2 beta latents, E0 future h, PC u). Prevents "
                             "encoder collapse via predictor pathway.")
    parser.add_argument("--vicreg-lambda", type=float, default=0.0,
                        help="(MambaAgent) VICReg variance regularizer weight on "
                             "aux loss targets. Forces per-dim std >= γ → anti-collapse. "
                             "0 disables. Recommended 0.1.")
    parser.add_argument("--vicreg-gamma", type=float, default=1.0,
                        help="(MambaAgent) VICReg target std threshold γ.")
    parser.add_argument("--pc-explicit-lambda", type=float, default=0.0,
                        help="(MambaAgent, PC) Weight for explicit MSE prediction "
                             "loss in PC pathway (alongside residual subtraction). "
                             "Required for true BYOL-style PC; recommend 0.3 with "
                             "--aux-detach + --vicreg-lambda > 0.")
    # 51_M5 (2026-05-01): simple comparator + cycle loss + PFC variant
    parser.add_argument("--simple-comparator", action="store_true",
                        help="(MambaAgent, 51_M5) Replace MLP compare_head with "
                             "single Linear(d, C) on (h_sp − h_β). Anti-symmetric. "
                             "Forces structural pressure onto scratch.")
    parser.add_argument("--cycle-loss-lambda", type=float, default=0.0,
                        help="(MambaAgent, 51_M5) Weight on L_cycle = "
                             "MSE(h_sp_avg, sg(h_alpha)). 0 disables. Recommend 0.5.")
    parser.add_argument("--pfc-variant", choices=["transformer", "mlp"], default="transformer",
                        help="(MambaAgent, 51_M5) PFC backbone: 'transformer' (M4 default) "
                             "or 'mlp' (DeepMLPPFC: flatten + 2-layer Linear).")
    parser.add_argument("--pfc-mlp-hidden", type=int, default=64,
                        help="(MambaAgent, 51_M5) Hidden dim for DeepMLPPFC.")
    # V6 (2026-05-04): Dual-pathway subitizing CNN bypass
    parser.add_argument("--dual-pathway", action="store_true",
                        help="(MambaAgent, V6) Replace single read_cnn with multi-scale "
                             "CNN. Adds CNN→PFC bypass token (subitizing pathway). "
                             "Detaches CNN→Mamba backward to keep CNN gradient clean.")
    parser.add_argument("--dp-kernels", type=str, default="5,21,51",
                        help="(V6) Comma-separated multi-scale kernel sizes.")
    parser.add_argument("--dp-channels-per-scale", type=int, default=4,
                        help="(V6) Conv channels per kernel scale.")
    parser.add_argument("--dp-detach-to-mamba", action="store_true", default=True,
                        help="(V6) Detach CNN output before Mamba (default: on).")
    parser.add_argument("--no-dp-detach-to-mamba", dest="dp_detach_to_mamba",
                        action="store_false",
                        help="(V6) Disable detach (joint train).")
    parser.add_argument("--cnn-pfc-only", action="store_true",
                        help="(V6 BRANCH) Pure CNN+PFC, NO Mamba. Forces "
                             "dual-pathway. PFC sees cnn_alpha_summary at write "
                             "and cnn_alpha_readback + cnn_beta_summary at "
                             "compare. Test on small N (≤3 or 4); small N MUST "
                             "give higher precision than large N.")
    parser.add_argument("--v7-mode", action="store_true",
                        help="(V7 2026-05-04) Convenience flag: enable "
                             "dual-pathway + Mamba on readback + CNN tokens at "
                             "compare. Equivalent to --dual-pathway with "
                             "cnn_pfc_only=False, dp_scratch_skip_mamba=False, "
                             "dp_detach_to_mamba=True (cut Mamba→CNN backward).")
    parser.add_argument("--dp-t-target", type=int, default=0,
                        help="(V7+stride) Adaptive avg pool CNN outputs to this "
                             "T_target. Forces Mamba to see fixed-length tokens "
                             "across stages. 0 = disable. Recommended 8 for "
                             "v7+stride mode (each token = 'saccade window').")
    parser.add_argument("--rbf-compare-head", action="store_true",
                        help="(V8 2026-05-05) Replace simple_comparator's "
                             "Linear(d, 3) with Gaussian RBF readout. 3 learnable "
                             "centers in d-space; logits = -dist^2/(2σ²). Bounded "
                             "logits prevent 'confidently wrong' divergence.")
    parser.add_argument("--rbf-init-sigma", type=float, default=1.0,
                        help="(V8) Initial sigma for RBF readout. Default 1.0.")
    parser.add_argument("--kwta-k", type=int, default=0,
                        help="(V8_kWTA) k for top-k WTA on PFC outputs. "
                             "Each PFC token (B, d) keeps top-k abs activations. "
                             "0 = disable. Recommended k=5 for d=16 (≈ 1/3 sparse).")
    # V9-AR (2026-05-05): GRUCell autoregressive write head
    parser.add_argument("--ar-write-head", action="store_true",
                        help="(V9-AR) Replace Linear write_head with GRUCell-based "
                             "autoregressive head: cell_k conditioned on h_for_write "
                             "AND prev cell. Enables cell-cell correlation needed for "
                             "place-value structure. Total codes preserved (Q^W).")
    parser.add_argument("--ar-write-embed-dim", type=int, default=4,
                        help="(V9-AR) embed dim for prev_cell scalar (default 4).")
    parser.add_argument("--ar-write-prev-dropout", type=float, default=0.3,
                        help="(V9-AR) Random mask prev_cell with this prob in training. "
                             "Forces each step to also use h_for_write directly, "
                             "preventing c1-dominance / chain degeneracy. Default 0.3.")
    parser.add_argument("--ar-write-tau-init", type=float, default=1.5,
                        help="(V9-AR) Gumbel-softmax tau at training step 0.")
    parser.add_argument("--ar-write-tau-final", type=float, default=0.8,
                        help="(V9-AR) Gumbel-softmax tau at >=warmup steps. Eval also uses this.")
    parser.add_argument("--ar-write-tau-warmup", type=int, default=5000,
                        help="(V9-AR) Linear-anneal steps from tau_init to tau_final.")
    # V10 (2026-05-05): preserve time/position info into Mamba/PFC
    parser.add_argument("--cnn-stride-no-pool", action="store_true",
                        help="(V10) Replace adaptive_avg_pool1d (within-window mean, "
                             "destroys within-window position) with stride sampling: "
                             "each kept position is the FULL conv output (kernel-pattern "
                             "preserved at fixation point). Mamba sees T_target snapshots "
                             "each carrying within-window structure.")
    # P3 (V11) — IF write controller (model decides write timing)
    parser.add_argument("--if-write-gate", action="store_true",
                        help="(V11/P3) Replace W-step write loop with Integrate-and-Fire "
                             "controller: state accumulates encoder output, fires when "
                             "learnable threshold exceeded, partial reset on fire. "
                             "Top-W of T fire-urgency steps become scratch writes. "
                             "Model learns WHEN to write.")
    parser.add_argument("--if-init-threshold", type=float, default=1.0,
                        help="(V11/P3) Initial fire threshold (learnable). Default 1.0.")
    parser.add_argument("--if-init-reset", type=float, default=0.5,
                        help="(V11/P3) Initial reset strength (∈[0,1], sigmoid'd, learnable). "
                             "Default 0.5 (half-reset on fire).")
    parser.add_argument("--if-fire-tau", type=float, default=1.0,
                        help="(V11/P3) Gumbel-sigmoid temperature for fire decision. Default 1.0.")
    parser.add_argument("--if-use-mamba-state", action="store_true",
                        help="(V11/P3) Use h_A (Mamba sequential state) as IF input. "
                             "Default: use CNN sequence directly (no Mamba).")
    parser.add_argument("--write-head-ln", action="store_true",
                        help="(V12, ABANDONED) LayerNorm before write_head — was the "
                             "first attempt at saturation protection but smoke showed "
                             "it destroys the magnitude axis that carries N. Use "
                             "--write-head-clip instead.")
    parser.add_argument("--write-head-clip", type=float, default=0.0,
                        help="(V12) Tanh-clip raw output to [±clip] before sigmoid: "
                             "raw = clip · tanh(raw/clip). Bounds magnitude smoothly "
                             "without destroying h_for_write structure. Prevents "
                             "sigmoid saturation seen in V10_PFCseq (raw ±5-16). "
                             "0 = disabled. Recommended: 3.0–4.0.")
    parser.add_argument("--use-lru", action="store_true",
                        help="(V18) Replace Mamba blocks with complex-eigenvalue "
                             "LRU (Orvieto 2023). Preserves oscillation under "
                             "task pressure (vs Mamba which collapses to real "
                             "eigenvalues on trio_wide). Pairs with --multi-modular-aux "
                             "for grid-cell-like representation.")
    parser.add_argument("--lru-d-state", type=int, default=16,
                        help="(V18) LRU complex state dim per block.")
    parser.add_argument("--lru-init-r-min", type=float, default=0.4,
                        help="(V18) LRU |λ| init lower bound.")
    parser.add_argument("--lru-init-r-max", type=float, default=0.9,
                        help="(V18) LRU |λ| init upper bound.")
    parser.add_argument("--lru-event-gated", action="store_true",
                        help="(V19) Event-gated LRU: LRU sees unpooled (B, L, d) "
                             "instead of pooled (B, 8, d). spike_detector → gate ∈ "
                             "[0,1]. gate≈1: phase rotates; gate≈0: h frozen. "
                             "Phase axis becomes spike-count, not spatial step.")
    parser.add_argument("--gate-sharpness-init", type=float, default=5.0,
                        help="(V19) gate_sharpness init (learnable). Init 5.0 makes "
                             "initial gate near-binary (sigmoid(5x) ≈ step).")
    parser.add_argument("--detection-aux-weight", type=float, default=0.0,
                        help="(V19) BCE aux loss weight on spike_score vs ground "
                             "truth spike centers. 0=off (let end-task learn). "
                             "Recommend 0.05 only if gate doesn't learn from end-task.")
    parser.add_argument("--use-oracle-spike-gate", action="store_true",
                        help="(V19 oracle ablation) Bypass spike_detector for "
                             "the LRU gate; build a hard-binary gate from scene "
                             "meta spike centers. Detector still runs (gradients "
                             "via detection BCE if enabled) but does NOT drive "
                             "the LRU gate. Used to disambiguate detection "
                             "learning vs path-integration mechanism.")
    parser.add_argument("--scratch-mod-aux-moduli", type=str, default="",
                        help="(V20) Comma-separated moduli for scratch-domain "
                             "modular aux head. e.g. '2,3'. Linear(W, n_mod) "
                             "reads (scratch_alpha - scratch_beta_virtual) and "
                             "predicts (α mod m == β mod m) per modulus. Forces "
                             "scratch_pad cells to host modular structure.")
    parser.add_argument("--scratch-mod-aux-weight", type=float, default=0.0,
                        help="(V20) BCE loss weight for scratch_mod_aux. V20 "
                             "canonical: 0.05 (combined with --multi-modular-aux"
                             "-lambda 0.05 to keep total mod aux at 0.10).")
    parser.add_argument("--e-pair-consistency-weight", type=float, default=0.0,
                        help="(V22) MSE loss weight forcing virtual β scratch "
                             "to match α scratch when δ=N_α-N_β=0. Drives "
                             "code-N 1-to-1 mapping. V22 canonical: 0.05.")
    parser.add_argument("--successor-loss-weight", type=float, default=0.0,
                        help="(V22) MSE loss weight on SuccessorFunction S "
                             "applied to within-episode |δ|=1 G/L pairs. V22 "
                             "canonical: 0.05.")
    parser.add_argument("--successor-hidden", type=int, default=32,
                        help="(V22) Hidden dim of SuccessorFunction MLP.")
    parser.add_argument("--successor-detached", action="store_true",
                        help="(V25-V22) Detach raw scratch before passing to S, "
                             "so gradient flows only through S not back into write_head.")
    parser.add_argument("--h-pair-consistency-weight", type=float, default=0.0,
                        help="(V25-V22) PFC-level E pair consistency MSE on "
                             "(h_α_pfc, h_β_pfc) when N_α=N_β. V25-V22 canonical: 0.05.")
    parser.add_argument("--write-head-type", choices=["linear", "sequential", "lru_decoder", "sinusoidal"], default="linear",
                        help="(V25/V28) write_head architecture. linear=V19 ext default, "
                             "sequential=V25 shared-MLP rolled out over W cells, "
                             "lru_decoder=V28-LRU state-init free response (B=D=0 disabled), "
                             "sinusoidal=V28-Sin explicit-ω sin(ω·proj+φ) per cell.")
    parser.add_argument("--write-head-hidden", type=int, default=32,
                        help="(V25) Hidden dim of SequentialWriteHead MLP.")
    parser.add_argument("--write-head-pending-marker", action="store_true",
                        help="(PM) Init `written` buffer with quantize midpoint (0.5 for "
                             "unit, 0 for symmetric) instead of zeros. Breaks cell cascade "
                             "where cell w sees identical input to cell 0 (because all "
                             "default to lv0). Same-medium-faithful: midpoint is a legal "
                             "alphabet value, not a new symbol.")
    parser.add_argument("--lru-decoder-d-state", type=int, default=16,
                        help="(V28-LRU) d_state of decoder LRU complex eigenvalue spectrum.")
    parser.add_argument("--sin-omega-init", type=str, default="0.3,0.7,1.2,1.7,2.3",
                        help="(V28-Sin) explicit ω init per cell, comma-separated. "
                             "Length must == W (5). Spread across orders of magnitude → "
                             "different mod periods. Default linspace [3, 21] period range.")
    parser.add_argument("--sin-init-std", type=float, default=0.5,
                        help="(V28-Sin) direction Linear weight init std. Affects projection "
                             "magnitude → effective arg range. Default 0.5. Tune via init "
                             "check warning (cycles < 0.5 → too small, > 5 → too big).")
    parser.add_argument("--sin-predict-hidden", type=int, default=64,
                        help="(V28-Sin) MultiHotMLPSuccessorHead hidden dim. Default 64.")
    parser.add_argument("--n-class-ce-weight", type=float, default=0.0,
                        help="(V23b) CE loss weight on NClassHead predicting "
                             "N_α from scratch_α. V23b canonical: 0.10.")
    parser.add_argument("--n-class-max-n", type=int, default=0,
                        help="(V23b) Max N for NClassHead output dim. 0=disable. "
                             "V23b canonical: 90 (stage-4 max).")
    parser.add_argument("--n-class-head-hidden", type=int, default=32,
                        help="(V23b) Hidden dim of NClassHead MLP.")
    parser.add_argument("--successor-task-k-max", type=int, default=5,
                        help="(V24) Max k for successor_prediction task. β = α + k, k ∈ [1, k_max].")
    parser.add_argument("--successor-predict-input-level", choices=["scratch", "pfc_hidden"], default="scratch",
                        help="(V25-V24) Input level for successor_predict_head. "
                             "'scratch'=Linear(2W,k_max) (V24 default), "
                             "'pfc_hidden'=Linear(2*d,k_max) (V25-V24, supervisory at PFC level).")
    parser.add_argument("--successor-bidirectional", action="store_true",
                        help="(V27) Allow β = α ± k (sign random). Output classes = 2*k_max "
                             "(k ∈ {-k_max..-1, +1..+k_max}, no 0). Scene must use "
                             "successor_only_positive=False. Removes lazy-N_α-prediction trap.")
    parser.add_argument("--beta-range-preset", choices=["successor", "balanced"], default="successor",
                        help="(V35 PSS leak-kill) successor=|k|<=k_max only (current); "
                             "balanced=EQUAL+NEAR+SCALE+FAR full-range mixture that forces alpha's high digits.")
    parser.add_argument("--scale-weight", type=float, default=0.30,
                        help="(V35 balanced) SCALE regime weight: per-scale hard negatives near theta^k.")
    parser.add_argument("--successor-far-bands", type=int, default=1,
                        help="(V35 balanced) FAR ordinal bands per direction. 1=COARSE (sign-only far). "
                             "n_classes = 2*k_max + 1 + 2*far_bands.")
    parser.add_argument("--cot-pfc-n-chunks", type=int, default=0,
                        help="(V27) Chain-of-thought PFC: split α/β scan into n chunks. "
                             "PFC self-attn over [pfc_state, cnn_chunk, lru_h_chunk] per chunk. "
                             "0=disabled (use V19/V25 path). V27 canonical: 4.")
    parser.add_argument("--cot-pfc-cnn-kernels", type=str, default="5,51",
                        help="(V27) Multi-scale CNN kernel sizes for chunked CoT path. "
                             "Default 5,51 (drops k=21 vs V19 ext).")
    # V29: R&F SNN substrate (replaces all V27/V28 modules when --use-v29)
    parser.add_argument("--use-v29", action="store_true",
                        help="(V29) Enable R&F SNN pipeline. Replaces all CNN/Mamba/CoTPFC/write_head with V29Pipeline.")
    parser.add_argument("--v29-encoder-n-neurons", type=int, default=16)
    parser.add_argument("--v29-write-n-neurons", type=int, default=5)
    parser.add_argument("--v29-d-id", type=int, default=16)
    parser.add_argument("--v29-d-pe", type=int, default=16)
    parser.add_argument("--v29-pfc-layers", type=int, default=2)
    parser.add_argument("--v29-pfc-heads", type=int, default=4)
    parser.add_argument("--v29-pfc-passes", type=int, default=2)
    parser.add_argument("--v29-threshold", type=float, default=1.0)
    parser.add_argument("--v29-surrogate-alpha", type=float, default=4.0)
    parser.add_argument("--v29-omega-enc-min", type=float, default=0.5)
    parser.add_argument("--v29-omega-enc-max", type=float, default=2.0)
    parser.add_argument("--v29-omega-write-min", type=float, default=0.7)
    parser.add_argument("--v29-omega-write-max", type=float, default=1.8)
    parser.add_argument("--v29-b-init-min", type=float, default=-0.3)
    parser.add_argument("--v29-b-init-max", type=float, default=-0.05)
    parser.add_argument("--v29-time-max-init", type=float, default=100.0)
    parser.add_argument("--v29-soft-reset", action="store_true", default=True)
    parser.add_argument("--v29-hard-reset", dest="v29_soft_reset", action="store_false")
    parser.add_argument("--v29-t-write", type=int, default=0,
                        help="(V29) 0 = auto from 2π/median(ω_write_init).")
    parser.add_argument("--v29-max-events", type=int, default=256)
    parser.add_argument("--v29-query-orth-lambda", type=float, default=0.01,
                        help="(V29) Orthogonality reg weight on alpha-side query tokens.")
    parser.add_argument("--v29-omega-clamp-decay", type=float, default=1e-4,
                        help="(V29) Soft clamp on encoder/write ω + b drift via L2 toward init.")
    parser.add_argument("--v29-firing-rate-lambda", type=float, default=0.01,
                        help="(V29.2) Weight for encoder firing rate regularization (prevents dead neurons).")
    parser.add_argument("--v29-target-firing-rate", type=float, default=1.5,
                        help="(V29.2) Target firing rate per encoder neuron per α episode (smoothed mean).")
    # V33: HH-SSM substrate (replaces V27 LRU+CoT-PFC core when --use-v33)
    parser.add_argument("--use-v33", action="store_true",
                        help="(V33) Enable HH-SSM substrate. Replaces V27 LRU+CoT-PFC core with multi-channel closed-loop SSM.")
    parser.add_argument("--v33-d-model", type=int, default=16,
                        help="(V33) Hidden state dim. Must equal --d-model for encoder compatibility.")
    parser.add_argument("--v33-n-layers", type=int, default=1,
                        help="(V33) Number of stacked HHSSMLayer (default 1).")
    parser.add_argument("--v33-v-perturbation-scale", type=float, default=0.3,
                        help="(V33) Per-channel v_c perturbation magnitude. Larger -> more orthogonal init.")
    parser.add_argument("--v33-ortho-lambda", type=float, default=0.0,
                        help="(V33-A) Lambda on channel-direction orthogonality regularizer "
                             "(mean of cos² over c<c'). 0 = baseline (5/20 v33 run, collapsed). "
                             "0.1 = first ablation try.")
    parser.add_argument("--v33-input-to-gate", action="store_true",
                        help="(V33-D5) Kill direct input bypass (u_seq=B_leak(x)) and route "
                             "input through W_input_to_gate into gate equilibria. Forces all "
                             "input through channel machinery; tests bypass-vs-dynamics hypothesis.")
    parser.add_argument("--v33-same-medium", action="store_true",
                        help="(V33-SM) Same-medium closed loop: clear ch1/ch2 phase flags "
                             "(CNN medium-agnostic) + scratch re-injection through CNN+substrate "
                             "between α and β. Computes real sc_β from h_β_end (not α placeholder). "
                             "Restores V8-V19-style closed-loop gradient pressure on scratch.")
    parser.add_argument("--predict-coding-enable", action="store_true",
                        help="(V34) Build h_predictor Linear(W, d_model) for self-supervised "
                             "symbol emergence. Required for L_predict / L_commit losses to have "
                             "effect. Architecture flag — changes saved ckpt schema.")
    parser.add_argument("--unsupervised-symbol-emergence", action="store_true",
                        help="(V34) Skip task CE entirely; train only via L_predict + L_commit "
                             "+ L_diversity + L_raw_l2 aux losses. Requires --predict-coding-enable "
                             "and --v33-same-medium. β scan still runs but no loss flows through it.")
    parser.add_argument("--predict-coding-lambda", type=float, default=1.0,
                        help="(V34) Weight for L_predict = ‖ĥ - sg[h_readback]‖². "
                             "Default 1.0; 0 disables.")
    parser.add_argument("--commit-lambda", type=float, default=0.25,
                        help="(V34) Weight for L_commit = ‖σ(raw) - sg[nearest_level]‖² "
                             "(VQ-VAE style). Default 0.25.")
    parser.add_argument("--diversity-lambda", type=float, default=0.05,
                        help="(V34) Weight for L_diversity = -mean pairwise (σ(raw_w)-σ(raw_w'))². "
                             "Default 0.05. Anti-collapse pressure across the W cells.")
    parser.add_argument("--raw-l2-reg-lambda", type=float, default=0.001,
                        help="(V34) Weight for L_raw_l2 = ‖raw_preSTE‖². "
                             "Weak L2 prevents raw → ±∞ when commit pushes to bin boundaries.")
    parser.add_argument("--v33-vicreg-lambda-var", type=float, default=0.0,
                        help="(V33-VICReg) Weight for L_var = mean_d hinge(γ - σ_d) "
                             "applied to substrate h_seq every timestep. Anti-dead-dim.")
    parser.add_argument("--v33-vicreg-lambda-cov", type=float, default=0.0,
                        help="(V33-VICReg) Weight for L_cov = Σ_{i≠j} C_ij² / d "
                             "on substrate h_seq covariance per timestep. Decorrelates dims.")
    parser.add_argument("--v33-vicreg-gamma", type=float, default=1.0,
                        help="(V33-VICReg) Target std γ in L_var hinge. Default 1.0.")
    # V35 (5/27 pivot): d=1 HH-SSM + fire-and-reset cascade. Mutually
    # exclusive with --use-v29 / --use-v33. Requires --dual-pathway,
    # --task successor_prediction, --quantize-levels 3 --quantize-range unit.
    parser.add_argument("--use-v35", action="store_true",
                        help="V35 d=1 HH-SSM + fire-and-reset cascade pivot.")
    parser.add_argument("--v35-n-recursions", type=int, default=5,
                        help="V35 number of Layer-B carry recursions (fixed; default 5).")
    parser.add_argument("--v35-fire-temp", type=float, default=0.5,
                        help="V35 STE temperature for fire-and-reset threshold.")
    parser.add_argument("--v35-write-d-attn", type=int, default=16,
                        help="V35 write attention embed dim. Must be divisible by --v35-write-n-heads.")
    parser.add_argument("--v35-write-n-heads", type=int, default=2,
                        help="V35 write attention heads.")
    parser.add_argument("--v35-compare-d-attn", type=int, default=16,
                        help="V35 compare attention embed dim. Must be divisible by --v35-compare-n-heads.")
    parser.add_argument("--v35-compare-n-heads", type=int, default=2,
                        help="V35 compare attention heads.")
    parser.add_argument("--v35-compare-n-layers", type=int, default=1,
                        help="V35 compare attention transformer layers.")
    parser.add_argument("--v35-compare-concat", action="store_true",
                        help="(V35 Rung 2, 2026-06-07) Replace compare-head mean-pool with a "
                             "position-indexed concat readout (keep all 2*W tokens). R1 probe "
                             "showed mean-pool is the comparison wall (real-code allpair 0.17-0.70 "
                             "mean vs 0.92-1.00 concat); concat lets the classifier learn per-slot "
                             "place weights. Same-medium compatible (read-side only).")
    parser.add_argument("--v35-threshold-layer-a", action="store_true",
                        help="(V35 ablation II) Replace HH-based layer A with deterministic "
                             "raw-signal threshold detector. Bypasses CNN + HHSSMLayer1D(A); "
                             "diagnostic for cascade carry math. NOT same-medium compatible.")
    parser.add_argument("--v35-layer-a-threshold", type=float, default=0.5,
                        help="(V35 ablation II) Threshold for layer-A raw-signal detector.")
    parser.add_argument("--v35-layer-a-detect", type=str, default="level",
                        choices=["level", "food_pattern", "learned"],
                        help="(V35) raw-signal Layer-A detector under --v35-threshold-layer-a: "
                             "'level' (sig>thr, over-counts), 'food_pattern' (3-cell count_foods gate, spkA=N, HAND-CODED), "
                             "or 'learned' (direction-1: SGD-trained conv detector + local-max + STE spike, de-scaffolds counting).")
    parser.add_argument("--v35-layer-a-no-localmax", action="store_true",
                        help="(V35 learned detector) disable the local-max gate (plain threshold spike; jitters).")
    parser.add_argument("--scene-clean-world", action="store_true",
                        help="(2026-06-17) Decoy-free complex_world (n_distractors=0, n_mid_noise=0). On a clean "
                             "world a LEARNED counter is near-exact (es_pretrain --clean-scene 31/31) -> enables the "
                             "scoped counting-de-scaffold V35 test (frozen clean-scene detector + clean world).")
    parser.add_argument("--v35-layer-a-center-thresh", type=float, default=0.9,
                        help="(V35 food_pattern) center-cell threshold (mirrors scene FOOD_CENTER_THRESH).")
    parser.add_argument("--v35-layer-a-neighbor-thresh", type=float, default=0.8,
                        help="(V35 food_pattern) both-neighbor threshold (mirrors scene FOOD_NEIGHBOR_THRESH).")
    parser.add_argument("--v35-pss-readback-beta", action="store_true",
                        help="(V35 PSS leak-kill, spec 2026-06-05) Write beta to its OWN scratch "
                             "-> readback so no fresh-magnitude beta reaches compare (severs the leak). "
                             "Requires --task successor_prediction; threshold path requires --v35-readback-direct.")
    parser.add_argument("--v35-readback-direct", action="store_true",
                        help="(V35 TEMPORARY scaffolding) Differentiable readback: compare reads scratch "
                             "directly (STE soft path -> task gradient reaches the write head). Requires "
                             "--v35-threshold-layer-a. NOT same-medium; make-it-work only.")
    parser.add_argument("--v35-local-pc", action="store_true",
                        help="(V35 PC v0) Use local predictive-coding update for substrate "
                             "B_leak weights instead of SGD. Substrate gates init to "
                             "biological HH values. Spec: docs/design.md")
    parser.add_argument("--v35-pc-lr", type=float, default=0.1,
                        help="(V35 PC v0) Learning rate for substrate B_leak local PC update. "
                             "Default 0.1.")
    parser.add_argument("--v35-theta-a", type=float, default=None,
                        help="(V35 PC v1) Fixed theta for layer A (translator). Default None uses "
                             "shared learnable theta_raw (init 3.0). Typical: 1.0 (peak detector).")
    parser.add_argument("--v35-theta-b", type=float, default=None,
                        help="(V35 PC v1) Fixed theta for layer B (carry counter). Default None uses "
                             "shared learnable theta_raw (init 3.0). Typical: 3.0 (base-q carry).")
    parser.add_argument("--v35-theta-shared-init", type=float, default=3.0,
                        help="(V35 Rung B2 de-scaffold 'invent' test) Init of the SHARED learnable "
                             "theta (=base) used when --v35-theta-a/--v35-theta-b are BOTH omitted. "
                             "Default 3.0 = prior behavior. Set away from 3 (e.g. 2.0 or 5.0) to test "
                             "whether the learnable base DISCOVERS 3 vs only MAINTAINS it.")
    parser.add_argument("--v35-theta-in-sgd", action="store_true", default=False,
                        help="(V35 Rung B1') Keep the shared learnable theta (=base) IN the SGD "
                             "optimizer even under --v35-local-pc (which otherwise freezes all "
                             "substrate params at init). Makes the base genuinely gradient-learnable: "
                             "tests whether it FUNCTION-SELECTS toward the task-optimal radix.")
    parser.add_argument("--v35-log-g-leak-a", type=float, default=None,
                        help="(V35 Layer-A leak, spec 2026-06-02) Override Layer A's leak "
                             "channel (idx 3) log_g init for a membrane leak (a_eff knob). "
                             "None keeps -4 (no leak). Layer B always stays -4. Sweep {-1.5,-1,-0.5}.")
    parser.add_argument("--v35-warm-b-leak-a", type=float, default=1.0,
                        help="(V35 Layer-A leak, spec 2026-06-02) Multiplicative scale on Layer A's "
                             "B_leak init (cold-start firing; replaces IP). 1.0 = no-op. Cut-1: 4.0.")
    parser.add_argument("--v35-b-leak-b-init", type=float, default=None,
                        help="(V35 hand-built cascade, spec 2026-06-02) Fill Layer B's B_leak with this "
                             "constant (per-spike ≈ 0.495·val; 2.0 → ≈1, makes theta_b the true base). "
                             "None = default init.")
    parser.add_argument("--v35-log-g-b", type=float, default=None,
                        help="(V35 timing-invariant form, 6/03 TEMPORARY) Fill ALL of Layer B's per-channel "
                             "log-conductance with this constant and FREEZE it. Very negative (-20) turns "
                             "channels off -> event-driven counter -> residual = base-3 digits, "
                             "timing-invariant (probe-verified). Pair with --v35-b-leak-b-init 2.0 "
                             "--v35-theta-b 3. None = default init (-4).")
    parser.add_argument("--v35-recon-aux-lambda", type=float, default=0.0,
                        help="(V35 codebook-fidelity recon aux, spec 2026-06-04) Weight of MSE "
                             "reconstruction loss D(scratch_alpha) ~ residuals_alpha. >0 builds a "
                             "decoder forcing scratch to losslessly encode all residual digits "
                             "(counters codebook collapse where weak |k|<=5 pressure kills high cells). "
                             "Zero-Prior: target is the substrate's own residuals, not an N label. 0 = off.")
    parser.add_argument("--v35-cheap-arith-aux-lambda", type=float, default=0.0,
                        help="(V35 cheap-arithmetic aux, INSIGHTS §9A) Weight of a CHEAP (single "
                             "Linear) successor head over the quantized alpha scratch code, "
                             "trained with the SAME k_target CE as the main task. Low CE is "
                             "reachable only if the scratch supports linear arithmetic readout, "
                             "pressuring the codebook toward the positional class (an expressive "
                             "compare head exerts no such pressure). Quantize-STE carries the "
                             "gradient back to the write head. 0 = exact no-op.")
    parser.add_argument("--v35-count-consist-lambda", type=float, default=0.0,
                        help="(V35 direction-1 counting pressure, 2026-06-16) Weight of a within-N "
                             "scratch-CONSISTENCY (same N_alpha -> same code) + across-N INJECTIVITY "
                             "(different N codes >= margin apart) aux on the quantized alpha scratch. "
                             "Ports count+carry's proven objective so a LEARNED Layer-A counter has a "
                             "gradient toward consistent injective per-N codes (the successor task alone "
                             "gives none -> over-fire -> collapse). Unsupervised; does NOT hand N. 0 = no-op.")
    parser.add_argument("--v35-detector-init", type=str, default=None,
                        help="(V35 direction-1 goal-step) Path to an ES-pretrained LearnedLayerADetector "
                             "state_dict (es_pretrain_detector.py) to initialize the learned Layer-A counter. "
                             "ES finds a clean spkA=N counter where SGD collapses (optimizer lever).")
    parser.add_argument("--v35-freeze-detector", action="store_true",
                        help="(V35 direction-1 goal-step) Freeze the learned Layer-A detector (no SGD grad) "
                             "-> V35 trains the rest on a LEARNED (ES-found), not hand-coded, counter without "
                             "the SGD-through-cascade collapse.")
    parser.add_argument("--v35-freeze-layer-b-bleak", action="store_true",
                        help="(V35 hand-built cascade, spec 2026-06-02) pc_step skips the Layer B B_leak "
                             "update (frozen hand-set counter).")
    parser.add_argument("--v35-freeze-layer-a-bleak", action="store_true",
                        help="(V35 full frozen cascade, spec 2026-06-02) pc_step skips BOTH the Layer A "
                             "B_leak update and the IP update — Layer A fully frozen. Pair with "
                             "--init-from-ckpt (loaded forward detector) + --v35-freeze-layer-b-bleak.")
    parser.add_argument("--v35-write-per-cell", action="store_true",
                        help="(V35 route A, spec 2026-06-02) Per-cell write readout: scratch cell k "
                             "reads ONLY cascade residual k via its own tiny MLP (structural "
                             "decoupling, breaks cell-lockstep). Replaces the cross-attention write "
                             "head; pairs with the frozen cascade.")
    parser.add_argument("--v35-pc-h-upstream", action="store_true",
                        help="(V35 PC v1) Use preceding layer's continuous h_seq instead of binary "
                             "spike train as PC update input multiplier. Layer B only (layer A unaffected). "
                             "Breaks no-fire attractor in cold-start.")
    parser.add_argument("--v35-ip-enable", action="store_true",
                        help="(V35 PC IP pilot) Enable homeostatic intrinsic-plasticity bias on "
                             "Layer A (regulates avg firing toward r*, breaks no-fire attractor). "
                             "In pilot, also freezes Layer B PC. Requires --v35-local-pc. "
                             "Spec: docs/design.md")
    parser.add_argument("--v35-ip-lr", type=float, default=0.02,
                        help="(V35 PC IP) IP bias learning rate. Default 0.02 (slower than pc-lr).")
    parser.add_argument("--v35-ip-target", type=float, default=None,
                        help="(V35 PC IP) Target avg Layer A spike count r*. Default None -> "
                             "auto (N_max+1)/2 per stage.")
    parser.add_argument("--v35-ip-bias-clamp", type=float, default=0.0,
                        help="(V35 PC IP) Optional ±clamp safety net on ip_bias. 0 = disabled "
                             "(the IP rule is self-bounding).")
    parser.add_argument("--v35-pc-weight-decay", type=float, default=0.0,
                        help="(V35 PC anchor, spec 2026-05-29) Plain L2 decay on Layer A B_leak "
                             "in pc_step: w <- w*(1-wd) + lr*Hebbian. Anchors the Hebbian "
                             "common-mode integrator (stops ±91 drift). 0 = off. Pilot: 1e-3.")
    parser.add_argument("--v35-ip-leak", type=float, default=0.0,
                        help="(V35 PC anchor, spec 2026-05-29) Soft leak on ip_bias (anti-windup): "
                             "Δb = ip_lr*(r*-r_obs) - ip_leak*b. Bounds controller to fixed point "
                             "b* = ip_lr*(r*-r_obs)/ip_leak. 0 = off. Pilot: 0.01.")
    parser.add_argument("--v35-pc-weight-clip", type=float, default=0.0,
                        help="(V35 PC fix, spec 2026-05-29) Max L2 norm of each substrate B_leak; "
                             "after the local update, renorm w to this cap if exceeded (preserves "
                             "direction, caps scale). Bounds u_seq/h runaway without weight-decay's "
                             "shrinkage of small weights. 0 = off. Curriculum re-run: 5.0.")
    parser.add_argument("--v35-three-factor", action="store_true",
                        help="(V35 PC fix 3, spec 2026-05-29) Three-factor task-modulated Hebbian: "
                             "ΔW ~ e*input*M with per-sample signed advantage M = p_target - EMA-baseline "
                             "from the compare head, on layer A AND the (unfrozen) layer B. Requires "
                             "--v35-local-pc.")
    parser.add_argument("--v35-tf-baseline-ema", type=float, default=0.99,
                        help="(V35 PC fix 3) EMA rate for the three-factor advantage baseline. Default 0.99.")
    parser.add_argument("--v35-state-discretize", type=int, default=0,
                        help="(V35 state-discreteness fix, INSIGHTS sec.13) STE-quantize the "
                             "post-fire-reset membrane state to this many levels across [0, theta] "
                             "in BOTH layer A and B. 0 (or 1) = OFF (default V35 unchanged); K>1 = "
                             "K-level discrete state so the inter-spike level is discrete (the "
                             "fire-and-reset carry is already discrete). Try 3.")
    parser.add_argument("--pfc-recurrent", action="store_true",
                        help="(V24R) Enable recurrent scratch refinement (3-iter default).")
    parser.add_argument("--pfc-n-iterations", type=int, default=3,
                        help="(V24R) Number of recurrent refinement iterations.")
    parser.add_argument("--pfc-recurrent-cnn-scratch-kernel", type=int, default=3,
                        help="(V24R) Kernel size of cnn_scratch (Conv1d).")
    parser.add_argument("--pfc-recurrent-cnn-scratch-channels", type=int, default=4,
                        help="(V24R) Output channels of cnn_scratch.")
    parser.add_argument("--multi-modular-aux-moduli", type=str, default="",
                        help="(V17) Comma-separated co-prime moduli for "
                             "auxiliary modular tasks. e.g. '2,3,5'. Each modulus "
                             "adds a binary head predicting (α mod m == β mod m). "
                             "RNS pressure makes 1D thermometer insufficient. "
                             "Empty string = disabled.")
    parser.add_argument("--multi-modular-aux-lambda", type=float, default=0.1,
                        help="(V17) λ weight for multi-modular aux BCE loss. "
                             "Default 0.1 (small enough not to disrupt main task).")
    parser.add_argument("--multi-modular-aux-pool", type=str, default="",
                        help="(V19) Pool of moduli to sample from per batch (random "
                             "moduli reduce specific-frequency prior). e.g. '2,3,4,5,6,7'. "
                             "If set, --multi-modular-aux-moduli specifies the COUNT only. "
                             "Each batch samples count moduli from pool, sorted ascending "
                             "(positional binding to head index).")
    parser.add_argument("--write-query-token", action="store_true",
                        help="(V16) Prepend learnable [WRITE_QUERY] token at "
                             "position 0 of write phase PFC seq. Output[0] becomes "
                             "query + Attn (small init) instead of h_t + Attn → "
                             "breaks h_t-residual-dominance, restores σ/|h_t| ratio "
                             "to V9 sweet spot.")
    parser.add_argument("--recurrent-latent-pfc", action="store_true",
                        help="(V15) Coconut-style recurrent latent PFC: replace "
                             "cnn_alpha_summary (mean over T) with sequential "
                             "transformer-cell processing of cnn_alpha_seq (state "
                             "token + 1 CNN token per step, T steps, shared block). "
                             "State accumulates info in latent space, avoids "
                             "Mamba's 1D-magnitude collapse.")
    parser.add_argument("--recurrent-latent-state-tokens", type=int, default=1,
                        help="(V15) K = number of [STATE] tokens in recurrent PFC. "
                             "Default 1 (single state token).")
    parser.add_argument("--recurrent-latent-sparse-k", type=int, default=0,
                        help="(V15) kWTA on state per step (anti signal washout). "
                             "0 = no sparsity. Recommended: 5 for d=16 if magnitude "
                             "explodes during recurrence.")
    parser.add_argument("--recurrent-latent-steps", type=int, default=0,
                        help="(V15) Override step count (0 = use all dp_t_target). "
                             "E.g. 4 to truncate 8-token cnn_alpha_seq to first 4.")
    parser.add_argument("--gumbel-write-head", action="store_true",
                        help="(V14) Replace sigmoid+STE quantize with per-cell "
                             "gumbel-softmax over Q levels. Output bounded by "
                             "construction → no sigmoid saturation. Drop-in "
                             "replacement for default Linear write_head + STE.")
    parser.add_argument("--gumbel-tau", type=float, default=1.0,
                        help="(V14) gumbel-softmax temperature. Higher = softer/noisier "
                             "categorical sample. Default 1.0.")
    parser.add_argument("--cross-attn-write", action="store_true",
                        help="(V13) Replace V10_PFCseq's W-step self-attention loop "
                             "with single parallel cross-attention: W learnable "
                             "queries cross-attend cnn_alpha_seq. Each query learns "
                             "own attention pattern → W cells naturally differentiate. "
                             "Hypothesis: solves saturation by removing the burden "
                             "of cell-differentiation from write_head.")
    parser.add_argument("--cross-attn-write-heads", type=int, default=2,
                        help="(V13) num_heads for cross-attention write. Default 2 "
                             "(matches pfc_heads).")
    parser.add_argument("--write-head-wd", type=float, default=0.0,
                        help="(V12-fix2) Per-group weight_decay for write_head "
                             "parameters specifically. Higher wd suppresses large "
                             "write_head weights → bounds raw output magnitude → "
                             "prevents sigmoid saturation. 0 = use global wd. "
                             "Recommended: 1.0 (vs global 0.05) for V10_PFCseq base.")
    parser.add_argument("--pfc-see-cnn-seq", action="store_true",
                        help="(V10) PFC at write/compare consumes T_target CNN tokens "
                             "(with positional + side embeddings) instead of single "
                             "mean-summary, so PFC self-attention can extract within-α "
                             "positional structure to drive scratch differentiation.")
    parser.add_argument("--ar-entropy-bonus", type=float, default=0.0,
                        help="(V9-AR) λ for anti-collapse bonus: total_loss -= λ * mean "
                             "softmax-entropy of AR head logits. λ>0 prevents collapse "
                             "to '[0.5,...] middle-level basin' seen in first attempt. "
                             "Recommended start: 0.05.")
    parser.add_argument("--delta-h-diagnostic-every", type=int, default=0,
                        help="(MambaAgent, 51_M5) Run |h[t]-h[t-1]| diagnostic "
                             "every N epochs. 0 = disabled. Recommend 2.")
    parser.add_argument("--agent-seed", type=int, default=None,
                        help="Override TrainConfig.agent_seed (default 42). Use for multi-seed verification.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Override TrainConfig.data_seed (default 0). Affects training data sampling.")
    parser.add_argument("--arch", choices=["gru", "transformer", "attention", "rglru", "mamba"], default="gru",
                        help="'gru': baseline GRU cell. "
                             "'transformer': recurrent Transformer cell (y_t = F(u_t, y_{t-1})). "
                             "'attention': full-attention perception + shallow recurrent think + "
                             "attention-based compare (Geiping-style). "
                             "'rglru': Griffin-style state-independent linear-recurrent cell.")
    parser.add_argument("--event-gate", action="store_true",
                        help="(transformer only) Enable STE-based event gate "
                             "(discrete hidden-state accumulator).")
    parser.add_argument("--scratch-attn", action="store_true",
                        help="(gru/rglru only) Add MultiheadAttention over the W "
                             "scratch slots at compare timesteps for content-based addressing.")
    parser.add_argument("--gaussian-attn", action="store_true",
                        help="(gru only, requires --scratch-attn) Replace MHA with PFC-style "
                             "Gaussian receptive-field attention over the 1-D scratch axis.")
    parser.add_argument("--cell-type", choices=["gru", "lif", "none", "rcnn", "lstm"], default="gru",
                        help="(--arch gru) cell type: gru | lif | none | rcnn (Liang-Hu 2015 K-iter RCNN) | lstm (Conv-LSTM degenerate to LSTM in 1D)")
    parser.add_argument("--rcnn-iter", type=int, default=4,
                        help="RCNN inner-iteration count (K-step self-feedback)")
    parser.add_argument("--lif-decay", type=float, default=0.9,
                        help="LIF leak rate (smaller = faster forgetting)")
    parser.add_argument("--lif-threshold", type=float, default=1.0,
                        help="LIF firing threshold")
    parser.add_argument("--read-cnn-kernel", type=int, default=0,
                        help="(Stage 3) Causal 1D conv kernel size for V1/V2 perceptual front-end. 0=disable.")
    parser.add_argument("--read-cnn-layers", type=int, default=1,
                        help="(Stage 3 CNN sweep) Number of stacked Conv1d layers in the perceptual front-end. "
                             "1=V1 only (current). 2=V1+V2. 3=V1+V2+V4. GELU activation between layers. "
                             "Each layer has its own causal buffer; effective receptive field grows linearly.")
    parser.add_argument("--pfc-layers", type=int, default=0,
                        help="(Stage 3) Number of transformer encoder layers for PFC executive on [h, scratch] at compare time. 0=disable.")
    parser.add_argument("--pfc-heads", type=int, default=2,
                        help="(Stage 3) PFC transformer attention heads.")
    parser.add_argument("--pfc-iter", type=int, default=1,
                        help="(Day 4 Universal-Transformer style) Apply the pfc encoder K times in a loop "
                             "with shared weights. K=1 is current behavior. Increasing K trades compute for depth. "
                             "Both write-time PFC and compare-time PFC benefit since they share the same module.")
    parser.add_argument("--reset-h-at-compare", action="store_true",
                        help="(Stage 3) Reset hidden state h (and LSTM c, CNN buffer) at the start of "
                             "each compare block. Forces alpha info to flow ONLY through scratch_pad — "
                             "the architectural pressure for symbolic emergence.")
    parser.add_argument("--pfc-split-ab", action="store_true",
                        help="(Stage 3 v3) Snapshot h at beta_start (h_alpha) and reset h second time. "
                             "PFC at compare reads [h_alpha, h_beta, scratch_slots]. Forces clean "
                             "separation: alpha only via scratch_pad, beta only via fresh GRU read.")
    parser.add_argument("--pfc-at-write", action="store_true",
                        help="(Stage 3 v3-w) PFC also fires at write timesteps with [h, prior_scratch]. "
                             "Shared weights with compare-time PFC. Tests whether PFC reasoning at write "
                             "improves symbolic emergence (Diester-Nieder PFC binding hypothesis).")
    parser.add_argument("--pfc-compare-drop-raw", action="store_true",
                        help="(Day 5) When --pfc-split-ab, drop PFC's direct access to raw scratch slots at "
                             "compare time. PFC sees only [h_alpha, h_beta]. Forces alpha info to route through "
                             "CNN-GRU-h_alpha path. Cleaner emergence pressure but smaller PFC seq length (2).")
    parser.add_argument("--compare-via-raw-scratch", action="store_true",
                        help="(E_F) Skip [h_alpha, h_beta] compare and use compare_head(concat(scratch_alpha, "
                             "scratch_beta)) instead. scratch_beta is generated by W internal GRU+write_head steps "
                             "on h_beta with zero inputs. Symmetrizes the alpha/beta encoding through quantize bottleneck.")
    parser.add_argument("--compare-mlp-hidden", type=int, default=None,
                        help="(gru/rglru only) If set, compare_head becomes 2-layer MLP "
                             "with this hidden width.")
    parser.add_argument("--inner-iter", type=int, default=1,
                        help="(gru only) >1: K fixed-point iterations per timestep "
                             "(attractor convergence).")
    parser.add_argument("--hidden-noise-std", type=float, default=0.0,
                        help="(gru only) Training-time Gaussian noise on hidden "
                             "(attractor basin selection).")
    parser.add_argument("--lateral-inhibition", action="store_true",
                        help="(gru only) Soft winner-take-all across hidden dims "
                             "(Gaussian-peak tuning specialization).")
    parser.add_argument("--lateral-alpha", type=float, default=2.0,
                        help="Softmax temperature for lateral inhibition.")
    parser.add_argument("--max-epochs-per-stage", type=str, default=None,
                        help="Override max_epochs per stage. Single int for "
                             "uniform, OR comma-sep list matching curriculum "
                             "length (e.g., '50,3000' for warmup vs grokking).")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Custom curriculum: comma-sep n_max values. "
                             "E.g. '15,100' = 2-stage (N=15 then N=100). "
                             "Overrides default 4-stage [5,15,40,100].")
    parser.add_argument("--curriculum-L", type=str, default=None,
                        help="(V7) Per-stage L override (parallel to --curriculum). "
                             "E.g. '60,200,300,400' for stages [5,30,60,90]. "
                             "Keeps spike density roughly constant across stages "
                             "(L scales with n_max). Each L_i must satisfy "
                             "L_i >= 2*n_max_i+1.")
    parser.add_argument("--target-accuracy", type=float, default=0.70,
                        help="Target val_acc for early-stop curriculum advance. "
                             "Set to 1.01 to DISABLE early-stop entirely "
                             "(use for grokking — train past convergence).")
    # 51_M6 (2026-05-01): weighted CE + noise anneal + scene equal_weight
    parser.add_argument("--ce-class-weights", type=str, default=None,
                        help="(loss=ce only) comma-sep 3 floats for [G, L, E] CE "
                             "weights. e.g., '0.83,0.83,1.67' to upweight EQUAL.")
    parser.add_argument("--scene-equal-weight", type=float, default=None,
                        help="Override SceneConfig.equal_weight (default 0.111 for "
                             "trio_wide). Set to 0.20 for gap uniform [0,4] sampling.")
    parser.add_argument("--task-e-pair-weight", type=float, default=None,
                        help="(V22) Alias for --scene-equal-weight: probability "
                             "of E pair (δ=0) sampling. e.g. 0.25 for V22.")
    parser.add_argument("--write-noise-anneal", type=str, default=None,
                        help="Anneal write_noise_std over training. Format: "
                             "'start,end,start_pct,end_pct'. e.g., '0.1,0.02,0.3,0.7' "
                             "= keep at 0.1 until 30%% epochs, linear anneal to 0.02 "
                             "by 70%%, hold 0.02 after.")
    args = parser.parse_args()

    # V35 (5/27 pivot) CLI guards
    if args.use_v35:
        if args.use_v29 or args.use_v33:
            parser.error("--use-v35 is mutually exclusive with --use-v29 and --use-v33")
        if args.quantize_levels != 3:
            parser.error("--use-v35 requires --quantize-levels 3 (q=3 alphabet)")
        if args.quantize_range != "unit":
            parser.error("--use-v35 requires --quantize-range unit (same-medium [0,1])")
        if not args.dual_pathway:
            parser.error("--use-v35 requires --dual-pathway (CNN encoder)")
        if args.task != "successor_prediction":
            parser.error("--use-v35 currently only supports --task successor_prediction")
    if args.v35_layer_a_detect in ("food_pattern", "learned") and not args.v35_threshold_layer_a:
        parser.error(f"--v35-layer-a-detect {args.v35_layer_a_detect} requires --v35-threshold-layer-a")
    if args.v35_readback_direct and not args.v35_threshold_layer_a:
        parser.error("--v35-readback-direct requires --v35-threshold-layer-a")
    if args.v35_local_pc and not args.use_v35:
        parser.error("--v35-local-pc requires --use-v35")
    if args.v35_ip_enable and not args.v35_local_pc:
        parser.error("--v35-ip-enable requires --v35-local-pc")
    if args.v35_ip_enable and args.v35_threshold_layer_a:
        parser.error(
            "--v35-ip-enable is incompatible with --v35-threshold-layer-a: the "
            "threshold-LA path stores no Layer A capture, so r_obs pins to 0 and the "
            "IP controller diverges. Use the HH Layer A path for the IP pilot."
        )
    if args.v35_three_factor and not args.v35_local_pc:
        parser.error("--v35-three-factor requires --v35-local-pc")

    # V34 / supervised-commit CLI dependency guards — fail fast at arg-parse time
    # rather than at batch-1 runtime (saves pod startup waste on misconfigured runs).
    if args.unsupervised_symbol_emergence:
        # Full V34 unsupervised stack requires predictor + same-medium.
        if not args.predict_coding_enable:
            parser.error(
                "--unsupervised-symbol-emergence requires --predict-coding-enable "
                "(predictor must be built for L_predict to have a target)."
            )
        if not args.v33_same_medium:
            parser.error(
                "--unsupervised-symbol-emergence requires --v33-same-medium "
                "(closed-loop readback substrate scan is the source of h_readback_end)."
            )
    # commit / raw_l2 aux losses (used in both unsupervised AND supervised paths)
    # have nearest-level formula hardcoded for q=3 unit. Any nonzero lambda must
    # satisfy the alphabet contract, regardless of mode.
    aux_needs_q3 = (
        args.unsupervised_symbol_emergence
        or args.commit_lambda > 0
        or args.raw_l2_reg_lambda > 0
        or args.diversity_lambda > 0
    )
    if aux_needs_q3:
        if args.quantize_levels != 3 or args.quantize_range != "unit":
            parser.error(
                f"--commit-lambda / --raw-l2-reg-lambda / --diversity-lambda / "
                f"--unsupervised-symbol-emergence require --quantize-levels 3 "
                f"--quantize-range unit (V34 aux loss formulas hardcoded for q=3 unit "
                f"alphabet {{0, 0.5, 1.0}}); got levels={args.quantize_levels}, "
                f"range={args.quantize_range}."
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    if args.task == "discrim":
        scene_cfg = SceneConfig.discrimination_preset(
            L=args.scene_L, complex_world=args.complex_world
        )
    elif args.task == "trio":
        scene_cfg = SceneConfig.trio_preset(
            L=args.scene_L, complex_world=args.complex_world
        )
    elif args.task == "trio_wide":
        scene_cfg = SceneConfig.trio_wide_preset(
            L=args.scene_L, complex_world=args.complex_world,
            gap_max=args.trio_wide_gap_max,
        )
    elif args.task == "trio_wide_extended":
        scene_cfg = SceneConfig.trio_wide_extended_preset(
            L=args.scene_L, complex_world=args.complex_world,
            gap_max_near=args.trio_wide_gap_max,
        )
    elif args.task == "successor_prediction":
        if args.beta_range_preset == "balanced":
            # V35 PSS leak-kill (spec 2026-06-05): EQUAL+NEAR+SCALE+FAR mixture.
            equal_w = args.scene_equal_weight if args.scene_equal_weight is not None else 0.10
            near_w = 0.30
            scale_w = args.scale_weight
            far_w = 1.0 - equal_w - near_w - scale_w
            scene_cfg = SceneConfig.successor_balanced_preset(
                L=args.scene_L, complex_world=args.complex_world,
                k_max=args.successor_task_k_max,
                equal_weight=equal_w, near_weight=near_w,
                scale_weight=scale_w, far_weight=far_w,
            )
            scene_cfg.successor_far_bands = args.successor_far_bands
        else:
            scene_cfg = SceneConfig.successor_prediction_preset(
                L=args.scene_L, complex_world=args.complex_world,
                k_max=args.successor_task_k_max,
            )
            if args.successor_bidirectional:
                # V27: relax positivity constraint to break N_α-boundary lazy solution
                scene_cfg.successor_only_positive = False
    elif args.task == "binary":
        scene_cfg = SceneConfig.binary_balanced_preset(
            L=args.scene_L, complex_world=args.complex_world,
            gap_max=args.trio_wide_gap_max,
        )
    else:
        scene_cfg = SceneConfig(L=args.scene_L, complex_world=args.complex_world)
    scene_cfg.alpha_distribution = args.alpha_dist
    scene_cfg.clean_world = args.scene_clean_world   # 2026-06-17: decoy-free world (counting-de-scaffold test)
    # 51_M6: override equal_weight (default 0.111 → set 0.20 for gap-uniform [0,4])
    # V22: --task-e-pair-weight is an alias for --scene-equal-weight.
    e_pair_override = args.scene_equal_weight if args.scene_equal_weight is not None else args.task_e_pair_weight
    if e_pair_override is not None and getattr(scene_cfg, "beta_range_preset", "successor") != "balanced":
        scene_cfg.equal_weight = e_pair_override
        scene_cfg.near_weight = 1.0 - e_pair_override - scene_cfg.far_weight
    # args.K is now str (single int or comma-list); apply single-int case here so
    # scene_cfg.K reflects intent for run_start logging. Per-stage list is parsed
    # later (after curriculum construction) and passed to train() as per_stage_K.
    if args.K is not None and "," not in args.K:
        scene_cfg.K = int(args.K)

    # V35 PSS leak-kill guard (spec 2026-06-05 §7.1).
    if args.v35_pss_readback_beta:
        if args.task != "successor_prediction":
            parser.error("--v35-pss-readback-beta requires --task successor_prediction")
        if args.v35_threshold_layer_a and not args.v35_readback_direct:
            parser.error("--v35-pss-readback-beta on the threshold-Layer-A path requires "
                         "--v35-readback-direct (hard-threshold beta readback blocks the write-head gradient)")

    # Loss-type / use_gap_head consistency check.
    if args.loss_type in ("gap_mse4", "rl_top1", "rl_perblock") and not args.use_gap_head:
        print(
            f"[warn] --loss-type {args.loss_type} expects --use-gap-head; enabling implicitly.",
            file=sys.stderr,
        )
        args.use_gap_head = True

    if args.arch == "transformer":
        agent_cfg = TransformerAgentConfig(
            d_model=args.d_model,
            use_event_gate=args.event_gate,
            quantize_levels=args.quantize_levels,
            quantize_range=args.quantize_range,
        )
    elif args.arch == "attention":
        if args.event_gate:
            print("--event-gate is transformer-only; ignoring", file=sys.stderr)
        agent_cfg = AttentionAgentConfig(
            d_model=args.d_model,
            quantize_levels=args.quantize_levels,
            quantize_range=args.quantize_range,
        )
    elif args.arch == "rglru":
        if args.event_gate:
            print("--event-gate is transformer-only; ignoring", file=sys.stderr)
        agent_cfg = RGLRUAgentConfig(
            d_model=args.d_model,
            quantize_levels=args.quantize_levels,
            quantize_range=args.quantize_range,
            scratch_attn=args.scratch_attn,
            compare_mlp_hidden=args.compare_mlp_hidden,
        )
    elif args.arch == "mamba":
        if not _HAS_MAMBA_AGENT:
            raise ImportError("mamba_ssm not available — install on Linux+CUDA pod")
        # Regression mode: compare_head outputs single scalar (channel 0 used).
        # Other channels exist (compare_classes=3) but unused; loss/eval read [..., 0].
        agent_cfg = MambaAgentConfig(
            d_model=args.d_model,
            quantize_levels=args.quantize_levels,
            quantize_range=args.quantize_range,
            read_cnn_kernel=args.read_cnn_kernel,
            mamba_layers=args.mamba_layers,
            d_state=args.mamba_d_state,
            compare_mlp_hidden=args.compare_mlp_hidden,
            pfc_layers=args.pfc_layers,
            pfc_heads=args.pfc_heads,
            pfc_iter=args.pfc_iter,
            pfc_at_write=args.pfc_at_write,
            pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
            write_noise_std=args.write_noise_std,
            use_gap_head=args.use_gap_head,
            world_pred_t_pred=args.world_pred_t_pred,
            beta_pred_t_pred=args.beta_pred_t_pred,
            compare_pred_enabled=args.compare_pred,
            predictive_coding=args.predictive_coding,
            rs_pred_t_pred=args.rs_pred_t_pred,
            scratch_self_pred_enabled=args.scratch_self_pred,
            cf_beta_pred_t_pred=args.cf_beta_pred_t_pred,
            review_iter=args.review_iter,
            review_layers=args.review_layers,
            review_heads=args.review_heads,
            vq_enabled=args.vq_bottleneck,
            vq_codebook_size=args.vq_codebook_size,
            vq_d_z=args.vq_d_z,
            vq_commitment=args.vq_commitment,
            dual_loss_enabled=args.dual_loss,
            dual_pred_t_p=args.dual_pred_t_p,
            aux_detach_targets=args.aux_detach,
            vicreg_lambda=args.vicreg_lambda,
            vicreg_gamma=args.vicreg_gamma,
            pc_explicit_lambda=args.pc_explicit_lambda,
            simple_comparator=args.simple_comparator,
            cycle_loss_lambda=args.cycle_loss_lambda,
            pfc_variant=args.pfc_variant,
            pfc_mlp_hidden=args.pfc_mlp_hidden,
            dual_pathway=args.dual_pathway or args.v7_mode,
            dp_kernels=tuple(int(k) for k in args.dp_kernels.split(",")),
            dp_channels_per_scale=args.dp_channels_per_scale,
            dp_detach_to_mamba=args.dp_detach_to_mamba,
            cnn_pfc_only=args.cnn_pfc_only and not args.v7_mode,
            dp_scratch_skip_mamba=False if args.v7_mode else True,
            dp_t_target=args.dp_t_target,
            rbf_compare_head=args.rbf_compare_head,
            rbf_init_sigma=args.rbf_init_sigma,
            kwta_k=args.kwta_k,
            cnn_stride_no_pool=args.cnn_stride_no_pool,
            pfc_see_cnn_seq=args.pfc_see_cnn_seq,
            if_write_gate=args.if_write_gate,
            if_init_threshold=args.if_init_threshold,
            if_init_reset=args.if_init_reset,
            if_fire_tau=args.if_fire_tau,
            if_use_mamba_state=args.if_use_mamba_state,
            write_head_ln=args.write_head_ln,
            write_head_clip=args.write_head_clip,
            cross_attn_write=args.cross_attn_write,
            cross_attn_write_heads=args.cross_attn_write_heads,
            gumbel_write_head=args.gumbel_write_head,
            gumbel_tau=args.gumbel_tau,
            recurrent_latent_pfc=args.recurrent_latent_pfc,
            write_query_token=args.write_query_token,
            multi_modular_aux_moduli=tuple(
                int(x) for x in args.multi_modular_aux_moduli.split(",") if x.strip()
            ) if args.multi_modular_aux_moduli else (),
            use_lru=args.use_lru,
            lru_d_state=args.lru_d_state,
            lru_init_r_min=args.lru_init_r_min,
            lru_init_r_max=args.lru_init_r_max,
            lru_event_gated=args.lru_event_gated,
            gate_sharpness_init=args.gate_sharpness_init,
            detection_aux_weight=args.detection_aux_weight,
            use_oracle_spike_gate=args.use_oracle_spike_gate,
            scratch_mod_aux_moduli=tuple(
                int(x) for x in args.scratch_mod_aux_moduli.split(",") if x.strip()
            ) if args.scratch_mod_aux_moduli else (),
            successor_hidden=args.successor_hidden if args.successor_loss_weight > 0 else 0,
            successor_predict_k_max=(
                balanced_n_classes(args.successor_task_k_max, args.successor_far_bands)
                if args.beta_range_preset == "balanced"
                else (2 * args.successor_task_k_max if args.successor_bidirectional else args.successor_task_k_max)
            ) if args.task == "successor_prediction" else 0,
            successor_predict_input_level=args.successor_predict_input_level,
            write_head_type=args.write_head_type,
            write_head_hidden=args.write_head_hidden,
            write_head_pending_marker=args.write_head_pending_marker,
            cot_pfc_n_chunks=args.cot_pfc_n_chunks,
            cot_pfc_cnn_kernels=tuple(int(k) for k in args.cot_pfc_cnn_kernels.split(",")) if args.cot_pfc_n_chunks > 0 else (5, 51),
            # V29 fields
            use_v29=args.use_v29,
            # V33 fields
            use_v33=args.use_v33,
            v33_d_model=args.v33_d_model,
            v33_n_layers=args.v33_n_layers,
            v33_v_perturbation_scale=args.v33_v_perturbation_scale,
            v33_ortho_lambda=args.v33_ortho_lambda,
            v33_input_to_gate=args.v33_input_to_gate,
            v33_same_medium=args.v33_same_medium,
            predict_coding_enable=args.predict_coding_enable,
            unsupervised_mode=args.unsupervised_symbol_emergence,
            predict_coding_lambda=args.predict_coding_lambda,
            commit_lambda=args.commit_lambda,
            diversity_lambda=args.diversity_lambda,
            raw_l2_reg_lambda=args.raw_l2_reg_lambda,
            v33_vicreg_lambda_var=args.v33_vicreg_lambda_var,
            v33_vicreg_lambda_cov=args.v33_vicreg_lambda_cov,
            v33_vicreg_gamma=args.v33_vicreg_gamma,
            # V35 fields
            use_v35=args.use_v35,
            v35_n_recursions=args.v35_n_recursions,
            v35_fire_temp=args.v35_fire_temp,
            v35_write_d_attn=args.v35_write_d_attn,
            v35_write_n_heads=args.v35_write_n_heads,
            v35_compare_d_attn=args.v35_compare_d_attn,
            v35_compare_n_heads=args.v35_compare_n_heads,
            v35_compare_n_layers=args.v35_compare_n_layers,
            v35_compare_concat=args.v35_compare_concat,
            v35_threshold_layer_a=args.v35_threshold_layer_a,
            v35_layer_a_threshold=args.v35_layer_a_threshold,
            v35_layer_a_detect=args.v35_layer_a_detect,
            v35_layer_a_center_thresh=args.v35_layer_a_center_thresh,
            v35_layer_a_neighbor_thresh=args.v35_layer_a_neighbor_thresh,
            v35_readback_direct=args.v35_readback_direct,
            v35_pss_readback_beta=args.v35_pss_readback_beta,
            v35_local_pc=args.v35_local_pc,
            v35_pc_lr=args.v35_pc_lr,
            v35_theta_a=args.v35_theta_a,
            v35_theta_b=args.v35_theta_b,
            v35_theta_shared_init=args.v35_theta_shared_init,
            v35_theta_in_sgd=args.v35_theta_in_sgd,
            v35_log_g_leak_a=args.v35_log_g_leak_a,
            v35_warm_b_leak_a=args.v35_warm_b_leak_a,
            v35_b_leak_b_init=args.v35_b_leak_b_init,
            v35_log_g_b=args.v35_log_g_b,
            v35_recon_aux_lambda=args.v35_recon_aux_lambda,
            v35_cheap_arith_aux_lambda=args.v35_cheap_arith_aux_lambda,
            v35_count_consist_lambda=args.v35_count_consist_lambda,
            v35_detector_init=args.v35_detector_init,
            v35_freeze_detector=args.v35_freeze_detector,
            v35_freeze_layer_b_bleak=args.v35_freeze_layer_b_bleak,
            v35_freeze_layer_a_bleak=args.v35_freeze_layer_a_bleak,
            v35_write_per_cell=args.v35_write_per_cell,
            v35_pc_h_upstream=args.v35_pc_h_upstream,
            v35_ip_enable=args.v35_ip_enable,
            v35_ip_lr=args.v35_ip_lr,
            v35_ip_target=args.v35_ip_target,
            v35_ip_bias_clamp=args.v35_ip_bias_clamp,
            v35_pc_weight_decay=args.v35_pc_weight_decay,
            v35_ip_leak=args.v35_ip_leak,
            v35_three_factor=args.v35_three_factor,
            v35_tf_baseline_ema=args.v35_tf_baseline_ema,
            v35_pc_weight_clip=args.v35_pc_weight_clip,
            v35_state_discretize=args.v35_state_discretize,
            v29_encoder_n_neurons=args.v29_encoder_n_neurons,
            v29_write_n_neurons=args.v29_write_n_neurons,
            v29_d_id=args.v29_d_id,
            v29_d_pe=args.v29_d_pe,
            v29_pfc_layers=args.v29_pfc_layers,
            v29_pfc_heads=args.v29_pfc_heads,
            v29_pfc_passes=args.v29_pfc_passes,
            v29_threshold=args.v29_threshold,
            v29_surrogate_alpha=args.v29_surrogate_alpha,
            v29_omega_enc_min=args.v29_omega_enc_min,
            v29_omega_enc_max=args.v29_omega_enc_max,
            v29_omega_write_min=args.v29_omega_write_min,
            v29_omega_write_max=args.v29_omega_write_max,
            v29_b_init_min=args.v29_b_init_min,
            v29_b_init_max=args.v29_b_init_max,
            v29_time_max_init=args.v29_time_max_init,
            v29_soft_reset=args.v29_soft_reset,
            v29_t_write=args.v29_t_write,
            v29_max_events=args.v29_max_events,
            v29_query_orth_lambda=args.v29_query_orth_lambda,
            v29_omega_clamp_decay=args.v29_omega_clamp_decay,
            v29_firing_rate_lambda=args.v29_firing_rate_lambda,
            v29_target_firing_rate=args.v29_target_firing_rate,
            lru_decoder_d_state=args.lru_decoder_d_state,
            sin_omega_init=tuple(float(o) for o in args.sin_omega_init.split(",")),
            sin_init_std=args.sin_init_std,
            sin_predict_hidden=args.sin_predict_hidden,
            pfc_recurrent_n_iter=args.pfc_n_iterations if args.pfc_recurrent else 0,
            pfc_recurrent_cnn_scratch_kernel=args.pfc_recurrent_cnn_scratch_kernel,
            pfc_recurrent_cnn_scratch_channels=args.pfc_recurrent_cnn_scratch_channels,
            n_class_max_n=args.n_class_max_n if args.n_class_ce_weight > 0 else 0,
            n_class_head_hidden=args.n_class_head_hidden,
            recurrent_latent_state_tokens=args.recurrent_latent_state_tokens,
            recurrent_latent_sparse_k=args.recurrent_latent_sparse_k,
            recurrent_latent_steps=args.recurrent_latent_steps,
            ar_write_head=args.ar_write_head,
            ar_write_embed_dim=args.ar_write_embed_dim,
            ar_write_prev_dropout=args.ar_write_prev_dropout,
            ar_write_tau_init=args.ar_write_tau_init,
            ar_write_tau_final=args.ar_write_tau_final,
            ar_write_tau_warmup=args.ar_write_tau_warmup,
        )
    else:
        if args.event_gate:
            print("--event-gate is only supported for --arch transformer; ignoring",
                  file=sys.stderr)
        agent_cfg = AgentConfig(
            d_model=args.d_model,
            quantize_levels=args.quantize_levels,
            quantize_range=args.quantize_range,
            scratch_attn=args.scratch_attn,
            gaussian_attn=args.gaussian_attn,
            compare_mlp_hidden=args.compare_mlp_hidden,
            inner_iter=args.inner_iter,
            hidden_noise_std=args.hidden_noise_std,
            lateral_inhibition=args.lateral_inhibition,
            lateral_alpha=args.lateral_alpha,
            cell_type=args.cell_type,
            lif_decay=args.lif_decay,
            lif_threshold=args.lif_threshold,
            rcnn_iter=args.rcnn_iter,
            read_cnn_kernel=args.read_cnn_kernel,
            read_cnn_layers=args.read_cnn_layers,
            pfc_layers=args.pfc_layers,
            pfc_heads=args.pfc_heads,
            pfc_iter=args.pfc_iter,
            reset_h_at_compare_block=args.reset_h_at_compare,
            pfc_split_ab=args.pfc_split_ab,
            pfc_at_write=args.pfc_at_write,
            pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
            compare_via_raw_scratch=args.compare_via_raw_scratch,
            write_noise_std=args.write_noise_std,
            use_gap_head=args.use_gap_head,
        )
    train_cfg = TrainConfig()
    if args.agent_seed is not None:
        train_cfg.agent_seed = args.agent_seed
    if args.data_seed is not None:
        train_cfg.data_seed = args.data_seed

    if args.smoke:
        curriculum = TrainConfig.smoke_curriculum()
        tag = "smoke"
    else:
        if args.curriculum is not None:
            n_maxes = [int(x) for x in args.curriculum.split(",")]
            tgt = args.target_accuracy
            curriculum = [
                CurriculumStage(f"stage{i+1}_n{n}", n_max=n, target_accuracy=tgt)
                for i, n in enumerate(n_maxes)
            ]
        else:
            curriculum = TrainConfig.default_curriculum()
        if args.stage4_n_max is not None:
            curriculum[-1].n_max = args.stage4_n_max
        # V7: per-stage L override via parallel list
        if args.curriculum_L is not None:
            L_list = [int(x) for x in args.curriculum_L.split(",")]
            if len(L_list) != len(curriculum):
                raise ValueError(
                    f"--curriculum-L list len {len(L_list)} != "
                    f"curriculum len {len(curriculum)}"
                )
            for s, L_val in zip(curriculum, L_list):
                s.L = L_val
        # 51_M5: per-stage max_epochs via comma-separated list
        # e.g., --max-epochs-per-stage 50,3000 for warmup vs grokking-long
        if args.max_epochs_per_stage is not None:
            mep_str = str(args.max_epochs_per_stage)
            if "," in mep_str:
                meps = [int(x) for x in mep_str.split(",")]
                if len(meps) != len(curriculum):
                    raise ValueError(
                        f"--max-epochs-per-stage list len {len(meps)} != "
                        f"curriculum len {len(curriculum)}"
                    )
                for s, mep in zip(curriculum, meps):
                    s.max_epochs = mep
            else:
                for s in curriculum:
                    s.max_epochs = int(mep_str)
        tag = "full"

    # Parse per-stage K (A3 K-curriculum). Single int = uniform across stages;
    # comma-list must match curriculum length. None = use scene_cfg.K throughout.
    per_stage_K: Optional[List[int]] = None
    if args.K is not None and "," in args.K:
        per_stage_K = [int(x) for x in args.K.split(",")]
        if len(per_stage_K) != len(curriculum):
            raise ValueError(
                f"--K list length {len(per_stage_K)} does not match curriculum length "
                f"{len(curriculum)}. Either pass single int (e.g. --K 12) or list "
                f"matching curriculum stages exactly."
            )
    arch_tag = args.arch
    if args.arch == "transformer" and args.event_gate:
        arch_tag = "txfm_gated"
    elif args.arch == "transformer":
        arch_tag = "txfm"
    elif args.arch == "attention":
        arch_tag = "attn"
    elif args.arch == "rglru":
        arch_tag = "rglru"
    extras = []
    if args.complex_world:
        extras.append("cw")
    if args.quantize_levels is not None:
        extras.append(f"q{args.quantize_levels}{args.quantize_range[0]}")  # e.g. q6u
    if args.scratch_attn:
        extras.append("attn")
    extras_tag = ("_" + "_".join(extras)) if extras else ""
    tag = f"{tag}_{args.task}_{arch_tag}{extras_tag}"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or f"logs/{timestamp}_{tag}")
    ckpt_dir = Path(args.checkpoint_dir or f"data/checkpoints/{timestamp}_{tag}")

    train(scene_cfg, agent_cfg, train_cfg, curriculum, device, log_dir, ckpt_dir,
          loss_type=args.loss_type,
          rl_entropy_coef=args.rl_entropy_coef,
          rl_baseline_ema=args.rl_baseline_ema,
          init_from_ckpt=args.init_from_ckpt,
          init_from_ckpt_nonstrict=args.init_from_ckpt_nonstrict,
          rl_reward_shape=args.rl_reward_shape,
          rl_adv_normalize=args.rl_adv_normalize,
          rl_perblock_reward=args.rl_perblock_reward,
          per_stage_K=per_stage_K,
          optimizer_name=args.optimizer,
          weight_decay=args.weight_decay,
          write_head_wd=args.write_head_wd,
          sgd_momentum=args.sgd_momentum,
          world_pred_lambda=args.world_pred_lambda,
          beta_pred_lambda=args.beta_pred_lambda,
          compare_pred_lambda=args.compare_pred_lambda,
          rs_pred_lambda=args.rs_pred_lambda,
          scratch_self_pred_lambda=args.scratch_self_pred_lambda,
          cf_beta_pred_lambda=args.cf_beta_pred_lambda,
          vq_commit_lambda=args.vq_commit_lambda,
          dual_world_lambda=args.dual_world_lambda,
          dual_sp_lambda=args.dual_sp_lambda,
          vicreg_lambda=args.vicreg_lambda,
          pc_explicit_lambda=args.pc_explicit_lambda,
          cycle_loss_lambda=args.cycle_loss_lambda,
          ar_entropy_bonus=args.ar_entropy_bonus,
          multi_modular_aux_lambda=args.multi_modular_aux_lambda,
          multi_modular_aux_moduli=tuple(
              int(x) for x in args.multi_modular_aux_moduli.split(",") if x.strip()
          ) if args.multi_modular_aux_moduli else (),
          multi_modular_aux_pool=tuple(
              int(x) for x in args.multi_modular_aux_pool.split(",") if x.strip()
          ) if args.multi_modular_aux_pool else (),
          detection_aux_weight=args.detection_aux_weight,
          scratch_mod_aux_weight=args.scratch_mod_aux_weight,
          scratch_mod_aux_moduli=tuple(
              int(x) for x in args.scratch_mod_aux_moduli.split(",") if x.strip()
          ) if args.scratch_mod_aux_moduli else (),
          e_pair_consistency_weight=args.e_pair_consistency_weight,
          successor_loss_weight=args.successor_loss_weight,
          successor_detached=args.successor_detached,
          h_pair_consistency_weight=args.h_pair_consistency_weight,
          n_class_ce_weight=args.n_class_ce_weight,
          delta_h_diagnostic_every=args.delta_h_diagnostic_every,
          ce_class_weights=(
              torch.tensor([float(x) for x in args.ce_class_weights.split(",")])
              if args.ce_class_weights else None
          ),
          write_noise_anneal=(
              tuple(float(x) for x in args.write_noise_anneal.split(","))
              if args.write_noise_anneal else None
          ),
          task=args.task)


if __name__ == "__main__":
    main()
