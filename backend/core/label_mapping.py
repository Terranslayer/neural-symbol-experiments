# -*- coding: utf-8 -*-
"""Single source of truth for successor offset k -> class index mapping.

Training (`train_phase1` V35/V24 branch) and eval (`collect`, `scripts/v35_inspect`)
MUST agree on how a signed offset k = N_beta - N_alpha becomes a class label.

History (the bug this module fixes): commit fbdd5e5 switched the training V35
branch to the COARSE *balanced* label (`k_to_class`: EQUAL->center class, dedicated
FAR classes) but left the eval copies on the legacy bidirectional mapping
(EQUAL->0 colliding with FAR-, fine k>0 off-by-one). ~30% of eval episodes were
then scored against a wrong ground-truth label, corrupting every V35 acc number
(rollout_accuracy + inspect per-N acc). Consolidating both sides here makes a
future divergence a test failure (see test_v35_eval_label_consistency.py).

Lives in backend.core (no deps on training/evaluation) so both layers can import
it without a circular import (train_phase1 imports evaluation.collect).
"""


def balanced_n_classes(k_max: int, far_bands: int = 1) -> int:
    """Number of relational classes for the V35 balanced preset (spec 2026-06-05 §3.4).

    2*k_max+1 fine signed-offset classes (k in [-k_max, +k_max]) plus far_bands
    dedicated FAR classes per direction. COARSE default (far_bands=1) => 2*k_max+3.
    """
    return 2 * k_max + 1 + 2 * far_bands


def k_to_class(k: int, k_max: int, far_bands: int = 1) -> int:
    """Map signed offset k = N_beta - N_alpha to a relational class (spec 2026-06-05 §3.4).

    COARSE (far_bands=1): FAR- -> 0; fine k in [-k_max, +k_max] -> 1..2k_max+1
    (class = k + k_max + 1, so EQUAL k=0 -> center class k_max+1); FAR+ -> 2k_max+2.
    FAR classes are sign-only (magnitude-agnostic) so the label never encodes integer
    N as a magnitude class (Zero-Prior). Fixes the old inline mapping where k==0
    collided with FAR-negative onto class 0 and far offsets clamped onto fine classes.
    """
    if far_bands != 1:
        raise NotImplementedError(
            "MULTI-BAND far labels (far_bands>=2) not yet implemented; "
            "ships COARSE (far_bands=1) per spec default."
        )
    if k < -k_max:
        return 0                       # FAR-
    if k > k_max:
        return 2 * k_max + 2           # FAR+
    return k + k_max + 1               # fine band (EQUAL -> center class k_max+1)


def k_to_class_legacy_bidir(k: int, n_classes: int) -> int:
    """Legacy bidirectional mapping for the (non-balanced) successor preset.

    k_max = n_classes // 2; k<0 -> [0, k_max), k>0 -> [k_max, n_classes), k==0 -> 0.
    Preserved verbatim for the legacy successor preset so its labels are unchanged;
    NOT used by the balanced preset (which has dedicated FAR classes + EQUAL center).
    """
    k_max_val = n_classes // 2
    if k < 0:
        return max(0, k + k_max_val)
    elif k > 0:
        return min(n_classes - 1, k + k_max_val - 1)
    return 0


def eval_k_to_class(k: int, scene_cfg, n_classes: int) -> int:
    """Eval-side class for offset k, matching the TRAINING label for this scene.

    Branches on the scene preset so eval reproduces exactly what the training loss
    used: balanced -> COARSE `k_to_class`; legacy bidirectional -> `k_to_class_legacy_bidir`;
    legacy unidirectional -> clamp(k-1).
    """
    balanced = getattr(scene_cfg, "beta_range_preset", "successor") == "balanced"
    if balanced:
        far_bands = getattr(scene_cfg, "successor_far_bands", 1)
        k_max_val = (n_classes - 1 - 2 * far_bands) // 2
        return k_to_class(k, k_max_val, far_bands)
    scene_bidir = not getattr(scene_cfg, "successor_only_positive", True)
    if scene_bidir:
        return k_to_class_legacy_bidir(k, n_classes)
    return max(0, min(n_classes - 1, k - 1))
