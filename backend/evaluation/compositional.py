# -*- coding: utf-8 -*-
"""
Base-free compositional probe (the MAIN metric for symbol emergence).

Trains linear probes on progressively more note positions to predict N,
and reports the marginal gain each position contributes. If the agent
has compositional encoding (like place-value), each position carries
independent information → gain stays positive across positions.

If the agent does magnitude estimation (scalar encoding), later positions
are redundant with earlier ones → gain collapses to ~0 quickly.

This metric does NOT assume any base / digit structure — it just tests
whether positions carry independent information.
"""
from typing import Dict, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _probe_accuracy(
    features: np.ndarray,
    targets: np.ndarray,
    cv_folds: int = 5,
    alpha: float = 1.0,
    seed: int = 0,
) -> float:
    """
    Cross-validated R² of a linear probe predicting targets from features.
    Returns 0.0 if cv fails (e.g., too few samples).
    """
    if features.shape[0] < cv_folds * 2:
        return 0.0
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    r2_scores = []
    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        ss_res = float(((y_test - pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        r2_scores.append(r2)
    return float(np.mean(r2_scores))


def compositional_probe_score(
    n_alpha: np.ndarray,
    notes: np.ndarray,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Compute per-position probe accuracy and marginal gains.

    Args:
        n_alpha: (num_episodes,) integer N values
        notes: (num_episodes, W) float note vectors

    Returns a dict with:
        per_position_r2: list of W floats — cumulative R² using notes[:, :j+1]
        marginal_gains: list of W floats — R² gain adding position j
        total_r2: float — R² using all W positions
        composition_index: float — mean of positive marginal gains /
            max possible (1 - single_position_r2). Higher = more compositional.
    """
    num, W = notes.shape
    per_position_r2: List[float] = []
    for j in range(W):
        feats = notes[:, : j + 1]
        r2 = _probe_accuracy(feats, n_alpha.astype(np.float64), cv_folds, ridge_alpha, seed)
        per_position_r2.append(r2)

    marginal_gains = [per_position_r2[0]]
    for j in range(1, W):
        marginal_gains.append(per_position_r2[j] - per_position_r2[j - 1])

    # Composition index: how much of the "remaining explainable variance"
    # after the first note position does the rest of the notes explain?
    # If first position captures everything → index = 0 (no composition).
    # If each subsequent position adds independent info → index close to 1.
    remaining = max(1.0 - per_position_r2[0], 1e-6)
    rest_gain = per_position_r2[-1] - per_position_r2[0]
    composition_index = max(0.0, rest_gain) / remaining

    return {
        "per_position_r2": per_position_r2,
        "marginal_gains": marginal_gains,
        "total_r2": per_position_r2[-1],
        "composition_index": composition_index,
    }
