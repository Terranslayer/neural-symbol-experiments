# -*- coding: utf-8 -*-
"""
Topographic similarity (Lazaridou et al. 2018).

Measures correlation between pairwise distances in the note-space and
pairwise distances in N-space. High topo_sim means "similar N values
produce similar notes" — a necessary condition for systematic encoding,
but NOT sufficient for place-value (continuous magnitude encoding also
scores high).
"""
from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def topographic_similarity(
    n_alpha: np.ndarray,
    notes: np.ndarray,
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Args:
        n_alpha: (num_episodes,) integer N values
        notes: (num_episodes, W) float note vectors
        metric: distance metric for notes ("cosine" or "euclidean")

    Returns:
        {"topo_sim": float, "p_value": float}
    """
    if n_alpha.shape[0] < 2:
        return {"topo_sim": float("nan"), "p_value": float("nan")}

    dist_n = pdist(n_alpha.reshape(-1, 1).astype(np.float64), metric="euclidean")
    dist_notes = pdist(notes.astype(np.float64), metric=metric)

    # Handle degenerate cases (all notes identical, all N identical)
    if dist_notes.std() < 1e-12 or dist_n.std() < 1e-12:
        return {"topo_sim": 0.0, "p_value": 1.0}

    rho, pval = spearmanr(dist_n, dist_notes)
    return {"topo_sim": float(rho), "p_value": float(pval)}
