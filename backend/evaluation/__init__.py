"""Evaluation metrics for Stage 1 symbol emergence study."""
from backend.evaluation.collect import collect_notes_and_predictions
from backend.evaluation.compositional import compositional_probe_score
from backend.evaluation.extrapolation import extrapolation_accuracy
from backend.evaluation.mi_matrix import base_scanning_mi, mutual_information_matrix
from backend.evaluation.topo_sim import topographic_similarity
from backend.evaluation.weber import WeberCurve, compute_weber_curve

__all__ = [
    "collect_notes_and_predictions",
    "compositional_probe_score",
    "extrapolation_accuracy",
    "base_scanning_mi",
    "mutual_information_matrix",
    "topographic_similarity",
    "WeberCurve",
    "compute_weber_curve",
]
