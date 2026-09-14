# -*- coding: utf-8 -*-
"""TDD: V35 eval-side label mapping must AGREE with the training-side label.

Bug (fbdd5e5): train_phase1 V35 branch switched to the COARSE balanced label
(`k_to_class`, EQUAL->center, dedicated FAR classes), but the eval copies
(`collect._k_target_from_metas`, `scripts/v35_inspect._k_to_class`) kept the
legacy bidirectional mapping (EQUAL->0 colliding with FAR-, fine k>0 off-by-one).
Result: ~30% of eval episodes were scored against a wrong ground-truth label, so
every V35 acc number (rollout_accuracy + inspect per-N acc) was corrupted.

This test pins the contract: under the balanced preset, the eval target mapper
reproduces the training `k_to_class` exactly across the full offset range.
"""
import torch

from backend.core.scene import SceneConfig
from backend.training.train_phase1 import k_to_class, balanced_n_classes
from backend.evaluation.collect import _k_target_from_metas


def _balanced_scene(k_max=5, far_bands=1):
    return SceneConfig.successor_balanced_preset(L=200, k_max=k_max)


def test_eval_mapper_matches_training_k_to_class_balanced():
    k_max, far_bands = 5, 1
    scene = _balanced_scene(k_max, far_bands)
    n_classes = balanced_n_classes(k_max, far_bands)  # 13
    device = torch.device("cpu")

    # Span the full offset range incl. EQUAL, fine, and both FAR tails.
    metas = []
    expected = []
    for N_alpha in range(1, 91):
        for beta in range(1, 91):
            metas.append({"N_alpha": N_alpha, "beta_counts": [beta]})
            expected.append(k_to_class(beta - N_alpha, k_max, far_bands))

    got = _k_target_from_metas(metas, scene, n_classes, device)
    assert got.tolist() == expected, (
        "eval label mapping diverged from training k_to_class under balanced preset"
    )


def test_eval_mapper_equal_is_center_not_far_negative():
    # k=0 (EQUAL) must be the center fine class (6), NOT class 0 (FAR-).
    scene = _balanced_scene()
    n_classes = balanced_n_classes(5, 1)
    device = torch.device("cpu")
    metas = [{"N_alpha": 10, "beta_counts": [10]}]  # k=0
    got = _k_target_from_metas(metas, scene, n_classes, device)
    assert got.tolist() == [6]


def test_eval_mapper_fine_positive_not_off_by_one():
    # k=+1 -> class 7 under COARSE (legacy bug gave 6).
    scene = _balanced_scene()
    n_classes = balanced_n_classes(5, 1)
    device = torch.device("cpu")
    metas = [{"N_alpha": 10, "beta_counts": [11]}]  # k=+1
    got = _k_target_from_metas(metas, scene, n_classes, device)
    assert got.tolist() == [7]
