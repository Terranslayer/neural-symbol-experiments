# -*- coding: utf-8 -*-
"""TDD: V35 PSS leak-kill — relational label mapping (spec 2026-06-05 §3.4).

COARSE class scheme (far_bands=1), n_classes = 2*k_max + 3:
  class 0              = FAR-  (alpha much greater; k < -k_max)
  classes 1..2k_max+1  = fine signed offset k in [-k_max, +k_max], class = k + k_max + 1
  class 2*k_max+2      = FAR+  (alpha much less; k > +k_max)

The bug this replaces (train_phase1.py:576-582): k==0 mapped to class 0, colliding
with FAR-negative; far offsets clamped onto the extreme fine classes (no dedicated
FAR classes); k_max derived from n_classes//2 instead of passed through.
"""
from backend.training.train_phase1 import k_to_class, balanced_n_classes


def test_balanced_n_classes_coarse():
    # 2*k_max + 1 fine + 2*far_bands far  => COARSE (far_bands=1): 11 + 2 = 13
    assert balanced_n_classes(k_max=5, far_bands=1) == 13
    assert balanced_n_classes(k_max=2, far_bands=1) == 7


def test_k_to_class_equal_is_own_center_class():
    # EQUAL (k=0) gets the CENTER fine class, NOT class 0 (which is FAR-).
    assert k_to_class(0, k_max=5) == 6


def test_k_to_class_fine_band():
    # fine signed offsets map to classes 1..2k_max+1, monotone in k
    assert k_to_class(-5, k_max=5) == 1
    assert k_to_class(-1, k_max=5) == 5
    assert k_to_class(1, k_max=5) == 7
    assert k_to_class(5, k_max=5) == 11


def test_k_to_class_far_dedicated_classes():
    # FAR- (k < -k_max) -> 0 ; FAR+ (k > +k_max) -> 2*k_max+2 (=12), magnitude-agnostic
    assert k_to_class(-6, k_max=5) == 0
    assert k_to_class(-30, k_max=5) == 0
    assert k_to_class(6, k_max=5) == 12
    assert k_to_class(90, k_max=5) == 12


def test_k_to_class_no_collisions_full_coverage():
    # over a wide k range, classes span EXACTLY {0..12}; EQUAL never collides with FAR-
    classes = {k_to_class(k, k_max=5) for k in range(-90, 91)}
    assert classes == set(range(13))
    assert k_to_class(0, k_max=5) != k_to_class(-6, k_max=5)
