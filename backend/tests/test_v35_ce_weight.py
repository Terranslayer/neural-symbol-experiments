# -*- coding: utf-8 -*-
"""TDD: V35 PSS leak-kill — inverse-frequency CE class weighting (spec 2026-06-05 §3.5).

The COARSE balanced label makes the two FAR classes (0, 12) the majority, so a
constant-predict-FAR scorer beats chance using no scratch and no beta (a class-prior
bypass). Inverse-frequency CE weighting removes that floor without distorting sampling.
"""
import torch
from backend.training.train_phase1 import inverse_freq_class_weights


def test_inverse_freq_upweights_minority():
    # class 0 x3 (majority), class 6 x1 (minority); n_classes=13
    kt = torch.tensor([0, 0, 0, 6])
    w = inverse_freq_class_weights(kt, n_classes=13)
    assert w.shape == (13,)
    assert w[6] > w[0]                                  # minority weighted higher
    # 1/freq normalized so present-class mean == 1: w0=0.5, w6=1.5
    assert abs(w[0].item() - 0.5) < 1e-5
    assert abs(w[6].item() - 1.5) < 1e-5
    present = torch.tensor([0, 6])
    assert abs(w[present].mean().item() - 1.0) < 1e-5
    assert w[1].item() == 0.0                           # absent class -> 0 (unused)


def test_inverse_freq_uniform_gives_ones():
    kt = torch.tensor([0, 1, 2, 3])
    w = inverse_freq_class_weights(kt, n_classes=4)
    assert torch.allclose(w, torch.ones(4), atol=1e-5)
