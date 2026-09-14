"""V35-PC three-factor task-modulated Hebbian rule tests (fix 3).

All tests run on the pod (project rule: feedback_tests_on_pod).
Spec: docs/design.md

The modulator M (per-sample signed advantage) multiplies the local Hebbian
contrib: delta_W ~ e * input * M, on both Layer A and (unfrozen) Layer B.
"""
import numpy as np
import torch
from backend.core.v35 import V35Substrate


def _batch_bump_input(bump_counts, L=60, d_input=1):
    """One sample per entry in bump_counts; sample i gets bump_counts[i] unit
    pulses spaced across L. Returns (B, L, d_input)."""
    B = len(bump_counts)
    x = torch.zeros(B, L, d_input)
    for i, n in enumerate(bump_counts):
        if n > 0:
            positions = np.linspace(0, L - 1, n, dtype=int)
            x[i, positions.tolist(), 0] = 1.0
    return x


def _mk_sub(seed=0):
    torch.manual_seed(seed)
    sub = V35Substrate(d_input=1, n_recursions=5, biological_init=True)
    sub.cascade.layer_A.B_leak.weight.data.fill_(0.2)
    # positive ip_bias so Layer A actually fires -> nonzero e and dW
    sub.cascade.layer_A.ip_bias.data.fill_(2.0)
    return sub


def _layerA_delta(seed, modulator, lr=0.3, weight_decay=0.0, x=None):
    """Run one capture + pc_step from a fresh sub; return (w0, w_after) for Layer A."""
    sub = _mk_sub(seed)
    if x is None:
        x = _batch_bump_input([8, 8])
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    w0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    sub.pc_step(lr=lr, ip_enable=True, ip_lr=0.0, r_target=4.0,
                weight_decay=weight_decay, modulator=modulator)
    return w0, sub.cascade.layer_A.B_leak.weight.detach().clone()


def test_modulator_sign_scales_update():
    """Constant M scales the Layer-A update by M relative to the unmodulated rule;
    negative M flips its sign."""
    x = _batch_bump_input([8, 8])
    w0_u, w_u = _layerA_delta(0, modulator=None, x=x)
    d_u = w_u - w0_u                                  # unmodulated Hebbian step
    assert d_u.abs().sum() > 1e-6, "need a nonzero unmodulated update to test scaling"

    w0_2, w_2 = _layerA_delta(0, modulator=torch.tensor([2.0, 2.0]), x=x)
    assert torch.allclose(w_2 - w0_2, 2.0 * d_u, atol=1e-6), "M=2 should give 2x update"

    w0_n, w_n = _layerA_delta(0, modulator=torch.tensor([-1.0, -1.0]), x=x)
    assert torch.allclose(w_n - w0_n, -1.0 * d_u, atol=1e-6), "M=-1 should flip update"


def test_per_sample_weighting():
    """Per-sample M weights each sample's contrib independently:
    contrib([1,0]) + contrib([0,1]) == contrib(None), and the two halves differ
    (distinct samples -> distinct per-sample contribs)."""
    x = _batch_bump_input([3, 9])  # distinct samples
    lr = 0.3
    w0, w_none = _layerA_delta(1, modulator=None, lr=lr, x=x)
    d_none = (w_none - w0) / lr
    _, w_10 = _layerA_delta(1, modulator=torch.tensor([1.0, 0.0]), lr=lr, x=x)
    _, w_01 = _layerA_delta(1, modulator=torch.tensor([0.0, 1.0]), lr=lr, x=x)
    d_10 = (w_10 - w0) / lr
    d_01 = (w_01 - w0) / lr
    assert torch.allclose(d_10 + d_01, d_none, atol=1e-6), (
        "per-sample contribs must sum to the full unmodulated contrib"
    )
    assert not torch.allclose(d_10, d_01, atol=1e-4), (
        "distinct samples should give distinct per-sample contribs"
    )


def test_layer_b_unfrozen_under_three_factor():
    """modulator provided -> Layer B B_leak updates (unfrozen) even under ip_enable;
    modulator=None under ip_enable -> Layer B stays frozen."""
    x = _batch_bump_input([8, 8])

    sub = _mk_sub(2)
    b0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    for _ in range(3):
        sub.clear_pc_history()
        sub(x, capture_for_pc=True)
        sub.pc_step(lr=0.3, ip_enable=True, ip_lr=0.05, r_target=4.0,
                    modulator=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(sub.cascade.layer_B.B_leak.weight, b0), (
        "three-factor mode must UNFREEZE and update Layer B"
    )

    sub2 = _mk_sub(2)
    b0_2 = sub2.cascade.layer_B.B_leak.weight.detach().clone()
    for _ in range(3):
        sub2.clear_pc_history()
        sub2(x, capture_for_pc=True)
        sub2.pc_step(lr=0.3, ip_enable=True, ip_lr=0.05, r_target=4.0, modulator=None)
    assert torch.allclose(sub2.cascade.layer_B.B_leak.weight, b0_2), (
        "modulator=None under ip_enable must keep Layer B frozen (pilot behavior)"
    )


def test_modulator_none_is_noop():
    """pc_step(modulator=None) reproduces the plain two-factor Layer-A update
    (w0 + lr*contrib) and leaves Layer B frozen under ip_enable."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(3)
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    wA0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    wB0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    sub.pc_step(lr=0.3, ip_enable=True, ip_lr=0.0, r_target=4.0, modulator=None)
    assert not torch.allclose(sub.cascade.layer_A.B_leak.weight, wA0), "Layer A should update"
    assert torch.allclose(sub.cascade.layer_B.B_leak.weight, wB0), "Layer B frozen (modulator=None, ip)"


def test_zero_modulator_only_weight_decay():
    """M = 0 vector -> Hebbian contrib is 0 for both layers -> only weight decay
    applies (w *= 1-wd) on Layer A; with wd=0 the weights are unchanged."""
    x = _batch_bump_input([8, 8])
    # wd = 0: zero modulator -> Layer A unchanged
    w0, w_after = _layerA_delta(4, modulator=torch.tensor([0.0, 0.0]), weight_decay=0.0, x=x)
    assert torch.allclose(w_after, w0, atol=1e-7), "M=0, wd=0 should leave Layer A unchanged"


def test_weight_clip_caps_norm_and_preserves_direction():
    """weight_clip renormalizes Layer A B_leak to the cap when exceeded, keeping
    direction. (lr=0 -> no Hebbian change -> isolate the clip.)"""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(8)
    # set a deterministically large weight (randn could draw tiny for a 1-elem tensor)
    big = torch.full_like(sub.cascade.layer_A.B_leak.weight, -30.0)
    sub.cascade.layer_A.B_leak.weight.data.copy_(big)
    unit_before = (big / big.norm()).clone()
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.0, r_target=4.0, weight_clip=5.0)
    w = sub.cascade.layer_A.B_leak.weight
    assert abs(float(w.norm()) - 5.0) < 1e-4, f"clip should cap norm to 5, got {float(w.norm())}"
    assert torch.allclose(w / w.norm(), unit_before, atol=1e-5), "clip must preserve direction"


def test_weight_clip_noop_under_threshold():
    """When ||w|| < clip, the clip does nothing (only Hebbian/decay applies)."""
    x = _batch_bump_input([8, 8])
    # run with clip far above the natural norm -> identical to no clip
    _, w_clip = _layerA_delta(9, modulator=torch.tensor([1.0, 1.0]), x=x)
    sub = _mk_sub(9)
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.3, ip_enable=True, ip_lr=0.0, r_target=4.0,
                modulator=torch.tensor([1.0, 1.0]), weight_clip=1000.0)
    # _layerA_delta used modulator but no clip; with clip=1000 (>> norm) result must match
    assert torch.allclose(sub.cascade.layer_A.B_leak.weight, w_clip, atol=1e-6), (
        "clip above norm must be a no-op"
    )


def test_weight_clip_default_off():
    """weight_clip=0 (default) -> no clipping; large weights stay large."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(10)
    sub.cascade.layer_A.B_leak.weight.data.mul_(50.0)  # norm ~ 50*0.2
    n0 = float(sub.cascade.layer_A.B_leak.weight.norm())
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.0, r_target=4.0, weight_clip=0.0)
    assert abs(float(sub.cascade.layer_A.B_leak.weight.norm()) - n0) < 1e-4, (
        "weight_clip=0 must not clip"
    )


def test_weight_clip_both_layers():
    """Under three-factor (layer B updating), an over-norm layer B B_leak is also
    clamped to weight_clip."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(11)
    sub.cascade.layer_B.B_leak.weight.data.fill_(40.0)  # |wB| = 40 > clip
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.0, ip_enable=True, ip_lr=0.0, r_target=4.0,
                modulator=torch.tensor([1.0, 1.0]), weight_clip=5.0)
    assert abs(float(sub.cascade.layer_B.B_leak.weight.norm()) - 5.0) < 1e-4, (
        "layer B B_leak should be clipped to 5 under three-factor"
    )


def test_weight_decay_both_layers_under_three_factor():
    """With M=0 (zero Hebbian) and weight_decay>0, BOTH Layer A and Layer B B_leak
    shrink by (1 - weight_decay)."""
    x = _batch_bump_input([8, 8])
    wd = 0.1
    sub = _mk_sub(5)
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    wA0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    wB0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    sub.pc_step(lr=0.3, ip_enable=True, ip_lr=0.0, r_target=4.0,
                weight_decay=wd, modulator=torch.tensor([0.0, 0.0]))
    assert torch.allclose(sub.cascade.layer_A.B_leak.weight, wA0 * (1 - wd), atol=1e-6), (
        "Layer A weight decay under three-factor"
    )
    assert torch.allclose(sub.cascade.layer_B.B_leak.weight, wB0 * (1 - wd), atol=1e-6), (
        "Layer B weight decay under three-factor"
    )


def test_ip_off_three_factor_updates_both_layers_no_ip_bias_change():
    """Cut-1 config (spec 2026-06-02): IP dropped, three-factor on. pc_step must
    update BOTH layers' B_leak and leave ip_bias untouched."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(20)
    wA0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    wB0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    ip0 = sub.cascade.layer_A.ip_bias.detach().clone()
    sub.clear_pc_history()
    sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.3, ip_enable=False, modulator=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(sub.cascade.layer_A.B_leak.weight, wA0), "Layer A updates"
    assert not torch.allclose(sub.cascade.layer_B.B_leak.weight, wB0), "Layer B updates (ip off)"
    assert torch.equal(sub.cascade.layer_A.ip_bias, ip0), "ip_bias untouched when ip_enable=False"


def test_freeze_layer_b_blocks_layer_b_update():
    """freeze_layer_b=True leaves layer_B.B_leak unchanged under three-factor; False updates it."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(30)
    b0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    sub.clear_pc_history(); sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.3, modulator=torch.tensor([1.0, 1.0]), freeze_layer_b=True)
    assert torch.equal(sub.cascade.layer_B.B_leak.weight, b0), "freeze_layer_b must leave layer_B unchanged"

    sub2 = _mk_sub(30)
    b0_2 = sub2.cascade.layer_B.B_leak.weight.detach().clone()
    sub2.clear_pc_history(); sub2(x, capture_for_pc=True)
    sub2.pc_step(lr=0.3, modulator=torch.tensor([1.0, 1.0]), freeze_layer_b=False)
    assert not torch.allclose(sub2.cascade.layer_B.B_leak.weight, b0_2), "without freeze, layer_B updates"


def test_freeze_layer_a_blocks_layer_a_update():
    """freeze_layer_a=True leaves layer_A.B_leak AND ip_bias unchanged; False updates B_leak."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(31)
    a0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    ip0 = sub.cascade.layer_A.ip_bias.detach().clone()
    sub.clear_pc_history(); sub(x, capture_for_pc=True)
    # Large ip_lr + r_target gap so ip_bias WOULD move a lot if not frozen.
    sub.pc_step(lr=0.3, ip_enable=True, ip_lr=0.5, r_target=20.0,
                modulator=torch.tensor([1.0, 1.0]), freeze_layer_a=True)
    assert torch.equal(sub.cascade.layer_A.B_leak.weight, a0), "freeze_layer_a must leave layer_A B_leak unchanged"
    assert torch.equal(sub.cascade.layer_A.ip_bias, ip0), "freeze_layer_a must leave ip_bias unchanged"

    sub2 = _mk_sub(31)
    a0_2 = sub2.cascade.layer_A.B_leak.weight.detach().clone()
    sub2.clear_pc_history(); sub2(x, capture_for_pc=True)
    sub2.pc_step(lr=0.3, ip_enable=True, ip_lr=0.5, r_target=20.0,
                 modulator=torch.tensor([1.0, 1.0]), freeze_layer_a=False)
    assert not torch.allclose(sub2.cascade.layer_A.B_leak.weight, a0_2), "without freeze, layer_A updates"


def test_freeze_both_layers_no_substrate_change():
    """freeze_layer_a + freeze_layer_b together: NO substrate B_leak changes (full frozen cascade)."""
    x = _batch_bump_input([8, 8])
    sub = _mk_sub(32)
    a0 = sub.cascade.layer_A.B_leak.weight.detach().clone()
    b0 = sub.cascade.layer_B.B_leak.weight.detach().clone()
    sub.clear_pc_history(); sub(x, capture_for_pc=True)
    sub.pc_step(lr=0.5, modulator=torch.tensor([1.0, 1.0]),
                freeze_layer_a=True, freeze_layer_b=True)
    assert torch.equal(sub.cascade.layer_A.B_leak.weight, a0), "layer_A frozen"
    assert torch.equal(sub.cascade.layer_B.B_leak.weight, b0), "layer_B frozen"
