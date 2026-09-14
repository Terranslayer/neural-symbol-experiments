import torch
from backend.core.v35_heads import V35WriteAttention


def test_write_attention_shape_and_alphabet():
    head = V35WriteAttention(n_slots=5, d_attn=16, n_heads=2)
    residuals = torch.randn(4, 5)  # (B, n_recursions)
    raw, scratch = head(residuals)
    assert raw.shape == (4, 5)
    assert scratch.shape == (4, 5)
    # After quantize-STE, hard values are exactly {0.0, 0.5, 1.0}
    uniq = scratch.detach().unique()
    for v in uniq.tolist():
        assert min(abs(v - q) for q in (0.0, 0.5, 1.0)) < 1e-5


def test_write_attention_grad_through_ste():
    head = V35WriteAttention(n_slots=5, d_attn=16, n_heads=2)
    residuals = torch.randn(4, 5, requires_grad=True)
    raw, scratch = head(residuals)
    (scratch.sum()).backward()
    assert residuals.grad is not None
    assert residuals.grad.abs().sum().item() > 0


def test_compare_attention_shape():
    from backend.core.v35_heads import V35CompareAttention
    head = V35CompareAttention(n_slots=5, d_attn=16, n_heads=2, n_classes=10)
    readback = torch.randn(4, 5)
    beta = torch.randn(4, 5)
    logits = head(readback, beta)
    assert logits.shape == (4, 10)


def test_compare_attention_grad_flow():
    from backend.core.v35_heads import V35CompareAttention
    head = V35CompareAttention(n_slots=5, d_attn=16, n_heads=2, n_classes=10)
    readback = torch.randn(4, 5, requires_grad=True)
    beta = torch.randn(4, 5, requires_grad=True)
    logits = head(readback, beta)
    logits.sum().backward()
    assert readback.grad is not None and readback.grad.abs().sum().item() > 0
    assert beta.grad is not None and beta.grad.abs().sum().item() > 0


# --- Rung 2 (2026-06-07): position-indexed concat readout replacing mean-pool. ---
# Probe R1 showed the symmetric mean-pool is the dominant comparison wall (real
# codes: mean-pool 0.17-0.70 vs concat 0.92-1.00). pool="concat" keeps every
# (position, source) token separate before the classifier so place-value weights
# are learnable per slot.

def test_compare_attention_concat_shape_and_clsdim():
    from backend.core.v35_heads import V35CompareAttention
    head = V35CompareAttention(n_slots=5, d_attn=16, n_heads=2, n_classes=13, pool="concat")
    readback = torch.randn(4, 5)
    beta = torch.randn(4, 5)
    logits = head(readback, beta)
    assert logits.shape == (4, 13)
    # concat keeps all 2*n_slots tokens: classifier input dim = 2*n_slots*d_attn
    assert head.cls[0].in_features == 2 * 5 * 16


def test_compare_attention_mean_default_unchanged():
    # Default pool stays "mean" with the original classifier input dim = d_attn.
    from backend.core.v35_heads import V35CompareAttention
    head = V35CompareAttention(n_slots=5, d_attn=16, n_heads=2, n_classes=13)
    assert head.cls[0].in_features == 16
    logits = head(torch.randn(4, 5), torch.randn(4, 5))
    assert logits.shape == (4, 13)


def test_compare_attention_concat_grad_flow():
    from backend.core.v35_heads import V35CompareAttention
    head = V35CompareAttention(n_slots=5, d_attn=16, n_heads=2, n_classes=13, pool="concat")
    readback = torch.randn(4, 5, requires_grad=True)
    beta = torch.randn(4, 5, requires_grad=True)
    logits = head(readback, beta)
    logits.sum().backward()
    assert readback.grad is not None and readback.grad.abs().sum().item() > 0
    assert beta.grad is not None and beta.grad.abs().sum().item() > 0
