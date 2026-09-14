"""V35 attention heads: write (residuals -> scratch) and compare
(readback + beta residuals -> k_offset logits).

Write attention is the learnable map psi: substrate_state -> codeword
that makes scratch a symbol system (rather than a deterministic state
exposure). Compare attention reads two 5-residual sets (one from
scratch-readback pass through cascade, one from beta-signal pass) and
predicts the successor-task k_offset.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from backend.core.agent import _quantize_ste


class V35WriteAttention(nn.Module):
    """Cross-attention from 5 learnable scratch-slot queries over 5 residual
    tokens (residual scalar + position embedding) -> 5 scratch values in
    alphabet {0, 0.5, 1.0} via sigmoid + quantize-STE.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_attn: int = 16,
        n_heads: int = 2,
        quantize_levels: int = 3,
    ):
        super().__init__()
        if d_attn % n_heads != 0:
            raise ValueError(
                f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})"
            )
        self.n_slots = n_slots
        self.d_attn = d_attn
        self.quantize_levels = quantize_levels

        # Learnable scratch-slot query embeddings (one per scratch cell)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, d_attn) * 0.1)
        # Position embeddings for residual tokens (one per cascade recursion)
        self.pos_emb = nn.Parameter(torch.randn(n_slots, d_attn) * 0.1)
        # Project scalar residual into d_attn space
        self.residual_proj = nn.Linear(1, d_attn)
        # Standard multi-head attention block
        self.attn = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=n_heads, batch_first=True
        )
        # Per-slot MLP to scalar pre-quantize logit
        self.out_mlp = nn.Sequential(
            nn.Linear(d_attn, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, 1),
        )

    def forward(
        self, residuals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        residuals: (B, n_slots) — 5 cascade residuals h_k(L)
        Returns:
          raw:     (B, n_slots) — pre-quantize logits (post-sigmoid is for grad)
          scratch: (B, n_slots) — quantized to alphabet {0, 0.5, 1.0} via STE
        """
        B = residuals.shape[0]
        # Build key/value tokens: residual scalar -> d_attn, add positional embed
        tok = self.residual_proj(residuals.unsqueeze(-1))   # (B, n_slots, d_attn)
        tok = tok + self.pos_emb.unsqueeze(0)
        # Build query tokens: replicate slot_queries across batch
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # (B, n_slots, d_attn)
        out, _ = self.attn(q, tok, tok, need_weights=False)
        raw = self.out_mlp(out).squeeze(-1)  # (B, n_slots)
        scratch = _quantize_ste(raw, levels=self.quantize_levels, output_range="unit")
        return raw, scratch


class V35WritePerCellReadout(nn.Module):
    """Per-cell write readout (route A, spec 2026-06-02): scratch cell k reads
    ONLY cascade residual k (recursion-k output) through its own tiny MLP, then
    sigmoid+quantize-STE. Structural decoupling — raw[:, k] depends mathematically
    only on residuals[:, k]. Drop-in for V35WriteAttention (same
    (residuals) -> (raw, scratch) signature).
    """

    def __init__(
        self,
        n_slots: int = 5,
        hidden: int = 8,
        quantize_levels: int = 3,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.hidden = hidden
        self.quantize_levels = quantize_levels
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_slots)
        ])

    def forward(
        self, residuals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        residuals: (B, n_slots) — 5 cascade residuals h_k(L)
        Returns:
          raw:     (B, n_slots) — pre-sigmoid logits, cell k = MLP_k(res_k)
          scratch: (B, n_slots) — quantized to alphabet {0, 0.5, 1.0} via STE
        """
        cols = [self.mlps[k](residuals[:, k:k + 1]) for k in range(self.n_slots)]
        raw = torch.cat(cols, dim=-1)                       # (B, n_slots)
        scratch = _quantize_ste(raw, levels=self.quantize_levels, output_range="unit")
        return raw, scratch


class V35CompareAttention(nn.Module):
    """Self-attention over 10 tokens (5 readback residuals + 5 beta residuals),
    each enriched with position + source embeddings, pooled to predict
    successor-task k_offset class logits.
    """

    def __init__(
        self,
        n_slots: int = 5,
        d_attn: int = 16,
        n_heads: int = 2,
        n_classes: int = 10,
        n_layers: int = 1,
        pool: str = "mean",
    ):
        super().__init__()
        if d_attn % n_heads != 0:
            raise ValueError(
                f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})"
            )
        if pool not in ("mean", "concat"):
            raise ValueError(f"pool must be 'mean' or 'concat', got {pool!r}")
        self.n_slots = n_slots
        self.d_attn = d_attn
        self.pool = pool
        self.residual_proj = nn.Linear(1, d_attn)
        self.pos_emb = nn.Parameter(torch.randn(n_slots, d_attn) * 0.1)
        self.source_emb = nn.Parameter(torch.randn(2, d_attn) * 0.1)  # 0=readback, 1=beta
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_attn, nhead=n_heads, dim_feedforward=d_attn * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # pool="mean": permutation-invariant average (original; the symmetric
        # comparison wall — R1 probe 2026-06-07). pool="concat": keep every
        # (position, source) token separate -> position-indexed readout so the
        # classifier can learn per-slot place weights (the C3 escape, allpair
        # 0.92-1.00 on real codes vs 0.17-0.70 for mean-pool).
        cls_in = (2 * n_slots * d_attn) if pool == "concat" else d_attn
        cls_hidden = (4 * d_attn) if pool == "concat" else d_attn
        self.cls = nn.Sequential(
            nn.Linear(cls_in, cls_hidden),
            nn.GELU(),
            nn.Linear(cls_hidden, n_classes),
        )

    def forward(
        self, readback_residuals: torch.Tensor, beta_residuals: torch.Tensor
    ) -> torch.Tensor:
        """
        readback_residuals: (B, n_slots) — cascade residuals from scratch readback pass
        beta_residuals:     (B, n_slots) — cascade residuals from beta raw signal pass
        Returns:
          logits: (B, n_classes)
        """
        B = readback_residuals.shape[0]
        rb = self.residual_proj(readback_residuals.unsqueeze(-1))  # (B, n_slots, d)
        bt = self.residual_proj(beta_residuals.unsqueeze(-1))
        rb = rb + self.pos_emb.unsqueeze(0) + self.source_emb[0].unsqueeze(0).unsqueeze(0)
        bt = bt + self.pos_emb.unsqueeze(0) + self.source_emb[1].unsqueeze(0).unsqueeze(0)
        tokens = torch.cat([rb, bt], dim=1)  # (B, 2*n_slots, d)
        enc = self.encoder(tokens)
        if self.pool == "concat":
            pooled = enc.reshape(enc.shape[0], -1)  # (B, 2*n_slots*d) position-indexed
        else:
            pooled = enc.mean(dim=1)                 # (B, d) symmetric
        logits = self.cls(pooled)            # (B, n_classes)
        return logits


class V35ResidualDecoder(nn.Module):
    """Codebook-fidelity reconstruction decoder (spec 2026-06-04): maps the quantized
    scratch code back to the substrate residuals. MSE-trained as an autoencoder
    bottleneck (residuals -> write head -> scratch -> THIS -> residuals) so scratch
    must losslessly carry every residual digit. Counters the collapse where the weak
    |k|<=5 successor task leaves the high digits unencoded (high cells dead -> codebook
    not 1-to-1). Zero-Prior: the target is the substrate's OWN residuals, not an N label
    — we require invertibility, not a preset N->code assignment, so the code still
    self-organizes.
    """

    def __init__(self, n_slots: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_slots, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_slots),
        )

    def forward(self, scratch: torch.Tensor) -> torch.Tensor:
        """scratch: (B, n_slots) quantized code -> (B, n_slots) reconstructed residuals."""
        return self.net(scratch)
