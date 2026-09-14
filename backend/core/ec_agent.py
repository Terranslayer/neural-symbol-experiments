"""V32 EC Agent: composite forward with shared multi-gate encoder block.

Two-phase pipeline:

Phase 1 (encode signal → scratch):
    signal: (B, L) → repeat to (B, L, input_dim=3) input feature ([signal, 0, 0]
        following scene.py format) → MultiGateCoTPFC n_chunks=4 →
        final_pfc_state → MultiGateSequentialWriteHead → 5-cell quantized scratch

Phase 3 (read own + partner scratch → δ logits):
    own_scratch, partner_scratch: each (B, W=5) quantized in {0, 0.5, 1.0} →
        construct token sequence (B, 10, input_dim) with row 0..4 = own,
        row 5..9 = partner → MultiGateCoTPFC n_chunks=2 → final_pfc_state →
        Linear(d_model, 5) → δ logits (5-class)

Param sharing: LRU block is shared between phase1 and phase3 encoders.
PFC step and CNN are per-phase (different chunk counts + kernel sizes).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from backend.core.multigate import (
    MultiGateCoTPFC,
    MultiGateLRUBlock,
    MultiGateSequentialWriteHead,
)


@dataclass
class ECAgentConfig:
    L: int = 200
    W: int = 3                  # 2026-05-19: changed 5→3 (+ quantize 3→5) → cap 5^3=125 (was 3^5=243)
    quantize_levels: int = 5
    d_model: int = 16
    d_state: int = 16
    d_role: int = 4
    n_chunks_phase1: int = 4
    n_chunks_phase3: int = 2
    cnn_kernels_phase1: tuple = (5, 51)
    cnn_kernels_phase3: tuple = (2, 3)   # narrower kernel since W=3 (was (2,5) for W=5)
    cnn_channels_per_scale: int = 8
    write_head_hidden: int = 32
    write_head_d_role: int = 8
    n_pfc_heads: int = 4
    n_delta_classes: int = 5
    aux_n_classes: int = 31              # F2: aux head predicts N ∈ [0, aux_n_classes-1]
    aux_detach: bool = False             # F2: if True, aux head reads pfc_state.detach() — gradient does not flow into encoder
    output_scale_init: float = 2.0       # F3c-2: write_head raw scale. With F3c-1 clamp (active range [-1,+1]),
                                         # scale=2.0 maps init raw_pre ~[-0.5, +0.5] → raw ~[-1, +1] spanning all 5 bins.
                                         # Previous 8.0 was for sigmoid mapping and amplified out_proj weights into saturation.


class ECAgent(nn.Module):
    """Single EC agent with ONE shared encoder + write/predict heads.

    Phase 1 (signal → scratch) and Phase 3 (mutual scratch read → δ) share the
    same MultiGateCoTPFC instance. Differ only in (n_chunks, token_type_id)
    passed at forward time. Multiple CNN heads registered via kernels_per_type.
    """
    def __init__(self, cfg: ECAgentConfig):
        super().__init__()
        self.cfg = cfg
        # Shared LRU block (used by both Phase 1 and Phase 3 via the single encoder)
        self.lru_block = MultiGateLRUBlock(
            d_model=cfg.d_model, d_state=cfg.d_state, d_role=cfg.d_role,
        )
        # Single shared encoder block. Multiple CNN heads (one per token_type)
        # registered inside via kernels_per_type. Phase 1 vs Phase 3 differ only
        # in (n_chunks, token_type_id) passed at forward time — actual model
        # params shared across phases.
        self.encoder = MultiGateCoTPFC(
            d_model=cfg.d_model,
            kernels_per_type={
                0: cfg.cnn_kernels_phase1,  # SIGNAL
                1: cfg.cnn_kernels_phase3,  # SCRATCH
            },
            cnn_channels_per_scale=cfg.cnn_channels_per_scale,
            lru_block=self.lru_block, input_dim=3, d_role=cfg.d_role,
            n_pfc_heads=cfg.n_pfc_heads,
        )
        # Write head: PFC state → 5-cell scratch
        self.write_head = MultiGateSequentialWriteHead(
            d_pfc=cfg.d_model, W=cfg.W,
            hidden=cfg.write_head_hidden, d_role=cfg.write_head_d_role,
            output_scale_init=cfg.output_scale_init,
        )
        # Predict head: Phase 3 PFC state → δ logits (5-class)
        self.predict_head = nn.Linear(cfg.d_model, cfg.n_delta_classes)
        # F2 aux head: Phase 1 PFC state → N class logits. Supervised by ground-truth N
        # to break EC mutual-zero-info equilibrium (encoder gets a direct grounding signal
        # independent of the listener's random init). aux_detach controls whether gradient
        # flows back through encoder (False) or not (True, ablation).
        self.aux_n_head = nn.Linear(cfg.d_model, cfg.aux_n_classes)

    def _signal_to_input_feature(self, signal: torch.Tensor) -> torch.Tensor:
        """(B, L) → (B, L, 3). Channel 0 = signal value, channels 1-2 = 0
        (input_valid / post_gap_flag placeholders, following scene.py format).
        """
        B, L = signal.shape
        feat = torch.zeros(B, L, 3, device=signal.device, dtype=signal.dtype)
        feat[:, :, 0] = signal
        return feat

    def _scratch_to_input_feature(self, scratch: torch.Tensor) -> torch.Tensor:
        """(B, W) → (B, W, 3). Channel 0 = scratch cell value (∈ {0, 0.5, 1.0}).
        Same channel layout as signal for same-medium consistency.
        """
        B, W = scratch.shape
        feat = torch.zeros(B, W, 3, device=scratch.device, dtype=scratch.dtype)
        feat[:, :, 0] = scratch
        return feat

    def encode_signal(self, signal: torch.Tensor):
        """Phase 1: signal (B, L) → (raw_scratch, q_scratch, aux_logits).

        aux_logits: (B, aux_n_classes) — used by train loop for supervised N CE loss.
        If cfg.aux_detach, gradient does not flow from aux head into encoder.
        """
        feat = self._signal_to_input_feature(signal)
        pfc_state = self.encoder(feat, n_chunks=self.cfg.n_chunks_phase1, token_type_id=0)
        raw, q = self.write_head(pfc_state, quantize_levels=self.cfg.quantize_levels)
        aux_in = pfc_state.detach() if self.cfg.aux_detach else pfc_state
        aux_logits = self.aux_n_head(aux_in)
        return raw, q, aux_logits

    def read_scratches_predict(self,
                                own_scratch: torch.Tensor,
                                partner_scratch: torch.Tensor) -> torch.Tensor:
        """Phase 3: own + partner scratch (B, W) → δ logits (B, n_classes).

        Construct 10-token sequence: [own_0, ..., own_4, partner_0, ..., partner_4].
        Encoder n_chunks=2 → 2-chunk PFC chain (one chunk per scratch source).
        """
        # Concat scratches along sequence dim
        combined = torch.cat([own_scratch, partner_scratch], dim=-1)  # (B, 2W)
        feat = self._scratch_to_input_feature(combined)  # (B, 2W, 3)
        pfc_state = self.encoder(feat, n_chunks=self.cfg.n_chunks_phase3, token_type_id=1)
        return self.predict_head(pfc_state)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
