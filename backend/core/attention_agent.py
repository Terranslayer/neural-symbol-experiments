"""Full-attention agent (Geiping-style) for Stage 2 Phase 4.

Unlike the per-timestep recurrent cells (GRUAgent, TransformerAgent), this
agent processes each observation window in *one shot* via a small transformer
encoder, then applies a short "think" recurrence over a single latent vector
(Geiping 2025, Hao 2024 / Coconut), and produces all W notes from the thought.

Architecture:
  Phase 1 - perceive alpha:   Transformer encoder over (B, L, 3) -> pooled (B, d)
  Phase 2 - think k steps:    length-2 sequence [perception, s] recursed k times
  Phase 3 - write notes:      d -> W quantized values (all at once)
  Phase 4 - per compare:      project notes + beta as (B, W+L, 3) via SHARED
                              input_proj (same-medium), encode, pool, classify.

Same-medium guarantee: notes are scalars on the world's value range and are
fed back through the same input_proj that world observations use, with the
notes' 3-channel representation filled as (note_value, valid=1, post_gap=1) to
mimic the read-phase markers the recurrent agents also see.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from backend.core.agent import _quantize_ste
from backend.core.scene import SceneConfig


@dataclass
class AttentionAgentConfig:
    d_model: int = 16
    n_heads: int = 2
    ff_dim: int = 16
    n_perception_layers: int = 1
    n_compare_layers: int = 1
    n_think_steps: int = 4
    input_dim: int = 3
    compare_classes: int = 3
    quantize_levels: Optional[int] = None
    quantize_range: str = "symmetric"


class _PreLNTransformerBlock(nn.Module):
    """Pre-LN transformer block. Preserves residual-stream variance better
    than post-LN across repeated iterations."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        seq_norm = self.ln1(seq)
        a, _ = self.attn(seq_norm, seq_norm, seq_norm, need_weights=False)
        x = seq + a
        f = self.ff(self.ln2(x))
        return x + f


class AttentionAgent(nn.Module):
    def __init__(self, agent_cfg: AttentionAgentConfig, scene_cfg: SceneConfig):
        super().__init__()
        self.agent_cfg = agent_cfg
        self.scene_cfg = scene_cfg
        d = agent_cfg.d_model

        self.input_proj = nn.Linear(agent_cfg.input_dim, d)

        self.perception_blocks = nn.ModuleList([
            _PreLNTransformerBlock(d, agent_cfg.n_heads, agent_cfg.ff_dim)
            for _ in range(agent_cfg.n_perception_layers)
        ])
        self.perception_ln = nn.LayerNorm(d)

        self.think_block = _PreLNTransformerBlock(d, agent_cfg.n_heads, agent_cfg.ff_dim)

        self.write_head = nn.Linear(d, scene_cfg.W)
        self.compare_blocks = nn.ModuleList([
            _PreLNTransformerBlock(d, agent_cfg.n_heads, agent_cfg.ff_dim)
            for _ in range(agent_cfg.n_compare_layers)
        ])
        self.compare_ln = nn.LayerNorm(d)
        self.compare_head = nn.Linear(d, agent_cfg.compare_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for _, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1 and "bias" in _:
                nn.init.zeros_(param)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _project_world_window(self, window: torch.Tensor) -> torch.Tensor:
        return self.input_proj(window)

    def _notes_to_faux_read_sequence(self, notes: torch.Tensor) -> torch.Tensor:
        """Expand (B, W) scalar notes into (B, W, 3) tokens with phase markers
        matching the read-phase the recurrent agents receive: channel 0 holds
        the note value, channels 1/2 are (valid, post_gap) = (1, 1)."""
        B, W = notes.shape
        faux = notes.new_zeros(B, W, 3)
        faux[:, :, 0] = notes
        faux[:, :, 1] = 1.0
        faux[:, :, 2] = 1.0
        return faux

    def forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        initial_hidden: Optional[torch.Tensor] = None,
    ):
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        d = self.agent_cfg.d_model

        # Phase 1: perceive alpha
        alpha_window = inputs[:, 0:cfg.L, :]
        alpha_tokens = self._project_world_window(alpha_window)
        for block in self.perception_blocks:
            alpha_tokens = block(alpha_tokens)
        alpha_tokens = self.perception_ln(alpha_tokens)
        perception = alpha_tokens.mean(dim=1)

        # Phase 2: think for k steps (Geiping-style recurrence over length-2 seq)
        s = perception
        for _ in range(self.agent_cfg.n_think_steps):
            seq = torch.stack([perception, s], dim=1)
            out = self.think_block(seq)
            s = out[:, 1]

        # Phase 3: write W notes at once
        note_logits = self.write_head(s)
        if self.agent_cfg.quantize_levels is not None:
            note_logits_expanded = note_logits.unsqueeze(-1)
            quantized = _quantize_ste(
                note_logits_expanded,
                self.agent_cfg.quantize_levels,
                self.agent_cfg.quantize_range,
            )
            notes = quantized.squeeze(-1)
        else:
            notes = note_logits
        scratch_pad = notes

        # Phase 4: per compare block, attend over notes+beta jointly
        faux_read = self._notes_to_faux_read_sequence(notes)
        compare_logits_list: List[torch.Tensor] = []
        hidden_at_compares_list: List[torch.Tensor] = []
        for i in range(cfg.K):
            _, beta_start, compare_step = cfg.compare_block_phases(i)
            beta_window = inputs[:, beta_start:compare_step, :]
            combined = torch.cat([faux_read, beta_window], dim=1)
            tokens = self._project_world_window(combined)
            for block in self.compare_blocks:
                tokens = block(tokens)
            tokens = self.compare_ln(tokens)
            pooled = tokens.mean(dim=1)
            compare_logits_list.append(self.compare_head(pooled))
            hidden_at_compares_list.append(pooled)

        compare_logits = torch.stack(compare_logits_list, dim=1)
        hidden_at_compares = torch.stack(hidden_at_compares_list, dim=1)
        return compare_logits, scratch_pad, hidden_at_compares
