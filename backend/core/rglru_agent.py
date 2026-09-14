"""RG-LRU agent — state-independent linear-recurrent ablation against GRUAgent.

Tests the hypothesis: does the GRU S2-P3 漏桶温度计 emerge from GRU's
state-dependent z_t gate (z_t = σ(W_z x + U_z h_{t-1}))? Or does any input-
dependent gated recurrence — including state-independent linear recurrence
like Griffin's RG-LRU — produce the same encoding?

If RG-LRU produces equivalent compositional metrics, the gating-feedback
loop is NOT essential — any input-dependent EMA suffices. If RG-LRU
collapses to single-rate / pos-0 hoarding, then state-dependent feedback
WAS essential and is a real PFC-like inductive bias.

Cell math (Griffin 2024, eq paraphrased):
    r_t = sigmoid(W_a x_t)            # recurrence gate (state-INDEPENDENT)
    i_t = sigmoid(W_x x_t)            # input gate (state-INDEPENDENT)
    a   = sigmoid(Lambda)             # learnable per-dim base decay in (0, 1)
    a_t = a ** (c · r_t)              # log-space gate, c=8
    h_t = a_t · h_{t-1} + sqrt(1 - a_t²) · (i_t · x_t)

The linear-in-h recurrence + state-independent gate makes parallel scan
training feasible, but we don't bother — same train_phase1 loop,
sequential cell, just a different attractor structure.

The sqrt(1-a²) factor preserves variance regardless of decay strength,
giving stable gradients across deep stacks (no tanh saturation).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from backend.core.agent import _quantize_ste
from backend.core.scene import SceneConfig


@dataclass
class RGLRUAgentConfig:
    d_model: int = 16
    input_dim: int = 3
    signal_output_dim: int = 1
    compare_classes: int = 3
    quantize_levels: Optional[int] = None
    quantize_range: str = "symmetric"
    rglru_c: float = 8.0
    scratch_attn: bool = False
    scratch_attn_heads: int = 2
    compare_mlp_hidden: Optional[int] = None


class RGLRUCell(nn.Module):
    """One-step RG-LRU update. Stateless module — state passed externally."""

    def __init__(self, dim: int, c: float = 8.0):
        super().__init__()
        self.dim = dim
        self.c = c
        # Two state-independent gates: only see input x_t (in projected form).
        self.W_a = nn.Linear(dim, dim)
        self.W_x = nn.Linear(dim, dim)
        # Lambda learns the base decay a = sigmoid(Lambda) per-dimension.
        # Init at 0 → a = 0.5 → moderate decay; training pushes toward 0/1.
        self.Lambda = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_a(x))                              # (B, d)
        i = torch.sigmoid(self.W_x(x))                              # (B, d)
        a = torch.sigmoid(self.Lambda)                              # (d,)
        # Element-wise effective decay; r close to 0 → a^0 = 1 (full retain),
        # r close to 1 → a^c (strong forget when a small).
        a_t = a.unsqueeze(0) ** (self.c * r)                        # (B, d)
        # Variance-normalized linear update. Clamp keeps sqrt finite when a→1.
        gain = torch.sqrt(torch.clamp(1.0 - a_t ** 2, min=1e-8))    # (B, d)
        h_t = a_t * h_prev + gain * (i * x)
        return h_t


class RGLRUAgent(nn.Module):
    """Drop-in replacement for GRUAgent with same forward signature.

    Reuses input_proj, write_head, compare_head, scratch-pad read-back
    semantics. Only the recurrent cell differs.
    """

    def __init__(self, agent_cfg: RGLRUAgentConfig, scene_cfg: SceneConfig):
        super().__init__()
        self.agent_cfg = agent_cfg
        self.scene_cfg = scene_cfg
        self.input_proj = nn.Linear(agent_cfg.input_dim, agent_cfg.d_model)
        self.processor = RGLRUCell(agent_cfg.d_model, c=agent_cfg.rglru_c)
        self.write_head = nn.Linear(agent_cfg.d_model, agent_cfg.signal_output_dim)
        if agent_cfg.scratch_attn:
            self.scratch_proj = nn.Linear(agent_cfg.signal_output_dim, agent_cfg.d_model)
            self.scratch_pos_emb = nn.Parameter(
                torch.randn(scene_cfg.W, agent_cfg.d_model) * 0.02
            )
            self.scratch_mha = nn.MultiheadAttention(
                agent_cfg.d_model, agent_cfg.scratch_attn_heads, batch_first=True
            )
            compare_in_dim = 2 * agent_cfg.d_model
        else:
            compare_in_dim = agent_cfg.d_model
        if agent_cfg.compare_mlp_hidden is not None:
            self.compare_head = nn.Sequential(
                nn.Linear(compare_in_dim, agent_cfg.compare_mlp_hidden),
                nn.ReLU(),
                nn.Linear(agent_cfg.compare_mlp_hidden, agent_cfg.compare_classes),
            )
        else:
            self.compare_head = nn.Linear(compare_in_dim, agent_cfg.compare_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _write_timesteps(self) -> List[int]:
        cfg = self.scene_cfg
        return list(range(cfg.L, cfg.write_end))

    def _read_timesteps(self) -> List[Tuple[int, int]]:
        cfg = self.scene_cfg
        blocks = []
        for i in range(cfg.K):
            read_start = cfg.compare_block_start(i)
            for j in range(cfg.W):
                blocks.append((read_start + j, j))
        return blocks

    def _compare_timesteps(self, compare_indices: torch.Tensor) -> List[int]:
        if compare_indices.dim() == 2:
            return compare_indices[0].tolist()
        return compare_indices.tolist()

    def forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        initial_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = inputs.shape
        d = self.agent_cfg.d_model
        W = self.scene_cfg.W
        cfg = self.scene_cfg

        h = inputs.new_zeros(B, d) if initial_hidden is None else initial_hidden

        write_set = set(self._write_timesteps())
        read_map = dict(self._read_timesteps())
        compare_set = set(self._compare_timesteps(compare_indices))

        scratch_pad_list: List[Optional[torch.Tensor]] = [None] * W
        compare_logits_list: List[torch.Tensor] = []
        hidden_at_compares_list: List[torch.Tensor] = []

        for t in range(T):
            raw_t = inputs[:, t, :]
            if t in read_map:
                scratch_idx = read_map[t]
                stored = scratch_pad_list[scratch_idx]
                if stored is not None:
                    phase_channels = raw_t[:, 1:]
                    x_t = torch.cat([stored, phase_channels], dim=-1)
                else:
                    x_t = raw_t
            else:
                x_t = raw_t

            u_t = self.input_proj(x_t)
            h = self.processor(u_t, h)

            if t in write_set:
                scratch_idx = t - cfg.L
                raw = self.write_head(h)
                if self.agent_cfg.quantize_levels is not None:
                    raw = _quantize_ste(
                        raw,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
                scratch_pad_list[scratch_idx] = raw

            if t in compare_set:
                if self.agent_cfg.scratch_attn:
                    sp_slots: List[torch.Tensor] = []
                    B_local = h.shape[0]
                    for s in scratch_pad_list:
                        if s is None:
                            sp_slots.append(h.new_zeros(B_local, self.agent_cfg.signal_output_dim))
                        else:
                            sp_slots.append(s)
                    sp = torch.stack(sp_slots, dim=1)  # (B, W, 1)
                    kv = self.scratch_proj(sp) + self.scratch_pos_emb.unsqueeze(0)
                    retrieved, _ = self.scratch_mha(
                        h.unsqueeze(1), kv, kv, need_weights=False
                    )
                    compare_in = torch.cat([h, retrieved.squeeze(1)], dim=-1)
                else:
                    compare_in = h
                compare_logits_list.append(self.compare_head(compare_in))
                hidden_at_compares_list.append(h)

        compare_logits = torch.stack(compare_logits_list, dim=1)
        hidden_at_compares = torch.stack(hidden_at_compares_list, dim=1)
        scratch_pad = torch.stack(scratch_pad_list, dim=1).squeeze(-1)

        return compare_logits, scratch_pad, hidden_at_compares
