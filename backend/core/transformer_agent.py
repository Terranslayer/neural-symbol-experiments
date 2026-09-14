# -*- coding: utf-8 -*-
"""
Recurrent Transformer agent for Stage 2 Phase 2 experiments.

At each timestep t the agent applies a Transformer cell F to the
length-2 sequence [u_t, y_{t-1}] where:
    u_t = InputProj(x_t)          — projection of the 3-channel input
    y_{t-1} = previous hidden state
The Transformer cell's output at position 0 is taken as y_t.

Two variants controlled by `use_event_gate`:
    False (S2-P2-A, control):
        y_t = F(u_t, y_{t-1})                 — continuous update each step
    True  (S2-P2-B, anti-log):
        g_t = STE(sigmoid(gate_net(u_t, y_{t-1})))   binary in [0, 1]
        Delta_t = F(u_t, y_{t-1}) - y_{t-1}
        y_t = y_{t-1} + g_t * Delta_t         — discrete "event accumulator"

Same scratch-pad / write-head / compare-head structure as GRUAgent, so
train_phase1.py's per-step logic is reusable.

Parameter budget targets ~1700 params (matches GRU d=16 baseline).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from backend.core.scene import SceneConfig


@dataclass
class TransformerAgentConfig:
    d_model: int = 16
    n_heads: int = 2
    ff_dim: int = 16            # no expansion -> hits ~1830 params,
                                # close to GRU d=16 baseline (1764)
    input_dim: int = 3
    signal_output_dim: int = 1
    compare_classes: int = 3
    use_event_gate: bool = False
    quantize_levels: Optional[int] = None
    quantize_range: str = "symmetric"  # "symmetric" [-1,1] or "unit" [0,1]


def _gate_ste(raw_logit: torch.Tensor) -> torch.Tensor:
    """Binary event gate with straight-through estimator.
    Forward:  g = 1 if sigmoid(raw) > 0.5 else 0
    Backward: grad flows as if g = sigmoid(raw).
    """
    soft = torch.sigmoid(raw_logit)
    hard = (soft > 0.5).float()
    return soft + (hard - soft).detach()


class TransformerCell(nn.Module):
    """Minimal Transformer block operating on a length-2 sequence."""

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
        # seq: (B, 2, d_model)
        a, _ = self.attn(seq, seq, seq, need_weights=False)
        x = self.ln1(seq + a)
        f = self.ff(x)
        return self.ln2(x + f)


class TransformerAgent(nn.Module):
    def __init__(self, agent_cfg: TransformerAgentConfig, scene_cfg: SceneConfig):
        super().__init__()
        self.agent_cfg = agent_cfg
        self.scene_cfg = scene_cfg

        self.input_proj = nn.Linear(agent_cfg.input_dim, agent_cfg.d_model)
        self.cell = TransformerCell(
            agent_cfg.d_model, agent_cfg.n_heads, agent_cfg.ff_dim,
        )
        self.write_head = nn.Linear(agent_cfg.d_model, agent_cfg.signal_output_dim)
        self.compare_head = nn.Linear(agent_cfg.d_model, agent_cfg.compare_classes)

        if agent_cfg.use_event_gate:
            # Gate takes [u_t; y_{t-1}] concatenated, outputs scalar logit
            self.gate_net = nn.Linear(2 * agent_cfg.d_model, 1)
        else:
            self.gate_net = None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Bias event gate to "mostly open" initially so gradients flow early
        if self.gate_net is not None:
            nn.init.constant_(self.gate_net.bias, 1.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---- Forward-pass helpers (mirror GRUAgent) ----------------------------

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

    # ---- Main forward pass --------------------------------------------------

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

        y = inputs.new_zeros(B, d) if initial_hidden is None else initial_hidden

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
                    x_t = torch.cat([stored, raw_t[:, 1:]], dim=-1)
                else:
                    x_t = raw_t
            else:
                x_t = raw_t

            u_t = self.input_proj(x_t)                              # (B, d)

            # Length-2 sequence: [u_t, y_{t-1}]. We take position 1 (the
            # processed previous state) as the new state because the cell
            # has residual connections — position 0 would be u_t + attn,
            # dominated by the current input, so using it as "new state"
            # effectively overwrites long-term memory each step.
            seq = torch.stack([u_t, y], dim=1)                       # (B, 2, d)
            out = self.cell(seq)                                     # (B, 2, d)
            y_new = out[:, 1]                                        # processed y_{t-1}

            if self.gate_net is not None:
                gate_input = torch.cat([u_t, y], dim=-1)             # (B, 2d)
                gate_logit = self.gate_net(gate_input)               # (B, 1)
                g = _gate_ste(gate_logit)                            # (B, 1), in {0, 1}
                y = y + g * (y_new - y)                              # event-driven update
            else:
                y = y_new                                            # continuous update

            if t in write_set:
                scratch_idx = t - cfg.L
                raw = self.write_head(y)
                if self.agent_cfg.quantize_levels is not None:
                    from backend.core.agent import _quantize_ste
                    raw = _quantize_ste(
                        raw,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
                scratch_pad_list[scratch_idx] = raw

            if t in compare_set:
                compare_logits_list.append(self.compare_head(y))
                hidden_at_compares_list.append(y)

        compare_logits = torch.stack(compare_logits_list, dim=1)
        hidden_at_compares = torch.stack(hidden_at_compares_list, dim=1)
        scratch_pad = torch.stack(scratch_pad_list, dim=1).squeeze(-1)

        return compare_logits, scratch_pad, hidden_at_compares
