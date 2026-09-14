"""Hierarchical active-agent policy (Stage 2 Phase 5 — option 2).

Extends ActivePolicy with multi-level latent tokens connected by
predictive-coding-style bottom-up error flow:

  Level 0 (perception):  predicts the current window; innovation = window - pred
  Level 1 (counting):    predicts L0's corrected state
  Level 2 (task):        predicts L1's corrected state
  ...

Each level has its own latent token (shape (B, d_model)) that is corrected by
a Kalman-gain-like mapping of its sub-level's innovation before the
transformer attention mixes everything. Policy output heads read the pooled
attention output as before.

This gives the agent a structured belief-update pathway that does not rely
on PPO backprop through reward alone — perception mismatches immediately
drive latent updates at the appropriate level.

Same action space, same rollout/PPO code (with a latent shape change from
(B, d) to (B, K, d)).
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from backend.core.active_agent import (
    ACTION_KINDS,
    _PreLNBlock,
    sample_action,
)


@dataclass
class HierActivePolicyConfig:
    d_model: int = 16
    n_heads: int = 2
    ff_dim: int = 32
    n_layers: int = 2
    window_w: int = 5
    active_page_len: int = 4
    past_page_len: int = 5
    n_review_pages: int = 3
    saccade_std_init: float = 1.0
    write_value_std_init: float = 0.2
    saccade_abs_max: float = 10.0
    n_latent_tokens: int = 2           # depth of hierarchy; 1 == flat
    kalman_gain_init: float = 0.1      # xavier gain for kalman heads


class HierActivePolicy(nn.Module):
    def __init__(self, cfg: HierActivePolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        K = cfg.n_latent_tokens

        self.window_emb = nn.Linear(1, d)
        self.active_page_emb = nn.Linear(1, d)
        self.target_emb = nn.Linear(1, d)
        self.past_page_emb = nn.Linear(cfg.past_page_len, d)

        # Level-0 latent init + per-level additional init vectors
        self.latent_init = nn.Parameter(torch.zeros(K, d))

        # Level-0 predictor outputs a window (w floats); level-k (k>=1)
        # predicts the level-(k-1) latent (d floats).
        self.level_predictors = nn.ModuleList()
        self.level_predictors.append(nn.Linear(d, cfg.window_w))
        for _ in range(K - 1):
            self.level_predictors.append(nn.Linear(d, d))

        # Kalman gains: for level 0, map window-error (w) -> latent (d).
        # For level k>=1, map lower-level latent error (d) -> latent (d).
        self.kalman_gains = nn.ModuleList()
        self.kalman_gains.append(nn.Linear(cfg.window_w, d, bias=False))
        for _ in range(K - 1):
            self.kalman_gains.append(nn.Linear(d, d, bias=False))
        for g in self.kalman_gains:
            nn.init.xavier_uniform_(g.weight, gain=cfg.kalman_gain_init)

        # Position embedding across the token set
        n_tokens = K + cfg.window_w + cfg.active_page_len + 1 + cfg.n_review_pages
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.n_tokens = n_tokens
        self.K = K

        self.blocks = nn.ModuleList(
            [_PreLNBlock(d, cfg.n_heads, cfg.ff_dim) for _ in range(cfg.n_layers)]
        )
        self.ln_final = nn.LayerNorm(d)

        # Action heads (identical to flat policy)
        self.kind_head = nn.Linear(d, len(ACTION_KINDS))
        with torch.no_grad():
            emit_idx = ACTION_KINDS.index("emit")
            self.kind_head.bias.fill_(0.0)
            self.kind_head.bias[emit_idx] = 1.0
        self.saccade_mu_head = nn.Linear(d, 1)
        self.saccade_logstd = nn.Parameter(
            torch.full((1,), float(np.log(cfg.saccade_std_init)))
        )
        self.write_slot_head = nn.Linear(d, cfg.active_page_len)
        self.write_value_mu_head = nn.Linear(d, 1)
        self.write_value_logstd = nn.Parameter(
            torch.full((1,), float(np.log(cfg.write_value_std_init)))
        )
        self.read_index_head = nn.Linear(d, cfg.n_review_pages)
        self.emit_class_head = nn.Linear(d, 3)
        self.value_head = nn.Linear(d, 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        latent: torch.Tensor,           # (B, K, d)
        window: torch.Tensor,           # (B, w)
        active_page: torch.Tensor,      # (B, W_content)
        target_cue: torch.Tensor,       # (B,)
        past_pages: torch.Tensor,       # (B, n_review, past_page_len)
    ) -> Dict[str, torch.Tensor]:
        B, K, d = latent.shape
        assert K == self.K, f"latent tokens {K} != configured {self.K}"

        # ----- Hierarchical Kalman correction (bottom-up) -----
        # Level 0: predict window from L0 prior; correct L0 with innovation.
        pred_window = self.level_predictors[0](latent[:, 0])       # (B, w)
        innovation_0 = window - pred_window                         # (B, w)
        L0_corr = latent[:, 0] + self.kalman_gains[0](innovation_0)

        corrected: List[torch.Tensor] = [L0_corr]
        innovations: List[torch.Tensor] = [innovation_0]
        for k in range(1, K):
            pred_lower = self.level_predictors[k](latent[:, k])    # (B, d)
            # Innovation at level (k-1) = corrected_lower - Lk's prediction_lower
            innov_k = corrected[k - 1] - pred_lower
            Lk_corr = latent[:, k] + self.kalman_gains[k](innov_k)
            corrected.append(Lk_corr)
            innovations.append(innov_k)

        corrected_stack = torch.stack(corrected, dim=1)            # (B, K, d)

        # ----- Token assembly + attention -----
        window_tok = self.window_emb(window.unsqueeze(-1))          # (B, w, d)
        page_tok = self.active_page_emb(active_page.unsqueeze(-1))  # (B, W_content, d)
        target_tok = self.target_emb(target_cue.unsqueeze(-1).unsqueeze(-1))  # (B,1,d)
        past_tok = self.past_page_emb(past_pages)                   # (B, n_review, d)

        tokens = torch.cat(
            [corrected_stack, window_tok, page_tok, target_tok, past_tok], dim=1
        )
        tokens = tokens + self.pos_emb.unsqueeze(0)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.ln_final(tokens)

        new_latents = tokens[:, :K]                                # (B, K, d)
        pooled = tokens.mean(dim=1)

        # Predictions for next-tick aux loss (level-0 predicts next window).
        next_window_pred = self.level_predictors[0](new_latents[:, 0])

        innov_norm_per_level = torch.stack(
            [i.detach().pow(2).mean(dim=-1).sqrt() for i in innovations], dim=1
        )  # (B, K)

        return {
            "new_latent": new_latents,
            "kind_logits": self.kind_head(pooled),
            "saccade_mu": self.saccade_mu_head(pooled).squeeze(-1) * self.cfg.saccade_abs_max,
            "saccade_logstd": self.saccade_logstd,
            "write_slot_logits": self.write_slot_head(pooled),
            "write_value_mu": torch.sigmoid(self.write_value_mu_head(pooled).squeeze(-1)),
            "write_value_logstd": self.write_value_logstd,
            "read_index_logits": self.read_index_head(pooled),
            "emit_class_logits": self.emit_class_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
            "window_prediction": next_window_pred,
            "innovation_norm_per_level": innov_norm_per_level,
        }
