"""Active-agent policy network (Stage 2 Phase 5 MVP).

Input set-of-tokens:
  [LATENT, WINDOW×w, ACTIVE_PAGE×W_content, TARGET_CUE, PAST_PAGE×n_review]

Transformer encoder (pre-LN, flat) fuses them; LATENT token is read back as
the agent's persistent latent (updated each tick). Pooled encoder output
feeds parallel action heads:
  - action_kind: categorical{saccade, think, write, read_page, emit, noop}
  - saccade_delta: Gaussian continuous
  - write_slot: categorical{0..3}
  - write_value: Gaussian (later clipped + quantized to 6 levels)
  - read_index: categorical{0..N_review-1}
  - emit_class: categorical{0, 1, 2}
  - value: scalar critic

For PPO the log-prob of an action only includes the head that matches its
kind (others conditionally ignored).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_KINDS = ["saccade", "think", "write", "read_page", "emit", "noop"]


@dataclass
class ActivePolicyConfig:
    d_model: int = 16
    n_heads: int = 2
    ff_dim: int = 32
    n_layers: int = 2
    window_w: int = 5
    active_page_len: int = 4           # content slots (feedback slot is read-only past)
    past_page_len: int = 5              # full past page: 4 content + 1 feedback
    n_review_pages: int = 3             # how many past pages included per obs
    saccade_std_init: float = 1.0
    write_value_std_init: float = 0.2
    saccade_abs_max: float = 10.0       # policy output clipped here, engine clips world position separately


class _PreLNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.ln1(x)
        a, _ = self.attn(n, n, n, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class ActivePolicy(nn.Module):
    def __init__(self, cfg: ActivePolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Token embeddings (each input scalar / token-source gets its own projection)
        self.window_emb = nn.Linear(1, d)
        self.active_page_emb = nn.Linear(1, d)
        self.target_emb = nn.Linear(1, d)
        self.past_page_emb = nn.Linear(cfg.past_page_len, d)
        self.latent_init = nn.Parameter(torch.zeros(1, d))

        # Position embedding across the token set (flat positional, single
        # learned parameter table keyed by token index).
        n_tokens = 1 + cfg.window_w + cfg.active_page_len + 1 + cfg.n_review_pages
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.n_tokens = n_tokens

        self.blocks = nn.ModuleList([
            _PreLNBlock(d, cfg.n_heads, cfg.ff_dim) for _ in range(cfg.n_layers)
        ])
        self.ln_final = nn.LayerNorm(d)

        # Action heads
        self.kind_head = nn.Linear(d, len(ACTION_KINDS))
        # Bias kind_head to favor 'emit' early (otherwise under reward-sparse
        # uniform init the agent mostly times out before trying emit).
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
        # World-predictor head: given latent_t, predict window_{t+1}. Trained
        # via aux MSE. Also reused below for a Kalman-style pre-attention
        # correction of the latent when the new observation arrives.
        self.window_predict_head = nn.Linear(d, cfg.window_w)
        # Kalman gain: maps the innovation (observation - prior prediction)
        # in window-space back into latent-space. Small init so the very
        # first forward passes don't blow up before the predictor has
        # learned anything useful.
        self.kalman_gain = nn.Linear(cfg.window_w, d, bias=False)
        nn.init.xavier_uniform_(self.kalman_gain.weight, gain=0.1)

    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_tokens(
        self,
        latent: torch.Tensor,       # (B, d)
        window: torch.Tensor,       # (B, w)
        active_page: torch.Tensor,  # (B, W_content)
        target_cue: torch.Tensor,   # (B,)
        past_pages: torch.Tensor,   # (B, n_review, past_page_len)
    ) -> torch.Tensor:
        B, d = latent.shape
        window_tok = self.window_emb(window.unsqueeze(-1))           # (B, w, d)
        page_tok = self.active_page_emb(active_page.unsqueeze(-1))   # (B, W_content, d)
        target_tok = self.target_emb(target_cue.unsqueeze(-1).unsqueeze(-1))  # (B, 1, d)
        past_tok = self.past_page_emb(past_pages)                     # (B, n_review, d)
        latent_tok = latent.unsqueeze(1)                              # (B, 1, d)
        tokens = torch.cat(
            [latent_tok, window_tok, page_tok, target_tok, past_tok], dim=1
        )
        # Add positional embedding
        tokens = tokens + self.pos_emb.unsqueeze(0)
        return tokens

    def forward(
        self,
        latent: torch.Tensor,
        window: torch.Tensor,
        active_page: torch.Tensor,
        target_cue: torch.Tensor,
        past_pages: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # --- Kalman-style belief update ---
        # latent is the posterior from the PREVIOUS tick. At that tick the
        # predictor guessed what the current window would look like; compare
        # to what we actually see and use the innovation to correct the latent
        # BEFORE letting the transformer make further action decisions.
        prior_pred_window = self.window_predict_head(latent)        # (B, w)
        innovation = window - prior_pred_window                      # (B, w)
        corrected_latent = latent + self.kalman_gain(innovation)     # (B, d)

        tokens = self.build_tokens(
            corrected_latent, window, active_page, target_cue, past_pages
        )
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.ln_final(tokens)
        new_latent = tokens[:, 0]                   # LATENT token position
        pooled = tokens.mean(dim=1)                 # pool for action heads

        return {
            "new_latent": new_latent,
            "kind_logits": self.kind_head(pooled),
            "saccade_mu": self.saccade_mu_head(pooled).squeeze(-1) * self.cfg.saccade_abs_max,
            "saccade_logstd": self.saccade_logstd,
            "write_slot_logits": self.write_slot_head(pooled),
            "write_value_mu": torch.sigmoid(self.write_value_mu_head(pooled).squeeze(-1)),
            "write_value_logstd": self.write_value_logstd,
            "read_index_logits": self.read_index_head(pooled),
            "emit_class_logits": self.emit_class_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
            # Predict NEXT window from the NEW latent (trained via aux MSE
            # against actual next_window in the rollout loop).
            "window_prediction": self.window_predict_head(new_latent),
            # Expose innovation magnitude for diagnostics/logging (optional).
            "innovation_norm": innovation.detach().pow(2).mean(dim=-1).sqrt(),
        }


# -------------------------------------------------------------------- sampling

def sample_action(
    policy_out: Dict[str, torch.Tensor], deterministic: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Sample one action per batch element. Returns the sampled parts and their
    log-probs (mask-compatible with PPO: log_prob per head).
    """
    kind_dist = torch.distributions.Categorical(logits=policy_out["kind_logits"])
    kind = kind_dist.probs.argmax(-1) if deterministic else kind_dist.sample()

    saccade_dist = torch.distributions.Normal(
        policy_out["saccade_mu"], policy_out["saccade_logstd"].exp()
    )
    saccade = policy_out["saccade_mu"] if deterministic else saccade_dist.sample()
    saccade = saccade.clamp(min=-float(torch.as_tensor(policy_out["saccade_mu"]).new_full((1,), 1e9)), max=1e9)  # no-op safeguard
    # Clamp softly to saccade_abs_max via tanh-like? We already scaled mu, so fine.

    write_slot_dist = torch.distributions.Categorical(logits=policy_out["write_slot_logits"])
    write_slot = write_slot_dist.probs.argmax(-1) if deterministic else write_slot_dist.sample()

    wv_dist = torch.distributions.Normal(
        policy_out["write_value_mu"], policy_out["write_value_logstd"].exp()
    )
    write_value = policy_out["write_value_mu"] if deterministic else wv_dist.sample()
    write_value = write_value.clamp(0.0, 1.0)

    read_dist = torch.distributions.Categorical(logits=policy_out["read_index_logits"])
    read_index = read_dist.probs.argmax(-1) if deterministic else read_dist.sample()

    emit_dist = torch.distributions.Categorical(logits=policy_out["emit_class_logits"])
    emit_class = emit_dist.probs.argmax(-1) if deterministic else emit_dist.sample()

    action = {
        "kind": kind,
        "saccade_delta": saccade,
        "write_slot": write_slot,
        "write_value": write_value,
        "read_index": read_index,
        "emit_class": emit_class,
    }
    logprob = {
        "kind": kind_dist.log_prob(kind),
        "saccade_delta": saccade_dist.log_prob(saccade),
        "write_slot": write_slot_dist.log_prob(write_slot),
        "write_value": wv_dist.log_prob(write_value),
        "read_index": read_dist.log_prob(read_index),
        "emit_class": emit_dist.log_prob(emit_class),
    }
    return action, logprob


def action_to_record(action: Dict[str, torch.Tensor], batch_idx: int):
    """Convert a sampled batched action into a single ActionRecord for the engine.

    The policy produces ALL head outputs at every step, but only the head
    selected by `kind` matters for world dynamics. The rest are "dummy" but
    still incur log-prob and should be regularized via entropy.
    """
    from backend.core.world_event import ActionRecord
    kind_idx = int(action["kind"][batch_idx].item())
    kind = ACTION_KINDS[kind_idx]
    return ActionRecord(
        kind=kind,
        saccade_delta=float(action["saccade_delta"][batch_idx].item()),
        write_slot=int(action["write_slot"][batch_idx].item()),
        write_value=float(action["write_value"][batch_idx].item()),
        read_index=int(action["read_index"][batch_idx].item()),
        emit_class=int(action["emit_class"][batch_idx].item()),
    )
