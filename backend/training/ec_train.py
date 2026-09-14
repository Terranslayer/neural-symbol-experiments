"""V32 EC pool training loop.

Pool of K agents (same architecture, different random init). Per batch:
- Sample batch_size EC episodes (random N_A, N_B with δ ∈ ±2)
- For each episode, random-sample 2 distinct agents from pool
- Run Phase 1 on both, Phase 3 on both (mutual read), compute joint loss
- Backward + per-agent optimizer step

Curriculum: list of (n_max, n_epochs) pairs. Smaller n_max first.

Logging: JSON-lines events to log_dir/run.jsonl. Checkpoints per stage end
to ckpt_dir/stageN_agent_K.pt.
"""
import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.ec_agent import ECAgent, ECAgentConfig
from backend.core.scene_ec import ECSceneConfig, sample_ec_batch


@dataclass
class ECTrainConfig:
    pool_size: int = 4
    batch_size: int = 32
    n_max_stages: tuple = (5, 10, 20, 30)
    max_epochs_per_stage: tuple = (20, 30, 40, 50)
    steps_per_epoch: int = 100
    L: int = 200
    W: int = 3                  # 2026-05-19: 5→3, q=3→5 → cap 5^3=125 (was 3^5=243)
    quantize_levels: int = 5
    d_model: int = 16
    d_state: int = 16
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    log_dir: str = "logs/v32_ec_default"
    ckpt_dir: str = "checkpoints/v32_ec_default"
    device: str = "cpu"
    seed: int = 0
    val_every_n_steps: int = 50
    # F2 aux supervised N prediction (break EC zero-info equilibrium)
    aux_weight: float = 1.0              # multiplier for aux N-prediction CE loss
    aux_detach: bool = False             # passed into ECAgentConfig; True = no gradient from aux into encoder
    aux_n_classes: int = 31              # N ∈ [0, 30] covered for n_max ≤ 30 curriculum
    # F3c-2 + F3c-3: write_head scale + targeted weight decay
    output_scale_init: float = 2.0       # write_head raw scale init (passed to ECAgentConfig)
    write_head_weight_decay: float = 1e-4  # AdamW wd on write_head.out_proj.{weight,bias} + output_scale only


def _build_pool(cfg: ECTrainConfig) -> List[ECAgent]:
    """Create pool_size ECAgents with distinct random inits.

    Uses per-agent torch.manual_seed inside a save/restore of global RNG state
    so the pool construction doesn't pollute global RNG for downstream training.
    """
    agent_cfg = ECAgentConfig(
        L=cfg.L, W=cfg.W, quantize_levels=cfg.quantize_levels,
        d_model=cfg.d_model, d_state=cfg.d_state,
        aux_n_classes=cfg.aux_n_classes, aux_detach=cfg.aux_detach,
        output_scale_init=cfg.output_scale_init,
    )
    # Save global RNG state so we can restore it after per-agent seeding
    rng_state_before = torch.random.get_rng_state()
    agents = []
    for k in range(cfg.pool_size):
        torch.manual_seed(cfg.seed + k * 1000)
        agent = ECAgent(agent_cfg).to(cfg.device)
        agents.append(agent)
    # Restore global state so subsequent stochastic ops are deterministic w.r.t. cfg.seed
    torch.random.set_rng_state(rng_state_before)
    return agents


def _build_optimizers(agents: List[ECAgent], cfg: ECTrainConfig):
    """Build per-agent AdamW optimizers with targeted weight decay.

    F3c-3: write_head.out_proj.{weight,bias} + write_head.output_scale receive
    weight_decay = cfg.write_head_weight_decay. All other params: wd=0.
    Reason: out_proj weights training to norm ~11 (vs init ~0.18) combined with
    output_scale ~8 produced raw_w ≈ -125 → sigmoid dead zone. Targeted wd
    keeps these scalars near init scale; rest of model trains unconstrained.
    """
    optimizers = []
    wd_param_keys = {
        "write_head.out_proj.weight",
        "write_head.out_proj.bias",
        "write_head.output_scale",
    }
    for a in agents:
        wd_params = []
        no_wd_params = []
        for name, p in a.named_parameters():
            if name in wd_param_keys:
                wd_params.append(p)
            else:
                no_wd_params.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": no_wd_params, "weight_decay": 0.0},
                {"params": wd_params, "weight_decay": cfg.write_head_weight_decay},
            ],
            lr=cfg.learning_rate,
        )
        optimizers.append(opt)
    return optimizers


def _train_step(
    agents: List[ECAgent], optimizers,
    batch: dict, cfg: ECTrainConfig, rng: np.random.Generator,
) -> dict:
    """One training step. Random-sample 2 agents per episode for pairing.

    Returns dict with keys: total_loss, ec_loss, aux_loss (each a python float,
    averaged across batch). total_loss = ec_loss + aux_weight * aux_loss.
    """
    B = batch["signal_A"].shape[0]
    device = cfg.device
    sig_A = torch.from_numpy(batch["signal_A"]).to(device)
    sig_B = torch.from_numpy(batch["signal_B"]).to(device)
    target_A = torch.from_numpy(batch["delta_class_A"]).to(device)
    target_B = torch.from_numpy(batch["delta_class_B"]).to(device)
    n_A = torch.from_numpy(batch["N_A"]).to(device).long()
    n_B = torch.from_numpy(batch["N_B"]).to(device).long()

    # Per-episode random pair (without replacement within episode)
    pair_idx = np.zeros((B, 2), dtype=np.int64)
    for i in range(B):
        pair = rng.choice(cfg.pool_size, size=2, replace=False)
        pair_idx[i] = pair

    for opt in optimizers:
        opt.zero_grad()

    # Group episodes by (a_idx, b_idx) pair so each group can run as a single
    # batched forward (amortizes GPU launch cost). Different pairs within a
    # batch still see different protocols (preserves random-pairing semantics).
    groups = defaultdict(list)
    for i in range(B):
        groups[(int(pair_idx[i, 0]), int(pair_idx[i, 1]))].append(i)

    losses_sum = []
    ec_losses_sum = []
    aux_losses_sum = []
    for (a_idx, b_idx), eps in groups.items():
        idx = torch.tensor(eps, dtype=torch.long, device=device)
        agent_a, agent_b = agents[a_idx], agents[b_idx]
        # Phase 1: encode signals (batched per group)
        _, sc_A, aux_logits_A = agent_a.encode_signal(sig_A[idx])
        _, sc_B, aux_logits_B = agent_b.encode_signal(sig_B[idx])
        # Phase 3: cross-read + predict
        logits_A = agent_a.read_scratches_predict(sc_A, sc_B)
        logits_B = agent_b.read_scratches_predict(sc_B, sc_A)
        # EC δ-prediction CE
        ec_loss_A = F.cross_entropy(logits_A, target_A[idx], reduction='sum')
        ec_loss_B = F.cross_entropy(logits_B, target_B[idx], reduction='sum')
        # F2 aux N-prediction CE (per-agent, on each agent's encoder output)
        aux_loss_A = F.cross_entropy(aux_logits_A, n_A[idx], reduction='sum')
        aux_loss_B = F.cross_entropy(aux_logits_B, n_B[idx], reduction='sum')
        ec_losses_sum.append(ec_loss_A + ec_loss_B)
        aux_losses_sum.append(aux_loss_A + aux_loss_B)
        losses_sum.append((ec_loss_A + ec_loss_B) + cfg.aux_weight * (aux_loss_A + aux_loss_B))
    total_loss = torch.stack(losses_sum).sum() / B
    ec_loss_mean = torch.stack(ec_losses_sum).sum().item() / B
    aux_loss_mean = torch.stack(aux_losses_sum).sum().item() / B
    total_loss.backward()
    for agent, opt in zip(agents, optimizers):
        torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
        opt.step()
    return {
        "total_loss": float(total_loss.item()),
        "ec_loss": ec_loss_mean,
        "aux_loss": aux_loss_mean,
    }


def run_ec_training(cfg: ECTrainConfig) -> List[dict]:
    """Main entry. Returns list of step-level log records."""
    log_dir = Path(cfg.log_dir)
    ckpt_dir = Path(cfg.ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.jsonl"

    # Validate config: stages and epochs must match length
    if len(cfg.n_max_stages) != len(cfg.max_epochs_per_stage):
        raise ValueError(
            f"n_max_stages length {len(cfg.n_max_stages)} != "
            f"max_epochs_per_stage length {len(cfg.max_epochs_per_stage)}"
        )

    agents = _build_pool(cfg)
    optimizers = _build_optimizers(agents, cfg)
    rng = np.random.default_rng(cfg.seed)

    history: List[dict] = []
    global_step = 0
    t_start = time.time()

    with open(log_path, "w") as fout:
        fout.write(json.dumps({"event": "run_start", "cfg": asdict(cfg)}) + "\n")

        for stage_idx, (n_max, n_epochs) in enumerate(
            zip(cfg.n_max_stages, cfg.max_epochs_per_stage)
        ):
            scene_cfg = ECSceneConfig(N_min=1, N_max=n_max, L=cfg.L)
            for epoch in range(n_epochs):
                for step in range(cfg.steps_per_epoch):
                    batch = sample_ec_batch(scene_cfg, cfg.batch_size, rng)
                    losses = _train_step(agents, optimizers, batch, cfg, rng)
                    rec = {
                        "event": "train_step",
                        "stage": stage_idx, "epoch": epoch, "step": step,
                        "global_step": global_step,
                        "loss": losses["total_loss"],
                        "ec_loss": losses["ec_loss"],
                        "aux_loss": losses["aux_loss"],
                        "elapsed_sec": time.time() - t_start,
                    }
                    history.append(rec)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    global_step += 1
            # End of stage: save ckpt per agent
            for k, agent in enumerate(agents):
                ckpt_path = ckpt_dir / f"stage{stage_idx}_agent{k}.pt"
                torch.save({
                    "agent_state_dict": agent.state_dict(),
                    "agent_cfg": asdict(agent.cfg),
                    "stage": stage_idx, "n_max": n_max,
                }, ckpt_path)
            fout.write(json.dumps({
                "event": "stage_end", "stage": stage_idx, "n_max": n_max,
            }) + "\n")
        fout.write(json.dumps({"event": "run_end"}) + "\n")
    return history
