# -*- coding: utf-8 -*-
# DIAG_FOR: any | ANSWERS: codebook,cell_decouple,write_head_raw | INPUTS: <ckpt> --q --L --n-max
"""Inspect a Phase-1 GRU agent checkpoint: dump scratch-pad encoding analysis.

Replaces the older inspect_s2p3_stage*.py / inspect_s2p5_gru_n100.py / inspect_agent.py
trio. Combines:
  - thermometer-style mean-note table per N
  - per-position correlation with N (linear & log)
  - per-position alphabet-usage entropy (does the agent use all q levels?)
  - tuple uniqueness (how many distinct (lv0,...,lv_{W-1}) codes? collisions?)
  - modal tuple per N
  - per-position N-range each level covers
  - optional PNG plots (scatter / heatmap / t-SNE / MI matrix)

Phase-1 ckpts only store {agent_state_dict, optimizer_state_dict, stage_idx,
global_step}, so scene/agent config must be supplied via CLI. d_model and
the presence of scratch-attention are auto-detected from the state dict.

Usage:
  python scripts/inspect_checkpoint.py CKPT --q 3 --L 250 --complex --n-max 100
  python scripts/inspect_checkpoint.py CKPT --q 6 --L 150 --complex --n-max 40 --plots
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.agent import AgentConfig, GRUAgent
from backend.core.scene import SceneConfig, sample_training_batch


def _build_levels(q: int, qrange: str) -> np.ndarray:
    if qrange == "unit":
        return np.linspace(0.0, 1.0, q)
    elif qrange == "symmetric":
        return np.linspace(-1.0, 1.0, q)
    raise ValueError(f"Unknown quantize_range: {qrange}")


def _detect_d_model(state_dict) -> int:
    if "dp_proj.weight" in state_dict:
        # V6 dual_pathway: dp_proj is Linear(total_ch, d_model)
        return state_dict["dp_proj.weight"].shape[0]
    if "input_proj.weight" in state_dict:
        return state_dict["input_proj.weight"].shape[0]
    if "read_cnn.weight" in state_dict:
        return state_dict["read_cnn.weight"].shape[0]
    raise ValueError("Cannot detect d_model from state dict")


def _detect_read_cnn_kernel(state_dict) -> int:
    if "read_cnn.weight" in state_dict:
        return state_dict["read_cnn.weight"].shape[2]
    return 0


def _detect_read_cnn_layers(state_dict) -> int:
    extras = {int(k.split(".")[1]) for k in state_dict if k.startswith("read_cnn_extra.")}
    return 1 + len(extras) if "read_cnn.weight" in state_dict else 1


def _detect_scratch_attn(state_dict) -> bool:
    return any(k.startswith("scratch_mha") for k in state_dict.keys())


def _detect_pfc_layers(state_dict) -> int:
    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("pfc.layers."):
            layer_indices.add(int(k.split(".")[2]))
    return len(layer_indices)


def _detect_compare_split_2d(state_dict, d_model) -> bool:
    """If compare_head input is 2*d, agent was trained with pfc_split_ab."""
    if "compare_head.0.weight" in state_dict:  # MLP first layer
        return state_dict["compare_head.0.weight"].shape[1] == 2 * d_model
    if "compare_head.weight" in state_dict:
        return state_dict["compare_head.weight"].shape[1] == 2 * d_model
    return False


def _detect_lstm(state_dict) -> bool:
    return any(k.startswith("processor.weight_ih") for k in state_dict.keys())


def _detect_mamba(state_dict) -> int:
    """Return number of mamba blocks if checkpoint is MambaAgent, else 0."""
    blocks = {int(k.split(".")[1]) for k in state_dict if k.startswith("mamba_blocks.")}
    return len(blocks)


def _load_mamba_agent(state, q, qrange, L, complex_world,
                      pfc_at_write=False, pfc_iter=1,
                      pfc_compare_drop_raw_scratch=False,
                      cnn_pfc_only=False, v7_mode=False,
                      saved_agent_cfg=None):
    """Load a MambaAgent checkpoint. Auto-detects layers, d_model, cnn vs linear, PFC.

    saved_agent_cfg: if provided (from ckpt's saved 'agent_cfg' dict),
    apply BEFORE agent construction so MambaAgent.__init__ sees correct
    cot_pfc_n_chunks, pfc_recurrent_n_iter, etc., and builds the right
    submodules. Without this, fields are restored AFTER construction → modules
    not built → forward dispatch wrong → silent acc divergence.
    """
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    n_layers = _detect_mamba(state)
    d_model = _detect_d_model(state)
    cnn_K = _detect_read_cnn_kernel(state)
    a_log_key = next((k for k in state if k.endswith("A_log")), None)
    d_state = state[a_log_key].shape[1] if a_log_key else 16
    has_cmp_mlp = "compare_head.0.weight" in state
    has_gap_head = any(k.startswith("gap_head.") for k in state)
    # V22 fix: when use_gap_head + rbf_compare_head, compare_head is RBF (not
    # Sequential), so has_cmp_mlp via compare_head is False. But gap_head may
    # still be Sequential (compare_mlp_hidden>0). Detect via gap_head shape.
    gap_head_is_sequential = "gap_head.0.weight" in state
    pfc_n_layers = len({int(k.split(".")[2]) for k in state if k.startswith("pfc.layers.")})
    # Auto-detect aux modules from state_dict so they're instantiated properly
    # (load_state_dict can find their weights).
    fp_t_pred = 0
    if "forward_predictor.fc2.weight" in state:
        # fc2 output dim = T_pred * d_model
        fp_t_pred = state["forward_predictor.fc2.weight"].shape[0] // d_model
    bp_t_pred = 0
    if "beta_predictor.fc2.weight" in state:
        bp_t_pred = state["beta_predictor.fc2.weight"].shape[0] // d_model
    has_compare_pred = any(k.startswith("compare_predictor.") for k in state)
    has_top_down = any(k.startswith("top_down_predictor.") for k in state)
    # 51_M5: detect simple_comparator (compare_head.weight shape (C, d) vs (C, 2d))
    simple_comp = False
    if "compare_head.weight" in state:
        shape = state["compare_head.weight"].shape
        # If input dim equals d_model → simple comparator (Linear(d, C))
        if shape[1] == d_model:
            simple_comp = True
    # V8: detect RBFCompareHead via compare_head.centers / compare_head.log_sigma
    rbf_head = "compare_head.centers" in state
    rbf_sigma = 1.0
    if rbf_head and "compare_head.log_sigma" in state:
        # init via mean of saved log_sigma (just for module init; load_state_dict overwrites)
        rbf_sigma = float(state["compare_head.log_sigma"].exp().mean().item())
    # V11/P3: detect IFWriteHead via write_head.threshold / fire_proj
    if_write_gate_detected = "write_head.threshold" in state
    # V12: detect write-head LayerNorm via write_head_ln.weight
    write_head_ln_detected = "write_head_ln.weight" in state
    # V10: detect pfc_see_cnn_seq via cnn_pos_emb / alpha_side_emb / beta_side_emb
    pfc_see_cnn_seq_detected = (
        "cnn_pos_emb" in state or "alpha_side_emb" in state or "beta_side_emb" in state
    )
    # V10: detect dp_t_target from cnn_pos_emb shape (T, d)
    # If pfc_see_cnn_seq is on, exact value recoverable. Else, default to 8
    # for dual_pathway runs (V8_kWTA / V18 / etc convention).
    dp_t_target_detected = 0
    if "cnn_pos_emb" in state:
        dp_t_target_detected = int(state["cnn_pos_emb"].shape[0])
    elif "dp_convs.0.weight" in state:
        # V8_kWTA / V18 default; without this LRU/Mamba sees full L tokens (OOD)
        dp_t_target_detected = 8
    # V13: detect cross-attn write head
    cross_attn_write_detected = any(k.startswith("cross_attn_write_module") for k in state)
    # V14: detect gumbel write head via write_head.logit_proj
    gumbel_write_head_detected = "write_head.logit_proj.weight" in state
    # V15: detect recurrent latent PFC + recover state-token count
    recurrent_latent_pfc_detected = any(
        k.startswith("recurrent_latent_pfc_module") for k in state
    )
    # V16: detect write_query_token via write_query_emb param
    write_query_token_detected = "write_query_emb" in state
    # V17: detect multi-modular aux head; recover modulus count from output dim
    # V18: detect LRU via mamba_blocks.0.nu_log (LRU has nu_log/theta_log; Mamba doesn't)
    use_lru_detected = "mamba_blocks.0.nu_log" in state
    lru_d_state_detected = 16
    if use_lru_detected and "mamba_blocks.0.nu_log" in state:
        lru_d_state_detected = int(state["mamba_blocks.0.nu_log"].shape[0])
    # V19: detect event-gated LRU via spike_detector + gate_sharpness params
    lru_event_gated_detected = (
        "spike_detector.weight" in state and "gate_sharpness" in state
    )
    multi_modular_aux_moduli_detected = ()
    if "multi_modular_aux_head.weight" in state:
        n_mod_detected = int(state["multi_modular_aux_head.weight"].shape[0])
        # We can't recover the actual modulus values from ckpt, so use a
        # default like (2, 3, 5) if 3 heads, else placeholder; user passes
        # them explicitly when re-loading for analysis
        if n_mod_detected == 3:
            multi_modular_aux_moduli_detected = (2, 3, 5)
        else:
            multi_modular_aux_moduli_detected = tuple(range(2, 2 + n_mod_detected))
    # V20: detect scratch_mod_aux_head; default to (2, 3) for n_mod=2
    scratch_mod_aux_moduli_detected = ()
    if "scratch_mod_aux_head.weight" in state:
        n_smod_detected = int(state["scratch_mod_aux_head.weight"].shape[0])
        if n_smod_detected == 2:
            scratch_mod_aux_moduli_detected = (2, 3)
        else:
            scratch_mod_aux_moduli_detected = tuple(range(2, 2 + n_smod_detected))
    # V22: detect SuccessorFunction (Sequential: Linear / LayerNorm / GELU / Linear)
    successor_hidden_detected = 0
    if "successor.net.0.weight" in state:
        # successor.net.0 = Linear(W, hidden); .weight is (hidden, W)
        successor_hidden_detected = int(state["successor.net.0.weight"].shape[0])
    # V25: detect SequentialWriteHead via write_head.cell_mlp.0.weight
    # (Linear(d_pfc+W, hidden); shape (hidden, d_pfc+W))
    write_head_type_detected = "linear"
    write_head_hidden_detected = 32
    if "write_head.cell_mlp.0.weight" in state:
        write_head_type_detected = "sequential"
        write_head_hidden_detected = int(state["write_head.cell_mlp.0.weight"].shape[0])
    # V23b: detect NClassHead (Linear-ReLU-Linear). max_n = output dim of net.2
    n_class_max_n_detected = 0
    n_class_head_hidden_detected = 32
    if "n_class_head.net.2.weight" in state:
        n_class_max_n_detected = int(state["n_class_head.net.2.weight"].shape[0])
        n_class_head_hidden_detected = int(state["n_class_head.net.2.weight"].shape[1])
    # V22: detect gap_head (use_gap_head=True). Could be Linear or Sequential.
    use_gap_head_detected = (
        "gap_head.weight" in state or "gap_head.0.weight" in state
    )
    recurrent_latent_state_tokens_detected = 1
    if "recurrent_latent_pfc_module.init_state" in state:
        recurrent_latent_state_tokens_detected = int(
            state["recurrent_latent_pfc_module.init_state"].shape[0]
        )
    # 51_M5: detect pfc_variant
    pfc_variant_detected = "transformer"
    pfc_mlp_h = 64
    if any(k.startswith("pfc.fc1.") for k in state):
        pfc_variant_detected = "mlp"
        # fc1 weight shape (hidden, num_tokens*d) → recover hidden
        pfc_mlp_h = state["pfc.fc1.weight"].shape[0]
    # 51_M5: detect cycle_loss_lambda from any non-zero h_alpha-cycle params (no
    # extra params; cycle is loss-only). Set to 0.5 if simple_comp present.
    cycle_lambda = 0.5 if simple_comp else 0.0
    # β-1: detect VQ codebook
    vq_enabled = "vq.codebook" in state
    vq_K, vq_d_z = 32, 8
    if vq_enabled:
        vq_K, vq_d_z = state["vq.codebook"].shape
    # β-1: detect review module
    review_iter = 0
    if "review_module.encoder.layers.0.self_attn.in_proj_weight" in state:
        review_iter = 1  # default; actual iter count not in state
    # idea2-related: detect scratch_self / cf_beta / rs predictors
    has_rs = any(k.startswith("rs_predictor.") for k in state)
    rs_t_pred = 0
    if has_rs:
        rs_t_pred = state["rs_predictor.fc2.weight"].shape[0]
    has_ssp = any(k.startswith("scratch_self_predictor.") for k in state)
    cfb_t_pred = 0
    if "cf_beta_predictor.fc3.weight" in state:
        cfb_t_pred = state["cf_beta_predictor.fc3.weight"].shape[0]
    # V6: detect dual_pathway via dp_convs presence
    has_dual = "dp_convs.0.weight" in state
    dp_kernels = (5, 21, 51)
    dp_cps = 4
    if has_dual:
        # Recover kernel sizes from each dp_conv weight shape (out_ch, in_ch, K)
        dp_idxs = sorted({int(k.split(".")[1]) for k in state if k.startswith("dp_convs.")})
        dp_kernels = tuple(state[f"dp_convs.{i}.weight"].shape[2] for i in dp_idxs)
        dp_cps = state["dp_convs.0.weight"].shape[0]
        # When V6 dual_pathway, read_cnn=None — disable cnn_K detection
        cnn_K = 0  # placeholder; not used when dual_pathway

    scene_cfg = SceneConfig.trio_preset(L=L, complex_world=complex_world)
    # 51_M5: pfc_n_layers detection works for transformer; for MLP variant
    # need to set pfc_layers=2 explicitly (DeepMLPPFC has 2 fc layers internally
    # but counts as 1 module in our config — set pfc_layers=2 to enable PFC path).
    if pfc_variant_detected == "mlp":
        pfc_n_layers = 2
    agent_cfg = MambaAgentConfig(
        d_model=d_model, quantize_levels=q, quantize_range=qrange,
        read_cnn_kernel=cnn_K, mamba_layers=n_layers, d_state=d_state,
        compare_mlp_hidden=32 if (has_cmp_mlp or gap_head_is_sequential) else None,
        pfc_layers=pfc_n_layers, pfc_heads=2, pfc_iter=pfc_iter,
        pfc_at_write=pfc_at_write,
        pfc_compare_drop_raw_scratch=pfc_compare_drop_raw_scratch,
        use_gap_head=has_gap_head,
        world_pred_t_pred=fp_t_pred,
        beta_pred_t_pred=bp_t_pred,
        compare_pred_enabled=has_compare_pred,
        predictive_coding=has_top_down,
        rs_pred_t_pred=rs_t_pred,
        scratch_self_pred_enabled=has_ssp,
        cf_beta_pred_t_pred=cfb_t_pred,
        review_iter=review_iter,
        vq_enabled=vq_enabled,
        vq_codebook_size=vq_K,
        vq_d_z=vq_d_z,
        simple_comparator=simple_comp,
        cycle_loss_lambda=cycle_lambda,
        pfc_variant=pfc_variant_detected,
        pfc_mlp_hidden=pfc_mlp_h,
        dual_pathway=has_dual or v7_mode,
        dp_kernels=dp_kernels,
        dp_channels_per_scale=dp_cps,
        cnn_pfc_only=cnn_pfc_only and not v7_mode,
        dp_scratch_skip_mamba=False if v7_mode else True,
        rbf_compare_head=rbf_head,
        rbf_init_sigma=rbf_sigma,
        if_write_gate=if_write_gate_detected,
        write_head_ln=write_head_ln_detected,
        cross_attn_write=cross_attn_write_detected,
        gumbel_write_head=gumbel_write_head_detected,
        pfc_see_cnn_seq=pfc_see_cnn_seq_detected,
        dp_t_target=dp_t_target_detected,
        recurrent_latent_pfc=recurrent_latent_pfc_detected,
        recurrent_latent_state_tokens=recurrent_latent_state_tokens_detected,
        write_query_token=write_query_token_detected,
        multi_modular_aux_moduli=multi_modular_aux_moduli_detected,
        use_lru=use_lru_detected,
        lru_d_state=lru_d_state_detected,
        lru_event_gated=lru_event_gated_detected,
        scratch_mod_aux_moduli=scratch_mod_aux_moduli_detected,
        successor_hidden=successor_hidden_detected,
        n_class_max_n=n_class_max_n_detected,
        n_class_head_hidden=n_class_head_hidden_detected,
        write_head_type=write_head_type_detected,
        write_head_hidden=write_head_hidden_detected,
    )
    # CRITICAL: apply saved_agent_cfg BEFORE MambaAgent construction so __init__
    # builds correct submodules (cot_pfc, recurrent_refiner, etc.). Otherwise
    # restore-after-construction silently leaves modules None → forward dispatch
    # wrong → 8pp+ silent acc gap.
    if saved_agent_cfg is not None:
        pre_applied = []
        for k, v in saved_agent_cfg.items():
            if hasattr(agent_cfg, k) and getattr(agent_cfg, k) != v:
                pre_applied.append(k)
                setattr(agent_cfg, k, v)
        if pre_applied:
            print(f"  (pre-construction restored {len(pre_applied)} agent_cfg fields: {pre_applied[:8]}{'...' if len(pre_applied)>8 else ''})")
    agent = MambaAgent(agent_cfg, scene_cfg)
    # strict=False to tolerate auxiliary modules (e.g. forward_predictor) that
    # are training-only and not needed at eval / inspection time.
    missing, unexpected = agent.load_state_dict(state, strict=False)
    if unexpected:
        skipped = sorted({k.split(".")[0] for k in unexpected})
        print(f"  (skipped train-only modules: {skipped})")
    agent.train(False)
    if torch.cuda.is_available():
        agent = agent.cuda()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded MambaAgent: d_model={d_model}, layers={n_layers}, d_state={d_state}, cnn_K={cnn_K}, pfc_layers={pfc_n_layers}, pfc_at_write={pfc_at_write} ({dev})")
    return agent, agent_cfg, scene_cfg


def load_agent(ckpt_path: Path, q: int, qrange: str, L: int, complex_world: bool,
               cell_type: str = "gru", pfc_at_write: bool = False, pfc_iter: int = 1,
               pfc_compare_drop_raw_scratch: bool = False,
               cnn_pfc_only: bool = False,
               v7_mode: bool = False,
               use_oracle_spike_gate: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["agent_state_dict"]
    # NEW (5/10): if ckpt contains saved agent_cfg + scene_cfg, restore them after
    # construction. This makes forward path EXACTLY match training. Without this,
    # forward-affecting hyperparams without learnable params (kwta_k,
    # use_oracle_spike_gate, etc.) default to wrong values and cause silent
    # forward divergence.
    saved_agent_cfg = ckpt.get("agent_cfg") if isinstance(ckpt, dict) else None
    saved_scene_cfg = ckpt.get("scene_cfg") if isinstance(ckpt, dict) else None
    # V29 SNN: detect via saved_agent_cfg.use_v29. V29 ckpts have NO mamba_blocks
    # (V29Pipeline replaces all V27/V28 modules), so _detect_mamba returns 0.
    # We must short-circuit before the GRU fallback path (which calls
    # _detect_d_model and crashes since V29 has no dp_proj/input_proj/read_cnn).
    is_v29 = bool(saved_agent_cfg and saved_agent_cfg.get("use_v29", False))
    if is_v29:
        from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
        # Build a minimal default cfg; saved_agent_cfg fields applied below.
        # Must include use_v29=True + use_lru=True (V29 needs the import-guard
        # bypass even though it doesn't use the mamba blocks).
        agent_cfg = MambaAgentConfig(
            d_model=int(saved_agent_cfg.get("d_model", 16)),
            quantize_levels=int(saved_agent_cfg.get("quantize_levels", q)),
            quantize_range=str(saved_agent_cfg.get("quantize_range", qrange)),
        )
        # Apply ALL saved fields BEFORE construction (so V29Pipeline builds
        # with the right neuron counts / layers / passes / k_max etc.).
        for k, v in saved_agent_cfg.items():
            if hasattr(agent_cfg, k):
                setattr(agent_cfg, k, v)
        # Need a scene_cfg for MambaAgent.__init__; will be overwritten by caller.
        scene_cfg = SceneConfig.trio_preset(L=L, complex_world=complex_world)
        if saved_scene_cfg is not None:
            for sk, sv in saved_scene_cfg.items():
                if hasattr(scene_cfg, sk) and not isinstance(sv, list):
                    setattr(scene_cfg, sk, sv)
        agent = MambaAgent(agent_cfg, scene_cfg)
        missing, unexpected = agent.load_state_dict(state, strict=False)
        if unexpected:
            skipped = sorted({k.split(".")[0] for k in unexpected})
            print(f"  (V29 load: skipped train-only modules: {skipped})")
        if missing:
            # Filter mamba_blocks/cot_pfc/etc. — those are intentionally absent in V29
            real_missing = [m for m in missing if not any(
                m.startswith(p) for p in ("mamba_blocks.", "cot_pfc.", "recurrent_refiner.",
                                          "write_head.", "compare_head.", "pfc.", "dp_",
                                          "spike_detector", "gate_sharpness")
            )]
            if real_missing:
                print(f"  (V29 load: WARN missing keys: {real_missing[:5]}...)")
        agent.train(False)
        if torch.cuda.is_available():
            agent = agent.cuda()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loaded V29 MambaAgent: d_model={agent_cfg.d_model}, "
              f"v29_encoder_n={agent_cfg.v29_encoder_n_neurons}, "
              f"v29_write_n={agent_cfg.v29_write_n_neurons}, "
              f"v29_pfc_layers={agent_cfg.v29_pfc_layers} ({dev})")
        return agent, agent_cfg, scene_cfg
    # V33 HH-SSM: detect via saved_agent_cfg.use_v33. V33 ckpts have NO mamba_blocks
    # and NO V29 modules; instead they have v33_substrate.* + dp_convs.* + write_head.*.
    # Short-circuit BEFORE the mamba/GRU detection paths.
    is_v33 = bool(saved_agent_cfg and saved_agent_cfg.get("use_v33", False))
    if is_v33:
        from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
        agent_cfg = MambaAgentConfig(
            d_model=int(saved_agent_cfg.get("v33_d_model", saved_agent_cfg.get("d_model", 16))),
            quantize_levels=int(saved_agent_cfg.get("quantize_levels", q)),
            quantize_range=str(saved_agent_cfg.get("quantize_range", qrange)),
        )
        for k, v in saved_agent_cfg.items():
            if hasattr(agent_cfg, k):
                setattr(agent_cfg, k, v)
        scene_cfg = SceneConfig.trio_preset(L=L, complex_world=complex_world)
        if saved_scene_cfg is not None:
            for sk, sv in saved_scene_cfg.items():
                if hasattr(scene_cfg, sk) and not isinstance(sv, list):
                    setattr(scene_cfg, sk, sv)
        agent = MambaAgent(agent_cfg, scene_cfg)
        missing, unexpected = agent.load_state_dict(state, strict=False)
        if unexpected:
            skipped = sorted({k.split(".")[0] for k in unexpected})
            print(f"  (V33 load: skipped train-only modules: {skipped})")
        if missing:
            # Filter modules intentionally absent in V33: V27 LRU/PFC/CoT and V29 SNN
            real_missing = [m for m in missing if not any(
                m.startswith(p) for p in ("mamba_blocks.", "cot_pfc.", "recurrent_refiner.",
                                          "pfc.", "spike_detector", "gate_sharpness",
                                          "lru", "v29")
            )]
            if real_missing:
                print(f"  (V33 load: WARN missing keys: {real_missing[:5]}...)")
        agent.train(False)
        if torch.cuda.is_available():
            agent = agent.cuda()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loaded V33 MambaAgent: d_model={agent_cfg.d_model}, "
              f"v33_n_layers={agent_cfg.v33_n_layers} ({dev})")
        return agent, agent_cfg, scene_cfg
    # Mamba detection: if checkpoint has mamba_blocks.*, dispatch to MambaAgent loader
    if _detect_mamba(state) > 0:
        agent, agent_cfg, scene_cfg = _load_mamba_agent(
            state, q, qrange, L, complex_world,
            pfc_at_write=pfc_at_write, pfc_iter=pfc_iter,
            pfc_compare_drop_raw_scratch=pfc_compare_drop_raw_scratch,
            cnn_pfc_only=cnn_pfc_only, v7_mode=v7_mode,
            saved_agent_cfg=saved_agent_cfg,
        )
        if saved_agent_cfg is not None:
            applied = []
            for k, v in saved_agent_cfg.items():
                if hasattr(agent.agent_cfg, k) and getattr(agent.agent_cfg, k) != v:
                    applied.append(k)
                    setattr(agent.agent_cfg, k, v)
            if applied:
                print(f"  (restored {len(applied)} agent_cfg fields from ckpt: {applied[:8]}{'...' if len(applied)>8 else ''})")
        if saved_scene_cfg is not None:
            for k, v in saved_scene_cfg.items():
                if hasattr(agent.scene_cfg, k) and not isinstance(v, list):
                    setattr(agent.scene_cfg, k, v)
        if use_oracle_spike_gate:
            agent.agent_cfg.use_oracle_spike_gate = True
            print("  (oracle-spike-gate enabled at load time)")
        return agent, agent_cfg, scene_cfg
    d_model = _detect_d_model(state)
    has_attn = _detect_scratch_attn(state)
    cnn_K = _detect_read_cnn_kernel(state)
    cnn_layers = _detect_read_cnn_layers(state)
    pfc_layers = _detect_pfc_layers(state)
    has_split = _detect_compare_split_2d(state, d_model)
    has_pfc = pfc_layers > 0
    has_compare_raw = any(k.startswith("compare_head_raw.") for k in state)
    scene_cfg = SceneConfig.discrimination_preset(L=L, complex_world=complex_world)
    agent_cfg = AgentConfig(
        d_model=d_model,
        quantize_levels=q,
        quantize_range=qrange,
        scratch_attn=has_attn,
        compare_mlp_hidden=32 if ("compare_head.0.weight" in state or "compare_head_raw.0.weight" in state) else None,
        read_cnn_kernel=cnn_K,
        read_cnn_layers=cnn_layers,
        pfc_layers=pfc_layers,
        pfc_heads=2,
        reset_h_at_compare_block=True if (has_pfc or has_split) else False,
        pfc_split_ab=has_split,
        pfc_at_write=pfc_at_write,  # cannot detect; user must specify
        pfc_iter=pfc_iter,          # cannot detect; user must specify
        pfc_compare_drop_raw_scratch=pfc_compare_drop_raw_scratch,  # cannot detect; user must specify
        compare_via_raw_scratch=has_compare_raw,
        cell_type=cell_type,        # cannot detect rcnn; user must specify
    )
    agent = GRUAgent(agent_cfg, scene_cfg)
    agent.load_state_dict(state)
    agent.train(False)
    print(f"Loaded {ckpt_path.name}")
    print(f"  d_model={d_model}  cnn_K={cnn_K}  cnn_layers={cnn_layers}  scratch_attn={has_attn}")
    print(f"  pfc_layers={pfc_layers}  split_ab={has_split}  cell_type={cell_type}  pfc_at_write={pfc_at_write}")
    print(f"  L={L}  W={scene_cfg.W}  complex={complex_world}  q={q}")
    print(f"  params={agent.num_parameters()}  step={ckpt.get('global_step', '?')}")
    return agent, agent_cfg, scene_cfg


def collect_notes_per_N(agent, scene_cfg, n_max: int, n_batches: int, batch_size: int, seed: int):
    per_N = defaultdict(list)
    rng = np.random.default_rng(seed=seed)
    device = next(agent.parameters()).device
    with torch.no_grad():
        for _ in range(n_batches):
            inputs, _, cidx, metas = sample_training_batch(
                batch_size=batch_size, n_max_stage=n_max, config=scene_cfg, rng=rng,
            )
            _, scratch, _ = agent(inputs.to(device), cidx.to(device))
            scratch = scratch.cpu()
            for b in range(scratch.shape[0]):
                per_N[int(metas[b]["N_alpha"])].append(scratch[b].numpy())
    return per_N


def report_thermometer(per_N, levels: np.ndarray, n_max: int):
    W = len(next(iter(per_N.values()))[0])
    print("\n" + "=" * (40 + 8 * W))
    print("MEAN scratch-pad notes per N")
    print("=" * (40 + 8 * W))
    pos_hdr = " ".join(f"{'pos'+str(j):>6}" for j in range(W))
    lv_hdr = " ".join(f"{'lv'+str(j):>3}" for j in range(W))
    print(f"{'N':>4} {'ct':>4} | {pos_hdr} | {lv_hdr}")
    print("-" * (40 + 8 * W))
    step = max(1, n_max // 20)
    for N in [1] + list(range(step, n_max + 1, step)):
        if N not in per_N:
            continue
        arrs = np.stack(per_N[N], axis=0)
        mean = arrs.mean(axis=0)
        level_idx = np.abs(mean[:, None] - levels[None, :]).argmin(axis=1)
        vals_str = " ".join(f"{v:+.3f}" for v in mean)
        levels_str = " ".join(f"{int(l):>3}" for l in level_idx)
        print(f"{N:>4} {arrs.shape[0]:>4} | {vals_str} | {levels_str}")


def report_correlation(per_N):
    all_X, all_N = [], []
    for N, lst in per_N.items():
        for arr in lst:
            all_X.append(arr)
            all_N.append(N)
    X = np.array(all_X)
    y = np.array(all_N).astype(np.float32)
    print("\n--- Per-position correlation with N ---")
    for p in range(X.shape[1]):
        r_lin = np.corrcoef(X[:, p], y)[0, 1]
        r_log = np.corrcoef(X[:, p], np.log(y + 0.1))[0, 1]
        print(f"  pos{p}: corr(notes, N)={r_lin:+.3f}  corr(notes, log N)={r_log:+.3f}")
    return X, y


def report_alphabet(X, levels):
    q = len(levels)
    print("\n--- Per-position alphabet usage (fraction of writes hitting each level) ---")
    hdr_levels = " ".join(f"{'lv'+str(i)+'('+f'{levels[i]:.2f}'+')':>11}" for i in range(q))
    print(f"{'pos':>4} | {hdr_levels} | entropy")
    print("-" * (12 + 12 * q))
    W = X.shape[1]
    for p in range(W):
        col_lvls = np.abs(X[:, p:p+1] - levels[None, :]).argmin(axis=1)
        counts = np.bincount(col_lvls, minlength=q) / len(col_lvls)
        ent = -np.sum(counts * np.log(counts + 1e-10)) / np.log(q)
        cnts_str = " ".join(f"{c:>11.3f}" for c in counts)
        print(f"  {p:>2} | {cnts_str} | {ent:.3f}")


def report_uniqueness(per_N, levels, n_max: int):
    q = len(levels)
    W = len(next(iter(per_N.values()))[0])
    unique_tuples = defaultdict(set)
    tuple_to_Ns = defaultdict(set)
    for N, lst in per_N.items():
        for arr in lst:
            tup = tuple(int(l) for l in np.abs(arr[:, None] - levels[None, :]).argmin(axis=1))
            unique_tuples[N].add(tup)
            tuple_to_Ns[tup].add(N)
    total = set()
    for s in unique_tuples.values():
        total.update(s)
    print(f"\n--- Code uniqueness ---")
    print(f"  Distinct tuples observed: {len(total)} / {q**W} possible ({q}^{W})")
    print(f"  Needed for unique encoding of N in [1, {n_max}]: {n_max}")

    collisions = sorted(
        ((t, ns) for t, ns in tuple_to_Ns.items() if len(ns) > 1),
        key=lambda kv: -len(kv[1]),
    )
    print(f"  Tuples shared across multiple N values: {len(collisions)}")
    for tup, ns in collisions[:10]:
        sn = sorted(ns)
        show = sn if len(sn) <= 8 else sn[:4] + ['...'] + sn[-3:]
        print(f"    {tup}  ->  N in {show}  ({len(ns)} distinct N)")


def report_modal_per_N(per_N, levels, n_max: int):
    q = len(levels)
    print("\n--- Modal note pattern per N ---")
    print(f"{'N':>4} | {'modal tuple':>20} | {'freq':>5} | {'#distinct':>10}")
    print("-" * 60)
    sample = list(range(1, 11)) + list(range(20, n_max + 1, max(10, n_max // 10)))
    seen = set()
    for N in sample:
        if N in seen or N not in per_N:
            continue
        seen.add(N)
        c = Counter()
        for arr in per_N[N]:
            tup = tuple(int(l) for l in np.abs(arr[:, None] - levels[None, :]).argmin(axis=1))
            c[tup] += 1
        modal, n = c.most_common(1)[0]
        freq = n / len(per_N[N])
        print(f"  {N:>2} | {str(modal):>20} | {freq:>5.2f} | {len(c):>10}")


def report_pos_ranges(X, y, levels):
    q = len(levels)
    W = X.shape[1]
    print("\n--- Pos-by-pos: which N range maps to each level ---")
    for p in range(W):
        col_lvls = np.abs(X[:, p:p+1] - levels[None, :]).argmin(axis=1).flatten()
        print(f"  pos{p}:")
        for lvl in range(q):
            mask = col_lvls == lvl
            if mask.sum() == 0:
                print(f"    lvl{lvl} (~{levels[lvl]:.2f}): never used")
                continue
            ns = y[mask]
            print(f"    lvl{lvl} (~{levels[lvl]:.2f}): N in [{int(ns.min()):>3}, {int(ns.max()):>3}]  "
                  f"mean={ns.mean():.1f}  median={np.median(ns):.0f}  count={int(mask.sum())}")


def make_plots(X, y, out_dir: Path, ckpt_name: str):
    """Optional matplotlib visualizations — only loaded on demand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    W = X.shape[1]

    # Scatter per position
    fig, axes = plt.subplots(1, W, figsize=(3 * W, 3.2))
    for j in range(W):
        ax = axes[j] if W > 1 else axes
        ax.scatter(y, X[:, j], s=6, alpha=0.4)
        uniq = np.unique(y)
        means = [X[y == u, j].mean() for u in uniq]
        ax.plot(uniq, means, color="red", linewidth=1.5)
        ax.set_xlabel("N")
        ax.set_ylabel(f"notes[:,{j}]")
        ax.set_title(f"pos {j}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{ckpt_name}: per-position note vs N")
    fig.tight_layout()
    fig.savefig(out_dir / "notes_scatter.png", dpi=120)
    plt.close(fig)

    # Heatmap of mean note per N
    uniq = np.unique(y).astype(int)
    M = np.zeros((len(uniq), W))
    for i, u in enumerate(uniq):
        M[i] = X[y == u].mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(8, W * 1.2), max(6, len(uniq) * 0.12)))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(W)); ax.set_xticklabels([f"pos{j}" for j in range(W)])
    ax.set_yticks(range(len(uniq))); ax.set_yticklabels([str(u) for u in uniq], fontsize=6)
    ax.set_title(f"{ckpt_name}: mean notes by N")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "notes_heatmap.png", dpi=120)
    plt.close(fig)

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0,
                    perplexity=min(30, len(X) // 4))
        xy = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=y, cmap="viridis", s=12, alpha=0.7)
        ax.set_title(f"{ckpt_name}: notes t-SNE (color=N)")
        fig.colorbar(sc, ax=ax, label="N")
        fig.tight_layout()
        fig.savefig(out_dir / "notes_tsne.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("  (sklearn not available, skipping t-SNE)")

    # MI matrix at the best base
    try:
        from backend.evaluation.mi_matrix import base_scanning_mi, mutual_information_matrix
        scan = base_scanning_mi(y, X, candidate_bases=[2, 3, 4, 5, 6, 7, 8, 10])
        b = scan["best_base"]
        mi = mutual_information_matrix(y, X, b=b)["mi_matrix"]
        fig, ax = plt.subplots(figsize=(max(6, mi.shape[1] * 1.2), max(4, W)))
        im = ax.imshow(mi, aspect="auto", cmap="inferno")
        ax.set_xticks(range(mi.shape[1]))
        ax.set_xticklabels([f"digit {k}" for k in range(mi.shape[1])])
        ax.set_yticks(range(W)); ax.set_yticklabels([f"pos {j}" for j in range(W)])
        ax.set_title(f"{ckpt_name}: MI at best_base={b}")
        for j in range(W):
            for k in range(mi.shape[1]):
                ax.text(k, j, f"{mi[j,k]:.2f}", ha="center", va="center",
                        color="white" if mi[j,k] < mi.max()/2 else "black", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "mi_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  (MI plot skipped: {e})")
    print(f"  Plots written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--q", type=int, required=True, help="quantize_levels (e.g. 3, 6)")
    ap.add_argument("--range", dest="qrange", default="unit", choices=["unit", "symmetric"])
    ap.add_argument("--L", type=int, default=250, help="scene L")
    ap.add_argument("--complex", action="store_true", help="complex_world preset")
    ap.add_argument("--n-max", type=int, required=True, help="curriculum stage n_max for sampling")
    ap.add_argument("--n-batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=789)
    ap.add_argument("--plots", action="store_true", help="also write PNG plots next to ckpt")
    ap.add_argument("--cell-type", choices=["gru", "lif", "rcnn", "lstm", "none"], default="gru",
                    help="Override cell type for ckpts that can't be auto-detected (rcnn/lif).")
    ap.add_argument("--pfc-at-write", action="store_true",
                    help="Set if checkpoint was trained with --pfc-at-write (can't auto-detect).")
    ap.add_argument("--pfc-iter", type=int, default=1,
                    help="K iterations of PFC encoder (Universal Transformer style); cannot auto-detect.")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="path to training run.jsonl, restores agent_cfg for old ckpts "
                         "without saved cfg")
    ap.add_argument("--pfc-compare-drop-raw", action="store_true",
                    help="Set if checkpoint was trained with --pfc-compare-drop-raw (can't auto-detect).")
    args = ap.parse_args()

    agent, agent_cfg, scene_cfg = load_agent(
        args.ckpt, args.q, args.qrange, args.L, args.complex,
        cell_type=args.cell_type, pfc_at_write=args.pfc_at_write,
        pfc_iter=args.pfc_iter,
        pfc_compare_drop_raw_scratch=args.pfc_compare_drop_raw,
    )
    if args.jsonl is not None:
        from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl
        restore_agent_cfg_from_jsonl(agent, args.jsonl)
    levels = _build_levels(args.q, args.qrange)
    per_N = collect_notes_per_N(
        agent, scene_cfg, args.n_max, args.n_batches, args.batch_size, args.seed,
    )

    report_thermometer(per_N, levels, args.n_max)
    X, y = report_correlation(per_N)
    report_alphabet(X, levels)
    report_uniqueness(per_N, levels, args.n_max)
    report_modal_per_N(per_N, levels, args.n_max)
    report_pos_ranges(X, y, levels)

    if args.plots:
        out_dir = args.ckpt.parent / f"{args.ckpt.stem}_inspect"
        make_plots(X, y, out_dir, args.ckpt.stem)


if __name__ == "__main__":
    main()
