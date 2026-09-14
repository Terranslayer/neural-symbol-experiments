"""Helper: restore agent_cfg from training run.jsonl.

Ckpts don't save agent_cfg, but run.jsonl's first line (run_start event) does.
This module loads that and applies it to a freshly-loaded agent so forward
behavior matches training exactly.

Usage:
    from scripts._load_agent_cfg import restore_agent_cfg_from_jsonl
    restore_agent_cfg_from_jsonl(agent, "logs/v26_correct_v24r_seqwrite_continue/run.jsonl")
"""
import json
from pathlib import Path


def restore_agent_cfg_from_jsonl(agent, jsonl_path):
    """Parse run.jsonl run_start, override agent.agent_cfg fields that aren't in state_dict."""
    p = Path(jsonl_path)
    with open(p, "r") as f:
        first_line = f.readline()
    rec = json.loads(first_line)
    if rec.get("event") != "run_start":
        raise RuntimeError(f"First line of {p} is not run_start")
    saved = rec["agent_cfg"]
    # List of fields that AFFECT FORWARD but inspect_checkpoint does NOT auto-detect.
    # If detection is added later, remove from this list.
    forward_affecting = [
        "kwta_k", "use_oracle_spike_gate", "lru_event_gated", "detection_aux_weight",
        "write_noise_std", "pfc_recurrent_n_iter", "successor_predict_input_level",
        "successor_predict_k_max", "gate_sharpness_init",
        "lru_init_r_min", "lru_init_r_max", "lru_init_phase_max",
        "dp_t_target", "dp_detach_to_mamba", "dp_scratch_skip_mamba",
        "dp_channels_per_scale", "rbf_compare_head", "rbf_init_sigma",
        "pfc_compare_drop_raw_scratch", "pfc_at_write", "pfc_iter",
        "pfc_layers", "pfc_heads", "compare_mlp_hidden",
        "write_head_type", "write_head_hidden",
        "quantize_levels", "quantize_range",
    ]
    overrides = {}
    for k in forward_affecting:
        if k in saved:
            cur = getattr(agent.agent_cfg, k, None)
            if cur != saved[k]:
                overrides[k] = (cur, saved[k])
                setattr(agent.agent_cfg, k, saved[k])
    if overrides:
        print(f"[restore_agent_cfg] applied {len(overrides)} overrides from {p}:")
        for k, (cur, new) in overrides.items():
            print(f"  {k}: {cur!r} -> {new!r}")
    else:
        print(f"[restore_agent_cfg] no overrides needed from {p}")
    # Also restore scene_cfg (K, weights, etc).
    # Skip "L" because training uses --curriculum-L per-stage override (the preset
    # default L stored in scene_cfg is wrong post-training). User passes L via CLI.
    SCENE_SKIP = {"L"}
    saved_scene = rec.get("scene_cfg", {})
    if saved_scene:
        scene_overrides = {}
        for k, v in saved_scene.items():
            if k in SCENE_SKIP:
                continue
            cur = getattr(agent.scene_cfg, k, None)
            if cur != v and not isinstance(v, list):  # skip list types like near_range
                scene_overrides[k] = (cur, v)
                setattr(agent.scene_cfg, k, v)
        if scene_overrides:
            print(f"[restore_agent_cfg] also applied {len(scene_overrides)} scene overrides:")
            for k, (cur, new) in scene_overrides.items():
                print(f"  scene.{k}: {cur!r} -> {new!r}")
    return overrides
