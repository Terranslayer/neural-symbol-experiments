"""V25-V24 sanity check: validate_stage on V25-V24 ckpt — should match training's 0.418."""
import sys; sys.path.insert(0, ".")
import torch, numpy as np, torch.nn as nn
from backend.core.scene import SceneConfig
from backend.training.train_phase1 import validate_stage, TrainConfig, CurriculumStage
from scripts.inspect_checkpoint import load_agent

agent, _, _ = load_agent("checkpoints/v25_v24_seqwrite/stage1_n30.pt", q=3, qrange="unit", L=200, complex_world=True, pfc_at_write=True, pfc_compare_drop_raw_scratch=True)
device = next(agent.parameters()).device
raw = torch.load("checkpoints/v25_v24_seqwrite/stage1_n30.pt", map_location=device)["agent_state_dict"]

succ_w = raw["successor_predict_head.weight"]
k_max = succ_w.shape[0]
in_dim = succ_w.shape[1]
W = agent.scene_cfg.W
print(f"successor_predict_head: Linear({in_dim}, {k_max})  W={W}")
succ_head = nn.Linear(in_dim, k_max).to(device)
succ_head.load_state_dict({"weight": raw["successor_predict_head.weight"], "bias": raw["successor_predict_head.bias"]})
agent.successor_predict_head = succ_head
agent.agent_cfg.successor_predict_input_level = "pfc_hidden"
agent.agent_cfg.successor_predict_k_max = 5

cfg = SceneConfig.successor_prediction_preset(L=200, complex_world=True, k_max=5)
cfg.K = 1
cfg.alpha_distribution = "uniform"
agent.scene_cfg = cfg
agent.train(False)

train_cfg = TrainConfig(batch_size=64, validation_batches=4)
stage = CurriculumStage(name="s1", n_max=30, target_accuracy=0.7, max_epochs=50)

accs = []
for trial in range(10):
    rng = np.random.default_rng(trial)
    m = validate_stage(agent, cfg, train_cfg, stage, device, rng, loss_type="ce")
    a = m["validation_accuracy"]
    accs.append(a)
    print(f"trial {trial}: val_acc={a:.4f}  val_loss={m['validation_loss']:.4f}")
print(f"\nV25-V24 mean val_acc over 10 trials: {np.mean(accs):.4f} std {np.std(accs):.4f}")
print(f"(training jsonl reported V25-V24 ep49 val_acc = 0.418)")
