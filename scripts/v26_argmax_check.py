"""Quick check: what does predict_head output for various input types?
- Constant input (1,1,1,1,1, 0,0,0,0,0)
- Constant input (1,1,1,1,1, 1,1,1,1,1)
- Random input
- Actual refined scratch from training_data
"""
import sys, torch
sys.path.insert(0, ".")
import numpy as np
import torch.nn as nn
ckpt = torch.load("checkpoints/v26_correct_v24r_seqwrite_continue/stage1_n30.pt", map_location="cuda")["agent_state_dict"]

W = ckpt["successor_predict_head.weight"]  # (5, 10)
b = ckpt["successor_predict_head.bias"]    # (5,)
print(f"successor_predict_head W shape: {W.shape}, b shape: {b.shape}")
print(f"W:\n{W}")
print(f"b:\n{b}")

# Test various inputs
inputs = {
    "all-zeros": torch.zeros(10, device="cuda"),
    "all-ones": torch.ones(10, device="cuda"),
    "α=1, β=0": torch.tensor([1.0,1,1,1,1, 0,0,0,0,0], device="cuda"),
    "α=1, β=1": torch.tensor([1.0,1,1,1,1, 1,1,1,1,1], device="cuda"),
    "α=0.5, β=0.5": torch.tensor([0.5]*10, device="cuda"),
    "varying β [0,0.5,1,0.5,0]": torch.tensor([1.0,1,1,1,1, 0,0.5,1,0.5,0], device="cuda"),
}
for name, inp in inputs.items():
    logits = inp @ W.T + b
    print(f"  {name}: logits = {logits.cpu().numpy().tolist()}, argmax = {logits.argmax().item()}")
