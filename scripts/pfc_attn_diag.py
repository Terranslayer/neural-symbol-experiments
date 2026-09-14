"""V7 PFC compare diagnostics: attention weights + output PCA.

Two key questions:
  1. Attention pattern: when refining h_α (output token 0), how does PFC distribute
     attention across [h_α (Mamba), h_β (Mamba), cnn_α_readback (CNN), cnn_β_summary (CNN)]?
     If row-0 weights concentrate on token 0 (self) → PFC just passes Mamba through.
     If non-trivial weight on tokens 2,3 → PFC IS attending to CNN signals.

  2. Output PCA: for the refined h_α_pfc (PFC output token 0), do PC1 inversions
     improve over Mamba's pure h_α (45% inversions at stage 2)?
     If yes → PFC integration works, helps break Mamba's 1D collapse.
     If similar → PFC is passthrough, no benefit.
"""
import argparse, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.core.scene import SceneConfig, sample_training_batch  # noqa
from scripts.inspect_checkpoint import load_agent  # noqa


class CapturingMHA(nn.Module):
    """Wrap MultiheadAttention to force need_weights=True and capture per-head attn."""
    def __init__(self, mha):
        super().__init__()
        self.mha = mha
        self.captured = []  # list of (B, num_heads, num_tokens, num_tokens)
        # Expose attributes that TransformerEncoder accesses (batch_first, etc.)
        # nn.Module __getattr__ first looks in _parameters/_buffers/_modules then raises.
        # We forward unknown attributes to self.mha.

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mha, name)

    def forward(self, *args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False
        out, attn = self.mha(*args, **kwargs)
        self.captured.append(attn.detach())
        return out, attn


class CapturingPFC(nn.Module):
    """Wrap PFC encoder to capture its full output."""
    def __init__(self, pfc):
        super().__init__()
        self.pfc = pfc
        self.captured = []

    def forward(self, x, *args, **kwargs):
        out = self.pfc(x, *args, **kwargs)
        self.captured.append(out.detach())
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--n-max", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    agent, _, scene_cfg = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        v7_mode=True,
    )
    # Keep in train mode so PyTorch's TransformerEncoder slow-path (not fastpath)
    # routes through our CapturingMHA wrapper. no_grad below prevents updates.
    agent.train(True)
    # But disable scratch noise / dropout for clean diag
    if hasattr(agent, "agent_cfg"):
        agent.agent_cfg.write_noise_std = 0.0
    device = next(agent.parameters()).device

    # Wrap pfc_compare's MHA in each encoder layer
    assert isinstance(agent.pfc_compare, nn.TransformerEncoder), \
        f"Expected TransformerEncoder, got {type(agent.pfc_compare)}"
    wrapped_mhas = []
    for layer in agent.pfc_compare.layers:
        wrapper = CapturingMHA(layer.self_attn)
        layer.self_attn = wrapper
        wrapped_mhas.append(wrapper)

    # Wrap PFC compare itself for output capture
    pfc_wrapper = CapturingPFC(agent.pfc_compare)
    agent.pfc_compare = pfc_wrapper
    if agent.pfc is wrapped_mhas[0].mha or agent.pfc is agent.pfc_compare.pfc:
        agent.pfc = pfc_wrapper  # shared transformer encoder
    elif agent.pfc is not None:
        # PFC at write is shared with compare in transformer mode
        pass

    cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = scene_cfg.K
    cfg.equal_weight = 0.20
    cfg.near_weight = 0.80
    cfg.alpha_distribution = "uniform"
    rng = np.random.default_rng(0)

    # Per-N collection (use first compare block's data per sample)
    n_to_attn_per_layer = defaultdict(lambda: [[] for _ in wrapped_mhas])
    n_to_pfc_token0 = defaultdict(list)  # h_α_pfc
    n_to_pfc_token1 = defaultdict(list)  # h_β_pfc

    for _ in range(args.n_batches):
        for w in wrapped_mhas:
            w.captured.clear()
        pfc_wrapper.captured.clear()

        inputs, _, ci, metas = sample_training_batch(
            batch_size=args.batch_size, n_max_stage=args.n_max,
            config=cfg, rng=rng,
        )
        with torch.no_grad():
            _ = agent(inputs.to(device), ci.to(device))

        # Each compare block calls PFC encoder, which fires each MHA layer once.
        # K=8 compare blocks per sample → 8 entries per layer in captured.
        # Take first block's data (call index 0).
        # Note: PFC at write also fires the SAME wrapped MHAs (shared). Write fires W=5 times
        # before any compare block. So actually captured[0..4] = write blocks, [5..12] = compare blocks.
        # But with v7_mode and pfc_compare_drop_raw_scratch=True, PFC at write has 2+W tokens
        # while compare has 4 tokens. Different shapes!
        # Safer: filter for compare-shape entries.

        # Find first capture with token count = 4 (compare with v7_mode)
        attn_per_layer = []
        for layer_idx, w in enumerate(wrapped_mhas):
            cap = w.captured
            compare_caps = [c for c in cap if c.shape[-1] == 4]
            if not compare_caps:
                print(f"WARN: layer {layer_idx} no 4-token captures. Sizes: {[c.shape for c in cap]}")
                return
            attn_per_layer.append(compare_caps[0].cpu().numpy())  # (B, heads, 4, 4)

        # PFC output (compare): same logic, find 4-token output
        compare_outs = [o for o in pfc_wrapper.captured if o.shape[1] == 4]
        if not compare_outs:
            print(f"WARN: pfc no 4-token outputs. Sizes: {[o.shape for o in pfc_wrapper.captured]}")
            return
        pfc_out = compare_outs[0].cpu().numpy()  # (B, 4, d)

        for b, m in enumerate(metas):
            n = m["N_alpha"]
            for layer_idx in range(len(wrapped_mhas)):
                n_to_attn_per_layer[n][layer_idx].append(attn_per_layer[layer_idx][b])
            n_to_pfc_token0[n].append(pfc_out[b, 0])
            n_to_pfc_token1[n].append(pfc_out[b, 1])

    # ===== Report 1: Per-N attention pattern (last layer) =====
    print(f"\n{'='*78}\n=== ATTENTION ANALYSIS ({args.ckpt.name}) ===\n{'='*78}")
    print(f"  Tokens: 0=h_α(Mamba)  1=h_β(Mamba)  2=cnn_α_readback  3=cnn_β_summary")

    n_values = sorted(n_to_attn_per_layer.keys())
    n_to_show = [n for n in n_values if n in [1, 2, 3, 5, 10, 15, 20, 25, 30] or n == n_values[-1]]

    for layer_idx in range(len(wrapped_mhas)):
        print(f"\n--- Layer {layer_idx} ---")
        for n in n_to_show:
            arrs = np.stack(n_to_attn_per_layer[n][layer_idx], axis=0)  # (#samples, heads, 4, 4)
            mean_attn = arrs.mean(axis=0)  # (heads, 4, 4)
            num_heads = mean_attn.shape[0]
            print(f"\n  N={n} (n={arrs.shape[0]}), {num_heads} heads:")
            for h in range(num_heads):
                # Row 0: how token 0 (h_α output) attends to all tokens
                row0 = mean_attn[h, 0]
                # Row 1: how token 1 (h_β output) attends
                row1 = mean_attn[h, 1]
                print(f"    head{h} | row0(h_α): [α:{row0[0]:.2f} β:{row0[1]:.2f} cnn_α:{row0[2]:.2f} cnn_β:{row0[3]:.2f}]"
                      f" | row1(h_β): [α:{row1[0]:.2f} β:{row1[1]:.2f} cnn_α:{row1[2]:.2f} cnn_β:{row1[3]:.2f}]")

    # ===== Report 2: Per-head dominant attention destination =====
    print(f"\n{'='*78}\n=== HEAD SPECIALIZATION (across all N, layer {len(wrapped_mhas)-1}) ===\n{'='*78}")
    last_layer = len(wrapped_mhas) - 1
    all_attn_last = np.concatenate([
        np.stack(n_to_attn_per_layer[n][last_layer], axis=0)
        for n in n_values
    ], axis=0)  # (total_samples, heads, 4, 4)
    mean_attn_overall = all_attn_last.mean(axis=0)  # (heads, 4, 4)
    num_heads = mean_attn_overall.shape[0]
    print(f"  For each head, average over all samples + N values:")
    for h in range(num_heads):
        for src_token in range(4):
            row = mean_attn_overall[h, src_token]
            tok_names = ["h_α", "h_β", "cnn_α_rb", "cnn_β_sum"]
            dom_idx = int(np.argmax(row))
            dom_pct = row[dom_idx] * 100
            print(f"    head{h} {tok_names[src_token]:>10} → most attends to {tok_names[dom_idx]} ({dom_pct:.1f}%)")
        print()

    # ===== Report 3: PCA on h_α_pfc (PFC compare output token 0) =====
    print(f"\n{'='*78}\n=== PFC OUTPUT PCA (h_α_pfc = compare output token 0) ===\n{'='*78}")
    H_list = []
    N_list = []
    for n in n_values:
        for h_arr in n_to_pfc_token0[n]:
            H_list.append(h_arr)
            N_list.append(n)
    H = np.stack(H_list, axis=0)  # (total, d)
    N_arr = np.array(N_list)

    Hc = H - H.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    var_ratio = (S ** 2) / max(H.shape[0] - 1, 1)
    var_ratio /= var_ratio.sum()
    PC1 = Hc @ Vt[0]

    rho_N, _ = spearmanr(PC1, N_arr)
    rho_logN, _ = spearmanr(PC1, np.log(N_arr + 1e-3))

    print(f"  PC1 var: {var_ratio[0]:.3f}, PC2: {var_ratio[1]:.3f}, PC3: {var_ratio[2]:.3f}")
    print(f"  PC1 vs N spearman: {rho_N:+.3f}")
    print(f"  PC1 vs log(N) spearman: {rho_logN:+.3f}")

    # Within-N std vs adj-N gap on PC1 (resolution metric)
    pc1_per_n = {n: PC1[N_arr == n] for n in n_values}
    within_std = np.mean([v.std() for v in pc1_per_n.values() if len(v) > 1])
    means = np.array([pc1_per_n[n].mean() for n in n_values])
    adj_gaps = np.abs(np.diff(means))
    print(f"\n  within-N std (avg): {within_std:.3f}")
    print(f"  adj-N gap on PC1: mean={adj_gaps.mean():.3f}, min={adj_gaps.min():.3f}")
    print(f"  ratio (mean_gap / within_std): {adj_gaps.mean()/max(within_std,1e-9):.3f}  (>1 = separable)")

    # Inversions
    inversions = 0
    direction = 1 if means[-1] > means[0] else -1
    for i in range(len(means) - 1):
        if direction * (means[i+1] - means[i]) < 0:
            inversions += 1
    inv_pct = inversions / max(len(means) - 1, 1) * 100
    print(f"  PC1 inversions: {inversions}/{len(means)-1} = {inv_pct:.1f}%")
    print(f"  (compare to pure Mamba — Stage 2 was 45%; lower = PFC fixes Mamba's 1D collapse)")


if __name__ == "__main__":
    main()
