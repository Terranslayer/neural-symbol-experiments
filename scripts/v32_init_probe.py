"""V32 init probe: locate where N signal collapses in forward path.

DIAG_FOR: v32_multigate_ec

At fresh init (no training), runs forward on signals of varying N and measures
per-layer "N-explained variance":
    - PC1 variance ratio
    - |Spearman ρ(PC1_score, N)|
    - mean inter-N std (variance across N for fixed positions)
    - mean intra-N std (variance within fixed N)
    - ratio (inter / intra)

The layer where ratio drops to ~1.0 = layer where N info dies.
Also prints per-N mean+std of write_head raw[:, w] to see if cells are
constant across N at init (smoking-gun for "dead at init").

Optionally loads a trained ckpt for comparison via --ckpt PATH.

Usage:
    python scripts/v32_init_probe.py
    python scripts/v32_init_probe.py --ckpt checkpoints/v32_ec_default/agent_0.pt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from backend.core.ec_agent import ECAgent, ECAgentConfig
from backend.core.scene import _scan_world_complex


# ----------------------------------------------------------------------------
# Activation capture
# ----------------------------------------------------------------------------

@dataclass
class CapturedLayer:
    """One layer's activations across all (N, repeat) samples.

    acts: (n_samples, dim) numpy array. Each row corresponds to one (N, repeat)
        sample in the order produced by `run_probe`.
    N_per_sample: (n_samples,) numpy array of N for each row.
    """
    name: str
    acts: np.ndarray  # (n_samples, dim)
    N_per_sample: np.ndarray  # (n_samples,)


def capture_forward(agent: ECAgent, signal_batch: torch.Tensor):
    """Run signal_batch through agent.encode_signal, capturing intermediates.

    Returns dict with:
        cnn_chunks: (B, n_chunks, d)
        pfc_states_per_chunk: list of (B, d) — len n_chunks
        lru_h_per_chunk: list of (B, d) — len n_chunks
        pfc_final: (B, d)
        raw_scratch: (B, W)
        q_scratch: (B, W)
    """
    feat = agent._signal_to_input_feature(signal_batch)
    stash: dict = {}
    pfc_state = agent.encoder(
        feat, n_chunks=agent.cfg.n_chunks_phase1, token_type_id=0, stash=stash
    )
    raw, q = agent.write_head(pfc_state, quantize_levels=agent.cfg.quantize_levels)
    return {
        "cnn_chunks": stash["cnn_chunks"].cpu().numpy(),
        "pfc_states_per_chunk": [t.cpu().numpy() for t in stash["pfc_states_per_chunk"]],
        "lru_h_per_chunk": [t.cpu().numpy() for t in stash["lru_h_per_chunk"]],
        "pfc_final": pfc_state.detach().cpu().numpy(),
        "raw_scratch": raw.detach().cpu().numpy(),
        "q_scratch": q.detach().cpu().numpy(),
    }


def run_probe(agent: ECAgent, N_values, n_repeats: int, L: int, device: str, seed: int = 0):
    """Run forward on signals of varying N. Returns list[CapturedLayer]
    plus per-N raw scratch tensor for visualization."""
    rng = np.random.default_rng(seed)
    signals = []
    N_per_sample = []
    for N in N_values:
        for _ in range(n_repeats):
            sig = _scan_world_complex(N=N, L=L, rng=rng)  # (L,)
            signals.append(sig)
            N_per_sample.append(N)
    signals_np = np.stack(signals, axis=0).astype(np.float32)  # (n_total, L)
    N_per_sample = np.array(N_per_sample, dtype=np.int64)

    # Run forward in one batch
    sig_t = torch.from_numpy(signals_np).to(device)
    with torch.no_grad():
        captured = capture_forward(agent, sig_t)

    # Flatten per-chunk activations into layer-named entries
    layers: list[CapturedLayer] = []
    cnn = captured["cnn_chunks"]  # (n_total, n_chunks, d)
    for c in range(cnn.shape[1]):
        layers.append(CapturedLayer(
            name=f"CNN chunk {c}",
            acts=cnn[:, c, :],
            N_per_sample=N_per_sample,
        ))
    for c, h in enumerate(captured["lru_h_per_chunk"]):
        layers.append(CapturedLayer(
            name=f"LRU lru_token c{c}",
            acts=h,
            N_per_sample=N_per_sample,
        ))
    for c, p in enumerate(captured["pfc_states_per_chunk"]):
        layers.append(CapturedLayer(
            name=f"PFC state c{c}",
            acts=p,
            N_per_sample=N_per_sample,
        ))
    layers.append(CapturedLayer(
        name="PFC final",
        acts=captured["pfc_final"],
        N_per_sample=N_per_sample,
    ))
    layers.append(CapturedLayer(
        name="WriteHead raw",
        acts=captured["raw_scratch"],
        N_per_sample=N_per_sample,
    ))
    layers.append(CapturedLayer(
        name="WriteHead q",
        acts=captured["q_scratch"],
        N_per_sample=N_per_sample,
    ))
    return layers, captured["raw_scratch"], captured["q_scratch"], N_per_sample


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def pc1_variance_ratio_and_corr(acts: np.ndarray, N_per_sample: np.ndarray):
    """Compute (pc1_var_ratio, |spearman(pc1, N)|).

    acts: (n, d). If d == 1, treat acts[:, 0] as PC1.
    """
    n, d = acts.shape
    if n < 2:
        return float("nan"), float("nan")
    # Center
    X = acts - acts.mean(axis=0, keepdims=True)
    # SVD for PC1
    if d == 1:
        pc1 = X[:, 0]
        pc1_var = float(np.var(pc1))
        total_var = pc1_var if pc1_var > 0 else 1e-12
        ratio = 1.0
    else:
        # Use np.linalg.svd on covariance via X
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            total_var = float((S ** 2).sum() / max(1, n - 1))
            pc1_var = float((S[0] ** 2) / max(1, n - 1))
            ratio = pc1_var / total_var if total_var > 0 else float("nan")
            pc1 = U[:, 0] * S[0]
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
    try:
        rho, _ = spearmanr(pc1, N_per_sample)
        rho_abs = abs(rho) if not np.isnan(rho) else float("nan")
    except Exception:
        rho_abs = float("nan")
    return ratio, rho_abs


def inter_intra_std(acts: np.ndarray, N_per_sample: np.ndarray):
    """For each dim, compute:
        intra: mean over N of std-within-N (samples sharing same N)
        inter: std over N of mean-within-N
    Return mean over dims.
    """
    Ns = np.unique(N_per_sample)
    # Means per N: (len(Ns), d)
    means = np.stack([acts[N_per_sample == n].mean(axis=0) for n in Ns], axis=0)
    intras = np.stack([acts[N_per_sample == n].std(axis=0) for n in Ns], axis=0)
    # mean across N of within-N std (per-dim) → average across dims
    mean_intra = float(intras.mean())
    mean_inter = float(means.std(axis=0).mean())
    ratio = mean_inter / mean_intra if mean_intra > 1e-12 else float("inf")
    return mean_inter, mean_intra, ratio


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def format_table(rows):
    """rows: list of (name, pc1_var, rho, inter, intra, ratio)."""
    header = ["Layer", "PC1 var ratio", "|rho(N,PC1)|",
              "mean inter-N std", "mean intra-N std", "inter/intra"]
    widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    out_lines = []
    fmt_row = " | ".join("{:<" + str(w) + "}" for w in widths)
    out_lines.append(fmt_row.format(*header))
    out_lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        out_lines.append(fmt_row.format(*r))
    return "\n".join(out_lines)


def print_per_n_scratch(raw: np.ndarray, q: np.ndarray, N_per_sample: np.ndarray, W: int):
    Ns = np.unique(N_per_sample)
    print("\nPer-N write_head RAW (pre-sigmoid logits): mean ± std across repeats")
    print("  N  | " + "  ".join([f"  cell{w}_mean(std)  " for w in range(W)]))
    for n in Ns:
        mask = N_per_sample == n
        sub = raw[mask]
        cells_str = "  ".join([f"{sub[:, w].mean():+.3f}({sub[:, w].std():.3f})" for w in range(W)])
        print(f"  {n:2d} | {cells_str}")

    print("\nPer-N write_head Q (post-quantize): mean (= probability mass in each level via mean across repeats)")
    print("  N  | " + "  ".join([f"  cell{w}_mean(std)  " for w in range(W)]))
    for n in Ns:
        mask = N_per_sample == n
        sub = q[mask]
        cells_str = "  ".join([f"{sub[:, w].mean():.3f}({sub[:, w].std():.3f})" for w in range(W)])
        print(f"  {n:2d} | {cells_str}")

    # Distinct codes
    print("\nDistinct quantized codes observed per N:")
    for n in Ns:
        mask = N_per_sample == n
        sub = q[mask]
        codes = {tuple(np.round(row, 4)) for row in sub}
        print(f"  N={n:2d}: {len(codes)} distinct codes (out of {sub.shape[0]} samples)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Optional trained ckpt path. If omitted, fresh-init.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--L", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ECAgentConfig()  # uses defaults — W=3, q=5 per current code
    print(f"[probe] ECAgentConfig: W={cfg.W} q={cfg.quantize_levels} "
          f"L={cfg.L} d_model={cfg.d_model} d_state={cfg.d_state} "
          f"n_chunks_phase1={cfg.n_chunks_phase1}")
    agent = ECAgent(cfg).to(args.device)
    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
        if "agent_state_dict" in state:
            agent.load_state_dict(state["agent_state_dict"])
        else:
            agent.load_state_dict(state)
        print(f"[probe] loaded trained ckpt: {args.ckpt}")
    else:
        print("[probe] using FRESH-INIT agent (no training)")
    agent.train(False)
    print(f"[probe] num params: {agent.num_parameters():,}")

    N_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    print(f"[probe] N_values={N_values} n_repeats={args.n_repeats} "
          f"total samples = {len(N_values) * args.n_repeats}")

    layers, raw, q, N_per_sample = run_probe(
        agent, N_values, args.n_repeats, args.L, args.device, seed=args.seed
    )

    rows = []
    for layer in layers:
        pc1_var, rho = pc1_variance_ratio_and_corr(layer.acts, layer.N_per_sample)
        inter, intra, ratio = inter_intra_std(layer.acts, layer.N_per_sample)
        rows.append((
            layer.name,
            f"{pc1_var:.4f}",
            f"{rho:.4f}",
            f"{inter:.4f}",
            f"{intra:.4f}",
            f"{ratio:.3f}",
        ))
    print("\n=== Per-layer N-preservation summary ===")
    print(format_table(rows))

    # Identify collapse layer
    print("\n=== Collapse analysis ===")
    prev_ratio = None
    for layer, row in zip(layers, rows):
        ratio = float(row[5]) if row[5] != "inf" else float("inf")
        marker = ""
        if prev_ratio is not None and prev_ratio > 1.5 and ratio < 1.5:
            marker = "  <-- COLLAPSE HERE (ratio dropped below 1.5)"
        elif ratio < 1.5 and prev_ratio is None:
            marker = "  <-- DEAD at input (ratio < 1.5 from start)"
        print(f"  {layer.name:<25s}: ratio = {ratio:.3f}{marker}")
        prev_ratio = ratio

    print_per_n_scratch(raw, q, N_per_sample, W=cfg.W)

    # Summary verdict
    raw_per_n_means = []
    for n in np.unique(N_per_sample):
        mask = N_per_sample == n
        raw_per_n_means.append(raw[mask].mean(axis=0))
    raw_per_n_means = np.stack(raw_per_n_means)  # (n_N, W)
    spread_per_cell = raw_per_n_means.std(axis=0)  # (W,)
    print("\n=== Smoking-gun summary ===")
    print(f"Per-cell std of mean-raw-across-N: {spread_per_cell}")
    print(f"Max per-cell spread: {spread_per_cell.max():.4f}, "
          f"min: {spread_per_cell.min():.4f}")
    if spread_per_cell.max() < 0.05:
        print("VERDICT: write_head raw is ~CONSTANT across N — DEAD AT INIT")
    elif spread_per_cell.max() < 0.2:
        print("VERDICT: write_head raw shows weak N dependence — borderline")
    else:
        print("VERDICT: write_head raw IS N-dependent at init "
              "— signal makes it through forward path")


if __name__ == "__main__":
    main()
