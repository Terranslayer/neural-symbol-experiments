# -*- coding: utf-8 -*-
"""Throwaway (2026-06-08): V35 self-inverse eye-decode probe.

Localizes the same-medium decode failure across three layers by pushing hand-set
ideal base-3 codes through {multi-scale CNN eye A, single kernel=5 eye B} -> frozen
log_g=-20 substrate, measuring per-position / per-level / place-value recovery at two
stages (eye feat + substrate residual). See
docs/design.md
Diagnostic only: no training, no forward-path change. `--selftest` checks the pure
logic without a checkpoint.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------- input gen ----
def base3_digits(N, W=5):
    """Return the W base-3 digits of N, low place first: [(N//3^k) mod 3]."""
    return [(N // (3 ** k)) % 3 for k in range(W)]


def digits_to_levels(digits):
    """Map base-3 digits {0,1,2} to scratch alphabet levels {0, 0.5, 1.0}."""
    return [d / 2.0 for d in digits]


def make_ideal_codes(n_min, n_max, W=5):
    """Return (Ns (M,), levels (M, W)) for ideal base-3 codes over [n_min, n_max]."""
    Ns = list(range(n_min, n_max + 1))
    levels = np.array([digits_to_levels(base3_digits(N, W)) for N in Ns], dtype=np.float32)
    return np.array(Ns), levels


def make_onehot_probes(W=5):
    """W codes, code k = level 1.0 at cell k only. Shape (W, W)."""
    return np.eye(W, dtype=np.float32)


def make_perlevel_probes(W=5):
    """For each cell k, three codes with cell k at {0, 0.5, 1.0}, others 0.
    Returns dict cell_k -> levels array (3, W)."""
    out = {}
    for k in range(W):
        codes = np.zeros((3, W), dtype=np.float32)
        codes[1, k] = 0.5
        codes[2, k] = 1.0
        out[k] = codes
    return out


# ------------------------------------------------------------------ metrics ----
def offdiag_cosine(mat):
    """mat: (M, D). Row-normalize, return (mean_offdiag_cos, max_offdiag_cos).
    High off-diagonal cosine => the M rows are near-collinear (aggregated/blurred).
    Low => the rows are distinct (per-item structure preserved)."""
    x = np.asarray(mat, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(n, 1e-9, None)
    g = x @ x.T
    M = g.shape[0]
    off = g[~np.eye(M, dtype=bool)]
    return float(off.mean()), float(off.max())


def per_position_sep(vecs):
    """vecs: (W, F) one feature vector per one-hot position probe (flattened feat or
    residual). Returns (mean_offdiag_cos, max_offdiag_cos). Eye A (aggregating) -> high;
    eye B (local) -> low."""
    return offdiag_cosine(vecs)


def per_level_fidelity(level_vecs):
    """level_vecs: (3, F) outputs for cell levels {0, 0.5, 1.0}. Returns
    (min_pairwise_dist, monotone_ok). distinguishable if min dist > 0; monotone_ok if
    the 0.5 vector lies between 0 and 1.0 on the dominant axis."""
    v = np.asarray(level_vecs, dtype=np.float64)
    d01 = np.linalg.norm(v[0] - v[1])
    d12 = np.linalg.norm(v[1] - v[2])
    d02 = np.linalg.norm(v[0] - v[2])
    min_pair = float(min(d01, d12, d02))
    axis = v[2] - v[0]
    na = np.linalg.norm(axis)
    if na < 1e-9:
        return min_pair, False
    proj = (v - v[0]) @ (axis / na)            # scalar projection of each level
    monotone_ok = bool(proj[0] <= proj[1] <= proj[2] and proj[2] > proj[0])
    return min_pair, monotone_ok


def place_value_recovery(res, Ns, n_inrange_max=30):
    """res: (M, R) residuals; Ns: (M,). Fit Ridge res->N on N<=n_inrange_max, report
    R^2 in-range AND on the extrapolation split (N>n_inrange_max)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    res = np.asarray(res, dtype=np.float64)
    Ns = np.asarray(Ns)
    tr = Ns <= n_inrange_max
    ex = ~tr
    assert tr.sum() >= 2, f"too few in-range samples ({int(tr.sum())}) for Ridge fit (n_inrange_max={n_inrange_max})"
    model = Ridge(alpha=1e-3).fit(res[tr], Ns[tr])
    r2_in = r2_score(Ns[tr], model.predict(res[tr]))
    r2_ex = (r2_score(Ns[ex], model.predict(res[ex])) if ex.sum() >= 2 else float("nan"))
    return float(r2_in), float(r2_ex)


def bump_peak_slope(feats, Ns, thresh_frac=0.5):
    """feats: list of (T, d) per-signal feature maps. Count local peaks of the per-step
    feature magnitude above thresh_frac*max, regress count vs N, return slope.
    Eye that resolves bumps -> slope ~ 1; aggregating eye -> slope ~ 0 (saturates)."""
    counts = []
    for f in feats:
        mag = np.linalg.norm(np.asarray(f, dtype=np.float64), axis=1)   # (T,)
        if mag.max() < 1e-9:
            counts.append(0)
            continue
        thr = thresh_frac * mag.max()
        above = mag > thr
        # count rising edges = number of distinct peaks
        counts.append(int(np.sum(above[1:] & ~above[:-1]) + (1 if above[0] else 0)))
    counts = np.array(counts, dtype=np.float64)
    Ns = np.asarray(Ns, dtype=np.float64)
    slope = float(np.polyfit(Ns, counts, 1)[0])
    return slope


# --------------------------------------------------------------------- eyes ----
def encode_with(dp_convs, dp_proj, seg):
    """Replicates mamba_agent _v35_forward `encode()`. seg: (B, T, input_dim).
    dp_convs: nn.ModuleList of Conv1d; dp_proj: Linear(total_ch, d). Returns (B, T, d).
    Works for eye A (3 convs 5/21/51) and eye B (single k=5 conv) alike."""
    x = seg.transpose(1, 2)                                  # (B, in, T)
    conv_outs = [conv(x) for conv in dp_convs]
    feat = torch.cat(conv_outs, dim=1).transpose(1, 2)       # (B, T, total_ch)
    return dp_proj(feat)


def build_eye_b(input_dim, cps, d, device):
    """Single kernel=5 conv eye at LOCAL-IDENTITY (delta) init: output channel 0 passes
    input channel 0 (the signal/scratch channel) through unchanged at the center tap, so
    feat[t] depends only on the local input[t] (no cross-position aggregation). Returns
    (dp_convs (ModuleList of one conv), dp_proj) matching the eye-A interface.

    Biases zeroed. Only conv out-channel 0 is nonzero, so feat[t] = proj.weight[:,0] *
    input[t, ch0]: per-position locality is by CONSTRUCTION (the impulse stays at its
    timestep), not by an informed proj embedding. Hence eye B's pos_feat_cos is ~0 BY
    DESIGN -- the meaningful contrast is eye A's pos_feat_cos being HIGH (its wide 21/51
    kernels spread each impulse across timesteps, collapsing the positions)."""
    conv = nn.Conv1d(input_dim, cps, kernel_size=5, padding=2, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 2] = 1.0          # out-ch 0 = delta on in-ch 0 (center tap)
        conv.bias.zero_()
    proj = nn.Linear(cps, d)                # locality is carried by the conv
    with torch.no_grad():
        proj.bias.zero_()                   # zero bias so proj(0)=0 (neutral passthrough)
    dp_convs = nn.ModuleList([conv]).to(device)
    proj = proj.to(device)
    return dp_convs, proj


# ------------------------------------------------------------------ forward ----
def load_agent(ckpt_path, device):
    """Load MambaAgent + restore agent_cfg/scene_cfg (pattern from _v35_dump_codebook).
    Returns (agent, scene_cfg)."""
    from backend.core.scene import SceneConfig
    from backend.core.mamba_agent import MambaAgent, MambaAgentConfig
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    acfg, scfg = ckpt["agent_cfg"], ckpt["scene_cfg"]
    if isinstance(acfg, dict):
        acfg = MambaAgentConfig(**acfg)
    if isinstance(scfg, dict):
        scfg = SceneConfig(**scfg)
    agent = MambaAgent(acfg, scfg)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent = agent.to(device).train(False)
    return agent, scfg


def run_decode(dp_convs, dp_proj, substrate, levels, device):
    """levels: (M, W) scratch alphabet codes. Build seg_readback (M, W, 3) with ch0=levels,
    ch1=ch2=0 (mirrors mamba_agent.py:3041-3047), push through eye then substrate.
    Returns (feats (M, W, d) np, residuals (M, R) np)."""
    M, W = levels.shape
    lv = torch.tensor(levels, dtype=torch.float32, device=device)
    zeros = torch.zeros_like(lv)
    seg = torch.stack([lv, zeros, zeros], dim=-1)            # (M, W, 3)
    with torch.no_grad():
        feat = encode_with(dp_convs, dp_proj, seg)           # (M, W, d)
        residuals, _aux = substrate(feat, capture_for_pc=False)
    return feat.detach().cpu().numpy(), residuals.detach().cpu().numpy()


def make_world_signals(n_min, n_max, scene_cfg, seed=12345):
    """One raw alpha signal per N via build_episode. Returns (Ns, alpha_segs list of (L,3))."""
    from backend.core.scene import build_episode, sample_beta_count
    rng = np.random.default_rng(seed)
    L = scene_cfg.L
    Ns, segs = [], []
    for N in range(n_min, n_max + 1):
        betas = [sample_beta_count(N, n_max, scene_cfg, rng) for _ in range(scene_cfg.K)]
        inp, _l, _c, _m = build_episode(N, betas, scene_cfg, rng)   # (T, 3) numpy/tensor
        inp = inp if isinstance(inp, np.ndarray) else inp.numpy()
        seg = inp[:L].astype(np.float32).copy()
        seg[:, 1:3] = 0.0          # same-medium: zero phase markers (mirrors _v35_forward)
        segs.append(seg)
        Ns.append(N)
    return np.array(Ns), segs


def run_world(dp_convs, dp_proj, substrate, segs, device):
    """segs: list of (L, 3). Push each through eye -> substrate. Returns
    (feats list of (L, d) np, residuals (M, R) np)."""
    feats, res_list = [], []
    with torch.no_grad():
        for seg in segs:
            s = torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0)  # (1,L,3)
            feat = encode_with(dp_convs, dp_proj, s)         # (1, L, d)
            residuals, _aux = substrate(feat, capture_for_pc=False)
            feats.append(feat[0].detach().cpu().numpy())
            res_list.append(residuals[0].detach().cpu().numpy())
    return feats, np.array(res_list)


# ------------------------------------------------------------- orchestration ----
def eval_eye(name, dp_convs, dp_proj, substrate, scene_cfg, n_max, device):
    """Run one eye through decode + world paths; return a dict of metric scalars."""
    W = scene_cfg.W
    # decode: ideal base-3 codes
    Ns, levels = make_ideal_codes(1, n_max, W)
    feat_dec, res_dec = run_decode(dp_convs, dp_proj, substrate, levels, device)
    # per-position separability via one-hot probes (feat + residual stages)
    oh = make_onehot_probes(W)
    feat_oh, res_oh = run_decode(dp_convs, dp_proj, substrate, oh, device)
    pos_feat_mean, _ = per_position_sep(feat_oh.reshape(W, -1))
    pos_res_mean, _ = per_position_sep(res_oh.reshape(W, -1))
    # per-level fidelity (residual stage), averaged over cells
    perlevel = make_perlevel_probes(W)
    mindists, monos = [], []
    for k in range(W):
        _f, r = run_decode(dp_convs, dp_proj, substrate, perlevel[k], device)
        md, mono = per_level_fidelity(r)
        mindists.append(md); monos.append(mono)
    # place-value recovery from decode residuals
    r2_in, r2_ex = place_value_recovery(res_dec, Ns, n_inrange_max=30)
    # world-read structural: bump resolvability at feat stage
    wNs, segs = make_world_signals(1, min(30, n_max), scene_cfg)
    wfeats, _wres = run_world(dp_convs, dp_proj, substrate, segs, device)
    world_slope = bump_peak_slope(wfeats, wNs)
    return {
        "name": name,
        "pos_feat_cos": pos_feat_mean,     # LOWER = positions preserved by eye
        "pos_res_cos": pos_res_mean,       # LOWER = positions survive substrate
        "level_mindist": float(np.mean(mindists)),
        "level_monotone_frac": float(np.mean(monos)),
        "pv_r2_in": r2_in,
        "pv_r2_extrap": r2_ex,
        "world_bump_slope": world_slope,
    }


def selftest():
    # base-3 generator (the whole probe depends on this being correct)
    assert base3_digits(1, 5) == [1, 0, 0, 0, 0], base3_digits(1, 5)
    assert base3_digits(13, 5) == [1, 1, 1, 0, 0], base3_digits(13, 5)   # 1+3+9
    assert base3_digits(80, 5) == [2, 2, 2, 2, 0], base3_digits(80, 5)   # 2+6+18+54
    assert base3_digits(81, 5) == [0, 0, 0, 0, 1], base3_digits(81, 5)   # cell4 opens
    assert digits_to_levels([0, 1, 2, 0, 1]) == [0.0, 0.5, 1.0, 0.0, 0.5]
    Ns, lv = make_ideal_codes(1, 90, 5)
    assert lv.shape == (90, 5) and Ns.shape == (90,)
    assert make_onehot_probes(5).shape == (5, 5)
    assert make_perlevel_probes(5)[0].shape == (3, 5)

    # offdiag_cosine: identity rows separable (low), constant rows collapsed (high)
    sep_mean, _ = offdiag_cosine(np.eye(5))
    col_mean, _ = offdiag_cosine(np.ones((5, 7)))
    assert sep_mean < 0.1 and col_mean > 0.99, (sep_mean, col_mean)
    # per_level_fidelity: monotone scaled vectors -> distinguishable + monotone
    lv = np.stack([np.zeros(4), 0.5 * np.ones(4), 1.0 * np.ones(4)])
    mind, mono = per_level_fidelity(lv)
    assert mind > 0 and mono, (mind, mono)
    # collapsed levels (all same) -> not monotone, zero min dist
    mind0, mono0 = per_level_fidelity(np.ones((3, 4)))
    assert mind0 == 0.0 and not mono0
    # place_value_recovery: residuals == ideal base-3 digits -> R^2 ~ 1 both splits
    # NOTE: n_inrange_max=85 (not 30) so that cell-4 (activates at N>=81) appears in
    # training, making all 5 digit basis functions observable and extrap R^2 well-defined.
    # The real eval uses n_inrange_max=30 which is the in-range/extrap split for ckpt eval;
    # this selftest verifies only that the metric function itself is correct.
    Ns2, lv2 = make_ideal_codes(1, 90, 5)
    digits = lv2 * 2.0                              # exact base-3 digits as "residuals"
    r2_in, r2_ex = place_value_recovery(digits, Ns2, n_inrange_max=85)
    assert r2_in > 0.999 and r2_ex > 0.999, (r2_in, r2_ex)
    # bump_peak_slope: N delta-bumps spaced out -> slope ~ 1
    feats = []
    for N in range(1, 11):
        T = 60
        f = np.zeros((T, 2))
        for j in range(N):
            f[5 + j * 5, 0] = 1.0
        feats.append(f)
    s = bump_peak_slope(feats, list(range(1, 11)))
    assert s > 0.9, s

    # eye B delta init: a single-channel impulse at position k -> feat nonzero only near k
    dev = torch.device("cpu")
    dp_convs_b, proj_b = build_eye_b(input_dim=3, cps=4, d=8, device=dev)
    W = 5
    seg = torch.zeros(1, W, 3)
    seg[0, 2, 0] = 1.0                       # impulse at position 2, channel 0
    feat = encode_with(dp_convs_b, proj_b, seg)[0]       # (W, d)
    mag = feat.norm(dim=1)                   # per-position magnitude
    assert mag.argmax().item() == 2, mag      # eye B keeps the impulse local at pos 2
    # one-hot probes through eye B must be highly separable (low offdiag cosine)
    onehots = torch.tensor(make_onehot_probes(W))         # (W, W)
    seg2 = torch.zeros(W, W, 3); seg2[:, :, 0] = onehots
    feats2 = encode_with(dp_convs_b, proj_b, seg2)        # (W, W, d)
    flat = feats2.reshape(W, -1).detach().numpy()
    mean_cos, _ = per_position_sep(flat)
    assert mean_cos < 0.2, mean_cos          # delta eye keeps positions distinct

    print("Task 1 selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-max", type=int, default=90)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    device = torch.device(args.device)
    agent, scene_cfg = load_agent(args.ckpt, device)
    substrate = agent.v35_substrate
    # eye A: the ckpt's multi-scale CNN
    eyeA_convs, eyeA_proj = agent.dp_convs, agent.dp_proj
    # eye B: single kernel=5 delta-init eye, dims matched to eye A
    input_dim = eyeA_convs[0].in_channels
    cps = eyeA_convs[0].out_channels
    d = eyeA_proj.out_features
    eyeB_convs, eyeB_proj = build_eye_b(input_dim, cps, d, device)

    rows = [
        eval_eye("A multi-scale 5/21/51", eyeA_convs, eyeA_proj, substrate, scene_cfg, args.n_max, device),
        eval_eye("B single k=5 (delta)", eyeB_convs, eyeB_proj, substrate, scene_cfg, args.n_max, device),
    ]

    cols = ["pos_feat_cos", "pos_res_cos", "level_mindist",
            "level_monotone_frac", "pv_r2_in", "pv_r2_extrap", "world_bump_slope"]
    print(f"\nckpt: {args.ckpt}")
    print(f"{'eye':<26} | " + " | ".join(f"{c:>16}" for c in cols))
    print("-" * (28 + 19 * len(cols)))
    for r in rows:
        print(f"{r['name']:<26} | " + " | ".join(f"{r[c]:>16.4f}" for c in cols))

    a, b = rows[0], rows[1]
    print("\n--- VERDICT (3-layer localization) ---")
    layer1 = b["pos_feat_cos"] < a["pos_feat_cos"] - 0.2
    print(f"Layer 1 (eye aggregation): eye B preserves positions better at feat? "
          f"{'YES' if layer1 else 'no'} "
          f"(A pos_feat_cos={a['pos_feat_cos']:.3f} vs B={b['pos_feat_cos']:.3f})")
    # pv_r2_extrap is coverage-confounded: a linear reader trained on N<=30 never sees the
    # high digits (cell4 first active at N>=81), so it is low even for a perfect substrate.
    # Drive the layer-2 verdict from pos_res_cos (per-position separability surviving the
    # substrate), which is coverage-independent; pv_r2_* are reported for context only.
    layer2 = b["pos_res_cos"] > 0.8
    print(f"Layer 2 (substrate re-counts): positions LOST after substrate despite eye B? "
          f"{'YES (substrate is the wall)' if layer2 else 'no'} "
          f"(B pos_res_cos={b['pos_res_cos']:.3f}; pv_r2_in={b['pv_r2_in']:.3f}, "
          f"pv_r2_extrap={b['pv_r2_extrap']:.3f} [coverage-confounded, context only])")
    binar = b["level_monotone_frac"] < 0.8
    print(f"Sub-wall (Layer-A binarization): 0.5 vs 1.0 collapse under eye B? "
          f"{'YES' if binar else 'no'} (B level_monotone_frac={b['level_monotone_frac']:.3f})")
    print("\nReading: Layer1 YES + Layer2 YES => eye is over-heavy AND substrate-as-counter "
          "is the deeper wall (decode needs a dedicated tied un-fold reader). "
          "Layer1 YES + Layer2 no => same-medium decode viable with a simple eye + the "
          "existing concat compare head.")
    return


if __name__ == "__main__":
    main()
