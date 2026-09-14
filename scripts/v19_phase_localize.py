# DIAG_FOR: v19 | ANSWERS: fft_n,codebook | INPUTS: <ckpt> --tag
"""V19 phase-signal localization diagnostics.

Run on a single ckpt:
  1. LRU per-dim FFT top-3 peaks over N=1..n_max
  2. Compare-in (PFC mod-aux input) per-dim FFT over N_α
  3. mod 5 head weight inspection (top 3 abs-weight input dims)
  4. Scratch codebook variance per N

Compare two ckpts by running twice with --tag.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core.scene import SceneConfig, sample_training_batch
from scripts.inspect_checkpoint import load_agent


def fft_top_peaks(curve, top_k=3, n_max=20):
    """Return list of (period, power_share) for top_k FFT peaks (excluding DC)."""
    fft = np.fft.rfft(curve - curve.mean())
    power = np.abs(fft) ** 2
    nondc = power[1:].sum() + 1e-12
    if len(power) <= 1:
        return []
    nondc_power = power[1:]
    # argsort descending
    order = np.argsort(nondc_power)[::-1][:top_k]
    out = []
    for k_idx in order:
        k = int(k_idx) + 1
        period = n_max / k
        share = float(nondc_power[k_idx] / nondc)
        out.append((period, share, k))
    return out


def collect_per_n(agent, cfg, n_min, n_max, n_batches, batch_size, device, capture_lru=True, capture_compare=True, capture_scratch=True):
    """Collect per-N: LRU end-state (B,d), compare_in stash (B,d), scratch (B,W)."""
    use_oracle = getattr(agent.agent_cfg, "use_oracle_spike_gate", False)

    # Hook LRU block 1 output
    lru_buf = []
    def lru_hook(m, i, o): lru_buf.append(o.detach())
    h_lru = agent.mamba_blocks[1].register_forward_hook(lru_hook) if capture_lru else None

    # Stash compare_in via monkey-patching mod aux head
    cmp_buf = []
    if capture_compare and agent.multi_modular_aux_head is not None:
        orig_head = agent.multi_modular_aux_head
        class _Stash(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                cmp_buf.append(x.detach())
                return self.m(x)
        agent.multi_modular_aux_head = _Stash(orig_head)

    n_to_lru = defaultdict(list)
    n_to_compare = defaultdict(list)
    n_to_scratch = defaultdict(list)

    rng_master = np.random.default_rng(123)
    for n in range(n_min, n_max + 1):
        rng = np.random.default_rng(1000 + n)
        for _ in range(n_batches):
            inp, _, ci, metas = sample_training_batch(
                batch_size=batch_size, n_max_stage=max(n, 5), config=cfg, rng=rng,
            )
            mask = [i for i, m in enumerate(metas) if m["N_alpha"] == n]
            if not mask: continue

            # LRU per-N: alpha-only forward (no full episode)
            if capture_lru:
                lru_buf.clear()
                x_alpha = inp[torch.tensor(mask), : cfg.L, :].to(device)
                if use_oracle:
                    agent._current_oracle_centers = [metas[i]["alpha_centers"] for i in mask]
                with torch.no_grad():
                    agent._encode_segment_v6(x_alpha)
                if lru_buf:
                    end = lru_buf[0][:, -1, :].cpu().numpy()
                    for r in end:
                        n_to_lru[n].append(r)

            # full forward to populate compare_in + scratch
            if capture_compare or capture_scratch:
                cmp_buf.clear()
                with torch.no_grad():
                    out = agent(inp.to(device), ci.to(device), metas=metas)
                logits = out[0]
                scratch = out[1] if len(out) > 1 else None
                if capture_scratch and scratch is not None:
                    sc = scratch.cpu().numpy()
                    if sc.ndim == 3 and sc.shape[2] == 1: sc = sc[..., 0]
                    for b in mask:
                        n_to_scratch[n].append(sc[b])
                if capture_compare and cmp_buf:
                    # cmp_buf has K=8 entries, each (B, d). Stack and average over K.
                    cmp = torch.stack(cmp_buf, dim=1).mean(dim=1).cpu().numpy()  # (B, d)
                    for b in mask:
                        n_to_compare[n].append(cmp[b])

    if h_lru: h_lru.remove()

    return n_to_lru, n_to_compare, n_to_scratch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--L", type=int, default=200)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--task", choices=["trio_wide", "trio_wide_extended"], default="trio_wide_extended")
    p.add_argument("--tag", default="ckpt", help="Label to put in output headers")
    p.add_argument("--no-oracle", action="store_true")
    args = p.parse_args()

    agent, _, _ = load_agent(
        args.ckpt, q=3, qrange="unit", L=args.L, complex_world=True,
        pfc_at_write=True, pfc_compare_drop_raw_scratch=True,
        cnn_pfc_only=False, v7_mode=False,
        use_oracle_spike_gate=not args.no_oracle,
    )
    agent.train(False)
    device = next(agent.parameters()).device

    if args.task == "trio_wide_extended":
        cfg = SceneConfig.trio_wide_extended_preset(L=args.L, complex_world=True)
    else:
        cfg = SceneConfig.trio_wide_preset(L=args.L, complex_world=True)
    cfg.K = 8

    print(f"\n========= [{args.tag}] phase-localization on {args.ckpt.name} =========\n")

    # LRU eigenvalue phases
    with torch.no_grad():
        lam_r, lam_i, _ = agent.mamba_blocks[1].get_lambda()
    phases = torch.atan2(lam_i, lam_r).abs().detach().cpu().numpy()
    expected_periods = [(2 * np.pi / p_) if p_ > 0 else float("inf") for p_ in phases]

    n_to_lru, n_to_compare, n_to_scratch = collect_per_n(
        agent, cfg, args.n_min, args.n_max, args.n_batches, args.batch_size, device,
    )
    Ns = list(range(args.n_min, args.n_max + 1))
    Nrange = len(Ns)

    # ============== 1. LRU per-dim FFT top-3 peaks ==============
    print(f"=== [1] LRU per-dim FFT top-3 peaks over N=1..{args.n_max} ===\n")
    print(f"{'dim':>3} | {'phase':>5} | {'exp':>5} | {'top1 (period, share, k)':<25} | {'top2':<25} | {'top3':<25} | match5?")
    if any(n_to_lru.values()):
        d = next(iter(n_to_lru.values()))[0].shape[0]
        means = np.full((d, Nrange), np.nan)
        for i, n in enumerate(Ns):
            if n_to_lru[n]:
                means[:, i] = np.array(n_to_lru[n]).mean(axis=0)
        for di in range(d):
            curve = means[di]
            if np.any(np.isnan(curve)) or curve.std() < 1e-3:
                print(f"  {di:>2} | (constant or insufficient samples)"); continue
            peaks = fft_top_peaks(curve, top_k=3, n_max=Nrange)
            peak_strs = [f"({p[0]:.2f}, {p[1]:.2f}, k={p[2]})" for p in peaks]
            match5 = any(4.5 <= p[0] <= 5.5 for p in peaks)
            mark = "✓ in top3" if match5 else ""
            row = f"  {di:>2} | {phases[di]:>5.2f} | {expected_periods[di]:>5.2f} | "
            row += " | ".join([s.ljust(25) for s in peak_strs])
            row += f" | {mark}"
            print(row)
    print()

    # ============== 2. Compare-in (mod aux input) per-dim FFT ==============
    print(f"=== [2] PFC compare_in (mod aux input) per-dim FFT over N_α=1..{args.n_max} ===\n")
    if any(n_to_compare.values()):
        d_c = next(iter(n_to_compare.values()))[0].shape[0]
        cmp_means = np.full((d_c, Nrange), np.nan)
        for i, n in enumerate(Ns):
            if n_to_compare[n]:
                cmp_means[:, i] = np.array(n_to_compare[n]).mean(axis=0)
        print(f"{'dim':>3} | {'top1':<25} | {'top2':<25} | {'top3':<25} | match5?")
        match5_count = 0
        for di in range(d_c):
            curve = cmp_means[di]
            if np.any(np.isnan(curve)) or curve.std() < 1e-3: continue
            peaks = fft_top_peaks(curve, top_k=3, n_max=Nrange)
            peak_strs = [f"({p[0]:.2f}, {p[1]:.2f}, k={p[2]})" for p in peaks]
            match5 = any(4.5 <= p[0] <= 5.5 for p in peaks)
            if match5: match5_count += 1
            mark = "✓" if match5 else ""
            print(f"  {di:>2} | " + " | ".join([s.ljust(25) for s in peak_strs]) + f" | {mark}")
        print(f"\n  → compare_in dims with period-5 in top3: {match5_count} / {d_c}")
    print()

    # ============== 3. mod 5 head weight inspection ==============
    print("=== [3] mod 5 head weight inspection ===\n")
    if agent.multi_modular_aux_head is not None:
        # may be wrapped by Stash; unwrap
        head = agent.multi_modular_aux_head
        if hasattr(head, "m"): head = head.m
        W = head.weight.detach().cpu().numpy()  # (n_mod=3, d=16)
        print(f"  multi_modular_aux_head weight shape: {W.shape}")
        moduli = getattr(agent.agent_cfg, "multi_modular_aux_moduli", (2, 3, 5))
        try: idx5 = list(moduli).index(5)
        except ValueError: idx5 = 2
        mod5_w = W[idx5]
        absw = np.abs(mod5_w)
        order = np.argsort(absw)[::-1]
        print(f"  mod 5 row weights, top 5 abs:")
        for rank, di in enumerate(order[:5]):
            print(f"    rank {rank+1}: dim={di} weight={mod5_w[di]:+.4f} abs={absw[di]:.4f}")

        if any(n_to_compare.values()):
            d_c = next(iter(n_to_compare.values()))[0].shape[0]
            cmp_means = np.full((d_c, Nrange), np.nan)
            for i, n in enumerate(Ns):
                if n_to_compare[n]:
                    cmp_means[:, i] = np.array(n_to_compare[n]).mean(axis=0)
            print(f"\n  Per-N curves of top 3 mod-5 dims (compare_in):")
            for rank in range(3):
                di = int(order[rank])
                curve = cmp_means[di]
                if np.any(np.isnan(curve)): continue
                peaks = fft_top_peaks(curve, top_k=3, n_max=Nrange)
                peak_strs = ", ".join([f"({p[0]:.1f}@{p[1]:.2f})" for p in peaks])
                vals = ", ".join(f"{v:+.2f}" for v in curve)
                print(f"    dim {di} (w={mod5_w[di]:+.3f}): [{vals}]  peaks {peak_strs}")
    print()

    # ============== 4. Scratch codebook variance per N ==============
    print("=== [4] Scratch codebook per N: modal code, distinct, variance ===\n")
    if any(n_to_scratch.values()):
        W_scr = next(iter(n_to_scratch.values()))[0].shape[0]
        print(f"{'N':>3} | {'n_eps':>5} | {'modal':<20} | {'modal_share':>11} | {'pos_var':<22} | distinct_codes")
        all_distinct = set()
        for n in Ns:
            arr = np.array(n_to_scratch[n])
            if arr.size == 0: print(f"  {n:>3} | (no samples)"); continue
            sc_q = np.clip(np.round(arr * 2) / 2, 0.0, 1.0)
            codes = [tuple(c) for c in sc_q]
            cnt = Counter(codes)
            modal, share = cnt.most_common(1)[0]
            modal_str = "[" + ",".join(f"{v:.1f}" for v in modal) + "]"
            share_frac = share / len(codes)
            pos_var = sc_q.var(axis=0)
            pos_var_str = "[" + ",".join(f"{v:.3f}" for v in pos_var) + "]"
            distinct_n = len(cnt)
            all_distinct.update(codes)
            print(f"  {n:>3} | {len(codes):>5} | {modal_str:<20} | {share_frac:>11.2f} | {pos_var_str:<22} | {distinct_n}")
        print(f"\n  total distinct codes across all N: {len(all_distinct)}")
    print()


if __name__ == "__main__":
    main()
