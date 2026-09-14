# V32 multi-gate + EC — findings (2026-05-19)

Branch: `v32-multigate-ec` (latest commit d6dd8e7)

## Status

❌ **Architecture does not learn**. Loss stuck at chance baseline 3.219 (= 2 × ln(5)) across 4 training attempts despite progressive fixes.

## Training attempts

| Attempt | Config | Result | Steps | Final loss | Codebook |
|---|---|---|---|---|---|
| #1 (aborted) | W=5 q=3, no fixes | Flat at chance, std monotone shrinking (collapse) | 1467 | 3.20 | No ckpt saved |
| #2 (canonical) | W=5 q=3, post-refactor (I-2/I-3/I-8/I-1) | Same | — | — | — |
| #3 (diag) | W=3 q=5, no F5 fixes | Flat at chance, std stable, codebook all 1 distinct code | 500 | 3.23 | distinct=1, modal constant across N |
| #4 (canonical+diag) | W=3 q=5 + P1+P2 | Flat at chance, **3/4 agents collapsed to all-zero scratch** | 466 | 3.23 | distinct=1 for 3 agents, distinct=2 (but same modal) for 1 |

## Architectural diagnostics

`scripts/v32_init_probe.py` (committed at 264b6e6) probes layer-by-layer N-signal preservation at fresh init.

### Pre-fix (baseline)

| Layer | inter-N/intra-N ratio | |ρ(N, PC1)| |
|---|---|---|
| CNN chunk 0 | 0.44 | 0.48 |
| LRU chunk 3 | 0.36 | 0.04 |
| PFC final | 0.77 | 0.65 |
| WriteHead raw | **0.92** | 0.65 |
| WriteHead quantized | std=0 | NaN |

Per-cell raw spread across N=1..30: **0.006** (essentially constant).

Root cause: multi-gate `σ(W_role τ) ⊙ σ(W_cont x) ⊙ (W_v x)` at init gives ~0.25× gain per layer. After 4 stacked layers: ~0.004× cumulative gain. Matches observed raw spread of 0.005.

### After P1+P2 (commit d6dd8e7)

P1: `MultiGateLinear.bias = +2.0` constant init. sigmoid(2) ≈ 0.88 vs old 0.5.
P2: `MultiGateSequentialWriteHead.output_scale = nn.Parameter(8.0)` learnable scale before sigmoid+quantize.

Per-cell raw spread: 0.005 → **0.136** (24× improvement). Quantized: 1-2 codes per N at init (vs always 1).

VERDICT changed from "DEAD AT INIT" to "borderline N dependence — training-viable".

## Training collapses init signal back to zero

Despite P1+P2 making init differentiable, ~400 steps of training pushed 3/4 agents to write **all-zero scratch** for ALL N. The 4th agent kept `[0.75, 0.75, 0]` modal across all N.

### Why

EC mutual zero-info equilibrium:
1. Listener has no useful info from random-init speaker
2. Speaker's gradient about correct N comes through listener → listener output is random wrt N → gradient about scratch is random noise
3. Best response to random noise gradient: minimize output magnitude (L2-like)
4. Speaker → sigmoid(very negative) → quantize to 0 → constant scratch
5. Listener → "all scratches are zero, predict marginal class freq" → uniform output
6. Vicious cycle reinforces

P1+P2 prevented the trivial collapse-at-init but cannot break the EC training equilibrium.

## Included records

- `v32_diag_run/run.jsonl`: saved diagnostic run records.
- `v32_canonical_run/run.jsonl`: saved canonical run records.

Training checkpoints and separate console-output files are not included in this snapshot.

## Source to inspect

- [Design overview](../../docs/design.md)
- `scripts/v32_init_probe.py`: initialization diagnostics.
- `scripts/v32_codebook_diag.py`: codebook inspection.
- `scripts/v32_cross_agent_diag.py`: cross-agent diagnostics.
