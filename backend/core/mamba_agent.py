# -*- coding: utf-8 -*-
"""
Mamba-based agent for Stage 3 prior-break experiments.

Replaces the GRU+PFC stack with a 2-layer Mamba (selective state-space)
stack while keeping the rest of the v3 architecture (resetH, splitAB,
scratch_pad with quantize-STE, compare via [h_alpha, h_beta] MLP).

Two front-end variants controlled by read_cnn_kernel:
  * M2 (read_cnn_kernel > 0):  CNN front-end -> 2x Mamba -> heads
  * M3 (read_cnn_kernel == 0): Linear front-end -> 2x Mamba -> heads (purest)

Episode is processed in segments separated by reset boundaries:
  segment A:  [0, L+W+T_gap)         (alpha + write + forget gap)
  segment B:  [read_start_i, beta_start_i)            (read-back, W steps)
  segment C:  [beta_start_i, compare_step_i + 1)      (beta scan + compare)
Inside each segment Mamba.forward processes the whole sub-sequence with
internal state initialized to zero. Across segments the state is dropped,
which is the equivalent of resetH+splitAB in the GRU agent.

NOTE: requires mamba_ssm (Linux + CUDA). Forward path is deliberately the
minimal "pure Mamba" version: no PFC anywhere, write_head reads Mamba state
directly, compare reads the concat of Mamba state at read-back end and
compare step.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.core.agent import _quantize_ste
from backend.core.scene import SceneConfig

try:
    from mamba_ssm import Mamba  # type: ignore
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


@dataclass
class MambaAgentConfig:
    d_model: int = 16
    input_dim: int = 3
    signal_output_dim: int = 1
    compare_classes: int = 3
    quantize_levels: Optional[int] = 3
    quantize_range: str = "unit"
    # Front-end
    read_cnn_kernel: int = 5  # 0 = pure Linear (M3), >0 = CNN front-end (M2)
    # Mamba block hyperparams
    mamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    # Compare head
    compare_mlp_hidden: Optional[int] = 32
    # PFC integration (M4: Mamba + PFC)
    pfc_layers: int = 0  # 0 = no PFC; >0 = transformer encoder over scratch+h tokens
    pfc_heads: int = 2
    pfc_iter: int = 1
    pfc_at_write: bool = False     # PFC fires at every write timestep
    pfc_compare_drop_raw_scratch: bool = False  # at compare, PFC sees only [h_alpha, h_beta]
    write_noise_std: float = 0.0  # If >0, add Gaussian noise to write_head logit before quantize-STE
                                   # (training only). Sharpens scratch encoding to reduce off-by-one.
    use_gap_head: bool = False    # Replace 3-class compare_head with scalar gap_head (Linear -> 1).
                                   # Used by gap_mse4 (signed-gap MSE^4) and rl_top1 (preference score).
    # Predictive auxiliary loss (E0-prime / Predictive M5 ablation): per-timestep
    # multi-output ForwardPredictor predicts the next T_pred CNN latents during
    # alpha scan. MSE against actual latents (NOT detached) provides dense
    # self-supervised gradient → forces scratch_pad to encode trajectory pattern,
    # not just N count. Hypothesized to enable near-distance extrapolation.
    world_pred_t_pred: int = 0    # 0 disables predictor entirely. >0 enables.
    # A2 (Path α): BetaPredictor — predict first T_β steps of each compare-block
    # β scan from (h_alpha, scratch). Forces scratch to encode α trajectory.
    beta_pred_t_pred: int = 0     # 0 disables. >0 = predict T steps of β scan.
    # A5 (Path α): ComparePredictor — predict compare_logits from (h_α, h_β, scratch).
    # Self-consistency loss vs detached actual compare_head output. Forces scratch
    # to contain enough info to predict own compare behavior.
    compare_pred_enabled: bool = False
    # PC (Path α): predictive coding. During α scan, mamba operates on
    # (CNN(input) − top_down_prediction) instead of raw CNN. Predictor takes
    # u[t-1] (shifted CNN latent) and outputs predicted u[t].
    predictive_coding: bool = False
    # RS-1 (Path α-mirror): RawSignalPredictor — predict next T_pred raw signal
    # scalars from h_t. Targets the actual input stream, not internal latents.
    rs_pred_t_pred: int = 0       # 0 disables. >0 = predict T raw signal steps.
    # A3 (Path α-mirror): ScratchSelfPredictor — predict scratch_t from prior
    # scratch_list[0..t-1] + h_t. Forces self-coherent scratch dynamics.
    scratch_self_pred_enabled: bool = False
    # A4 (Path α-mirror): CounterfactualBetaPredictor — predict raw β signal
    # from (scratch, target_n_norm). Counterfactual: β_count is explicit input.
    cf_beta_pred_t_pred: int = 0  # 0 disables. >0 = predict T β signal steps.
    # idea2: self-review iterative refinement. After write phase, refine the
    # full scratch_list via a small ReviewModule (transformer) for K iterations.
    review_iter: int = 0          # 0 disables. >0 = K refinement passes.
    review_layers: int = 2
    review_heads: int = 2
    # β-1 (Path β): VQ bottleneck. Replaces per-slot scalar quantize with shared
    # learnable codebook of K codes × d_z dim. Forces symbol reuse across positions.
    vq_enabled: bool = False
    vq_codebook_size: int = 32    # K
    vq_d_z: int = 8               # codebook code dimension
    vq_commitment: float = 0.25
    vq_decay: float = 0.99
    # P1 (L1579 dual-loss): predict raw signal of imagined future + scratch
    # under that imagination. Two losses: L_world (signal vs β signal) +
    # L_sp (imagined scratch vs pfc(β) scratch).
    dual_loss_enabled: bool = False
    dual_pred_t_p: int = 30       # how many future signal scalars to predict
    # COLLAPSE FIXES (GPT recipe: L_z + L_h(sg) + L_reg)
    # When True, prediction targets are detached so encoder isn't pulled toward
    # collapse via the predictor's gradient pathway. Applies to A2, E0, PC.
    aux_detach_targets: bool = False
    # VICReg variance regularizer on prediction targets — anti-collapse via
    # forcing per-dim variance >= gamma in batch. λ=0 disables.
    vicreg_lambda: float = 0.0
    vicreg_gamma: float = 1.0
    # V33-VICReg (5/22): exploratory aux on substrate h_seq at every timestep.
    # Goal: cap PC1 dominance (cos overlap ~0.9 across channels → 95% PC1) by
    # pushing per-dim variance up (L_var, hinge on σ_d < gamma) AND off-diagonal
    # covariance to 0 (L_cov). Force info to spread across substrate dims rather
    # than ride one linear mode (μ(PC1) linear in N, capped at ~2 bits info).
    # Applied to h_alpha_seq + h_beta_seq + (if SM) h_readback_seq, averaged.
    v33_vicreg_lambda_var: float = 0.0
    v33_vicreg_lambda_cov: float = 0.0
    v33_vicreg_gamma: float = 1.0
    # PC: add explicit prediction MSE loss (sg) so PC is real BYOL-style
    # not just a "residual subtraction" trick. λ=0 keeps legacy behavior.
    pc_explicit_lambda: float = 0.0
    # 51_M5 (2026-05-01): simple comparator + L_cycle
    # - simple_comparator: replace 3-class MLP with Linear(d, 3) on (h_sp − h_β)
    # - cycle_loss_lambda: weight on ||h_sp − sg(h_alpha)||²
    # - pfc_variant: "transformer" (default M4) or "mlp" (deep MLP per-token mixer)
    simple_comparator: bool = False
    cycle_loss_lambda: float = 0.0
    pfc_variant: str = "transformer"
    pfc_mlp_hidden: int = 64
    # V6 (2026-05-04) Dual-pathway: subitizing CNN bypass to PFC.
    # When True, replaces single-kernel read_cnn with multi-scale CNN (V5b style).
    # CNN→Mamba forward pathway is detach()ed (Mamba grad doesn't update CNN).
    # CNN→PFC bypass: per-segment mean-pooled CNN features added as a token
    # to PFC at write + compare. PFC compare loss directly trains the CNN via
    # this bypass — that's the "subitizing" signal pathway.
    dual_pathway: bool = False
    dp_kernels: tuple = (5, 21, 51)
    dp_channels_per_scale: int = 4  # total channels = len(kernels) * cps
    dp_detach_to_mamba: bool = True  # cut Mamba→CNN backward via detach
    # V6 Q3: at readback (seg_B with scratch injected), skip Mamba and use
    # CNN-only path. Rationale: W=5 scratch is too short for Mamba accumulation;
    # CNN (multi-scale, large kernel) can directly read the scratch sequence
    # without imposing Mamba's 1D log-thermometer geometry.
    dp_scratch_skip_mamba: bool = True
    # V6 BRANCH (2026-05-04): pure CNN+PFC, no Mamba at all.
    # When True (forces dual_pathway=True), Mamba blocks are skipped entirely.
    # PFC at write sees [cnn_alpha_summary, scratch_kv] (h_t replaced).
    # PFC at compare sees [cnn_alpha_readback, cnn_beta_summary, scratch_kv]
    # (h_alpha, h_beta replaced with CNN means).
    # Test target: small N (<=3 or 4) where CNN receptive field spans all
    # spikes — should achieve near-100% precision. If doesn't, setup bug.
    cnn_pfc_only: bool = False
    # V7+stride (2026-05-04): adaptive avg pool CNN outputs to fixed T_target.
    # Forces Mamba to see SAME token count (T_eff) across all curriculum stages,
    # so PFC's attention learns position-invariant patterns. Each output token
    # represents an averaged "saccade window" — biological eye-movement analog.
    # 0 disables (V7 default). >0 enables stride-via-pooling.
    dp_t_target: int = 0
    # V8 (2026-05-05): RBF compare head replacing simple_comparator's Linear(d, 3).
    # Uses 3 learnable centers in d-space; logits[c] = -|diff - center_c|² / (2σ²).
    # Bounded outputs prevent "confidently wrong" mode (V7s stage 3 plateau);
    # forces decision to be locality-based rather than linear projection.
    rbf_compare_head: bool = False
    rbf_init_sigma: float = 1.0
    # V8_kWTA (2026-05-05): top-k Winner-Take-All on PFC outputs.
    # Each token (B, d) → keep top-k abs activations, zero rest. Forces sparse
    # representation. Applied at PFC compare (h_α_pfc, h_β_pfc) and PFC write
    # (h_for_write before write_head). 0 disables.
    kwta_k: int = 0
    # V10 (2026-05-05): preserve time/position info into Mamba/PFC.
    # cnn_stride_no_pool: replace adaptive_avg_pool1d (which means within each
    # window, destroying within-window position) with stride sampling
    # (every stride-th conv output, preserving kernel-pattern at fixation point).
    # pfc_see_cnn_seq: PFC at write/compare consumes T_target CNN tokens
    # (with positional embedding) instead of single mean-summary, so PFC can
    # extract within-α positional structure to drive scratch differentiation.
    cnn_stride_no_pool: bool = False
    pfc_see_cnn_seq: bool = False
    # P3 (2026-05-06): Integrate-and-Fire write controller. Replaces fixed W
    # write loop with dynamic IF dynamics: state accumulates encoder output,
    # fires (writes a cell) when learnable threshold exceeded, partial reset.
    # Top-W of T fire-urgency steps actually write to scratch.
    if_write_gate: bool = False
    if_init_threshold: float = 1.0
    if_init_reset: float = 0.5     # sigmoid'd, → reset_strength after train init
    if_fire_tau: float = 1.0       # gumbel-sigmoid temperature
    if_use_mamba_state: bool = False  # state_seq = h_A (Mamba) instead of cnn_alpha_seq
    # V12 (2026-05-06) — LayerNorm tried first but FAILED: LN per-sample normalizes
    # h_for_write to mean=0 std=1 across d=16, destroying the magnitude axis that
    # actually carries N information (V9 scratch PC1↔N +0.957 is in MAGNITUDE).
    # Replaced with tanh-clip on raw output: raw = clip · tanh(raw/clip). This
    # bounds raw to [±clip] smoothly without touching h_for_write structure.
    # 0 = disabled. Recommended: 3.0–4.0 (V9 raw [-3.5, 0.65] roughly fits at 4).
    write_head_ln: bool = False  # kept for ckpt-load compat, no longer used
    write_head_clip: float = 0.0
    # V13 (2026-05-06): cross-attention write head. Replaces W-step self-attention
    # PFC loop with single parallel cross-attention: W learnable "write queries"
    # cross-attend over cnn_alpha_seq (KV). Each query learns its own attention
    # pattern → 5 cells naturally differentiate, no need for write_head to learn
    # extreme weights to break ties → saturation pathology may resolve naturally.
    cross_attn_write: bool = False
    cross_attn_write_heads: int = 2
    # V14 (2026-05-06): per-cell gumbel-softmax write head. Replaces sigmoid+STE
    # quantize with categorical sampling over Q=3 levels. Output bounded by
    # construction → no sigmoid saturation pathology. Works with any base
    # (V9 / V10_PFCseq / V13 cross-attn). signal_output_dim still 1 (scalar/cell).
    gumbel_write_head: bool = False
    gumbel_tau: float = 1.0
    # V15 (2026-05-06): Coconut-style recurrent latent PFC. Replaces "mean over T
    # CNN tokens" with sequential transformer-cell processing: state token +
    # one CNN token per step, T steps total, shared transformer block across
    # iterations. State accumulates info in latent space (not magnitude).
    # Final state replaces cnn_alpha_summary as h_t in V9-style write loop.
    recurrent_latent_pfc: bool = False
    recurrent_latent_state_tokens: int = 1   # K, number of [STATE] tokens (default 1)
    recurrent_latent_sparse_k: int = 0       # 0 = no sparsity, else kWTA on state per step
    recurrent_latent_steps: int = 0          # 0 = use all dp_t_target tokens; else first N
    # V16 (2026-05-06): learnable [WRITE_QUERY] token at write phase PFC.
    # Replaces "h_t at position 0 with residual carry" with a small-init query
    # whose residual contribution to output[0] is negligible. Effectively makes
    # output[0] = query + Attn(LN([query, h_t, scratch_kv]))[0] — pure attention
    # output, not residual-h_t-dominated. Mathematical fix to phase-space
    # analysis: brings σ/|h_t| ratio closer to 1 (V9 sweet spot).
    write_query_token: bool = False
    # V17 (2026-05-07): Multi-modular auxiliary task. After main compare,
    # add binary heads predicting (α mod m == β mod m) for each m in moduli.
    # Co-prime moduli (2,3,5) make 1D thermometer NOT sufficient — model must
    # learn ≥3 independent oscillations to solve all heads. λ=0.1 small enough
    # to not disrupt main compare task. Implements the RNS pressure required
    # for modular structure to be advantageous.
    multi_modular_aux_moduli: tuple = ()  # e.g. (2, 3, 5); () = disabled
    # V18 (2026-05-07): complex-eigenvalue LRU replacing Mamba in dual pathway.
    # Orvieto 2023 stable parameterization: λ_i = exp(-ν_i + i·θ_i), ν > 0
    # for stability, θ ∈ [0, 2π] for frequency. Spectrum can stay COMPLEX
    # (oscillatory) if task has gradient pressure for it (e.g. multi-modular
    # aux). Avoids Mamba's collapse to real eigenvalues (1D thermometer).
    use_lru: bool = False  # if True, replace mamba_blocks with LRU blocks
    lru_d_state: int = 16   # complex state dim per LRU block (2 * d_state real params)
    lru_init_r_min: float = 0.4   # |λ| init range [r_min, r_max] = decay rate
    lru_init_r_max: float = 0.9   # closer to 1 = longer memory
    lru_init_phase_max: float = 6.283185  # 2π — phase init uniform [0, 2π]
    # V19 (2026-05-07): event-gated LRU. LRU sees unpooled (B, L, d) instead
    # of pooled (B, 8, d). spike_detector → gate ∈ [0,1]. When gate≈1, LRU
    # rotates phase normally; when gate≈0, h frozen (no decay, no rotation).
    # Goal: phase rotation occurs in spike-count dimension, not spatial step.
    # PFC bypass pathway still pools to dp_t_target=8 (unchanged).
    lru_event_gated: bool = False
    gate_sharpness_init: float = 5.0  # learnable; init ≈ binary gate
    detection_aux_weight: float = 0.0  # 0=off; if >0, BCE on spike_score vs GT
    # Oracle-gate ablation (oracle-gate branch only): when True, the LRU gate
    # is constructed from scene-meta spike-center positions instead of the
    # learned spike_detector output. spike_detector still runs (and its
    # detection BCE loss still applies if detection_aux_weight>0) — it is
    # simply bypassed for the LRU gate signal. This isolates "path
    # integration mechanism" from "detection learning quality".
    use_oracle_spike_gate: bool = False
    # V20 (oracle-gate branch): scratch_mod_aux_head reads (scratch_alpha −
    # scratch_beta_virtual) directly through Linear(W, n_mod) and pushes BCE
    # supervision into write_head. Goal: force scratch_pad cells to host
    # modular structure rather than only thermometer binning. Default empty
    # = disabled. V20 canonical: (2, 3).
    scratch_mod_aux_moduli: Tuple[int, ...] = ()
    # V22 (oracle-gate branch): SuccessorFunction S learns the +1 mapping in
    # raw (pre-quantize) scratch space. Trained on within-episode |δ|=1 G/L
    # pairs (no absolute N supervision). Residual prediction. 0 = disabled,
    # >0 = hidden dim of the MLP. V22 canonical: 32.
    successor_hidden: int = 0
    # V25 (sequential write head): replace Linear(d, signal_dim) write_head
    # with shared MLP applied sequentially over W cells. Each cell sees PFC h
    # plus the cells written so far. "linear" = V19 ext default.
    # V28: lru_decoder (state-init free response) | sinusoidal (true periodic)
    write_head_type: str = "linear"  # "linear" | "sequential" | "lru_decoder" | "sinusoidal"
    write_head_hidden: int = 32
    # PM (5/20): for SequentialWriteHead, init `written` buffer with quantize-alphabet
    # midpoint (0.5 for unit, 0 for symmetric) instead of zeros. Breaks the cell
    # cascade collapse where cell w sees the same all-zero buffer as cell 0 once
    # cells default-output to lv0. Same-medium-faithful (no new symbol introduced).
    write_head_pending_marker: bool = False
    # V28-LRU decoder params
    lru_decoder_d_state: int = 16
    # V28-Sin sinusoidal head params
    sin_omega_init: tuple = (0.3, 0.7, 1.2, 1.7, 2.3)
    sin_init_std: float = 0.5
    # V28-Sin predict_head: multi-hot + MLP (hidden=64). Triggered by sinusoidal.
    sin_predict_hidden: int = 64
    # V25-V22: when True, successor loss detaches sc_α / sc_β before passing
    # to S — gradient flows only through S, not back into write_head.
    successor_detached: bool = False
    # V25-V24: successor_predict_head input level. "scratch" = V24 default
    # (Linear(2W, k_max), reads concat sc_α + sc_β scratch). "pfc_hidden" =
    # V25-V24 (Linear(2*d_pfc, k_max), reads concat h_α_pfc + h_β_pfc) —
    # supervisory operates on PFC representation, scratch becomes side product.
    successor_predict_input_level: str = "scratch"  # "scratch" | "pfc_hidden"
    # V24 (oracle-gate / v24 branch): successor_predict_head Linear(2W, k_max)
    # for task=successor_prediction. Reads concat([sc_α, sc_β_virtual]) and
    # outputs k_max-class logits over k = N_β - N_α ∈ [1, k_max].
    successor_predict_k_max: int = 0
    # V24R: recurrent scratch refinement. Wraps post-forward sc_α/sc_β through
    # n_iter rounds of (cnn_scratch + iter_proj + write_head + quantize).
    # 0 = disabled. V24R canonical: n_iter=3, cnn_kernel=3, scratch_cnn_channels=4.
    pfc_recurrent_n_iter: int = 0
    pfc_recurrent_cnn_scratch_kernel: int = 3
    pfc_recurrent_cnn_scratch_channels: int = 4
    # V27 (5/10): chain-of-thought PFC over n chunks. Each chunk:
    #   PFC self-attn over [pfc_state, cnn_chunk_token, lru_h_chunk]
    # CNN strided to n_chunks tokens (kernels 5+51), LRU h persistent across chunks.
    # 0 = disabled (use V19/V25 path). V27 canonical: 4.
    cot_pfc_n_chunks: int = 0
    cot_pfc_cnn_kernels: tuple = (5, 51)   # multi-scale, drop k=21 vs V19 ext
    # V23b (oracle-gate branch): NClassHead reads scratch_α (W cells) and
    # predicts N_α as max_n-class CE. Strong supervisory pressure for
    # 1-to-1 N→scratch readability without dictating encoding form.
    # 0 = disabled. V23b canonical: max_n=90, hidden=32.
    n_class_max_n: int = 0
    n_class_head_hidden: int = 32
    # V9-AR (2026-05-05): GRUCell autoregressive write head. Replaces the
    # Linear(d, 1) per-write projection with a recurrent module that maintains
    # hidden state across the W=5 write iterations and conditions cell_k on
    # the previously-written cell_{k-1}. Goal: let cell-cell correlation
    # express place-value structure (carry, base) without expanding codebook.
    # Constraint preserved: total codes = quantize_levels^W (e.g. 3^5 = 243).
    ar_write_head: bool = False
    ar_write_embed_dim: int = 4
    ar_write_prev_dropout: float = 0.3   # drop prev_cell input with this prob (training)
    ar_write_tau_init: float = 1.5       # gumbel-softmax temp at step 0
    ar_write_tau_final: float = 0.8      # tau at >=warmup steps
    ar_write_tau_warmup: int = 5000      # linear anneal over this many train steps

    # V29 (2026-05-11): R&F SNN encoder + event attention PFC + R&F SNN write head.
    # Keeps V27 5-cell × q=3 scratch interface but replaces internal substrate
    # with spike-based dynamics. When use_v29=True, all CNN/Mamba/CoTPFC/write_head
    # construction is skipped; V29Pipeline takes over forward entirely.
    use_v29: bool = False
    v29_encoder_n_neurons: int = 16
    v29_write_n_neurons: int = 5
    v29_d_id: int = 16
    v29_d_pe: int = 16
    v29_pfc_layers: int = 2
    v29_pfc_heads: int = 4
    v29_pfc_passes: int = 2
    v29_threshold: float = 1.0
    v29_surrogate_alpha: float = 4.0
    v29_omega_enc_min: float = 0.5
    v29_omega_enc_max: float = 2.0
    v29_omega_write_min: float = 0.7
    v29_omega_write_max: float = 1.8
    v29_b_init_min: float = -0.3
    v29_b_init_max: float = -0.05
    v29_time_max_init: float = 100.0
    v29_soft_reset: bool = True
    v29_t_write: int = 0  # 0 = auto (2π/median(ω_write))
    v29_max_events: int = 256
    v29_query_orth_lambda: float = 0.01
    v29_omega_clamp_decay: float = 1e-4
    v29_firing_rate_lambda: float = 0.01
    v29_target_firing_rate: float = 1.5

    # V33 (2026-05-20): HH-SSM substrate — multi-channel selective SSM with closed-loop
    # gate dynamics. Replaces V27 (LRU + CoT-PFC) core; keeps V25 sequential write_head
    # + successor_prediction interface. Channels: Na/K/NMDA/leak with non-orthogonal v_c.
    use_v33: bool = False
    v33_d_model: int = 16
    v33_n_layers: int = 1
    v33_v_perturbation_scale: float = 0.3
    # V33-A (5/20): orthogonality regularizer on channel directions v̂_c. Penalty =
    # λ · mean_{c<c'} cos(v̂_c, v̂_{c'})². Counters baseline failure where SGD pulled
    # all 4 v_c toward one axis (pairwise cos > 0.91), collapsing substrate to rank-1.
    v33_ortho_lambda: float = 0.0
    # V33-D5 (5/20): kill input-to-h bypass (u_seq=B_leak(x)) and route input through
    # gate equilibria via W_input_to_gate. Tests whether channels are "unused because
    # bypass dominates" vs "unused because dynamics can't learn N-encoding".
    v33_input_to_gate: bool = False
    # V33-SM (5/20): same-medium closed loop. Clears ch1/ch2 phase flags (CNN
    # becomes medium-agnostic) AND inserts a W-step readback substrate scan
    # between α and β where ch0 = scratch_α_q (scratch re-injection). β scratch
    # is computed from h_β_end (which depends on scratch_α via the readback path)
    # instead of being a placeholder. Creates the gradient closed loop that the
    # V8-V19 forward path had but V24+/V25+/V27/V33 forward had dropped.
    v33_same_medium: bool = False
    # V34 (5/21): self-supervised symbol emergence on top of SM+PM. Predictor is
    # a rank-W linear bottleneck (Linear(W, d_model)) predicting h_readback_end
    # from raw_preSTE. Capacity is intentionally low: too weak to bypass quantize
    # via direct decoding, must route info through the closed loop.
    predict_coding_enable: bool = False
    # V34: skip task CE entirely. Used with predict_coding_enable + new aux
    # lambdas (predict / commit / diversity / raw_l2). No β / N_β / task acc
    # affects training; predict_head still built but receives no gradient.
    unsupervised_mode: bool = False
    # V34 loss weights (used only when unsupervised_mode=True).
    predict_coding_lambda: float = 1.0
    commit_lambda: float = 0.25
    diversity_lambda: float = 0.05
    raw_l2_reg_lambda: float = 0.001

    # V35 (5/27 pivot): d=1 HH-SSM substrate + fire-and-reset cascade.
    # Bypasses all V27/V28/V29/V33 forward paths.
    use_v35: bool = False
    v35_n_recursions: int = 5              # fixed for first-pass testing
    v35_fire_temp: float = 0.5             # STE temp for spike threshold
    v35_write_d_attn: int = 16             # write attention embed dim
    v35_write_n_heads: int = 2
    # 6/02 (route A, spec 2026-06-02): per-cell write readout. True -> cell k reads
    # only cascade residual k via its own tiny MLP (structural decoupling, breaks
    # cell-lockstep). False -> cross-attention V35WriteAttention (default).
    v35_write_per_cell: bool = False
    v35_compare_d_attn: int = 16
    v35_compare_n_heads: int = 2
    v35_compare_n_layers: int = 1
    # Rung 2 (2026-06-07): compare-head pooling. False -> mean-pool (symmetric, the
    # comparison wall identified by R1). True -> position-indexed concat readout
    # (the C3 escape: real-code allpair 0.92-1.00 vs 0.17-0.70 for mean-pool).
    v35_compare_concat: bool = False
    # 5/28 ablation (II): replace HH-based layer A with deterministic raw-signal
    # threshold detector. Bypasses CNN + HHSSMLayer1D(A); spike_train_A is
    # `(inputs[:, :L, 0] > threshold).float()`. Tests cascade carry math in
    # isolation from layer-A learning dynamics. Not same-medium compatible —
    # smoke-only flag for diagnosing layer B / cascade behavior.
    v35_threshold_layer_a: bool = False
    v35_layer_a_threshold: float = 0.5
    # 6/03: raw-signal Layer-A detector under --v35-threshold-layer-a.
    # "level" = (sig>thr) over-count (back-compat). "food_pattern" = 3-cell
    # count_foods gate -> spkA=N. center/neighbor mirror scene FOOD_*_THRESH.
    v35_layer_a_detect: str = "level"
    v35_layer_a_center_thresh: float = 0.9
    v35_layer_a_neighbor_thresh: float = 0.8
    # 6/03 (TEMPORARY scaffolding, NOT the goal): differentiable readback -- compare reads
    # the scratch code directly so the STE soft path carries the task gradient to the write
    # head. Spec: docs/design.md. NOT same-medium.
    v35_readback_direct: bool = False
    # V35 PSS leak-kill (spec 2026-06-05 §5): write beta to its OWN scratch -> readback,
    # so no fresh-magnitude beta reaches compare (severs the near-beta leak). Pairs with
    # the threshold-Layer-A + readback-direct scaffold on the canonical/runnable path.
    v35_pss_readback_beta: bool = False

    # 5/28 (PC v0): local predictive-coding update for substrate B_leak only.
    # When True: (a) substrate layers init with biological_init=True (specific
    # gate_w/gate_b values producing Rao-Ballard fast/slow error pair via
    # P_Na = m_Na · h_Na as transient bipolar prediction error);
    # (b) _v35_forward captures gate trajectories per pass; (c) training loop
    # excludes substrate from SGD optimizer and applies V35Substrate.pc_step
    # after each optimizer.step(). See docs/design.md.
    v35_local_pc: bool = False
    v35_pc_lr: float = 0.1
    # 5/28 PC v1: per-layer fixed theta (overrides shared learnable theta).
    # When both v35_theta_a and v35_theta_b are non-None, V35Substrate constructs
    # frozen per-layer theta buffers (theta_a for layer A peak detector,
    # theta_b for layer B base-q carry counter). Typical: theta_a=1.0, theta_b=3.0.
    # Default None means legacy shared learnable theta (init 3.0).
    v35_theta_a: Optional[float] = None
    v35_theta_b: Optional[float] = None
    # 6/15 Rung B2 (de-scaffold "invent" test): init of the SHARED learnable theta (= base)
    # when theta_a/theta_b are BOTH None (the peel path). Default 3.0 = prior behavior.
    # Set away from 3 to ask whether the learnable base DISCOVERS 3 vs only MAINTAINS it.
    v35_theta_shared_init: float = 3.0
    # 6/15 Rung B1' (genuinely-learnable base): keep the shared theta_raw IN the SGD optimizer
    # even under --v35-local-pc (which otherwise excludes ALL substrate params from SGD; PC only
    # updates B_leak, so theta is frozen at init). With this, the base is gradient-learnable -> tests
    # whether the base FUNCTION-SELECTS toward the task-optimal radix (base-2 here: acc 0.94>0.83).
    v35_theta_in_sgd: bool = False
    # 5/28 PC v1: replace binary spike train with preceding layer's continuous h_seq
    # in PC update's input multiplier (layer B recursions only; layer A unchanged).
    # Breaks no-fire attractor by providing non-zero PC signal even during cold-start.
    v35_pc_h_upstream: bool = False
    # 5/28 homeostatic intrinsic-plasticity (IP) bias on Layer A (pilot).
    # IP rule Δb = ip_lr·(r* − r_obs) regulates Layer A avg firing toward r*,
    # breaking the no-fire attractor. In pilot, enabling IP also freezes Layer B PC.
    # Spec: docs/design.md
    v35_ip_enable: bool = False
    v35_ip_lr: float = 0.02
    v35_ip_target: Optional[float] = None  # None -> auto (N_max+1)/2 per stage
    v35_ip_bias_clamp: float = 0.0         # 0 -> disabled (rule is self-bounding)
    # Anchor the two drifting integrators (spec 2026-05-29). weight_decay: plain L2
    # decay on Layer A B_leak (anchors Hebbian common mode, stops ±91 drift). ip_leak:
    # soft leak on ip_bias (anti-windup -> bounded fixed point). Both 0 -> off (no-op).
    v35_pc_weight_decay: float = 0.0
    v35_ip_leak: float = 0.0
    # Fix 3 (spec 2026-05-29): three-factor task-modulated Hebbian. Per-sample signed
    # advantage M = p_target - EMA-baseline (compare head) multiplies the Hebbian contrib
    # on layer A AND the unfrozen layer B. Supplies directional credit assignment IP lacks.
    v35_three_factor: bool = False
    v35_tf_baseline_ema: float = 0.99
    # NaN-divergence fix (spec 2026-05-29): max L2 norm of each substrate B_leak,
    # renorm after the local update. Bounds u_seq/h runaway without weight-decay's
    # proportional shrinkage of small weights. 0 = off.
    v35_pc_weight_clip: float = 0.0
    # 6/02 (Layer-A leak, spec 2026-06-02): membrane leak on Layer A only. log_g_leak_a
    # overrides Layer A's leak channel (idx 3) log_g init (None keeps -4); warm_b_leak_a
    # scales Layer A's B_leak init (cold-start firing, replaces IP). Layer B unaffected.
    v35_log_g_leak_a: Optional[float] = None
    v35_warm_b_leak_a: float = 1.0
    # 6/02 (hand-built cascade, spec 2026-06-02): hand-set frozen Layer B base-counter.
    # b_leak_b_init fills Layer B's B_leak with a constant (per-spike ≈ 0.495·val; 2.0 → ≈1, so
    # theta_b becomes the true base). freeze_layer_b_bleak makes pc_step skip the Layer B update.
    v35_b_leak_b_init: Optional[float] = None
    v35_freeze_layer_b_bleak: bool = False
    # 6/15 (INSIGHTS sec.13 state-discreteness fix): STE-quantize the post-fire-reset
    # membrane state to this many levels across [0, theta] in BOTH layer A and B,
    # making the inter-spike level discrete (the carry is already discrete). 0 (or 1)
    # = OFF -> default V35 byte-for-byte unchanged. K>1 = K-level discrete state.
    v35_state_discretize: int = 0
    # 6/03 (timing-invariant target form, TEMPORARY scaffolding): fill ALL of Layer B's
    # per-channel log-conductance with this constant. Very negative (e.g. -20) turns
    # channels off -> a_eff=0 + no standing drive -> h holds between spikes -> event-driven
    # base-theta counter -> residual = base-3 digits, timing-invariant (probe-verified at
    # -20). Frozen (requires_grad False) since it is a hand-set form. None = default (-4).
    v35_log_g_b: Optional[float] = None
    # 6/04 (codebook-fidelity recon aux, spec 2026-06-04): weight of an MSE
    # reconstruction loss D(scratch_alpha) ~ residuals_alpha. >0 builds a
    # V35ResidualDecoder and forces scratch to losslessly encode every residual digit
    # (counters codebook collapse where weak |k|<=5 task pressure kills the high cells).
    # Zero-Prior: target is the substrate's own residuals, not an N label. 0.0 = off.
    v35_recon_aux_lambda: float = 0.0
    # 6/15 (cheap-arithmetic aux, INSIGHTS §9A): weight of a CHEAP (single Linear)
    # successor head on the quantized scratch code alone, trained with the SAME
    # k_target CE as the main task. A cheap/bounded arithmetic readout can only reach
    # low loss if the scratch supports linear arithmetic, which selects the positional
    # class of codes (an expressive compare head exerts no such pressure). >0 builds
    # v35_cheap_arith_head and adds lambda * its CE to the loss. 0.0 = EXACT no-op.
    # Zero-Prior: target is the existing relational k_target label, not an N label.
    v35_cheap_arith_aux_lambda: float = 0.0
    # direction-1 counting pressure: within-N consistency + across-N injectivity aux on alpha scratch.
    v35_count_consist_lambda: float = 0.0
    # direction-1 goal-step: init the learned detector from an ES-pretrained ckpt + optionally freeze it.
    v35_detector_init: Optional[str] = None
    v35_freeze_detector: bool = False
    # 6/02 (full frozen cascade): freeze_layer_a_bleak makes pc_step skip BOTH the
    # Layer A B_leak update and the IP update — Layer A fully frozen (e.g. a forward
    # detector loaded via --init-from-ckpt). Pairs with b_leak_b_init + freeze_layer_b_bleak.
    v35_freeze_layer_a_bleak: bool = False


class ForwardPredictor(nn.Module):
    """Multi-output MLP predicting T_pred future CNN latents from (h_t, scratch).

    Vectorized over batch+time so a single forward computes predictions for ALL
    timesteps in a segment. Output is (B*T, T_pred, d) reshaped to (B, T, T_pred, d).
    """

    def __init__(
        self, d_model: int, t_pred: int, scratch_dim: int = 5,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.t_pred = t_pred
        self.scratch_dim = scratch_dim
        self.fc1 = nn.Linear(d_model + scratch_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, t_pred * d_model)

    def forward(
        self, h: torch.Tensor, scratch_flat: torch.Tensor
    ) -> torch.Tensor:
        """h: (B, d) or (N, d). scratch_flat: (B, scratch_dim) or (N, scratch_dim).
        Returns (B, T_pred, d) or (N, T_pred, d).
        """
        x = torch.cat([h, scratch_flat], dim=-1)
        x = self.act(self.fc1(x))
        out = self.fc2(x)
        return out.view(-1, self.t_pred, self.d_model)


class ComparePredictor(nn.Module):
    """A5: predict compare_head outputs from (h_α, h_β, scratch).

    Forces scratch to contain information sufficient for the model's own
    compare behavior to be predictable. Self-consistency loss = MSE vs
    detached actual compare_logits (no gradient through compare_head from
    this aux loss → encoder learns extra info, compare_head not perturbed).
    """

    def __init__(
        self, d_model: int, scratch_dim: int, output_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.fc1 = nn.Linear(2 * d_model + scratch_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, h_alpha: torch.Tensor, h_beta: torch.Tensor,
        scratch_flat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h_alpha, h_beta, scratch_flat], dim=-1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class TopDownPredictor(nn.Module):
    """PC-lite: predict u[t] from u[t-1].

    Used inside `_encode_segment_pc` to compute residual error pathway for
    mamba. Encoder learns to encode "what's surprising" rather than raw signal.
    """

    def __init__(self, d_model: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, u_prev: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(u_prev)))


class RawSignalPredictor(nn.Module):
    """RS-1: predict next T_pred RAW signal scalars from h_t.

    Target is inputs[..., 0] (the actual signal channel), NOT internal latents.
    Pushes encoder to model the world dynamics rather than self-state.
    """

    def __init__(self, d_model: int, t_pred: int, hidden_dim: int = 32):
        super().__init__()
        self.t_pred = t_pred
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, t_pred)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(h))
        return self.fc2(x)  # (..., t_pred)


class ScratchSelfPredictor(nn.Module):
    """A3: predict scratch_t from prior scratch slots + h_t.

    Forces scratch sequence to have systematic self-coherent dynamics.
    Detached target so predictor learns from h_t side, not just identity copy.
    """

    def __init__(
        self, d_model: int, signal_dim: int, max_W: int,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.signal_dim = signal_dim
        self.max_W = max_W
        self.fc1 = nn.Linear(d_model + max_W * signal_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, signal_dim)

    def forward(
        self, h_t: torch.Tensor, prior_scratch_padded: torch.Tensor
    ) -> torch.Tensor:
        """h_t: (B, d). prior_scratch_padded: (B, max_W*signal_dim) — zero-padded.
        Returns predicted (B, signal_dim).
        """
        x = torch.cat([h_t, prior_scratch_padded], dim=-1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class CounterfactualBetaPredictor(nn.Module):
    """A4: predict raw β signal sequence from (scratch, target_n_norm).

    Conditioning on explicit β_count enables generation of any β scene at test
    time. Forces scratch to encode generative content of α (enough to imagine
    arbitrary β counts).
    """

    def __init__(
        self, scratch_dim: int, t_pred: int, hidden_dim: int = 64,
    ):
        super().__init__()
        self.t_pred = t_pred
        # Input: scratch_dim + 1 (target_n_norm)
        self.fc1 = nn.Linear(scratch_dim + 1, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, t_pred)

    def forward(
        self, scratch_flat: torch.Tensor, target_n_norm: torch.Tensor
    ) -> torch.Tensor:
        """scratch_flat: (B, scratch_dim). target_n_norm: (B, 1) ∈ [0, 1].
        Returns predicted β signal (B, t_pred).
        """
        x = torch.cat([scratch_flat, target_n_norm], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class ReviewModule(nn.Module):
    """idea2: iterative refinement of complete scratch via transformer.

    After write phase produces scratch_list (W slots), this module re-mixes
    h_α + scratch tokens for K_review passes, outputting refined per-slot
    embeddings that go through write_head again.
    """

    def __init__(
        self, d_model: int, num_layers: int = 2, num_heads: int = 2,
        ff_dim: int = 64,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 1+W, d) — h_α as first token, refined-scratch tokens after.
        Returns same shape, refined.
        """
        return self.encoder(tokens)


class VectorQuantizerEMA(nn.Module):
    """β-1 Path β: shared VQ codebook with EMA update + commitment loss + STE.

    Forward: continuous z (B, d_z) → nearest code z_q (B, d_z) + indices.
    EMA updates codebook (no gradient on codebook). Commitment loss pulls
    encoder output z toward its assigned code.
    """

    def __init__(
        self, K: int = 32, d_z: int = 8,
        commitment: float = 0.25, decay: float = 0.99, eps: float = 1e-5,
    ):
        super().__init__()
        self.K = K
        self.d_z = d_z
        self.commitment = commitment
        self.decay = decay
        self.eps = eps
        # Codebook (buffer, EMA-updated, no grad)
        self.register_buffer("codebook", torch.randn(K, d_z) * 0.1)
        self.register_buffer("ema_count", torch.zeros(K))
        self.register_buffer("ema_sum", torch.randn(K, d_z) * 0.1)

    def forward(self, z: torch.Tensor):
        """z: (..., d_z). Returns (z_q, commit_loss, idx).
        z_q: (..., d_z) straight-through. idx: (...,) long.
        """
        flat = z.reshape(-1, self.d_z)
        # squared L2 distance to each code
        d2 = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(-1)
        )
        idx = d2.argmin(dim=-1)  # (B,)
        z_q = self.codebook[idx]  # (B, d_z)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.K).float()  # (B, K)
                self.ema_count.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay,
                )
                self.ema_sum.mul_(self.decay).add_(
                    onehot.t() @ flat.detach(),
                    alpha=1 - self.decay,
                )
                n = self.ema_count.sum()
                smoothed = (
                    (self.ema_count + self.eps)
                    / (n + self.K * self.eps) * n
                )
                self.codebook.copy_(self.ema_sum / smoothed.unsqueeze(1))

        commit_loss = F.mse_loss(flat, z_q.detach())
        z_q_st = flat + (z_q - flat).detach()
        # Reshape back to input shape
        z_q_out = z_q_st.reshape(z.shape)
        idx_out = idx.reshape(z.shape[:-1])
        return z_q_out, commit_loss, idx_out


def vicreg_variance_loss(
    h: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance regularizer: per-dim std should be >= gamma.

    Forces each feature dimension to have standard deviation >= gamma in
    the batch. Prevents representation collapse to constants. Applied to
    prediction targets in BYOL-style aux losses.

    h: (..., d) — flattened across leading dims to (-1, d) for std calc.
    Returns scalar loss.
    """
    h_flat = h.reshape(-1, h.shape[-1])
    std = (h_flat.var(dim=0) + eps).sqrt()  # (d,)
    return F.relu(gamma - std).mean()


class SequentialWriteHead(nn.Module):
    """V25: shared-MLP sequential write head.

    Each of W cells is written by the SAME small MLP (weight-tied). At cell w,
    input = concat([h_pfc, written_so_far]) where written_so_far is a (B, W)
    vector with cells [0..w-1] populated and rest 0. Each cell raw output goes
    through quantize STE and the quantized value is written into the buffer
    before the next cell. Capacity ~10× Linear(d, 1).

    Note: API differs from Linear write_head — instead of being called per
    write step with h_for_write, this is called ONCE per phase with h_pfc and
    returns (raw_all, q_all) of shape (B, W) each. Caller (agent.forward) must
    branch on write_head type.
    """
    def __init__(self, d_pfc: int = 16, W: int = 5, hidden: int = 32,
                 pending_marker: bool = False):
        super().__init__()
        self.W = W
        # PM (5/20, same-medium-faithful cell-symmetry break): init `written` with
        # the midpoint of the quantize alphabet (e.g. 0.5 for unit q=3 → {0, 0.5, 1.0})
        # rather than zeros. Each cell w sees buffer = [q_0, ..., q_{w-1}, 0.5,
        # 0.5, ...]; the number of trailing 0.5's encodes w implicitly. Breaks
        # the baseline cascade where cell 0 → q=0 makes cells 1..4 see identical
        # input as cell 0 and collapse to the same output.
        self.pending_marker = pending_marker
        self.cell_mlp = nn.Sequential(
            nn.Linear(d_pfc + W, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _initial_buffer(self, B: int, device, dtype, quantize_range: str) -> torch.Tensor:
        if not self.pending_marker:
            return torch.zeros(B, self.W, device=device, dtype=dtype)
        # Midpoint of the quantize alphabet: 0.5 for unit, 0.0 for symmetric.
        if quantize_range == "unit":
            mp = 0.5
        elif quantize_range == "symmetric":
            mp = 0.0
        else:
            mp = 0.0
        return torch.full((B, self.W), mp, device=device, dtype=dtype)

    def forward(self, h_pfc: torch.Tensor, quantize_levels: int, quantize_range: str):
        B = h_pfc.shape[0]
        device = h_pfc.device
        written = self._initial_buffer(B, device, h_pfc.dtype, quantize_range)
        cells_raw = []
        cells_q = []
        for w in range(self.W):
            input_w = torch.cat([h_pfc, written], dim=-1)  # (B, d+W)
            cell_w_raw = self.cell_mlp(input_w).squeeze(-1)  # (B,)
            if quantize_levels is not None:
                cell_w_q = _quantize_ste(
                    cell_w_raw.unsqueeze(-1), quantize_levels, quantize_range,
                ).squeeze(-1)  # (B,)
            else:
                cell_w_q = cell_w_raw
            cells_raw.append(cell_w_raw)
            cells_q.append(cell_w_q)
            # Append to written buffer for next iteration. Clone to avoid
            # in-place op on tensor that backprop needs.
            written = written.clone()
            written[:, w] = cell_w_q
        return torch.stack(cells_raw, dim=-1), torch.stack(cells_q, dim=-1)  # (B, W), (B, W)


class CoTPFC(nn.Module):
    """V27 (5/10): Chain-of-Thought PFC over n_chunks.

    🚨 STRICT RULE: PFC sees only [pfc_state, cnn_chunk, lru_h_chunk] (3 tokens)
    per iteration. NO merge of CNN/LRU sequence into single global token.

    For each chunk c ∈ [0, n_chunks):
        chunk_signal = signal[:, c*chunk_size:(c+1)*chunk_size, :]
        cnn_token = strided_conv(chunk)               # (B, d) — 1 token per chunk
        lru_state, h_complex = LRU(chunk, init_h=h_complex)  # h persistent
        lru_token = lru_state[:, -1, :]               # last LRU output of chunk
        pfc_input = stack([pfc_state, cnn_token, lru_token], dim=1)  # (B, 3, d)
        pfc_state = pfc_transformer(pfc_input)[:, 0, :]  # state slot

    LRU h_complex carries across chunks (== one continuous L-step LRU run,
    just sampled at chunk boundaries). PFC iterates only n_chunks times
    (4 chunks → 4 PFC forwards → attention signal not magnetically smeared).

    Multi-scale CNN: kernels in cot_pfc_cnn_kernels (default 5,51).
    Each kernel: stride=chunk_size, padding=(K-1)//2 → 1 output per chunk.
    Concat scales in channel dim, project to d_model.
    """

    def __init__(self, d_model: int, n_chunks: int, kernels: tuple,
                 cnn_channels_per_scale: int, lru_block: nn.Module,
                 input_dim: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_chunks = n_chunks
        self.kernels = kernels
        # Multi-scale strided conv (one per kernel size)
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, cnn_channels_per_scale, k, padding=(k - 1) // 2)
            for k in kernels
        ])
        self.cnn_proj = nn.Linear(len(kernels) * cnn_channels_per_scale, d_model)
        # LRU input projection: raw signal (input_dim) → d_model. LRU expects d_model.
        # In dual_pathway main path, CNN handles this; V27 needs own projection.
        self.lru_input_proj = nn.Linear(input_dim, d_model)
        # LRU block (shared with main agent, not owned)
        self.lru_block = lru_block
        # Learnable initial PFC state (1, d), broadcast to (B, d) per call
        self.init_state = nn.Parameter(torch.randn(d_model) * 0.02)

    def cnn_chunked(self, signal: torch.Tensor) -> torch.Tensor:
        """signal: (B, L_seg, input_dim) → (B, n_chunks, d_model).

        Multi-scale strided conv: each kernel produces n_chunks outputs.
        Concat scales in channel dim, project to d_model.
        """
        B, L_seg, _ = signal.shape
        chunk_size = L_seg // self.n_chunks
        xt = signal.transpose(1, 2)  # (B, input_dim, L_seg)
        outs = []
        for conv in self.convs:
            o = F.relu(conv(xt))  # (B, ch, ~L_seg+pad) full stride=1
            # Subsample to n_chunks outputs at chunk boundaries (chunk centers)
            # Use indices c*chunk_size + chunk_size//2 (chunk centers)
            idx = torch.arange(self.n_chunks, device=o.device) * chunk_size + chunk_size // 2
            idx = idx.clamp(max=o.shape[-1] - 1)
            o = o[:, :, idx]  # (B, ch, n_chunks)
            outs.append(o)
        cnn_combined = torch.cat(outs, dim=1).transpose(1, 2)  # (B, n_chunks, total_ch)
        return self.cnn_proj(cnn_combined)  # (B, n_chunks, d_model)

    def forward(self, signal: torch.Tensor, pfc_module: nn.Module,
                gate: Optional[torch.Tensor] = None,
                kwta_k: int = 0,
                stash: Optional[dict] = None) -> torch.Tensor:
        """signal: (B, L_seg, input_dim) → (B, d_model) final pfc_state.

        pfc_module: shared transformer encoder (self.pfc on agent).
        gate: optional (B, L_seg) for LRU event gating.
        stash: optional dict to record per-chunk intermediate states for diag.
        """
        B, L_seg, _ = signal.shape
        chunk_size = L_seg // self.n_chunks

        # CNN: 1 token per chunk (n_chunks tokens total)
        cnn_chunks = self.cnn_chunked(signal)  # (B, n_chunks, d_model)

        # PFC initial state, broadcast to batch
        pfc_state = self.init_state.unsqueeze(0).expand(B, -1)  # (B, d)
        # LRU complex hidden state (persistent across chunks)
        h_lru_r, h_lru_i = None, None

        if stash is not None:
            stash["cnn_chunks"] = cnn_chunks.detach()
            stash["pfc_states_per_chunk"] = []
            stash["lru_h_per_chunk"] = []

        for c in range(self.n_chunks):
            chunk_start = c * chunk_size
            chunk_end = (c + 1) * chunk_size if c < self.n_chunks - 1 else L_seg
            chunk = signal[:, chunk_start:chunk_end, :]

            # LRU on chunk continuing from prev complex h (persistent)
            # Project chunk signal to d_model first (LRU expects d-dim input).
            chunk_proj = self.lru_input_proj(chunk)  # (B, chunk_size, d)
            chunk_gate = gate[:, chunk_start:chunk_end] if gate is not None else None
            lru_out, h_lru_r, h_lru_i = self.lru_block(
                chunk_proj, gate=chunk_gate,
                init_h_r=h_lru_r, init_h_i=h_lru_i, return_state=True,
            )
            lru_token = lru_out[:, -1, :]  # last LRU output of this chunk (B, d)

            # PFC chain step
            cnn_token = cnn_chunks[:, c, :]  # (B, d)
            tokens = torch.stack([pfc_state, cnn_token, lru_token], dim=1)  # (B, 3, d)
            refined = pfc_module(tokens)
            pfc_state = refined[:, 0, :]
            if kwta_k > 0:
                pfc_state = kwta_topk(pfc_state, k=kwta_k)

            if stash is not None:
                stash["pfc_states_per_chunk"].append(pfc_state.detach())
                stash["lru_h_per_chunk"].append(lru_token.detach())

        return pfc_state


class LRUDecoder(nn.Module):
    """V28-LRU (5/10): LRU-based decoder write head — state-init only free response.

    h_pfc → state_init_proj(Linear d→d_state) → real part of complex h_0 (imag=0).
    Then 5-step pure recurrence: h_{t+1} = λ ⊙ h_t (no input B, no skip D).
    Each step: y_t = Re(C_real h_r - C_imag h_i) → 1 scalar cell.
    Quantize tanh-style 3 levels {-1, 0, +1} (sign-preserving for bipolar
    LRU output, vs V27 sigmoid {0, 0.5, 1}).

    Key design (per user 5 decisions):
      - Independent (NOT shared) from encoder LRU. encoder/decoder time
        substrates differ (~120 raw samples vs 5 cell steps).
      - State-init only: h_0 = state_init_proj(h_pfc), x_t = 0 ∀t.
      - B=0, D=0 disabled (frozen, never built — pure dynamical system).
      - Tight |λ| ∈ [0.97, 0.99] init so envelope decays <14% over 5 steps.
      - θ ~ Uniform(0, π) for spectrum spread.

    Caveat: cell_i(N) is LINEAR in N (since LRU dynamics are linear and h_pfc
    is linear in N). Phase oscillation is on STEP axis, NOT N axis. So V28-LRU
    is NOT a true modular encoder — it's a stepping stone that may decouple
    cells (different timing samples) but ultimately still thermometer per cell.
    For real modular emergence see V28-Sin.
    """
    def __init__(self, d_pfc: int, W: int, d_state: int = 16,
                 r_min: float = 0.97, r_max: float = 0.99,
                 phase_max: float = math.pi):
        super().__init__()
        self.W = W
        self.d_state = d_state
        # Eigenvalue parameterization (same as LRUBlock): |λ_i| = exp(-exp(ν_i))
        u1 = torch.rand(d_state)
        u2 = torch.rand(d_state)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * phase_max + 1e-6)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        # State-init projection: h_pfc → real part of complex state
        self.state_init_proj = nn.Linear(d_pfc, d_state)
        # Output projection (complex state → real scalar per step)
        self.C_real = nn.Linear(d_state, 1, bias=False)
        self.C_imag = nn.Linear(d_state, 1, bias=False)

    def get_lambda(self):
        decay = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)
        return decay * torch.cos(phase), decay * torch.sin(phase)

    def forward(self, h_pfc: torch.Tensor, quantize_levels: int, quantize_range: str):
        """🚨 quantize_range arg IGNORED. V28-LRU uses V27 standard sigmoid quantize:
        raw LRU output → sigmoid → quantize {0, 1/(q-1), ..., 1.0} (unit mode)
        → scratch ∈ [0, 1] ✓ same medium as signal channel.

        Symmetric with V27 SequentialWriteHead quantize backend (also sigmoid → unit).
        The 3 heads (Sequential / LRU / Sin) share the SAME quantize backend; only
        the pre-activation differs. Clean ablation surface.
        """
        from backend.core.agent import _quantize_ste_with_int
        B = h_pfc.shape[0]
        lam_r, lam_i = self.get_lambda()
        h_r = self.state_init_proj(h_pfc)
        h_i = torch.zeros_like(h_r)
        outputs = []
        for t in range(self.W):
            y = self.C_real(h_r).squeeze(-1) - self.C_imag(h_i).squeeze(-1)  # (B,)
            outputs.append(y)
            new_r = lam_r * h_r - lam_i * h_i
            new_i = lam_r * h_i + lam_i * h_r
            h_r, h_i = new_r, new_i
        raw = torch.stack(outputs, dim=-1)  # (B, W)
        # V27-standard sigmoid → unit-medium quantize (same backend as Sequential)
        q, q_int = _quantize_ste_with_int(raw, quantize_levels, "unit")
        return raw, q


class SinusoidalWriteHead(nn.Module):
    """V28-Sin (5/10): TRUE periodic-in-N write head.

    cell_i = sin(ω_i · (direction_i · h_pfc) + φ_i)

    Since h_pfc ≈ c·N·v_N, args_i ≈ ω_i · (direction_i · v_N) · c · N + φ_i.
    Different ω_i → different N period → multi-modular codebook (the project's
    north star — positional value / mod-m emergence).

    Design (per user 6 decisions, post-pushback):
      - Explicit ω parameterization (NOT implicit via Linear weight scale).
        Why: implicit init (std=0.1) gives all cells ω≈0.1 same period, just
        phase offsets. Explicit linspace([0.3..2.3]) forces different periods.
      - direction: Linear(d_pfc, W, bias=False) — N-direction projection per cell
      - ω: Parameter(W,) init linspace; trainable
      - φ: Parameter(W,) init Uniform(0, 2π); breaks symmetry
      - NO linear bypass (sin only). curriculum n=30/60/90 acts as warmup.
      - Quantize 5-level tanh-symmetric {-1, -0.5, 0, +0.5, +1}: phase
        resolution beats q=3 sign-only.

    Init check on first training forward: prints per-cell args range +
    cycles count (over batch). User watches for:
      - cycles < 0.5 → near-linear, std too small, retrain with larger init
      - cycles > 5 → high-freq landscape, quantize noise risk
      - 1-3 cycles → OK
    """
    def __init__(self, d_pfc: int, W: int,
                 omega_init: tuple = (0.3, 0.7, 1.2, 1.7, 2.3),
                 init_std: float = 0.5):
        super().__init__()
        self.W = W
        if len(omega_init) != W:
            raise ValueError(
                f"omega_init len {len(omega_init)} != W {W}. "
                f"Provide W omega values."
            )
        # direction: per-cell N-direction projection (no bias, no scale; scale = ω)
        self.direction = nn.Linear(d_pfc, W, bias=False)
        # Init direction with reasonable spread so projection magnitude meaningful
        nn.init.normal_(self.direction.weight, mean=0.0, std=init_std)
        # Explicit per-cell frequency
        self.omega = nn.Parameter(torch.tensor(omega_init, dtype=torch.float32))
        # Phase offset, breaks symmetry
        self.phi = nn.Parameter(torch.rand(W) * 2 * math.pi)
        self._init_check_done = False

    def forward(self, h_pfc: torch.Tensor, quantize_levels: int, quantize_range: str):
        from backend.core.agent import _quantize_ste_with_int
        proj = self.direction(h_pfc)  # (B, W)
        args = self.omega * proj + self.phi
        raw = torch.sin(args)  # (B, W) ∈ [-1, 1]
        # Init check on first training forward
        if self.training and not self._init_check_done:
            self._init_check_done = True
            with torch.no_grad():
                args_min = args.min(dim=0).values
                args_max = args.max(dim=0).values
                spans = args_max - args_min
                cycles = spans / (2 * math.pi)
                print(f"[V28-Sin init check, batch B={h_pfc.shape[0]}]")
                print(f"  per-cell args range:")
                for i in range(self.W):
                    print(f"    cell {i}: [{args_min[i].item():+.2f}, {args_max[i].item():+.2f}]  "
                          f"span {spans[i].item():.2f} rad  ≈ {cycles[i].item():.2f} cycles")
                print(f"  ω: {[f'{w:.3f}' for w in self.omega.detach().cpu().tolist()]}")
                print(f"  φ: {[f'{p:.3f}' for p in self.phi.detach().cpu().tolist()]}")
                low = (cycles < 0.5).sum().item()
                high = (cycles > 5).sum().item()
                if low > 0:
                    print(f"  ⚠️ {low}/{self.W} cells <0.5 cycles → near-linear (thermometer risk)")
                if high > 0:
                    print(f"  ⚠️ {high}/{self.W} cells >5 cycles → high-freq (quantize noise risk)")
        # 🚨 Map sin output [-1, 1] to [0, 1] for SAME-MEDIUM with signal channel.
        # quantize_range arg IGNORED. Force linear_unit (no extra sigmoid that
        # would compress sin's already-bounded output to (0.27, 0.73) → all 0.5).
        mapped = (raw + 1.0) * 0.5  # [0, 1]
        q, q_int = _quantize_ste_with_int(mapped, quantize_levels, "linear_unit")
        return raw, q


class MultiHotMLPSuccessorHead(nn.Module):
    """V28-Sin predict head: read scratch as multi-hot encoded categorical levels,
    not as scalars. Linear-on-scalar-quantized treats levels as ordinal magnitude
    (level 1 = 2× level 0.5), which is the WRONG inductive bias for sin output
    where levels are categorical phase labels.

    Multi-hot exposes categorical structure; MLP can learn lookup-style
    (cell_i level k) → contribution to k_logit, supporting multi-modular code.

    Input: sc_a_int, sc_b_int both (B, W) int ∈ [0, q_levels-1]
    Encoded: one_hot(q_levels) flatten cat → (B, 2*W*q_levels)
    MLP: Linear(2*W*q_levels, hidden) → GELU → Linear(hidden, k_max)
    """
    def __init__(self, W: int, q_levels: int, k_max: int, hidden: int = 64):
        super().__init__()
        self.W = W
        self.q_levels = q_levels
        in_dim = 2 * W * q_levels
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, k_max),
        )

    def forward(self, sc_a_int: torch.Tensor, sc_b_int: torch.Tensor) -> torch.Tensor:
        # sc_a_int / sc_b_int: (B, W) long
        sc_a_oh = F.one_hot(sc_a_int, num_classes=self.q_levels).float()  # (B, W, q)
        sc_b_oh = F.one_hot(sc_b_int, num_classes=self.q_levels).float()
        x = torch.cat([sc_a_oh.flatten(1), sc_b_oh.flatten(1)], dim=-1)
        return self.mlp(x)


class RecurrentScratchRefiner(nn.Module):
    """V24R: 3-iter recurrent refinement on scratch_α / scratch_β.

    Each iter:
      1. cnn_scratch(sc_q.unsqueeze(1)) → cnn_feat (B, channels*W)
      2. combined = cat([pfc_h, lru_h, cnn_feat])
      3. iter_proj(combined) → new pfc_h
      4. tokens = pfc_h.unsqueeze(1) + scratch_pos_emb → (B, W, d)
      5. raw = write_head(tokens).squeeze(-1) → (B, W)
      6. sc_q_new = quantize_ste(raw)
    Output: refined sc_q after n_iter rounds.

    Note: write_head + scratch_pos_emb + quantize_fn passed in (shared with
    main agent). This module only owns cnn_scratch + iter_proj.
    """
    def __init__(self, W: int, d_pfc: int, d_lru: int,
                 scratch_cnn_kernel: int = 3,
                 scratch_cnn_channels: int = 4,
                 n_iter: int = 3):
        super().__init__()
        self.W = W
        self.n_iter = n_iter
        pad = scratch_cnn_kernel // 2
        self.cnn_scratch = nn.Conv1d(
            in_channels=1, out_channels=scratch_cnn_channels,
            kernel_size=scratch_cnn_kernel, padding=pad,
        )
        in_dim = d_pfc + d_lru + scratch_cnn_channels * W
        self.iter_proj = nn.Linear(in_dim, d_pfc)

    def forward(self, sc_q: torch.Tensor, h_pfc: torch.Tensor, lru_h: torch.Tensor,
                write_head: nn.Module, scratch_pos_emb: torch.Tensor,
                quantize_levels: int, quantize_range: str):
        sc = sc_q
        pfc = h_pfc
        for _ in range(self.n_iter):
            cnn_in = sc.unsqueeze(1)  # (B, 1, W)
            cnn_feat = self.cnn_scratch(cnn_in).flatten(1)  # (B, channels*W)
            combined = torch.cat([pfc, lru_h, cnn_feat], dim=-1)
            pfc = self.iter_proj(combined)
            if isinstance(write_head, SequentialWriteHead):
                # V25 + V24R: sequential rollout
                _, sc = write_head(pfc, quantize_levels, quantize_range)
            else:
                B = pfc.shape[0]
                pos = scratch_pos_emb.unsqueeze(0).expand(B, -1, -1)
                tokens = pfc.unsqueeze(1) + pos  # (B, W, d_pfc)
                raw = write_head(tokens).squeeze(-1)  # (B, W)
                sc = _quantize_ste(raw, quantize_levels, quantize_range)
        return sc


class NClassHead(nn.Module):
    """V23b: predict N_α from scratch_α via Linear-ReLU-Linear classifier.

    max_n is fixed at architecture build time (e.g. 90 for stage-4 max).
    Earlier stages have ground truth only for N ∈ [1, current_n_max], higher
    class indices get no supervisory signal from those batches.
    """
    def __init__(self, W: int = 5, max_n: int = 90, hidden: int = 32):
        super().__init__()
        self.max_n = max_n
        self.net = nn.Sequential(
            nn.Linear(W, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_n),
        )

    def forward(self, scratch: torch.Tensor) -> torch.Tensor:
        return self.net(scratch)


class SuccessorFunction(nn.Module):
    """V22: residual MLP learning +1 mapping on raw (pre-quantize) scratch.

    Input/output shape: (B, W) where W = scene_cfg.W (number of scratch cells).
    Output is `x + Δ(x)` (residual). Training signal comes from within-episode
    |δ|=1 G/L pairs only — no absolute N label needed.
    """
    def __init__(self, W: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(W, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, W),
        )

    def forward(self, scratch_raw: torch.Tensor) -> torch.Tensor:
        return scratch_raw + self.net(scratch_raw)

    def apply_k(self, scratch_raw: torch.Tensor, k: int) -> torch.Tensor:
        out = scratch_raw
        for _ in range(k):
            out = self.forward(out)
        return out


class DeepMLPPFC(nn.Module):
    """51_M5: 2-layer flatten+linear PFC variant (vs transformer encoder).

    Replaces nn.TransformerEncoder. Takes (B, num_tokens, d) → flatten →
    Linear(d*num_tokens, hidden) → GELU → Linear(hidden, d*num_tokens) →
    reshape to (B, num_tokens, d). Per-token cross-mixing without attention.

    Same I/O contract as transformer PFC for drop-in replacement.
    """

    def __init__(self, d_model: int, num_tokens: int, hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.flat_dim = d_model * num_tokens
        self.fc1 = nn.Linear(self.flat_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.flat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_tokens, d). Squeeze and flatten then unflatten.
        B = x.shape[0]
        flat = x.reshape(B, -1)
        h = self.act(self.fc1(flat))
        out = self.fc2(h)
        return out.reshape(B, x.shape[1], x.shape[2])


class RBFCompareHead(nn.Module):
    """V8 (2026-05-05): Gaussian RBF compare head.

    Replaces simple_comparator's Linear(d, 3). For input diff ∈ R^d:
        logit[c] = -|diff - center_c|^2 / (2 * sigma_c^2)

    3 learnable centers (one per class G/L/E). Each class gets a learnable
    log-sigma. Output logits naturally bounded (≤ 0); softmax always sharp.

    Init: centers near 0 (small random); sigma = init_sigma.
    """
    def __init__(self, d_model: int, num_classes: int = 3, init_sigma: float = 1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, d_model) * 0.1)
        self.log_sigma = nn.Parameter(torch.full((num_classes,), float(math.log(init_sigma))))

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        # diff: (B, d_model) — assumed (h_α_pfc - h_β_pfc) for simple_comparator path
        sigma = self.log_sigma.exp().clamp(0.05, 100.0)  # avoid degenerate values
        # Pairwise: (B, num_classes, d_model)
        delta = diff.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = (delta ** 2).sum(dim=-1)  # (B, num_classes)
        logits = -dist_sq / (2.0 * sigma.unsqueeze(0) ** 2)
        return logits


def kwta_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """V8_kWTA: keep top-k abs activations per last-dim, zero rest.

    Differentiable: mask is built from indices (no grad), multiplication
    propagates gradient through unmasked positions (zeros block grad like ReLU).

    Args:
        x: (..., d) tensor
        k: number of dims to keep (must be <= d)
    Returns: (..., d) sparse tensor
    """
    if k <= 0 or k >= x.shape[-1]:
        return x
    topk_idx = x.abs().topk(k, dim=-1).indices
    mask = torch.zeros_like(x)
    mask.scatter_(-1, topk_idx, 1.0)
    return x * mask


class AutoRegWriteHead(nn.Module):
    """V9-AR: GRUCell autoregressive write head.

    Replaces Linear(d, 1) → STE-quantize per write step with a stateful module:
        h_rnn_0 = 0
        prev    = 0   (B, 1)
        for k in 1..W:
            x       = [h_for_write_k, embed(prev)]
            h_rnn_k = GRUCell(x, h_rnn_{k-1})
            logits  = Linear(h_rnn_k)               # (B, Q)
            cell_k  = Σ q_levels * gumbel_softmax(logits, hard)
            prev    = cell_k

    Output `cell_k` is in {0, 1/(Q-1), ..., 1} (unit range), forward uses hard
    one-hot × levels = exactly quantized scalar; backward uses soft gumbel grad.

    Mitigations vs naive AR:
    - tau anneal (init→final over warmup_steps train steps): smoother early gradient
    - prev_dropout: random masking of prev signal prevents c1-dominance / chain
      degeneracy by forcing each step to look at h_for_write directly sometimes.
    """

    def __init__(
        self,
        d_model: int,
        Q: int = 3,
        embed_dim: int = 4,
        prev_dropout: float = 0.3,
        tau_init: float = 1.5,
        tau_final: float = 0.8,
        tau_warmup: int = 5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.Q = Q
        self.embed_dim = embed_dim
        self.prev_dropout = prev_dropout
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.tau_warmup = max(1, tau_warmup)
        self.gru = nn.GRUCell(input_size=d_model + embed_dim, hidden_size=d_model)
        self.cell_embed = nn.Linear(1, embed_dim)
        self.logit = nn.Linear(d_model, Q)
        # V9-AR fix (2026-05-05): zero-init logit so initial output is uniform
        # over levels (1/Q each). Default Linear init had a soft bias that
        # collapsed all cells to the middle level [0.5,...] basin.
        nn.init.zeros_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)
        # q_levels: {0, 1/(Q-1), ..., 1} for unit range
        if Q > 1:
            levels = torch.linspace(0.0, 1.0, Q)
        else:
            levels = torch.tensor([0.5])
        self.register_buffer("q_levels", levels)
        # Per-instance step counter for tau anneal (training only)
        self.register_buffer("cur_step", torch.zeros((), dtype=torch.long))
        # Entropy accumulator (anti-collapse bonus). Reset in init_state, read
        # by trainer for total loss = ce_loss − λ * mean_entropy.
        self._entropy_sum: Optional[torch.Tensor] = None
        self._entropy_count: int = 0

    def init_state(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_rnn = torch.zeros(B, self.d_model, device=device)
        prev = torch.zeros(B, 1, device=device)
        # Reset entropy accumulator for the new forward pass.
        self._entropy_sum = torch.zeros((), device=device)
        self._entropy_count = 0
        return h_rnn, prev

    def get_mean_entropy(self) -> torch.Tensor:
        """Mean entropy of softmax(logits) across W steps. 0 if no steps run."""
        if self._entropy_sum is None or self._entropy_count == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._entropy_sum / float(self._entropy_count)

    def current_tau(self) -> float:
        if not self.training:
            return self.tau_final
        s = float(self.cur_step.item())
        if s >= self.tau_warmup:
            return self.tau_final
        frac = s / self.tau_warmup
        return self.tau_init + frac * (self.tau_final - self.tau_init)

    def maybe_increment_step(self):
        """Trainer or forward should call once per training batch (across all W steps)."""
        if self.training:
            self.cur_step += 1

    def step(
        self,
        h_for_write: torch.Tensor,   # (B, d_model)
        prev_cell: torch.Tensor,     # (B, 1)
        h_rnn: torch.Tensor,         # (B, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Prev-embed dropout (training only): forces the head to also use h_for_write
        if self.training and self.prev_dropout > 0:
            mask = (
                torch.rand(prev_cell.size(0), 1, device=prev_cell.device)
                > self.prev_dropout
            ).float()
            prev_cell_in = prev_cell * mask
        else:
            prev_cell_in = prev_cell
        prev_e = self.cell_embed(prev_cell_in)               # (B, embed)
        x = torch.cat([h_for_write, prev_e], dim=-1)         # (B, d + embed)
        h_new = self.gru(x, h_rnn)                           # (B, d_model)
        lg = self.logit(h_new)                               # (B, Q)
        # Track entropy of softmax(logits) for anti-collapse bonus (training only).
        if self.training and self._entropy_sum is not None:
            p = F.softmax(lg, dim=-1)
            H = -(p * (p + 1e-10).log()).sum(dim=-1).mean()
            self._entropy_sum = self._entropy_sum + H
            self._entropy_count += 1
        if self.training:
            tau = self.current_tau()
            # hard=True: forward = one-hot, backward = soft gumbel gradient (ST)
            soft = F.gumbel_softmax(lg, tau=tau, hard=True)
        else:
            idx = lg.argmax(dim=-1)
            soft = F.one_hot(idx, num_classes=self.Q).float()
        # Weighted sum to scalar in [0, 1]
        cell = (soft * self.q_levels.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (B, 1)
        return cell, h_new, lg


class IFWriteHead(nn.Module):
    """P3 (V11, 2026-05-06): Integrate-and-Fire write controller.

    Replaces the fixed W-step write loop with leaky-IF dynamics over the
    encoder output sequence. State accumulates input; learnable threshold
    determines when to "fire" (commit a value to scratch); soft reset on fire
    releases capacity for further accumulation.

    The model decides WHEN to write based on its internal state.

    Threshold = nn.Parameter (scalar), learned via backprop.
    Reset strength = sigmoid(nn.Parameter), learned (in [0, 1]).

    Top-W of T fire-urgency time steps actually become scratch writes (W is
    fixed scratch capacity; if more than W fires happen, top-urgency W are
    selected; if fewer than W, remaining slots are zero-padded).

    Forward I/O:
      Input  state_seq: (B, T, d) — encoder output (CNN or Mamba) per timestep
      Output (B, W, 1) — quantized scratch values
    """

    def __init__(
        self,
        d_model: int,
        W: int,
        Q: int = 3,
        init_threshold: float = 1.0,
        init_reset: float = 0.5,
        fire_tau: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.W = W
        self.Q = Q
        self.fire_tau = fire_tau
        # Fire urgency: ||accum||_2 (no learnable proj; ensures monotonically
        # positive urgency that grows with accumulation, like classic IF
        # membrane potential). Earlier learnable fire_proj degenerated to
        # negative weights → never fires → top-W collapses to first-W timesteps.
        # Per-cell value: state → scalar (raw, will quantize via STE)
        self.write_proj = nn.Linear(d_model, 1)
        # Learnable threshold (scalar) — model learns "what accum norm = full"
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        # Learnable reset strength (sigmoid'd to [0, 1])
        rl = math.log(init_reset / max(1e-6, 1.0 - init_reset))
        self.reset_logit = nn.Parameter(torch.tensor(rl))

    def forward(
        self, state_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run IF dynamics + top-W selection.

        Returns:
            cells: (B, W, 1) quantized scratch values
            fire_scores_raw: (B, T) urgency per timestep (for diagnostics)
            top_w_indices: (B, W) selected fire timesteps
        """
        B, T, d = state_seq.shape
        device = state_seq.device

        accum = torch.zeros(B, d, device=device)
        fire_scores: List[torch.Tensor] = []
        states: List[torch.Tensor] = []

        for t in range(T):
            # Accumulate
            accum = accum + state_seq[:, t, :]
            # Fire urgency = L2 norm of accumulated state (always ≥ 0)
            urgency = accum.norm(dim=-1)  # (B,)
            fire_logit = urgency - self.threshold
            if self.training:
                # Gumbel-sigmoid (binary concrete). hard=True is straight-through.
                # F.gumbel_softmax with 2 classes ≈ gumbel-sigmoid; manually:
                noise = -torch.log(-torch.log(
                    torch.rand_like(fire_logit) + 1e-10) + 1e-10)
                soft = torch.sigmoid((fire_logit + noise) / self.fire_tau)
                # Straight-through: hard forward, soft backward
                hard = (soft > 0.5).float()
                fire = hard + (soft - soft.detach())
            else:
                fire = (fire_logit > 0).float()
            fire_scores.append(urgency)
            states.append(accum)
            # Soft reset on fire
            reset_strength = torch.sigmoid(self.reset_logit)
            accum = accum * (1.0 - fire.unsqueeze(-1) * reset_strength)

        fire_scores_t = torch.stack(fire_scores, dim=1)  # (B, T)
        states_t = torch.stack(states, dim=1)            # (B, T, d)

        # Top-W selection per batch (sort by time after picking for natural order)
        if T <= self.W:
            # Use all T steps; pad zero if T < W
            sel = states_t  # (B, T, d)
            if T < self.W:
                pad = torch.zeros(B, self.W - T, d, device=device)
                sel = torch.cat([sel, pad], dim=1)
            top_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            if T < self.W:
                pad_idx = torch.full(
                    (B, self.W - T), -1, dtype=torch.long, device=device,
                )
                top_idx = torch.cat([top_idx, pad_idx], dim=1)
        else:
            top_idx_unsorted = fire_scores_t.topk(self.W, dim=1).indices  # (B, W)
            top_idx = top_idx_unsorted.sort(dim=1).values
            # Gather the d-dim states at those indices
            sel = states_t.gather(
                1, top_idx.unsqueeze(-1).expand(-1, -1, d)
            )  # (B, W, d)

        # Project + quantize each cell via STE
        cell_raw = self.write_proj(sel).squeeze(-1)  # (B, W)
        cell_q = _quantize_ste(cell_raw, self.Q, "unit")  # (B, W)
        return cell_q.unsqueeze(-1), fire_scores_t, top_idx


class LRUBlock(nn.Module):
    """V18 (2026-05-07): Complex-eigenvalue Linear Recurrent Unit.

    Orvieto et al. 2023 "Resurrecting RNNs". Stable parameterization:
        λ_i = exp(-exp(ν_i) + i·θ_i)   stability: |λ_i| < 1
    Recurrence (per dim):
        h_t = λ ⊙ h_{t-1} + γ ⊙ (B x_t)
    where γ_i = sqrt(1 - |λ_i|²) for input-gain normalization.
    Output: y_t = Re(C h_t) + skip_connection(x_t).

    KEY property vs Mamba: complex spectrum allows oscillation (θ ≠ 0).
    Mamba's selective scan collapses to real eigenvalues under trio_wide
    pressure (1D thermometer optimum). LRU's parameterization PRESERVES
    complex structure if task gradient pressure rewards it (e.g. mod task).

    Implementation: parallel scan via cumulative product. d-real-dim model
    embedded as d/2 complex pairs (d_state = d/2 complex). State dynamics
    in complex space, projection back to real for next layer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        r_min: float = 0.4,
        r_max: float = 0.9,
        phase_max: float = 2 * math.pi,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Stable parameterization:
        # ν = log(-log(r))  → exp(-exp(ν)) = r ∈ (0, 1)
        # θ ~ uniform[0, phase_max]
        u1 = torch.rand(d_state)
        u2 = torch.rand(d_state)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * phase_max + 1e-6)
        self.nu_log = nn.Parameter(nu_log)        # decay (exp transformed)
        self.theta_log = nn.Parameter(theta_log)  # frequency (exp transformed)
        # Complex input projection B: real → 2 * d_state (real, imag)
        self.B_real = nn.Linear(d_model, d_state, bias=False)
        self.B_imag = nn.Linear(d_model, d_state, bias=False)
        # Complex output projection C: 2 * d_state → d_model (real out)
        self.C_real = nn.Linear(d_state, d_model, bias=False)
        self.C_imag = nn.Linear(d_state, d_model, bias=False)
        # Skip connection
        self.D = nn.Parameter(torch.randn(d_model) * 0.02)

    def get_lambda(self):
        """Compute complex λ from learnable nu, theta."""
        # |λ| = exp(-exp(nu_log)), in (0, 1)
        # arg(λ) = exp(theta_log)
        decay = torch.exp(-torch.exp(self.nu_log))           # (d_state,) real
        phase = torch.exp(self.theta_log)                    # (d_state,) real, ≥0
        # Complex λ
        lam_real = decay * torch.cos(phase)
        lam_imag = decay * torch.sin(phase)
        return lam_real, lam_imag, decay

    def forward(
        self, x: torch.Tensor, gate: Optional[torch.Tensor] = None,
        init_h_r: Optional[torch.Tensor] = None,
        init_h_i: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        """x: (B, T, d_model) → returns (B, T, d_model).

        gate: optional (B, T) ∈ [0, 1]. If provided, event-gated update:
          gate ≈ 1 → normal LRU recurrence (phase rotates)
          gate ≈ 0 → h frozen (no decay, no rotation)
        Used for V19: phase rotation occurs in spike-count dimension.

        init_h_r/init_h_i: V27 — optional initial complex hidden state (B, d_state).
        If None, init from zero. Used for chunked sequential processing where
        h must persist across chunks.
        return_state: V27 — if True, return (y, h_r_final, h_i_final) tuple.
        """
        B, T, _ = x.shape
        lam_r, lam_i, decay = self.get_lambda()
        gamma = torch.sqrt(torch.clamp(1.0 - decay ** 2, min=1e-6))
        Bx_r = self.B_real(x) * gamma
        Bx_i = self.B_imag(x) * gamma
        if init_h_r is None:
            h_r = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h_r = init_h_r
        if init_h_i is None:
            h_i = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h_i = init_h_i
        outs = []
        for t in range(T):
            new_r = lam_r * h_r - lam_i * h_i + Bx_r[:, t, :]
            new_i = lam_r * h_i + lam_i * h_r + Bx_i[:, t, :]
            if gate is not None:
                # V19: gated update. g≈1 → take new (rotate); g≈0 → keep h (freeze)
                g = gate[:, t].unsqueeze(-1)             # (B, 1)
                h_r = g * new_r + (1.0 - g) * h_r
                h_i = g * new_i + (1.0 - g) * h_i
            else:
                h_r, h_i = new_r, new_i
            y_r = self.C_real(h_r) - self.C_imag(h_i)
            outs.append(y_r)
        y = torch.stack(outs, dim=1)
        out = y + self.D * x
        if return_state:
            return out, h_r, h_i
        return out


class RecurrentLatentPFC(nn.Module):
    """V15 (2026-05-06): Coconut-style recurrent latent transformer over CNN
    fixation tokens.

    Replaces V10/V13's "feed all 8 CNN tokens at once with pos_emb" with
    truly sequential processing: one CNN token per step, with a learnable
    [STATE] token persisting across steps. Same transformer block (params
    shared) applied at every step, similar to Geiping recurrent depth but
    accumulating fresh input each step rather than iterating on fixed seq.

    Per step: seq = [state, cnn_token_t] (K+1 tokens). After block:
    state ← refined[:, :K, :] (carry the state position only).
    After T steps: state has accumulated info from all T fixations.

    Avoids Mamba/RNN's "magnitude only" pathology because state is a
    transformer-block latent vector evolving in d-dim space, not a 1D
    EMA sum. Optional kWTA sparsity per step prevents signal washout.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_state_tokens: int = 1,
        sparse_top_k: int = 0,
        ff_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.K = n_state_tokens
        self.sparse_top_k = sparse_top_k
        ff = ff_dim if ff_dim is not None else d_model
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        # Learnable initial state token(s). Init scale 0.5 so state has
        # non-trivial initial direction (vs 0.02 which is too quiet).
        self.init_state = nn.Parameter(torch.randn(n_state_tokens, d_model) * 0.5)
        # ReZero-style step gate: state += tanh(α) * (new - state).
        # init 0 → tanh(0)=0 → identity at start, training learns step size.
        self.step_alpha = nn.Parameter(torch.tensor(0.0))
        # Output rescale: brings state norm down to V9 scale (~0.4-0.5) so
        # that downstream write-PFC sees state and scratch_kv on same scale,
        # allowing scratch_kv to drive cross-write differentiation.
        # Init 0.15 ≈ V9_scale / measured_V15_no_rescale_scale (0.44 / 3.39).
        self.output_scale = nn.Parameter(torch.tensor(0.15))

    def forward(self, cnn_seq: torch.Tensor) -> torch.Tensor:
        """cnn_seq: (B, T, d). Returns (B, K, d) or (B, d) if K=1.

        ReZero-style gated update: state ← state + tanh(α) · (refined − state).
        α is learnable scalar init 0 → initial pass is identity (no accumulation).
        Training learns how aggressively to update state per step. Bounds
        magnitude: |state_new| ≤ |state| + |α|·|delta|, controlled by α.
        """
        B, T, d = cnn_seq.shape
        state = self.init_state.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, K, d)
        gate = torch.tanh(self.step_alpha)                           # (), in (-1, 1), init 0
        for t in range(T):
            cnn_t = cnn_seq[:, t:t+1, :]                             # (B, 1, d)
            seq = torch.cat([state, cnn_t], dim=1)                   # (B, K+1, d)
            refined = self.block(seq)
            new_state = refined[:, :self.K, :]                       # carry state pos only
            # Gated update: state = state + gate * (new_state - state)
            state = state + gate * (new_state - state)
            if self.sparse_top_k > 0:
                B_, K_, D_ = state.shape
                flat = state.reshape(-1, D_)
                flat = kwta_topk(flat, k=self.sparse_top_k)
                state = flat.view(B_, K_, D_)
        # V15: output rescale to V9-style scale, so downstream write-PFC sees
        # state and scratch_kv on comparable magnitude.
        state = state * self.output_scale
        if self.K == 1:
            return state.squeeze(1)
        return state


class GumbelWriteHead(nn.Module):
    """V14 (2026-05-06): Per-cell gumbel-softmax write head.

    Replaces `Linear(d, 1) → sigmoid+STE quantize` with `Linear(d, Q) →
    gumbel-softmax → weighted sum to scalar`. Output is one-hot in forward
    (categorical sample), differentiable via straight-through gumbel gradient
    in backward.

    Bypasses the sigmoid saturation pathology entirely — output is bounded by
    construction (always ∈ {q_levels}), regardless of upstream magnitudes.
    Drop-in replacement for `nn.Linear(d, 1)`: forward(h) returns (B, 1)
    quantized scalar in [0, 1].
    """

    def __init__(self, d_model: int, Q: int = 3, tau: float = 1.0):
        super().__init__()
        self.Q = Q
        self.tau = tau
        # Project h_for_write to Q logits — model picks one of Q levels per cell
        self.logit_proj = nn.Linear(d_model, Q)
        if Q > 1:
            levels = torch.linspace(0.0, 1.0, Q)
        else:
            levels = torch.tensor([0.5])
        self.register_buffer("q_levels", levels)

    def forward(self, h_for_write: torch.Tensor) -> torch.Tensor:
        """h_for_write: (..., d_model). Returns (..., 1) quantized scalar."""
        lg = self.logit_proj(h_for_write)            # (..., Q)
        if self.training:
            # hard=True: forward = one-hot, backward = soft gumbel-softmax (ST)
            soft = F.gumbel_softmax(lg, tau=self.tau, hard=True)
        else:
            idx = lg.argmax(dim=-1)
            soft = F.one_hot(idx, num_classes=self.Q).float()
        cell = (soft * self.q_levels).sum(dim=-1, keepdim=True)
        return cell


class CrossAttnWriteHead(nn.Module):
    """V13 (2026-05-06): Cross-attention write head.

    Replaces V10_PFCseq's W-step self-attention PFC loop. W learnable "write
    queries" cross-attend over a key/value sequence (cnn_alpha_seq). Each query
    learns its own attention pattern → W cells naturally differentiate WITHOUT
    write_head needing extreme weights.

    Hypothesis: V10_PFCseq's saturation came from 5 cells receiving near-identical
    h_for_write (mean over same 8 CNN tokens), forcing write_head to amplify
    differences via large weights → sigmoid saturated → cells frozen at extremes.
    Cross-attention parallelizes the writes with diverse queries → cells already
    different at h_for_write level → write_head can stay in linear region.

    Forward I/O:
      Input: kv (B, T, d) — encoder sequence (e.g. cnn_alpha_seq)
      Output: (B, W, d) — W differentiated h_for_write vectors, one per cell
    """

    def __init__(self, d_model: int, W: int, num_heads: int = 2):
        super().__init__()
        self.W = W
        self.d_model = d_model
        # Larger init scale (0.5) so 5 queries start with meaningfully different
        # directions in d=16 space (vs 0.02 which would make them too similar).
        self.write_queries = nn.Parameter(torch.randn(W, d_model) * 0.5)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True,
        )
        # Pre-norm on KV (standard cross-attention pattern)
        self.kv_norm = nn.LayerNorm(d_model)
        self.q_norm = nn.LayerNorm(d_model)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        """kv: (B, T, d). Returns (B, W, d)."""
        B = kv.size(0)
        q = self.write_queries.unsqueeze(0).expand(B, -1, -1)  # (B, W, d)
        q_n = self.q_norm(q)
        kv_n = self.kv_norm(kv)
        attended, _ = self.cross_attn(q_n, kv_n, kv_n, need_weights=False)
        # Residual: q + attended (preserves query identity in output)
        return q + attended


class WorldSignalPredictor(nn.Module):
    """P1 (L1579): predict next T_p RAW signal scalars from end-of-α latent.

    Output is in [0, 1] via sigmoid (matches input signal range). Used to
    construct an "imagined future world" by appending to the actual α
    input stream and re-scanning through the encoder.
    """

    def __init__(self, d_model: int, t_p: int, hidden_dim: int = 32):
        super().__init__()
        self.t_p = t_p
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, t_p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(h))
        return torch.sigmoid(self.fc2(x))  # (B, t_p) ∈ [0, 1]


class MambaAgent(nn.Module):
    """Pure Mamba variant of the v3 / v3-w architecture.

    The mandatory architectural pressures (resetH, splitAB) are baked into
    the segmented forward — no flag, always on. PFC is removed entirely.
    """

    def __init__(self, agent_cfg: MambaAgentConfig, scene_cfg: SceneConfig):
        super().__init__()
        # V18+: when --use-lru is on, Mamba blocks are replaced by LRUBlock and
        # the mamba_ssm library is not needed. Only raise if neither LRU nor
        # cnn_pfc_only path can carry the forward (i.e. legacy Mamba path).
        if not _HAS_MAMBA and not (
            agent_cfg.use_lru or agent_cfg.cnn_pfc_only or agent_cfg.use_v29 or agent_cfg.use_v33
            or agent_cfg.use_v35
        ):
            raise ImportError(
                "mamba_ssm not available. MambaAgent requires Linux + CUDA "
                "with mamba-ssm 2.x installed, OR --use-lru / --cnn-pfc-only / "
                "--use-v29 / --use-v33 / --use-v35 to bypass the Mamba path."
            )
        self.agent_cfg = agent_cfg
        self.scene_cfg = scene_cfg

        # V35 (5/27 pivot): d=1 HH-SSM + fire-and-reset cascade. Bypasses
        # V27/V28/V29/V33 forward paths entirely. Mirrors V29's early-return
        # pattern: build only the modules V35 needs (CNN encoder + cascade +
        # heads), stub out everything else.
        if agent_cfg.use_v35:
            from backend.core.v35 import V35Substrate
            from backend.core.v35_heads import V35WriteAttention, V35CompareAttention, V35WritePerCellReadout, V35ResidualDecoder
            if not agent_cfg.dual_pathway:
                raise ValueError(
                    "use_v35 requires --dual-pathway (CNN encoder). "
                    "Set dual_pathway=True in agent_cfg."
                )
            if agent_cfg.v35_n_recursions != scene_cfg.W:
                raise ValueError(
                    f"use_v35 requires v35_n_recursions == scene_cfg.W "
                    f"(got {agent_cfg.v35_n_recursions} vs W={scene_cfg.W}). "
                    "Each cascade recursion provides one scratch-slot residual."
                )
            d = agent_cfg.d_model
            # Build dual_pathway CNN encoder (replicates the dual_pathway block).
            cps = agent_cfg.dp_channels_per_scale
            self.dp_convs = nn.ModuleList([
                nn.Conv1d(
                    agent_cfg.input_dim, cps,
                    kernel_size=k, padding=k // 2, bias=True,
                )
                for k in agent_cfg.dp_kernels
            ])
            total_ch = cps * len(agent_cfg.dp_kernels)
            self.dp_proj = nn.Linear(total_ch, d)
            # V35 cascade substrate (d=1 HH-SSM with fire-and-reset).
            self.v35_substrate = V35Substrate(
                d_input=d,
                n_recursions=agent_cfg.v35_n_recursions,
                fire_temp=agent_cfg.v35_fire_temp,
                biological_init=agent_cfg.v35_local_pc,
                theta_a=agent_cfg.v35_theta_a,
                theta_b=agent_cfg.v35_theta_b,
                log_g_leak_a=agent_cfg.v35_log_g_leak_a,
                b_leak_warm_a=agent_cfg.v35_warm_b_leak_a,
                b_leak_b_init=agent_cfg.v35_b_leak_b_init,
                state_discretize=agent_cfg.v35_state_discretize,
                theta_shared_init=agent_cfg.v35_theta_shared_init,
            )
            if agent_cfg.v35_log_g_b is not None:
                # Hand-set Layer B per-channel log-conductance (timing-invariance
                # scaffolding, 6/03 — TEMPORARY). log_g -> very negative turns channels
                # off -> h holds between spikes -> event-driven base-theta counter ->
                # residual = base-3 digits, timing-invariant (probe-verified at -20).
                # Frozen (requires_grad False): a hand-set form, not learned.
                self.v35_substrate.cascade.layer_B.log_g.data.fill_(
                    float(agent_cfg.v35_log_g_b)
                )
                self.v35_substrate.cascade.layer_B.log_g.requires_grad_(False)
            # Write head: 5 cascade residuals -> scratch alphabet {0, 0.5, 1.0}.
            # Per-cell readout (route A) structurally binds cell k <- residual k;
            # cross-attention (default) lets all cells attend over all residuals.
            if agent_cfg.v35_write_per_cell:
                self.v35_write_attn = V35WritePerCellReadout(
                    n_slots=scene_cfg.W,
                    hidden=8,
                    quantize_levels=(agent_cfg.quantize_levels or 3),
                )
            else:
                self.v35_write_attn = V35WriteAttention(
                    n_slots=scene_cfg.W,
                    d_attn=agent_cfg.v35_write_d_attn,
                    n_heads=agent_cfg.v35_write_n_heads,
                    quantize_levels=(agent_cfg.quantize_levels or 3),
                )
            # Learned Layer-A object counter (direction-1: de-scaffold counting; replaces the
            # hand-coded food_pattern). Built only when --v35-layer-a-detect learned.
            self.v35_learned_la = None
            if getattr(agent_cfg, "v35_layer_a_detect", "level") == "learned":
                from backend.core.v35 import LearnedLayerADetector
                self.v35_learned_la = LearnedLayerADetector(
                    kernel=5, hidden=8, temp=4.0,
                    local_max=not getattr(agent_cfg, "v35_layer_a_no_localmax", False),
                )
                # direction-1 goal-step: init from an ES-pretrained detector (clean counter, where SGD
                # collapsed) and optionally FREEZE it -> V35 gets a LEARNED (not hand-coded food_pattern)
                # counter without the SGD-through-cascade collapse.
                _init = getattr(agent_cfg, "v35_detector_init", None)
                if _init:
                    self.v35_learned_la.load_state_dict(
                        torch.load(_init, map_location="cpu", weights_only=True))
                if getattr(agent_cfg, "v35_freeze_detector", False):
                    for _p in self.v35_learned_la.parameters():
                        _p.requires_grad_(False)
            # Compare attention: (readback_residuals, beta_residuals) -> k_offset logits.
            # n_classes = successor_predict_k_max (already 2*k_max for bidirectional
            # per train_phase1 convention).
            n_classes = max(agent_cfg.successor_predict_k_max, 2)
            self.v35_compare_attn = V35CompareAttention(
                n_slots=scene_cfg.W,
                d_attn=agent_cfg.v35_compare_d_attn,
                n_heads=agent_cfg.v35_compare_n_heads,
                n_classes=n_classes,
                n_layers=agent_cfg.v35_compare_n_layers,
                pool=("concat" if getattr(agent_cfg, "v35_compare_concat", False) else "mean"),
            )
            # V35 PSS leak-kill (spec 2026-06-05 §5.4): scratch slots, cascade recursions,
            # and both heads' slot counts must all equal W so direct-readback shapes align.
            assert (
                self.v35_write_attn.n_slots
                == self.v35_compare_attn.n_slots
                == agent_cfg.v35_n_recursions
                == scene_cfg.W
            ), "V35 invariant: write/compare n_slots must equal v35_n_recursions and W"
            # Codebook-fidelity reconstruction decoder (spec 2026-06-04). Built only when
            # the recon aux is active; decodes scratch -> residuals so scratch must keep
            # every digit (counters codebook collapse under weak |k|<=5 task pressure).
            self.v35_recon_decoder = None
            if (agent_cfg.v35_recon_aux_lambda or 0.0) > 0.0:
                self.v35_recon_decoder = V35ResidualDecoder(n_slots=scene_cfg.W, hidden=32)
            # Cheap-arithmetic aux head (INSIGHTS §9A). A single Linear over the quantized
            # alpha scratch code (in_dim = W) predicting the SAME k_target as the main task.
            # Built only when the lambda is active. n_classes matches the compare head's class
            # count (defined just above) so its CE shares k_targets. Reads scratch_alpha_q only
            # (always stashed at _scratch_alpha_for_aux); beta scratch is not available in the
            # same-medium path. Cheapness (linear) is the whole point: low CE is reachable only
            # if the scratch code supports linear arithmetic readout, pressuring the codebook
            # toward the positional class. Quantize-STE in the write head carries the gradient
            # from this CE back into the scratch code.
            self.v35_cheap_arith_head = None
            if (agent_cfg.v35_cheap_arith_aux_lambda or 0.0) > 0.0:
                self.v35_cheap_arith_head = nn.Linear(scene_cfg.W, n_classes)
            # Stub-out attributes the rest of the class (and forward dispatchers) check.
            self.v29 = None
            self.v33_substrate = None
            self.h_predictor = None
            self.cot_pfc = None
            self.recurrent_refiner = None
            self.successor_predict_head = None
            self.compare_head = nn.Identity()
            self.mamba_blocks = nn.ModuleList([])
            self.pfc = None
            self.pfc_compare = None
            self.read_cnn = None
            self.input_proj = None
            self.spike_detector = None
            self.gate_sharpness = None
            self.write_head = None
            # Init CNN-related stashes to keep references resolvable.
            self._last_cnn_alpha_summary = None
            self._last_cnn_beta_summary_per_block = []
            self._last_cnn_alpha_readback_per_block = []
            self._last_cnn_alpha_seq_per_block = []
            self._last_cnn_beta_seq_per_block = []
            self._last_cnn_alpha_seq = None
            # V25/V27-style stashes for aux losses (referenced by collect/train).
            self._scratch_alpha_for_aux = None
            self._scratch_alpha_raw = None
            self._last_scratch_beta_K_q = None
            self._last_scratch_beta_K_raw = None
            self._h_alpha_pfc_blocks = []
            self._h_beta_pfc_blocks = []
            self._h_readback_alpha_end = None
            self._h_predicted_alpha = None
            self._v35_successor_logits = None
            self._v35_residuals_alpha = None
            self._scratch_beta_for_aux = None
            self._scratch_beta_raw = None
            self._v35_residuals_beta = None
            self._v35_lesion = None
            return
        self.v35_substrate = None
        self.v35_write_attn = None
        self.v35_compare_attn = None

        # V29: if use_v29, build SNN pipeline and skip all V27/V28 modules
        if agent_cfg.use_v29:
            from backend.core.v29 import V29Pipeline
            self.v29 = V29Pipeline(
                n_encoder_neurons=agent_cfg.v29_encoder_n_neurons,
                n_write_neurons=agent_cfg.v29_write_n_neurons,
                d_id=agent_cfg.v29_d_id,
                d_pe=agent_cfg.v29_d_pe,
                n_pfc_layers=agent_cfg.v29_pfc_layers,
                n_pfc_heads=agent_cfg.v29_pfc_heads,
                n_pfc_passes=agent_cfg.v29_pfc_passes,
                threshold=agent_cfg.v29_threshold,
                surrogate_alpha=agent_cfg.v29_surrogate_alpha,
                omega_enc_min=agent_cfg.v29_omega_enc_min,
                omega_enc_max=agent_cfg.v29_omega_enc_max,
                omega_write_min=agent_cfg.v29_omega_write_min,
                omega_write_max=agent_cfg.v29_omega_write_max,
                b_init_min=agent_cfg.v29_b_init_min,
                b_init_max=agent_cfg.v29_b_init_max,
                time_max_init=agent_cfg.v29_time_max_init,
                firing_rate_lambda=agent_cfg.v29_firing_rate_lambda,
                target_firing_rate=agent_cfg.v29_target_firing_rate,
                soft_reset=agent_cfg.v29_soft_reset,
                t_write=(agent_cfg.v29_t_write if agent_cfg.v29_t_write > 0 else None),
                max_events=agent_cfg.v29_max_events,
                n_levels=agent_cfg.quantize_levels or 3,
                # `successor_predict_k_max` from train_phase1 already represents
                # 2*k_max for bidirectional successor (so for k_max=3 bidir, value=6).
                # V29Pipeline computes its own n_classes = 2*k_max if bidirectional,
                # so we must pass the un-doubled k_max here.
                k_max=max(agent_cfg.successor_predict_k_max // 2 if agent_cfg.successor_predict_k_max > 0 else 5, 3),
                bidirectional=True,
            )
            # Stub-out attributes the rest of the class (and forward dispatchers) check:
            self.cot_pfc = None
            self.recurrent_refiner = None
            self.successor_predict_head = None
            self.compare_head = nn.Identity()
            return
        self.v29 = None

        # V33: HH-SSM substrate (replaces V27 LRU+CoT-PFC core, keeps V25 write_head)
        if agent_cfg.use_v33:
            from backend.core.v33 import V33Substrate
            d = agent_cfg.v33_d_model
            agent_cfg.d_model = d  # sync top-level d_model
            self.v33_substrate = V33Substrate(
                d_model=d,
                d_input=d,
                n_layers=agent_cfg.v33_n_layers,
                perturbation_scale=agent_cfg.v33_v_perturbation_scale,
                input_to_gate=agent_cfg.v33_input_to_gate,
            )
            # V34: rank-W linear predictor — Linear(W=5, d_model=16). bias=True for
            # init stability (constant shift doesn't affect rank-W capacity). Built
            # iff predict_coding_enable so old SM ckpts (no predictor) still load.
            if agent_cfg.predict_coding_enable:
                self.h_predictor = nn.Linear(scene_cfg.W, d, bias=True)
            else:
                self.h_predictor = None
            assert agent_cfg.dual_pathway, (
                "use_v33 requires --dual-pathway (CNN encoder). "
                "Pass --dual-pathway --dp-kernels '5,21,51' or similar."
            )
            self.cot_pfc = None
            self.recurrent_refiner = None
            # write_head etc. still build below (V33 reuses V25 write_head)
        else:
            self.v33_substrate = None
            self.h_predictor = None
        # END V33 build

        d = agent_cfg.d_model

        # β-1: if VQ enabled, override signal_output_dim to vq_d_z so write_head,
        # scratch_proj, etc. all size to the codebook dimension. Read-back into
        # seg_B[:, j, 0] still uses only the first dim (lossy, by design).
        if agent_cfg.vq_enabled:
            agent_cfg.signal_output_dim = agent_cfg.vq_d_z

        # V6 BRANCH: cnn_pfc_only requires dual_pathway and forces skip-Mamba-on-readback.
        if agent_cfg.cnn_pfc_only:
            agent_cfg.dual_pathway = True
            agent_cfg.dp_scratch_skip_mamba = True

        # Front-end: CNN (M2) or Linear (M3) or V6 multi-scale CNN
        if agent_cfg.dual_pathway:
            # V6: multi-scale CNN replaces single read_cnn. Output projects to d_model.
            cps = agent_cfg.dp_channels_per_scale
            self.dp_convs = nn.ModuleList([
                nn.Conv1d(
                    agent_cfg.input_dim, cps,
                    kernel_size=k, padding=k // 2, bias=True,
                )
                for k in agent_cfg.dp_kernels
            ])
            total_ch = cps * len(agent_cfg.dp_kernels)
            self.dp_proj = nn.Linear(total_ch, d)
            # V19: independent single-spike detector for event-gated LRU.
            # Conv1d(1→1, k=5) on raw signal channel (channel 0 of inputs).
            # Gate = sigmoid(spike_score · gate_sharpness). Learnable sharpness
            # init 5.0 makes initial gate near-binary.
            if agent_cfg.lru_event_gated:
                self.spike_detector = nn.Conv1d(1, 1, kernel_size=5, padding=2)
                self.gate_sharpness = nn.Parameter(
                    torch.tensor(agent_cfg.gate_sharpness_init)
                )
            else:
                self.spike_detector = None
                self.gate_sharpness = None
            # Stash for PFC bypass — gradient-active features (NOT detached)
            self._last_cnn_alpha_summary: Optional[torch.Tensor] = None
            self._last_cnn_beta_summary_per_block: List[torch.Tensor] = []
            # V7: CNN-only view of seg_B (with scratch injected) per compare block
            self._last_cnn_alpha_readback_per_block: List[torch.Tensor] = []
            # V10 (pfc_see_cnn_seq): full CNN sequences per compare block (B, T, d)
            self._last_cnn_alpha_seq_per_block: List[torch.Tensor] = []
            self._last_cnn_beta_seq_per_block: List[torch.Tensor] = []
            self._last_cnn_alpha_seq: Optional[torch.Tensor] = None  # full α scan
            # Disable legacy front-ends
            self.read_cnn = None
            self.input_proj = None
        elif agent_cfg.read_cnn_kernel > 0:
            self.read_cnn = nn.Conv1d(
                agent_cfg.input_dim, d,
                kernel_size=agent_cfg.read_cnn_kernel, bias=True,
            )
            self.input_proj = None
        else:
            self.read_cnn = None
            self.input_proj = nn.Linear(agent_cfg.input_dim, d)

        # Mamba stack with residual connections.
        # V18: optionally replace each Mamba block with an LRU block
        # (complex-eigenvalue, preserves oscillation under task pressure).
        # V33: skip entirely — v33_substrate replaces the recurrence stack.
        if agent_cfg.use_v33:
            self.mamba_blocks = nn.ModuleList([])
        elif agent_cfg.use_lru:
            self.mamba_blocks = nn.ModuleList([
                LRUBlock(
                    d_model=d,
                    d_state=agent_cfg.lru_d_state,
                    r_min=agent_cfg.lru_init_r_min,
                    r_max=agent_cfg.lru_init_r_max,
                    phase_max=agent_cfg.lru_init_phase_max,
                )
                for _ in range(agent_cfg.mamba_layers)
            ])
        else:
            self.mamba_blocks = nn.ModuleList([
                Mamba(
                    d_model=d,
                    d_state=agent_cfg.d_state,
                    d_conv=agent_cfg.d_conv,
                    expand=agent_cfg.expand,
                )
                for _ in range(agent_cfg.mamba_layers)
            ])

        # Optional PFC stack (transformer encoder OR deep MLP for write/compare).
        # 51_M5: pfc_variant="mlp" uses 2 separate DeepMLPPFC instances (write
        # site has 1+W tokens, compare site has 2 tokens with dropRaw). For
        # transformer (default M4), single shared instance handles both.
        if agent_cfg.pfc_layers > 0:
            if agent_cfg.pfc_variant == "transformer":
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d, nhead=agent_cfg.pfc_heads,
                    dim_feedforward=d, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.pfc = nn.TransformerEncoder(enc_layer, num_layers=agent_cfg.pfc_layers)
                self.pfc_compare = self.pfc  # shared
            elif agent_cfg.pfc_variant == "mlp":
                # V6: +1 token for cnn subitizing summary at WRITE only (not compare).
                # V7 (2026-05-04): compare also gets +2 CNN tokens
                # (cnn_α_readback + cnn_β_summary) when dual_pathway and Mamba is
                # used at readback (i.e., not cnn_pfc_only). PFC sees both
                # Mamba-derived h_α/h_β and CNN-derived cnn tokens, can attention-weight.
                write_n = 1 + scene_cfg.W + (1 if agent_cfg.dual_pathway else 0)
                v7_compare_extra = (
                    2 if (agent_cfg.dual_pathway and not agent_cfg.cnn_pfc_only) else 0
                )
                cmp_n = 2 + v7_compare_extra
                self.pfc = DeepMLPPFC(
                    d_model=d, num_tokens=write_n,
                    hidden_dim=agent_cfg.pfc_mlp_hidden,
                )
                self.pfc_compare = DeepMLPPFC(
                    d_model=d, num_tokens=cmp_n,
                    hidden_dim=agent_cfg.pfc_mlp_hidden,
                )
            else:
                raise ValueError(f"Unknown pfc_variant: {agent_cfg.pfc_variant}")
            self.scratch_proj = nn.Linear(agent_cfg.signal_output_dim, d)
            self.scratch_pos_emb = nn.Parameter(torch.randn(scene_cfg.W, d) * 0.02)
            # V16: learnable [WRITE_QUERY] token at position 0 of write seq_w.
            # Small init (0.02) so output[0] = query + Attn[0] + FFN[0] is
            # dominated by Attn (not residual carrying h_t). Breaks V10/V15's
            # h_t-residual-dominance pathology mathematically.
            if agent_cfg.write_query_token:
                self.write_query_emb = nn.Parameter(torch.randn(d) * 0.02)
            else:
                self.write_query_emb = None
            # V10 (pfc_see_cnn_seq): pos emb for the T_target CNN tokens fed to PFC,
            # plus α/β side embeddings for the compare seq (so PFC can tell which
            # side a token belongs to). Constant scale 0.02 init.
            if agent_cfg.pfc_see_cnn_seq:
                T_pos = max(1, agent_cfg.dp_t_target)
                self.cnn_pos_emb = nn.Parameter(torch.randn(T_pos, d) * 0.02)
                self.alpha_side_emb = nn.Parameter(torch.randn(d) * 0.02)
                self.beta_side_emb = nn.Parameter(torch.randn(d) * 0.02)
        else:
            self.pfc = None
            self.pfc_compare = None

        # V15: optional recurrent latent PFC over cnn_alpha_seq (Coconut-style).
        # Replaces cnn_alpha_summary (mean) with state-accumulated latent.
        if agent_cfg.recurrent_latent_pfc:
            assert agent_cfg.dual_pathway or agent_cfg.cnn_pfc_only, \
                "recurrent_latent_pfc needs cnn_alpha_seq (dual_pathway or cnn_pfc_only)"
            self.recurrent_latent_pfc_module = RecurrentLatentPFC(
                d_model=d, n_heads=agent_cfg.pfc_heads,
                n_state_tokens=agent_cfg.recurrent_latent_state_tokens,
                sparse_top_k=agent_cfg.recurrent_latent_sparse_k,
            )
        else:
            self.recurrent_latent_pfc_module = None
        # V13: optional cross-attention write head (parallel W writes via 5
        # learnable queries cross-attending CNN sequence)
        if agent_cfg.cross_attn_write:
            assert agent_cfg.cnn_pfc_only or agent_cfg.dual_pathway, \
                "cross_attn_write needs cnn_alpha_seq stash (cnn_pfc_only or dual_pathway)"
            self.cross_attn_write_module = CrossAttnWriteHead(
                d_model=d, W=scene_cfg.W,
                num_heads=agent_cfg.cross_attn_write_heads,
            )
        else:
            self.cross_attn_write_module = None
        # V12: LN approach abandoned (smoke showed it destroys magnitude axis).
        # Kept module slot for ckpt-load compat but as Identity (no-op).
        if agent_cfg.write_head_ln:
            self.write_head_ln = nn.LayerNorm(d)
        else:
            self.write_head_ln = nn.Identity()
        # Heads
        if agent_cfg.gumbel_write_head:
            assert not agent_cfg.if_write_gate, "gumbel_write_head ✗ if_write_gate"
            assert not agent_cfg.ar_write_head, "gumbel_write_head ✗ ar_write_head"
            assert not agent_cfg.vq_enabled, "gumbel_write_head ✗ vq_enabled"
            assert agent_cfg.signal_output_dim == 1
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1
            assert agent_cfg.quantize_range == "unit"
            self.write_head = GumbelWriteHead(
                d_model=d, Q=agent_cfg.quantize_levels, tau=agent_cfg.gumbel_tau,
            )
        elif agent_cfg.if_write_gate:
            assert not agent_cfg.ar_write_head, "if_write_gate incompatible with ar_write_head"
            assert not agent_cfg.vq_enabled, "if_write_gate incompatible with vq_enabled"
            assert agent_cfg.signal_output_dim == 1, "if_write_gate expects signal_output_dim=1"
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1
            assert agent_cfg.quantize_range == "unit"
            self.write_head = IFWriteHead(
                d_model=d,
                W=scene_cfg.W,
                Q=agent_cfg.quantize_levels,
                init_threshold=agent_cfg.if_init_threshold,
                init_reset=agent_cfg.if_init_reset,
                fire_tau=agent_cfg.if_fire_tau,
            )
        elif agent_cfg.ar_write_head:
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1, \
                "ar_write_head requires quantize_levels > 1"
            assert not agent_cfg.vq_enabled, "ar_write_head incompatible with vq_enabled"
            assert agent_cfg.signal_output_dim == 1, \
                "ar_write_head expects scalar per cell (signal_output_dim=1)"
            assert agent_cfg.quantize_range == "unit", \
                "ar_write_head currently assumes unit range [0,1]"
            self.write_head = AutoRegWriteHead(
                d_model=d,
                Q=agent_cfg.quantize_levels,
                embed_dim=agent_cfg.ar_write_embed_dim,
                prev_dropout=agent_cfg.ar_write_prev_dropout,
                tau_init=agent_cfg.ar_write_tau_init,
                tau_final=agent_cfg.ar_write_tau_final,
                tau_warmup=agent_cfg.ar_write_tau_warmup,
            )
        elif agent_cfg.write_head_type == "sequential":
            assert agent_cfg.signal_output_dim == 1, "sequential write head expects scalar per cell"
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1
            self.write_head = SequentialWriteHead(
                d_pfc=d, W=scene_cfg.W, hidden=agent_cfg.write_head_hidden,
                pending_marker=agent_cfg.write_head_pending_marker,
            )
        elif agent_cfg.write_head_type == "lru_decoder":
            # V28-LRU: state-init free response decoder
            assert agent_cfg.signal_output_dim == 1
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1
            self.write_head = LRUDecoder(
                d_pfc=d, W=scene_cfg.W, d_state=agent_cfg.lru_decoder_d_state,
            )
        elif agent_cfg.write_head_type == "sinusoidal":
            # V28-Sin: explicit-ω sin write head
            assert agent_cfg.signal_output_dim == 1
            assert agent_cfg.quantize_levels is not None and agent_cfg.quantize_levels > 1
            self.write_head = SinusoidalWriteHead(
                d_pfc=d, W=scene_cfg.W,
                omega_init=tuple(agent_cfg.sin_omega_init),
                init_std=agent_cfg.sin_init_std,
            )
        else:
            self.write_head = nn.Linear(d, agent_cfg.signal_output_dim)
        # 51_M5: simple_comparator replaces 2-layer MLP with single Linear(d, C)
        # applied to (h_sp − h_β) instead of cat([h_α, h_β]). Anti-symmetric.
        if agent_cfg.rbf_compare_head:
            # V8: Gaussian RBF over diff (h_α_pfc - h_β_pfc). Implies simple_comparator
            # input geometry but bounded logit output.
            self.compare_head = RBFCompareHead(
                d_model=d,
                num_classes=agent_cfg.compare_classes,
                init_sigma=agent_cfg.rbf_init_sigma,
            )
        elif agent_cfg.simple_comparator:
            self.compare_head = nn.Linear(d, agent_cfg.compare_classes)
        elif agent_cfg.compare_mlp_hidden:
            self.compare_head = nn.Sequential(
                nn.Linear(2 * d, agent_cfg.compare_mlp_hidden),
                nn.GELU(),
                nn.Linear(agent_cfg.compare_mlp_hidden, agent_cfg.compare_classes),
            )
        else:
            self.compare_head = nn.Linear(2 * d, agent_cfg.compare_classes)
        # Optional scalar gap_head (1-D output) used by gap_mse4 and rl_top1.
        # Match compare_in shape: simple_comparator / rbf_compare_head pass
        # diff (B, d); otherwise concat (B, 2d).
        if agent_cfg.use_gap_head:
            gap_in = d if (agent_cfg.simple_comparator or agent_cfg.rbf_compare_head) else 2 * d
            if agent_cfg.compare_mlp_hidden:
                self.gap_head = nn.Sequential(
                    nn.Linear(gap_in, agent_cfg.compare_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(agent_cfg.compare_mlp_hidden, 1),
                )
            else:
                self.gap_head = nn.Linear(gap_in, 1)
        else:
            self.gap_head = None

        # V17: Multi-modular aux head. Same input as compare_head (diff if
        # simple_comparator/RBF, else cat). One binary logit per modulus.
        if agent_cfg.multi_modular_aux_moduli:
            n_mod = len(agent_cfg.multi_modular_aux_moduli)
            in_dim = d if (agent_cfg.simple_comparator or agent_cfg.rbf_compare_head) else 2 * d
            self.multi_modular_aux_head = nn.Linear(in_dim, n_mod)
        else:
            self.multi_modular_aux_head = None

        # V20: scratch-domain modular aux head. Linear(W, n_mod) reads
        # (scratch_alpha - scratch_beta_virtual) over W cells and outputs
        # a binary logit per scratch modulus. Forces write_head + scratch
        # cells to carry modular residue structure.
        if agent_cfg.scratch_mod_aux_moduli:
            n_mod_s = len(agent_cfg.scratch_mod_aux_moduli)
            self.scratch_mod_aux_head = nn.Linear(scene_cfg.W, n_mod_s)
        else:
            self.scratch_mod_aux_head = None

        # V22: SuccessorFunction over raw scratch space. Built when
        # successor_hidden > 0; trainer applies loss on within-episode
        # |δ|=1 G/L pairs.
        if agent_cfg.successor_hidden > 0:
            self.successor = SuccessorFunction(
                W=scene_cfg.W, hidden=agent_cfg.successor_hidden,
            )
        else:
            self.successor = None

        # V24: successor_predict_head — predicts k. V25-V24 supports two input
        # levels: "scratch" reads concat([sc_α, sc_β_virtual]) (Linear(2W, k_max));
        # "pfc_hidden" reads concat([h_α_pfc, h_β_pfc]) (Linear(2*d, k_max)).
        if agent_cfg.successor_predict_k_max > 0:
            if agent_cfg.successor_predict_input_level == "pfc_hidden":
                in_dim = 2 * d
                self.successor_predict_head = nn.Linear(
                    in_dim, agent_cfg.successor_predict_k_max,
                )
            elif agent_cfg.write_head_type == "sinusoidal":
                # V28-Sin: MLP + multi-hot one-hot encoding of categorical levels
                # (sin output quantized levels are categorical, not ordinal magnitude)
                self.successor_predict_head = MultiHotMLPSuccessorHead(
                    W=scene_cfg.W,
                    q_levels=agent_cfg.quantize_levels,
                    k_max=agent_cfg.successor_predict_k_max,
                    hidden=agent_cfg.sin_predict_hidden,
                )
            else:
                in_dim = 2 * scene_cfg.W
                self.successor_predict_head = nn.Linear(
                    in_dim, agent_cfg.successor_predict_k_max,
                )
        else:
            self.successor_predict_head = None

        # V24R: recurrent scratch refiner. Takes sc_q + pfc_h + lru_h, runs
        # n_iter iterations of (cnn_scratch + combine + write_head). Reuses
        # outer write_head + scratch_pos_emb (parameter sharing).
        if agent_cfg.pfc_recurrent_n_iter > 0:
            d_lru = agent_cfg.lru_d_state if agent_cfg.use_lru else d
            self.recurrent_refiner = RecurrentScratchRefiner(
                W=scene_cfg.W, d_pfc=d, d_lru=d_lru,
                scratch_cnn_kernel=agent_cfg.pfc_recurrent_cnn_scratch_kernel,
                scratch_cnn_channels=agent_cfg.pfc_recurrent_cnn_scratch_channels,
                n_iter=agent_cfg.pfc_recurrent_n_iter,
            )
        else:
            self.recurrent_refiner = None

        # V27 (5/10): Chain-of-thought PFC. Requires LRU + dual_pathway.
        if agent_cfg.cot_pfc_n_chunks > 0:
            assert agent_cfg.use_lru, "V27 requires --use-lru"
            assert len(self.mamba_blocks) >= 1, "V27 needs at least 1 LRU block"
            self.cot_pfc = CoTPFC(
                d_model=d,
                n_chunks=agent_cfg.cot_pfc_n_chunks,
                kernels=agent_cfg.cot_pfc_cnn_kernels,
                cnn_channels_per_scale=agent_cfg.dp_channels_per_scale,
                lru_block=self.mamba_blocks[0],   # share first LRU layer
                input_dim=agent_cfg.input_dim,
            )
        else:
            self.cot_pfc = None

        # V23b: NClassHead over post-quantize scratch_α. Built when
        # n_class_max_n > 0; trainer applies CE loss against N_α-1 target.
        if agent_cfg.n_class_max_n > 0:
            self.n_class_head = NClassHead(
                W=scene_cfg.W,
                max_n=agent_cfg.n_class_max_n,
                hidden=agent_cfg.n_class_head_hidden,
            )
        else:
            self.n_class_head = None

        # Optional ForwardPredictor for E0-prime / world-prediction auxiliary loss.
        if agent_cfg.world_pred_t_pred > 0:
            self.forward_predictor = ForwardPredictor(
                d_model=d,
                t_pred=agent_cfg.world_pred_t_pred,
                scratch_dim=scene_cfg.W * agent_cfg.signal_output_dim,
            )
        else:
            self.forward_predictor = None
        # A2: BetaPredictor — predict T_β steps of each compare-block β scan.
        if agent_cfg.beta_pred_t_pred > 0:
            self.beta_predictor = ForwardPredictor(
                d_model=d,
                t_pred=agent_cfg.beta_pred_t_pred,
                scratch_dim=scene_cfg.W * agent_cfg.signal_output_dim,
            )
        else:
            self.beta_predictor = None
        # A5: ComparePredictor — predict compare_logits from (h_α, h_β, scratch).
        compare_out_dim = (
            1 if agent_cfg.use_gap_head else agent_cfg.compare_classes
        )
        if agent_cfg.compare_pred_enabled:
            self.compare_predictor = ComparePredictor(
                d_model=d,
                scratch_dim=scene_cfg.W * agent_cfg.signal_output_dim,
                output_dim=compare_out_dim,
            )
        else:
            self.compare_predictor = None
        # PC-lite: TopDownPredictor for u[t-1] -> predicted u[t].
        if agent_cfg.predictive_coding:
            self.top_down_predictor = TopDownPredictor(d_model=d)
        else:
            self.top_down_predictor = None
        # RS-1: RawSignalPredictor — predict raw input signal stream.
        if agent_cfg.rs_pred_t_pred > 0:
            self.rs_predictor = RawSignalPredictor(
                d_model=d, t_pred=agent_cfg.rs_pred_t_pred,
            )
        else:
            self.rs_predictor = None
        # A3: ScratchSelfPredictor — predict scratch_t from prior slots + h_t.
        if agent_cfg.scratch_self_pred_enabled:
            self.scratch_self_predictor = ScratchSelfPredictor(
                d_model=d, signal_dim=agent_cfg.signal_output_dim,
                max_W=scene_cfg.W,
            )
        else:
            self.scratch_self_predictor = None
        # A4: CounterfactualBetaPredictor — predict β signal from (scratch, count).
        if agent_cfg.cf_beta_pred_t_pred > 0:
            self.cf_beta_predictor = CounterfactualBetaPredictor(
                scratch_dim=scene_cfg.W * agent_cfg.signal_output_dim,
                t_pred=agent_cfg.cf_beta_pred_t_pred,
            )
        else:
            self.cf_beta_predictor = None
        # idea2: ReviewModule — iterative refinement of complete scratch.
        if agent_cfg.review_iter > 0:
            self.review_module = ReviewModule(
                d_model=d, num_layers=agent_cfg.review_layers,
                num_heads=agent_cfg.review_heads,
            )
            # Project signal_dim scratch back to d for transformer input.
            self.review_scratch_proj = nn.Linear(
                agent_cfg.signal_output_dim, d,
            )
            self.review_pos_emb = nn.Parameter(
                torch.zeros(scene_cfg.W, d)
            )
            # Project transformer output back to signal_dim for write.
            self.review_write_head = nn.Linear(d, agent_cfg.signal_output_dim)
        else:
            self.review_module = None
        # β-1: VectorQuantizerEMA shared codebook.
        if agent_cfg.vq_enabled:
            self.vq = VectorQuantizerEMA(
                K=agent_cfg.vq_codebook_size, d_z=agent_cfg.vq_d_z,
                commitment=agent_cfg.vq_commitment, decay=agent_cfg.vq_decay,
            )
        else:
            self.vq = None
        # P1 (L1579): WorldSignalPredictor for imagined future raw signal.
        if agent_cfg.dual_loss_enabled:
            self.world_signal_predictor = WorldSignalPredictor(
                d_model=d, t_p=agent_cfg.dual_pred_t_p,
            )
        else:
            self.world_signal_predictor = None
        # Stashes for run_batch to read.
        self._last_world_loss: Optional[torch.Tensor] = None
        self._last_beta_pred_loss: Optional[torch.Tensor] = None
        self._last_compare_pred_loss: Optional[torch.Tensor] = None
        self._last_rs_pred_loss: Optional[torch.Tensor] = None
        self._last_scratch_self_pred_loss: Optional[torch.Tensor] = None
        self._last_cf_beta_pred_loss: Optional[torch.Tensor] = None
        self._last_vq_commit_loss: Optional[torch.Tensor] = None
        self._last_vq_indices: Optional[torch.Tensor] = None
        # P1 (L1579) dual-loss stashes
        self._last_dual_world_loss: Optional[torch.Tensor] = None
        self._last_dual_sp_loss: Optional[torch.Tensor] = None
        # 51_M5: cycle loss + h_alpha stash
        self._last_h_alpha_for_cycle: Optional[torch.Tensor] = None
        self._last_cycle_loss: Optional[torch.Tensor] = None
        # COLLAPSE FIX: VICReg + PC explicit loss stashes
        self._last_world_vicreg: Optional[torch.Tensor] = None
        self._last_beta_vicreg: Optional[torch.Tensor] = None
        self._last_compare_vicreg: Optional[torch.Tensor] = None
        self._last_pc_explicit_loss: Optional[torch.Tensor] = None
        self._last_pc_vicreg: Optional[torch.Tensor] = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _run_write_loop(
        self, h_at_writes: torch.Tensor, W: int,
    ) -> torch.Tensor:
        """Re-runnable write loop for dual-loss (imagined / β paths).

        h_at_writes: (B, W, d) — Mamba latents at the write timesteps.
        Returns scratch (B, W, signal_output_dim) after PFC + write_head + quantize.

        NOTE: this is a SIMPLIFIED version of the real-path write loop:
        - Reuses self.pfc, self.scratch_proj, self.scratch_pos_emb, self.write_head
        - No write_noise (we want deterministic prediction reproduction)
        - No VQ (only used by real path for now; dual-loss assumes scalar scratch)
        - No A3 self-prediction
        """
        B = h_at_writes.shape[0]
        sig_dim = self.agent_cfg.signal_output_dim
        d = self.agent_cfg.d_model
        scratch_list: List[torch.Tensor] = []
        if self.agent_cfg.ar_write_head:
            ar_h_rnn, ar_prev_cell = self.write_head.init_state(B, h_at_writes.device)
        for j in range(W):
            h_t = h_at_writes[:, j, :]
            if self.pfc is not None and self.agent_cfg.pfc_at_write:
                sp_slots = []
                for k in range(W):
                    if k < len(scratch_list):
                        sp_slots.append(scratch_list[k])
                    else:
                        sp_slots.append(h_t.new_zeros(B, sig_dim))
                sp_w = torch.stack(sp_slots, dim=1)
                kv_w = self.scratch_proj(sp_w) + self.scratch_pos_emb.unsqueeze(0)
                seq_w = torch.cat([h_t.unsqueeze(1), kv_w], dim=1)
                seq_w_refined = seq_w
                for _ in range(self.agent_cfg.pfc_iter):
                    seq_w_refined = self.pfc(seq_w_refined)
                h_for_write = seq_w_refined[:, 0, :]
            else:
                h_for_write = h_t
            # V12: same LN as primary loop for consistency
            h_for_write = self.write_head_ln(h_for_write)
            if self.agent_cfg.ar_write_head:
                raw, ar_h_rnn, _ = self.write_head.step(h_for_write, ar_prev_cell, ar_h_rnn)
                ar_prev_cell = raw
            elif self.agent_cfg.gumbel_write_head:
                raw = self.write_head(h_for_write)  # already quantized
            else:
                raw = self.write_head(h_for_write)
                if self.agent_cfg.write_head_clip > 0:
                    c = self.agent_cfg.write_head_clip
                    raw = c * torch.tanh(raw / c)
                if self.agent_cfg.quantize_levels is not None:
                    raw = _quantize_ste(
                        raw,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
            scratch_list.append(raw)
        return torch.stack(scratch_list, dim=1)  # (B, W, sig_dim)

    def _encode_segment_pc(self, x_seg: torch.Tensor) -> torch.Tensor:
        """PC-lite encoder: mamba processes (CNN(x) − top_down_predicted_CNN).

        Top-down predictor predicts CNN latent u[t] from previous u[t-1]
        (autoregressive 1-step). Mamba then sees ERROR pathway (residual
        after the prior). End-to-end gradient lets predictor learn to
        minimize predictable parts so mamba can focus on surprises.
        """
        B, T_seg, _ = x_seg.shape
        if self.read_cnn is not None:
            K_cnn = self.agent_cfg.read_cnn_kernel
            xt = x_seg.transpose(1, 2)
            pad = xt.new_zeros(B, self.agent_cfg.input_dim, K_cnn - 1)
            xt = torch.cat([pad, xt], dim=-1)
            u = self.read_cnn(xt).transpose(1, 2)  # (B, T_seg, d)
        else:
            u = self.input_proj(x_seg)
        # u_prev[t] = u[t-1], u_prev[0] = zeros (no prior at start)
        u_prev = torch.cat([u.new_zeros(B, 1, u.shape[-1]), u[:, :-1, :]], dim=1)
        d = u.shape[-1]
        predicted = self.top_down_predictor(
            u_prev.reshape(B * T_seg, d)
        ).reshape(B, T_seg, d)
        error = u - predicted  # error pathway
        # COLLAPSE FIX: optional explicit BYOL-style prediction loss with sg.
        # Without this, PC has NO predictor training signal (only main task
        # loss flows back through residual). With sg + VICReg, this becomes
        # closer to true predictive coding (Rao-Ballard style).
        if (self.agent_cfg.pc_explicit_lambda > 0.0
                and self.agent_cfg.aux_detach_targets):
            target = u.detach()
            self._last_pc_explicit_loss = F.mse_loss(predicted, target)
            if self.agent_cfg.vicreg_lambda > 0.0:
                self._last_pc_vicreg = vicreg_variance_loss(
                    u.reshape(-1, d),
                    gamma=self.agent_cfg.vicreg_gamma,
                )
            else:
                self._last_pc_vicreg = None
        else:
            self._last_pc_explicit_loss = None
            self._last_pc_vicreg = None
        h = error
        for block in self.mamba_blocks:
            h = block(h) + h
        return h

    def _encode_v6_cnn_only(self, x_seg: torch.Tensor, apply_pool: bool = True) -> torch.Tensor:
        """V6 CNN-only encoder (no Mamba). Returns (B, T_eff, d_model) — gradient-active.

        V7+stride: when dp_t_target > 0 and apply_pool=True, each scale's conv
        output is adaptive_avg_pool1d'd to T_target. Different scales naturally
        align (same T_target). For short segments (e.g. seg_B with W=5), pass
        apply_pool=False to skip pooling.
        """
        xt = x_seg.transpose(1, 2)  # (B, input_dim, T_seg)
        T_target = self.agent_cfg.dp_t_target if apply_pool else 0
        T_seg = xt.shape[-1]
        cnn_outs = []
        for conv in self.dp_convs:
            if T_target > 0 and self.agent_cfg.cnn_stride_no_pool and T_seg >= T_target:
                # V10 (efficient): native strided conv with kernel-aware "same"
                # padding so all multi-scale kernels produce same output length,
                # then trim to exactly T_target. Functionally ≈ old stride=1 +
                # sub-sample, but ~7x less compute (especially for K=51).
                K = conv.weight.shape[-1]
                stride = max(1, T_seg // T_target)
                pad = (K - 1) // 2  # "same"-ish padding (centered kernel)
                o = F.relu(F.conv1d(
                    xt, conv.weight, conv.bias, stride=stride, padding=pad,
                ))
                # All scales now align (same output length); trim to T_target
                if o.shape[-1] > T_target:
                    o = o[:, :, :T_target]
                elif o.shape[-1] < T_target:
                    # Edge case: pad with last value
                    pad_n = T_target - o.shape[-1]
                    o = torch.cat(
                        [o, o[:, :, -1:].expand(-1, -1, pad_n)], dim=-1,
                    )
            else:
                o = F.relu(conv(xt))  # (B, cps, T_seg) full stride=1
                if T_target > 0 and o.shape[-1] >= T_target:
                    o = F.adaptive_avg_pool1d(o, T_target)
            cnn_outs.append(o)
        feats = torch.cat(cnn_outs, dim=1)  # (B, total_ch, T_eff)
        feats = feats.transpose(1, 2)  # (B, T_eff, total_ch)
        return self.dp_proj(feats)  # (B, T_eff, d)

    def _encode_segment_v6(self, x_seg: torch.Tensor, apply_pool: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """V6 dual-pathway encoder. Returns (h_mamba, cnn_features).

        cnn_features keeps gradient (for PFC bypass). h_mamba is computed from
        cnn_features.detach() (cuts Mamba→CNN backward), then mamba blocks.

        V7+stride (dp_t_target > 0): cnn_features pre-pooled to T_target,
        Mamba sees fixed T_target tokens.

        V19 (lru_event_gated=True): LRU pathway sees UNPOOLED (B, L, d). CNN
        bypass pathway still pools to dp_t_target. spike_detector → gate
        modulates LRU recurrence: gate≈1 events trigger phase rotation,
        gate≈0 background freezes h. Phase axis becomes spike-count, not
        spatial step.
        """
        if self.agent_cfg.lru_event_gated and self.agent_cfg.use_lru:
            # V19: LRU sees unpooled, PFC bypass sees pooled
            cnn_unpooled = self._encode_v6_cnn_only(x_seg, apply_pool=False)  # (B, L, d)
            # PFC bypass: pool to dp_t_target (or keep if no pool needed)
            if apply_pool and self.agent_cfg.dp_t_target > 0 and cnn_unpooled.shape[1] > self.agent_cfg.dp_t_target:
                cnn_pooled = F.adaptive_avg_pool1d(
                    cnn_unpooled.transpose(1, 2),
                    output_size=self.agent_cfg.dp_t_target,
                ).transpose(1, 2)
            else:
                cnn_pooled = cnn_unpooled
            # Spike detection on raw signal (channel 0 of x_seg)
            sig = x_seg[:, :, 0:1].transpose(1, 2)  # (B, 1, L)
            spike_score = self.spike_detector(sig).squeeze(1)  # (B, T_seg)
            self._last_spike_score = spike_score                # stash for aux loss
            # V19 detection aux: stash (score, signal) for BCE supervision in trainer
            if not hasattr(self, "_spike_score_history") or self._spike_score_history is None:
                self._spike_score_history = []
            self._spike_score_history.append((spike_score, x_seg[:, :, 0]))
            if self.agent_cfg.use_oracle_spike_gate:
                # Oracle gate: hard binary, 1.0 only at scene-meta spike centers.
                # detector still ran (gradients via BCE detection loss), but
                # the LRU gate signal here ignores its output entirely.
                B_seg, T_seg = spike_score.shape
                gate = torch.zeros(B_seg, T_seg, device=spike_score.device, dtype=spike_score.dtype)
                centers = getattr(self, "_current_oracle_centers", None)
                if centers is not None:
                    for b, ctrs in enumerate(centers):
                        for c in ctrs:
                            if 0 <= c < T_seg:
                                gate[b, c] = 1.0
            else:
                gate = torch.sigmoid(spike_score * self.gate_sharpness)  # (B, T_seg)
            # LRU on unpooled with event gate
            mamba_in = cnn_unpooled.detach() if self.agent_cfg.dp_detach_to_mamba else cnn_unpooled
            h = mamba_in
            for block in self.mamba_blocks:
                h = block(h, gate=gate) + h
            return h, cnn_pooled  # h: (B, L, d) for write-phase h_t indexing; cnn_pooled: (B, T, d) for PFC bypass
        # V18c (legacy) behavior
        cnn_out = self._encode_v6_cnn_only(x_seg, apply_pool=apply_pool)
        if self.agent_cfg.dp_detach_to_mamba:
            mamba_in = cnn_out.detach()
        else:
            mamba_in = cnn_out
        h = mamba_in
        for block in self.mamba_blocks:
            h = block(h) + h
        return h, cnn_out

    def _encode_segment(self, x_seg: torch.Tensor, apply_pool: bool = True) -> torch.Tensor:
        """Encode a contiguous (B, T_seg, input_dim) segment.

        Mamba state starts fresh at the start of each segment (the reset
        semantics). CNN buffer is also fresh — left-padded with zeros.

        apply_pool: V7+stride pool flag. False for short segments (e.g. seg_B,
        scratch readback) to skip downsampling.
        """
        B, T_seg, _ = x_seg.shape
        if self.agent_cfg.cnn_pfc_only:
            # V6 BRANCH: pure CNN, no Mamba. h ≡ cnn_features.
            cnn_out = self._encode_v6_cnn_only(x_seg, apply_pool=apply_pool)
            self._last_cnn_features = cnn_out
            return cnn_out
        if self.agent_cfg.dual_pathway:
            h, cnn_out = self._encode_segment_v6(x_seg, apply_pool=apply_pool)
            # Stash CNN features for PFC bypass (overwrites per call; caller
            # must capture summary into _last_cnn_alpha/beta_summary)
            self._last_cnn_features = cnn_out
            return h
        if self.read_cnn is not None:
            K_cnn = self.agent_cfg.read_cnn_kernel
            xt = x_seg.transpose(1, 2)  # (B, input_dim, T_seg)
            pad = xt.new_zeros(B, self.agent_cfg.input_dim, K_cnn - 1)
            xt = torch.cat([pad, xt], dim=-1)  # (B, input_dim, T_seg+K-1)
            u = self.read_cnn(xt).transpose(1, 2)  # (B, T_seg, d)
        else:
            u = self.input_proj(x_seg)  # (B, T_seg, d)
        h = u
        for block in self.mamba_blocks:
            h = block(h) + h  # residual
        return h  # (B, T_seg, d)

    def _v29_forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V29 R&F SNN forward.

        Splits inputs into alpha phase [0, L) and beta phase [beta_start, compare_step + 1).
        Calls V29Pipeline end-to-end. Stashes scratch + aux losses for compute_step.
        """
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K

        # Clear stashes
        self._scratch_alpha_for_aux = None
        self._scratch_alpha_raw = None
        self._last_scratch_beta_K_q = None
        self._last_scratch_beta_K_raw = None
        self._h_alpha_pfc_blocks = []
        self._h_beta_pfc_blocks = []
        self._v29_alpha_stash = {}
        self._v29_beta_stash = {}
        self._v29_orth_loss = None
        self._v29_reg_loss = None
        self._v29_firing_reg = None
        self._v29_logits = None
        self._v29_scratch_int = None

        # Extract alpha + beta signals (only signal channel — V29 ignores the
        # is_input_valid + post_gap_flag channels; phase is implicit via
        # explicit alpha/beta segmentation).
        alpha_signal = inputs[:, :L, 0:1]  # (B, L, 1)
        rstart, beta_start, compare_step = cfg.compare_block_phases(0)
        beta_signal = inputs[:, beta_start : compare_step, 0:1]  # (B, L_b, 1)

        out = self.v29(alpha_signal, beta_signal)
        # Stash everything for compute_step
        self._scratch_alpha_for_aux = out["scratch_q"]
        self._last_scratch_beta_K_q = out["scratch_q"].unsqueeze(1)  # (B, K=1, W) — placeholder
        self._v29_alpha_stash = {
            "spikes": out["alpha_spikes"], "tokens": out["alpha_tokens"], "mask": out["alpha_mask"],
        }
        self._v29_beta_stash = {
            "spikes": out["beta_spikes"], "tokens": out["beta_tokens"], "mask": out["beta_mask"],
        }
        self._v29_orth_loss = out["orth_loss"]
        self._v29_reg_loss = out["reg_loss"]
        self._v29_firing_reg = out.get("firing_reg", None)
        self._v29_logits = out["logits"]
        self._v29_scratch_int = out["scratch_int"]

        # For interface compat: returns (compare_logits, scratch_pad, hidden_at_compares)
        fake_compare_logits = torch.zeros(B, K, 3, device=inputs.device, dtype=inputs.dtype)
        fake_hidden = None
        return fake_compare_logits, out["scratch_q"], fake_hidden

    @staticmethod
    def _compute_h_seq_vicreg(
        h_seq: torch.Tensor, gamma: float, eps: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VICReg variance + covariance penalties on a (B, L, d) sequence.

        Applied per-timestep (treating B samples at fixed t as the "batch"),
        then averaged over L. Returns (L_var, L_cov) scalars.

        L_var = mean_d hinge(γ - σ_d)   pushes each dim's std ≥ γ (anti-collapse)
        L_cov = mean_t Σ_{i≠j} C_ij² / d  pushes off-diagonal cov → 0 (decorrelate dims)

        Designed for V33 substrate h_seq where channel cos overlap ~0.9 collapses
        rank to 1 (PC1 = 95%+ variance). Pulling off-diagonal cov down forces v_c
        directions apart, leaving residual dims free to carry orthogonal info.
        """
        B, L, d = h_seq.shape
        if B < 2:
            zero = h_seq.new_zeros(())
            return zero, zero
        z = h_seq.permute(1, 0, 2).contiguous()             # (L, B, d)
        z_c = z - z.mean(dim=1, keepdim=True)
        var_z = z_c.pow(2).sum(dim=1) / (B - 1)              # (L, d)
        std_z = torch.sqrt(var_z + eps)
        L_var = F.relu(gamma - std_z).mean()
        cov_z = torch.einsum('lbi,lbj->lij', z_c, z_c) / (B - 1)  # (L, d, d)
        eye = torch.eye(d, device=h_seq.device, dtype=torch.bool)
        off_diag = cov_z.masked_fill(eye, 0.0)
        L_cov = (off_diag ** 2).sum(dim=(-2, -1)).mean() / d
        return L_var, L_cov

    def _v35_forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        alpha_centers=None,
        beta_centers_list=None,
        metas=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V35 fire-and-reset cascade forward.

        Three cascade passes (all with shared CNN + cascade weights, fresh
        substrate state each pass):
          1. alpha pass:    CNN(alpha_signal) -> cascade -> 5 residuals -> write_attn -> scratch_alpha
          2. readback pass: CNN(scratch_alpha as 5-step signal, ch1=ch2=0) -> cascade -> 5 readback_residuals
          3. beta pass:     CNN(beta_signal) -> cascade -> 5 beta_residuals
        Compare attention: (readback_residuals, beta_residuals) -> k_offset logits.

        Strict SM: dp_convs/dp_proj + v35_substrate weights are shared across all 3 passes.
        """
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K
        assert K == 1, (
            f"V35 forward only supports K=1 (got K={K}). "
            "Use SceneConfig.successor_prediction_preset which enforces K=1."
        )

        # Clear V25/V27/V33-style aux stashes that downstream code may read.
        self._scratch_alpha_for_aux = None
        self._scratch_alpha_raw = None
        self._last_scratch_beta_K_q = None
        self._last_scratch_beta_K_raw = None
        self._h_alpha_pfc_blocks = []
        self._h_beta_pfc_blocks = []
        self._h_readback_alpha_end = None
        self._h_predicted_alpha = None
        self._v35_successor_logits = None
        self._v35_residuals_alpha = None
        self._scratch_beta_for_aux = None
        self._scratch_beta_raw = None
        self._v35_residuals_beta = None
        self._v35_spike_train_A = None

        # Same-medium: clear ch1/ch2 phase-marker channels (V33-SM convention).
        inputs = inputs.clone()
        inputs[:, :, 1:3] = 0.0

        alpha_in = inputs[:, :L, :]
        rstart, beta_start, compare_step = cfg.compare_block_phases(0)
        beta_in = inputs[:, beta_start:compare_step, :]

        def encode(seg: torch.Tensor) -> torch.Tensor:
            x = seg.transpose(1, 2)
            conv_outs = [conv(x) for conv in self.dp_convs]
            feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
            return self.dp_proj(feat)

        # PC mode: capture gate trajectories on every substrate forward call.
        # Disabled (False) in eval/validation forwards or when v35_local_pc=False.
        capture_pc = getattr(self.agent_cfg, "v35_local_pc", False) and self.training

        # Optional ablation: replace HH-based layer A with raw-signal threshold.
        # Bypasses CNN + HHSSMLayer1D(A); diagnostic for cascade carry math.
        threshold_la = getattr(self.agent_cfg, "v35_threshold_layer_a", False)
        la_thresh = getattr(self.agent_cfg, "v35_layer_a_threshold", 0.5)
        la_detect = getattr(self.agent_cfg, "v35_layer_a_detect", "level")
        la_center = getattr(self.agent_cfg, "v35_layer_a_center_thresh", 0.9)
        la_neighbor = getattr(self.agent_cfg, "v35_layer_a_neighbor_thresh", 0.8)
        readback_direct = getattr(self.agent_cfg, "v35_readback_direct", False)
        pss = getattr(self.agent_cfg, "v35_pss_readback_beta", False)

        def _detect_layer_a(raw: torch.Tensor) -> torch.Tensor:
            # raw: (B, T') raw signal channel -> (B, T') binary spike train.
            if la_detect == "learned":
                # LEARNED counter (direction-1): SGD-trained detector + local-max, STE spike.
                return self.v35_learned_la(raw)
            if la_detect == "food_pattern":
                from backend.core.v35 import food_pattern_spike_gate
                return food_pattern_spike_gate(raw, la_center, la_neighbor)
            return (raw > la_thresh).float()

        if threshold_la:
            # --- Pass 1: alpha (raw signal channel 0 -> spike train) ---
            spike_alpha = _detect_layer_a(inputs[:, :L, 0])  # (B, L)
            self._v35_spike_train_A = spike_alpha
            residuals_alpha, _aux_alpha = self.v35_substrate.forward_with_external_layer_a_spike(spike_alpha, capture_for_pc=capture_pc)
            self._v35_residuals_alpha = residuals_alpha  # for recon aux (codebook fidelity)
            scratch_raw, scratch_alpha_q = self.v35_write_attn(residuals_alpha)
            self._scratch_alpha_for_aux = scratch_alpha_q
            self._scratch_alpha_raw = scratch_raw

            # --- Pass 2: scratch readback ---
            if readback_direct:
                # Differentiable readback (TEMPORARY scaffolding): compare reads the scratch
                # code directly; STE soft path carries the task gradient to the write head.
                # Skips the (hard-thresholded, gradient-blocking) cascade readback. NOT same-medium.
                residuals_readback = scratch_alpha_q
            else:
                # Legacy hard-threshold readback (smoke-only; blocks write-head gradient).
                # scratch values are in {0, 0.5, 1.0}; threshold > 0.5 keeps only 1.0.
                spike_readback = (scratch_alpha_q > la_thresh).float()  # (B, W)
                residuals_readback, _aux_rb = self.v35_substrate.forward_with_external_layer_a_spike(spike_readback, capture_for_pc=capture_pc)

            # --- Pass 3: beta ---
            if pss:
                # PSS (spec 2026-06-05 §5.3 Edit A): beta written to its OWN scratch then
                # read back, so NO fresh-magnitude beta reaches compare (severs the leak).
                spike_beta_obs = _detect_layer_a(inputs[:, beta_start:compare_step, 0])  # (B, Lb)
                residuals_beta_obs, _aux_bobs = self.v35_substrate.forward_with_external_layer_a_spike(spike_beta_obs, capture_for_pc=capture_pc)
                scratch_beta_raw, scratch_beta_q = self.v35_write_attn(residuals_beta_obs)
                self._scratch_beta_for_aux = scratch_beta_q
                self._scratch_beta_raw = scratch_beta_raw
                self._v35_residuals_beta = residuals_beta_obs
                if readback_direct:
                    residuals_beta = scratch_beta_q  # STE soft path -> grad to write head (TEMP scaffold)
                else:
                    spike_beta_rb = (scratch_beta_q > la_thresh).float()  # (B, W) hard, grad-blocked
                    residuals_beta, _aux_brb = self.v35_substrate.forward_with_external_layer_a_spike(spike_beta_rb, capture_for_pc=capture_pc)
            else:
                spike_beta = _detect_layer_a(inputs[:, beta_start:compare_step, 0])
                residuals_beta, _aux_beta = self.v35_substrate.forward_with_external_layer_a_spike(spike_beta, capture_for_pc=capture_pc)
        else:
            # --- Pass 1: alpha ---
            feat_alpha = encode(alpha_in)
            residuals_alpha, _aux_alpha = self.v35_substrate(feat_alpha, capture_for_pc=capture_pc)
            self._v35_residuals_alpha = residuals_alpha  # for recon aux (codebook fidelity)
            scratch_raw, scratch_alpha_q = self.v35_write_attn(residuals_alpha)
            self._scratch_alpha_for_aux = scratch_alpha_q
            self._scratch_alpha_raw = scratch_raw

            # --- Pass 2: scratch readback (same-medium injection) ---
            zeros_ch = torch.zeros_like(scratch_alpha_q)
            seg_readback = torch.stack(
                [scratch_alpha_q, zeros_ch, zeros_ch], dim=-1
            )  # (B, W, 3)
            feat_readback = encode(seg_readback)
            residuals_readback, _aux_rb = self.v35_substrate(feat_readback, capture_for_pc=capture_pc)

            # --- Pass 3: beta ---
            feat_beta = encode(beta_in)
            residuals_beta, _aux_beta = self.v35_substrate(feat_beta, capture_for_pc=capture_pc)

        # V35 PSS leak-kill (spec 2026-06-05 §6 D7): in-forward scratch-lesion hook.
        # Bypass-free under PSS (no fresh beta) -> a single-side zero must crater the decision.
        lesion = getattr(self, "_v35_lesion", None)
        if lesion == "zero_alpha":
            residuals_readback = torch.zeros_like(residuals_readback)
        elif lesion == "shuffle_alpha":
            residuals_readback = residuals_readback[torch.randperm(residuals_readback.shape[0])]
        elif lesion == "zero_beta":
            residuals_beta = torch.zeros_like(residuals_beta)

        # Compare attention -> k_offset logits
        successor_logits = self.v35_compare_attn(residuals_readback, residuals_beta)
        self._v35_successor_logits = successor_logits

        # Return shape contract: (compare_logits, scratch_pad, hidden_at_compares).
        # V35 path doesn't use the 3-class compare task — supply placeholders so
        # downstream collect code doesn't crash.
        compare_logits = torch.zeros(B, K, 3, device=inputs.device)
        scratch_pad = scratch_alpha_q  # (B, W)
        hidden_at_compares = torch.zeros(B, K, self.agent_cfg.d_model, device=inputs.device)
        return compare_logits, scratch_pad, hidden_at_compares

    def _v33_forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        alpha_centers=None,
        beta_centers_list=None,
        metas=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V33 HH-SSM forward.

        Same overall flow as _v27_forward but replaces (LRU + CoT-PFC) recurrence
        with V33Substrate. Uses existing dual_pathway CNN encoder + V25
        sequential write_head + successor predict head.
        Sequential O(L) recurrence - no chunking, no CoT.
        """
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K
        d = self.agent_cfg.v33_d_model

        self._scratch_alpha_for_aux = None
        self._scratch_alpha_raw = None
        self._last_scratch_beta_K_q = None
        self._last_scratch_beta_K_raw = None
        self._h_alpha_pfc_blocks = []
        self._h_beta_pfc_blocks = []
        self._h_readback_alpha_end = None
        self._h_predicted_alpha = None

        same_medium = getattr(self.agent_cfg, "v33_same_medium", False)

        # V33-SM (5/20): clear ch1/ch2 phase flags so CNN treats α / readback / β
        # identically (no phase oracle). Channels 1-2 are V8-V19 legacy markers
        # that within a segment are constant — they only differentiate α (1,0)
        # from β (1,1), giving CNN a "you're in β now" bit that violates
        # same-medium. Substrate state continuity already supplies phase identity
        # to the recurrence, so dropping the CNN-side bit is information-neutral
        # for the model but enforces medium-agnostic CNN processing.
        if same_medium:
            inputs = inputs.clone()
            inputs[:, :, 1:3] = 0.0

        alpha_in = inputs[:, :L, :]
        rstart, beta_start, compare_step = cfg.compare_block_phases(0)
        beta_in = inputs[:, beta_start:compare_step, :]

        def encode(seg: torch.Tensor) -> torch.Tensor:
            """seg: (B, T, input_dim) -> features (B, T, d) via dual_pathway CNN."""
            x = seg.transpose(1, 2)
            conv_outs = [conv(x) for conv in self.dp_convs]
            feat = torch.cat(conv_outs, dim=1).transpose(1, 2)
            return self.dp_proj(feat)

        feat_alpha = encode(alpha_in)

        h_alpha_seq, alpha_state = self.v33_substrate(feat_alpha)
        h_alpha_for_write = h_alpha_seq[:, -1, :]

        scratch_alpha_raw, scratch_alpha_q = self.write_head(
            h_alpha_for_write,
            self.agent_cfg.quantize_levels,
            self.agent_cfg.quantize_range,
        )
        self._scratch_alpha_for_aux = scratch_alpha_q
        self._scratch_alpha_raw = scratch_alpha_raw

        if same_medium:
            # V33-SM (5/20): same-medium closed loop. Build readback segment of W
            # timesteps where ch0 = scratch_alpha_q (re-injection). CNN re-encodes
            # this with the SAME dp_convs/dp_proj that processed α. Substrate
            # carries state α→readback→β so h_β depends on scratch_α via the
            # readback recurrence. This creates the closed gradient loop:
            #   loss ← sc_β ← write_head(h_β) ← substrate(β | readback_state)
            #               ← substrate(readback | α_state) ← encode(sc_α-injected segment)
            # Forces sc_α to encode N because if it doesn't, sc_β reads garbage,
            # prediction breaks, gradient pushes back through readback.
            # Build seg_readback differentiably so gradient flows back to scratch_alpha_q
            # (in-place index assignment to a torch.zeros tensor breaks the graph since
            # the zeros tensor doesn't require grad — torch.stack is the safe construction).
            zeros_ch = torch.zeros_like(scratch_alpha_q)
            seg_readback = torch.stack(
                [scratch_alpha_q, zeros_ch, zeros_ch], dim=-1
            )  # (B, W, 3), ch0 = scratch, ch1=ch2=0 (medium-agnostic)
            assert seg_readback.shape == (B, W, inputs.shape[-1]), \
                f"seg_readback shape {seg_readback.shape} != ({B},{W},{inputs.shape[-1]})"
            feat_readback = encode(seg_readback)
            # 5/21: state reset at readback (was state=alpha_state, leaking α-info
            # into β via substrate hidden state — implicit cheat path bypassing
            # scratch). Now readback starts from zero, so readback_state carries
            # ONLY scratch-derived info, β's only α-bridge is the scratch symbol.
            _h_rb_seq, readback_state = self.v33_substrate(feat_readback)
            # V34 stash: h_readback_end is the "actual h after substrate processed
            # quantized scratch". h_predictor (Linear(W, d), no nonlinearity) gives
            # a rank-W prediction from raw_preSTE — the comparison target for L_predict.
            # Gated on predict_coding_enable so loss-dispatcher in run_batch can detect
            # 'V34 not active' via None stashes and skip aux losses gracefully.
            if self.agent_cfg.predict_coding_enable and self.h_predictor is not None:
                self._h_readback_alpha_end = _h_rb_seq[:, -1, :]
                self._h_predicted_alpha = self.h_predictor(scratch_alpha_raw)
            else:
                self._h_readback_alpha_end = None
                self._h_predicted_alpha = None
            feat_beta = encode(beta_in)
            h_beta_seq, _ = self.v33_substrate(feat_beta, state=readback_state)
            h_beta_for_predict = h_beta_seq[:, -1, :]
            # Now write_head re-runs on h_β_end to produce the real sc_β
            scratch_beta_raw, scratch_beta_q = self.write_head(
                h_beta_for_predict,
                self.agent_cfg.quantize_levels,
                self.agent_cfg.quantize_range,
            )
            self._last_scratch_beta_K_q = scratch_beta_q.unsqueeze(1)
            self._last_scratch_beta_K_raw = scratch_beta_raw.unsqueeze(1)
        else:
            # V33 baseline: no readback, β picks up directly from α state, β scratch
            # is α placeholder (V29 pattern). This is the design that produced
            # baseline + option A + D5 ckpts — kept here for ablation parity.
            self._h_readback_alpha_end = None
            self._h_predicted_alpha = None
            feat_beta = encode(beta_in)
            h_beta_seq, _ = self.v33_substrate(feat_beta, state=alpha_state)
            h_beta_for_predict = h_beta_seq[:, -1, :]
            self._last_scratch_beta_K_q = scratch_alpha_q.unsqueeze(1)
            self._last_scratch_beta_K_raw = scratch_alpha_raw.unsqueeze(1)

        # V33-VICReg (5/22): exploratory aux on substrate h_seq. Compute per-pass
        # then average across {alpha, beta, (SM:readback)} passes. Stashes are
        # None when both lambdas are 0 — run_batch detects None and skips.
        v33_vlv = getattr(self.agent_cfg, "v33_vicreg_lambda_var", 0.0)
        v33_vlc = getattr(self.agent_cfg, "v33_vicreg_lambda_cov", 0.0)
        if v33_vlv > 0.0 or v33_vlc > 0.0:
            gamma_v33 = getattr(self.agent_cfg, "v33_vicreg_gamma", 1.0)
            lv_a, lc_a = MambaAgent._compute_h_seq_vicreg(h_alpha_seq, gamma_v33)
            lv_b, lc_b = MambaAgent._compute_h_seq_vicreg(h_beta_seq, gamma_v33)
            if same_medium:
                lv_r, lc_r = MambaAgent._compute_h_seq_vicreg(_h_rb_seq, gamma_v33)
                self._v33_vicreg_var = (lv_a + lv_b + lv_r) / 3.0
                self._v33_vicreg_cov = (lc_a + lc_b + lc_r) / 3.0
            else:
                self._v33_vicreg_var = (lv_a + lv_b) / 2.0
                self._v33_vicreg_cov = (lc_a + lc_b) / 2.0
        else:
            self._v33_vicreg_var = None
            self._v33_vicreg_cov = None

        fake_compare_logits = torch.zeros(B, K, 3, device=inputs.device, dtype=inputs.dtype)
        return fake_compare_logits, scratch_alpha_q, h_beta_for_predict

    def _v27_forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        alpha_centers: Optional[List[List[int]]] = None,
        beta_centers_list: Optional[List[List[List[int]]]] = None,
        metas: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """V27 chain-of-thought PFC forward (replaces V19/V25 at top of forward()).

        Architecture:
          1. α scan: chunked CoT loop → final α_state → write_head → scratch_α
          2. forget gap (LRU h re-init for β automatically)
          3. β scan (K=1 for successor task): chunked CoT loop → β_state → scratch_β
          4. predict_head reads scratch_α + scratch_β (set via run_batch V24 path)

        🚨 NO MERGE OF SEQUENTIAL INFO. CNN per-chunk + LRU per-chunk both fed
        sequentially to PFC chain. PFC iterates n_chunks times only.

        Returns (placeholder_logits, scratch_pad, None) for interface compat.
        """
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K
        d = self.agent_cfg.d_model

        # Clear stashes
        self._spike_score_history = []
        self._h_alpha_pfc_blocks = []
        self._h_beta_pfc_blocks = []
        self._scratch_alpha_for_aux = None
        self._scratch_alpha_raw = None
        self._last_scratch_beta_K_q = None
        self._last_scratch_beta_K_raw = None
        self._cot_alpha_stash = {}
        self._cot_beta_stash = {}

        # === α scan ===
        alpha_signal = inputs[:, :L, :]   # (B, L, 3)
        alpha_state = self.cot_pfc(
            alpha_signal, pfc_module=self.pfc,
            kwta_k=self.agent_cfg.kwta_k,
            stash=self._cot_alpha_stash,
        )  # (B, d)

        # Write to scratch_α via write_head (V25 SequentialWriteHead, V28-LRU
        # LRUDecoder, or V28-Sin SinusoidalWriteHead). All return (raw, quantized).
        h_for_write = self.write_head_ln(alpha_state)
        if isinstance(self.write_head, (SequentialWriteHead, LRUDecoder, SinusoidalWriteHead)):
            scratch_alpha_raw, scratch_alpha_q = self.write_head(
                h_for_write,
                self.agent_cfg.quantize_levels, self.agent_cfg.quantize_range,
            )
        else:
            raise RuntimeError(
                f"V27 forward requires Sequential/LRU/Sin write_head; got {type(self.write_head).__name__}"
            )

        # Stash for compute_step (run_batch V24 path reads these)
        self._scratch_alpha_for_aux = scratch_alpha_q
        self._scratch_alpha_raw = scratch_alpha_raw

        # === β scan (K=1 for V27 successor task) ===
        # Episode layout: alpha [0, L), gap [L, L+W+T_gap), then β block(s)
        rstart, beta_start, compare_step = cfg.compare_block_phases(0)
        beta_signal = inputs[:, beta_start : compare_step + 1, :]  # (B, L+1, 3)

        beta_state = self.cot_pfc(
            beta_signal, pfc_module=self.pfc,
            kwta_k=self.agent_cfg.kwta_k,
            stash=self._cot_beta_stash,
        )  # (B, d)

        h_for_write_beta = self.write_head_ln(beta_state)
        scratch_beta_raw, scratch_beta_q = self.write_head(
            h_for_write_beta,
            self.agent_cfg.quantize_levels, self.agent_cfg.quantize_range,
        )

        # Stash β scratch as (B, K=1, W)
        self._last_scratch_beta_K_q = scratch_beta_q.unsqueeze(1)
        self._last_scratch_beta_K_raw = scratch_beta_raw.unsqueeze(1)

        # Stash PFC h for diagnostic (h_α_pfc / h_β_pfc — V27 uses pfc_state directly)
        self._h_alpha_pfc_blocks.append(alpha_state)
        self._h_beta_pfc_blocks.append(beta_state)
        self._last_h_alpha_pfc_for_refine = alpha_state
        self._last_h_beta_pfc_for_refine = beta_state
        self._last_lru_h_alpha_for_refine = alpha_state  # placeholder
        self._last_lru_h_beta_for_refine = beta_state    # placeholder

        # Placeholder logits for interface compat (run_batch V24 path overrides)
        fake_logits = torch.zeros(B, K, 3, device=inputs.device, dtype=inputs.dtype)
        # scratch_pad shape (B, W) for eval framework (collect.py:113 expects 2D)
        fake_scratch_pad = scratch_alpha_q  # (B, W)

        return fake_logits, fake_scratch_pad, None

    def forward(
        self,
        inputs: torch.Tensor,
        compare_indices: torch.Tensor,
        alpha_centers: Optional[List[List[int]]] = None,
        beta_centers_list: Optional[List[List[List[int]]]] = None,
        metas: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs: (B, T, 3)
        compare_indices: (B, K) — unused (compute compare points from cfg)
        alpha_centers / beta_centers_list (oracle-gate only): if
            use_oracle_spike_gate is True, these supply per-batch spike
            centers from scene meta. alpha_centers: list-of-B lists of int
            positions. beta_centers_list: list of K entries (one per beta
            block), each a list-of-B lists.
        metas: optional shortcut. If supplied and use_oracle_spike_gate is
            True, alpha_centers / beta_centers_list are auto-extracted from
            metas (so callers like collect / extrapolation paths that have
            metas don't need to know the routing structure).
        Returns (compare_logits, scratch_pad, hidden_at_compares).

        If forward_predictor is enabled, also stashes world_loss as
        self._last_world_loss (a scalar tensor) for run_batch to read.
        """
        B, T, _ = inputs.shape
        cfg = self.scene_cfg
        L, W, T_gap, K = cfg.L, cfg.W, cfg.T_gap, cfg.K
        d = self.agent_cfg.d_model

        # V35 dispatch (5/27 pivot): bypasses V27/V28/V29/V33 forward paths.
        if self.v35_substrate is not None:
            return self._v35_forward(
                inputs, compare_indices, alpha_centers, beta_centers_list, metas
            )

        # V29 dispatch: SNN pipeline replaces entire forward
        if self.v29 is not None:
            return self._v29_forward(inputs, compare_indices)

        # V33 dispatch: HH-SSM substrate
        if self.v33_substrate is not None:
            return self._v33_forward(inputs, compare_indices, alpha_centers, beta_centers_list, metas)

        # V27 dispatch: if cot_pfc is enabled, use chain-of-thought path entirely.
        # Bypasses all V19/V25 PFC at write + PFC compare merging.
        if self.cot_pfc is not None:
            return self._v27_forward(inputs, compare_indices, alpha_centers, beta_centers_list, metas)

        # V19: clear spike-score history at start of each forward
        self._spike_score_history = []
        # V25-V22: clear h_alpha/h_beta_pfc per-block stash
        self._h_alpha_pfc_blocks = []
        self._h_beta_pfc_blocks = []
        # Oracle-gate routing: set alpha centers before encoding seg_A
        if self.agent_cfg.use_oracle_spike_gate:
            if alpha_centers is None and metas is not None:
                alpha_centers = [m.get("alpha_centers", []) for m in metas]
                K_cb = len(metas[0].get("beta_centers", []))
                beta_centers_list = [
                    [m.get("beta_centers", [[]] * K_cb)[i] for m in metas]
                    for i in range(K_cb)
                ]
            self._current_oracle_centers = alpha_centers
        else:
            self._current_oracle_centers = None

        # ===== Segment A: alpha scan + write + forget gap =====
        seg_a_end = L + W + T_gap
        seg_A = inputs[:, 0:seg_a_end, :]
        if self.top_down_predictor is not None:
            # PC-lite: mamba processes residual error pathway
            h_A = self._encode_segment_pc(seg_A)
        else:
            h_A = self._encode_segment(seg_A)  # (B, seg_a_end, d)

        # V6 dual-pathway: capture CNN summary of alpha scan (with gradient)
        # for PFC bypass at write/compare.
        # V7+stride: when dp_t_target > 0, _last_cnn_features is already pooled
        # to T_target ≪ L, so [:L] becomes [:T_target]; mean over whole.
        if self.agent_cfg.dual_pathway:
            if self.agent_cfg.dp_t_target > 0:
                self._last_cnn_alpha_summary = self._last_cnn_features.mean(dim=1)
                # V10/V15: stash α full sequence (B, T_target, d) for downstream
                if self.agent_cfg.pfc_see_cnn_seq or self.agent_cfg.recurrent_latent_pfc:
                    self._last_cnn_alpha_seq = self._last_cnn_features
            else:
                self._last_cnn_alpha_summary = self._last_cnn_features[:, :L, :].mean(dim=1)
                if self.agent_cfg.pfc_see_cnn_seq or self.agent_cfg.recurrent_latent_pfc:
                    self._last_cnn_alpha_seq = self._last_cnn_features[:, :L, :]
            # V15: replace cnn_alpha_summary with sequentially-accumulated latent
            # state from Coconut-style recurrent PFC over cnn_alpha_seq.
            if self.agent_cfg.recurrent_latent_pfc and self._last_cnn_alpha_seq is not None:
                seq_in = self._last_cnn_alpha_seq
                if self.agent_cfg.recurrent_latent_steps > 0:
                    # Truncate to first N steps (e.g. 4 step recurrence on 8 tokens)
                    seq_in = seq_in[:, :self.agent_cfg.recurrent_latent_steps, :]
                latent_state = self.recurrent_latent_pfc_module(seq_in)  # (B, d) for K=1
                if latent_state.dim() == 3:
                    latent_state = latent_state.mean(dim=1)  # if K>1, mean to single
                self._last_cnn_alpha_summary = latent_state
            self._last_cnn_beta_summary_per_block = []
            self._last_cnn_alpha_readback_per_block = []
            self._last_cnn_alpha_seq_per_block = []
            self._last_cnn_beta_seq_per_block = []
        else:
            self._last_cnn_alpha_summary = None
            self._last_cnn_alpha_seq = None

        # ===== World prediction (E0-prime auxiliary loss) =====
        # Compute predicted next-T_pred latents for every timestep in the alpha
        # scan only (t in [0, L)). MSE against actual latents at t+1..t+T_pred.
        # Targets are NOT detached so encoder is also pushed to be predictable.
        if self.forward_predictor is not None:
            t_pred = self.agent_cfg.world_pred_t_pred
            valid_t = L - t_pred  # last valid prediction start index (exclusive)
            if valid_t > 0:
                # Predictor input: h at each t in [0, valid_t), scratch is empty (zeros)
                # during alpha scan.
                h_alpha = h_A[:, :valid_t, :]  # (B, valid_t, d)
                B_v, T_v, _ = h_alpha.shape
                scratch_dim = W * self.agent_cfg.signal_output_dim
                empty_scratch = h_alpha.new_zeros(B_v * T_v, scratch_dim)
                pred = self.forward_predictor(
                    h_alpha.reshape(B_v * T_v, d),
                    empty_scratch,
                )  # (B_v*T_v, t_pred, d)
                pred = pred.view(B_v, T_v, t_pred, d)  # (B, valid_t, t_pred, d)

                # Build target tensor: target[b, t, k] = h_A[b, t+1+k]
                # Use unfold over the time dim of h_A[:, 1:1+valid_t+t_pred-1]
                # Shape: (B, valid_t + t_pred - 1, d). We want sliding windows of size t_pred.
                tgt_src = h_A[:, 1:1 + valid_t + t_pred - 1, :]  # (B, valid_t + t_pred - 1, d)
                # unfold dim=1 size=t_pred step=1 → (B, valid_t, d, t_pred)
                target = tgt_src.unfold(dimension=1, size=t_pred, step=1)
                target = target.transpose(-1, -2)  # (B, valid_t, t_pred, d)

                # COLLAPSE FIX: detach target so encoder isn't pulled toward
                # collapse via this loss (BYOL-style stop-gradient).
                if self.agent_cfg.aux_detach_targets:
                    target_for_loss = target.detach()
                else:
                    target_for_loss = target
                self._last_world_loss = F.mse_loss(pred, target_for_loss)
                # VICReg variance regularizer on target (anti-collapse).
                if self.agent_cfg.vicreg_lambda > 0.0:
                    self._last_world_vicreg = vicreg_variance_loss(
                        target.reshape(-1, d),
                        gamma=self.agent_cfg.vicreg_gamma,
                    )
                else:
                    self._last_world_vicreg = None
            else:
                self._last_world_loss = h_A.new_zeros(())
                self._last_world_vicreg = None
        else:
            self._last_world_loss = None
            self._last_world_vicreg = None

        # ===== RS-1: Raw signal prediction (alpha scan only) =====
        # Predict next T_pred RAW input signal scalars from h_t at each t in [0, L).
        if self.rs_predictor is not None:
            t_pred_rs = self.agent_cfg.rs_pred_t_pred
            valid_t = L - t_pred_rs
            if valid_t > 0:
                h_alpha_rs = h_A[:, :valid_t, :]  # (B, valid_t, d)
                B_v, T_v, _ = h_alpha_rs.shape
                pred_signal = self.rs_predictor(
                    h_alpha_rs.reshape(B_v * T_v, d)
                ).view(B_v, T_v, t_pred_rs)
                # Target: inputs[:, t+1:t+1+t_pred_rs, channel=0]
                signal_src = inputs[:, 1:1 + valid_t + t_pred_rs - 1, 0]
                target_signal = signal_src.unfold(
                    dimension=1, size=t_pred_rs, step=1,
                )  # (B, valid_t, t_pred_rs)
                self._last_rs_pred_loss = F.mse_loss(pred_signal, target_signal)
            else:
                self._last_rs_pred_loss = h_A.new_zeros(())
        else:
            self._last_rs_pred_loss = None

        # Apply write_head + quantize STE at each write timestep.
        # M4: if PFC + pfc_at_write, refine h via PFC([h, prior_scratch]) before write.
        scratch_list: List[torch.Tensor] = []
        # V22: pre-quantize α scratch (for successor function input)
        scratch_alpha_raw_list: List[torch.Tensor] = []
        # A3 accumulator
        scratch_self_pred_losses: List[torch.Tensor] = []
        # β-1 accumulators
        vq_commit_losses: List[torch.Tensor] = []
        vq_indices_list: List[torch.Tensor] = []
        # 51_M5: collect h_for_write across W steps for L_cycle target h_alpha.
        h_for_write_list: List[torch.Tensor] = []

        # V13: cross-attention write head — single parallel call, W queries
        # cross-attend cnn_alpha_seq, each query gets its own h_for_write.
        if self.agent_cfg.cross_attn_write:
            # Choose KV sequence
            if self.agent_cfg.cnn_pfc_only and self._last_cnn_alpha_seq is not None:
                kv = self._last_cnn_alpha_seq            # (B, T_target, d)
            elif self._last_cnn_features is not None:
                if self.agent_cfg.dp_t_target > 0:
                    kv = self._last_cnn_features
                else:
                    kv = self._last_cnn_features[:, :L, :]
            else:
                kv = h_A[:, :L, :]                       # fallback
            write_features = self.cross_attn_write_module(kv)  # (B, W, d)
            # Apply kWTA per cell
            if self.agent_cfg.kwta_k > 0:
                B_w = write_features.shape[0]
                wf_flat = write_features.reshape(-1, write_features.size(-1))
                wf_flat = kwta_topk(wf_flat, k=self.agent_cfg.kwta_k)
                write_features = wf_flat.view(write_features.shape)
            # write_head per cell (vectorized)
            raw_all = self.write_head(write_features)    # (B, W, signal_output_dim)
            if self.agent_cfg.write_head_clip > 0:
                c = self.agent_cfg.write_head_clip
                raw_all = c * torch.tanh(raw_all / c)
            if self.training and self.agent_cfg.write_noise_std > 0.0:
                raw_all = raw_all + torch.randn_like(raw_all) * self.agent_cfg.write_noise_std
            if self.agent_cfg.quantize_levels is not None:
                raw_all = _quantize_ste(
                    raw_all, self.agent_cfg.quantize_levels,
                    self.agent_cfg.quantize_range,
                )
            for k in range(W):
                scratch_list.append(raw_all[:, k, :])
                h_for_write_list.append(write_features[:, k, :])
            scratch_self_pred_losses = []
            vq_commit_losses = []
            vq_indices_list = []

        # P3 (V11): IF write controller — model decides write timing.
        # Replace W-step write loop entirely with single IF call over encoder seq.
        if self.agent_cfg.if_write_gate:
            # Choose state sequence: Mamba (h_A, sequential through CNN snapshots)
            # or CNN-only (cnn_alpha_seq) depending on if_use_mamba_state.
            if self.agent_cfg.if_use_mamba_state:
                if self.agent_cfg.dp_t_target > 0:
                    state_seq = h_A    # (B, T_target, d) when stride/pool active
                else:
                    state_seq = h_A[:, :L, :]  # (B, L, d)
            else:
                # Use CNN sequence directly (no Mamba). cnn_alpha_seq stashed by
                # _encode_segment via _last_cnn_features path.
                if (self.agent_cfg.cnn_pfc_only
                        and self._last_cnn_alpha_seq is not None):
                    state_seq = self._last_cnn_alpha_seq
                elif self._last_cnn_features is not None:
                    if self.agent_cfg.dp_t_target > 0:
                        state_seq = self._last_cnn_features
                    else:
                        state_seq = self._last_cnn_features[:, :L, :]
                else:
                    state_seq = h_A[:, :L, :]  # fallback

            # IF dynamics: returns (B, W, 1) quantized cells
            cells_qWd, fire_scores, top_idx = self.write_head(state_seq)
            # Stash diagnostics
            self._last_if_fire_scores = fire_scores
            self._last_if_top_idx = top_idx
            # Convert to scratch_list format (list of W tensors of (B, sig_dim))
            for j in range(W):
                scratch_list.append(cells_qWd[:, j, :])
                # PFC bypass / cycle target tracking — just stash zero placeholder
                h_for_write_list.append(state_seq.new_zeros(state_seq.shape[0], d))
            # Skip the per-step write loop entirely
            scratch_self_pred_losses = []
            vq_commit_losses = []
            vq_indices_list = []
        else:
            self._last_if_fire_scores = None
            self._last_if_top_idx = None

        # V9-AR: init recurrent state once before the W loop (only if AR head),
        # and bump tau-anneal step counter once per forward (training only).
        if self.agent_cfg.ar_write_head:
            B_now = h_A.shape[0]
            ar_h_rnn, ar_prev_cell = self.write_head.init_state(B_now, h_A.device)
            self.write_head.maybe_increment_step()
        else:
            ar_h_rnn = ar_prev_cell = None
        # V25: SequentialWriteHead replaces W loop with single sequential rollout.
        # Compute h_for_write once at j=0 (scratch_kv empty), then sequential
        # write_head produces all W cells.
        is_sequential_write = isinstance(self.write_head, SequentialWriteHead)
        if is_sequential_write:
            # Build h_t (constant across W in V19 ext config)
            if self.agent_cfg.cnn_pfc_only and self._last_cnn_alpha_summary is not None:
                h_t_seq = self._last_cnn_alpha_summary
            elif self.agent_cfg.dp_t_target > 0:
                h_t_seq = h_A[:, -1, :]
            else:
                h_t_seq = h_A[:, L, :]
            # PFC-at-write with empty scratch_kv (j=0 state)
            if self.pfc is not None and self.agent_cfg.pfc_at_write:
                Bw = h_t_seq.shape[0]
                sig_dim = self.agent_cfg.signal_output_dim
                empty_scratch = torch.zeros(Bw, W, sig_dim, device=h_t_seq.device, dtype=h_t_seq.dtype)
                kv = self.scratch_proj(empty_scratch) + self.scratch_pos_emb.unsqueeze(0)
                if self.write_query_emb is not None:
                    q_tok = self.write_query_emb.view(1, 1, -1).expand(Bw, 1, -1)
                    seq_w = torch.cat([q_tok, h_t_seq.unsqueeze(1), kv], dim=1)
                else:
                    seq_w = torch.cat([h_t_seq.unsqueeze(1), kv], dim=1)
                seq_w_refined = seq_w
                for _ in range(self.agent_cfg.pfc_iter):
                    seq_w_refined = self.pfc(seq_w_refined)
                h_for_write_seq = seq_w_refined[:, 0, :]
            else:
                h_for_write_seq = h_t_seq
            if self.agent_cfg.kwta_k > 0:
                h_for_write_seq = kwta_topk(h_for_write_seq, k=self.agent_cfg.kwta_k)
            h_for_write_seq = self.write_head_ln(h_for_write_seq)
            # Sequential rollout — all W cells at once
            raw_all_seq, q_all_seq = self.write_head(
                h_for_write_seq,
                self.agent_cfg.quantize_levels,
                self.agent_cfg.quantize_range,
            )  # both (B, W)
            # Push into scratch_list / raw_list for downstream consumers
            for j in range(W):
                scratch_list.append(q_all_seq[:, j].unsqueeze(-1))  # (B, 1)
                scratch_alpha_raw_list.append(raw_all_seq[:, j].unsqueeze(-1))
                h_for_write_list.append(h_for_write_seq)
            W_iter_range = range(0)
        # Skip W loop entirely if IF gate, cross_attn_write, or Sequential write head.
        elif self.agent_cfg.if_write_gate or self.agent_cfg.cross_attn_write:
            W_iter_range = range(0)
        else:
            W_iter_range = range(W)
        for j in W_iter_range:
            # V6 BRANCH: cnn_pfc_only replaces h_t (Mamba state at write step) with
            # cnn_alpha_summary (mean CNN over alpha scan). Constant across writes;
            # PFC differentiates slots via scratch_pos_emb in scratch_kv.
            # V7+stride (dp_t_target > 0): h_A pooled to T_target ≪ L, so per-write
            # index L+j doesn't exist. Use last Mamba state for all writes; PFC
            # differentiates via scratch_pos_emb (same as cnn_pfc_only).
            if self.agent_cfg.cnn_pfc_only and self._last_cnn_alpha_summary is not None:
                h_t = self._last_cnn_alpha_summary
            elif self.agent_cfg.dp_t_target > 0:
                h_t = h_A[:, -1, :]
            else:
                h_t = h_A[:, L + j, :]
            if self.pfc is not None and self.agent_cfg.pfc_at_write:
                # Build current scratch view (already-written + zero placeholders)
                Bw = h_t.shape[0]
                sp_slots = []
                for k in range(W):
                    if k < len(scratch_list):
                        sp_slots.append(scratch_list[k])
                    else:
                        sp_slots.append(h_t.new_zeros(Bw, self.agent_cfg.signal_output_dim))
                sp_w = torch.stack(sp_slots, dim=1)
                kv_w = self.scratch_proj(sp_w) + self.scratch_pos_emb.unsqueeze(0)
                # V10 (pfc_see_cnn_seq): replace single h_t token with full
                # T_target CNN tokens (with positional embedding), letting PFC
                # extract within-α positional structure. h_for_write = mean
                # over T CNN-output tokens after PFC self-attention.
                if (self.agent_cfg.pfc_see_cnn_seq
                        and self._last_cnn_alpha_seq is not None):
                    cnn_seq = self._last_cnn_alpha_seq + self.cnn_pos_emb.unsqueeze(0)
                    T_seq = cnn_seq.shape[1]
                    seq_w = torch.cat([cnn_seq, kv_w], dim=1)  # (B, T+W, d)
                    seq_w_refined = seq_w
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_w_refined = self.pfc(seq_w_refined)
                    # h_for_write = mean over the T CNN-output positions
                    h_for_write = seq_w_refined[:, :T_seq, :].mean(dim=1)
                # V6: insert CNN subitizing token. Skip in cnn_pfc_only mode
                # (h_t already IS cnn_alpha_summary, redundant).
                elif (self.agent_cfg.dual_pathway
                        and not self.agent_cfg.cnn_pfc_only
                        and self._last_cnn_alpha_summary is not None):
                    cnn_tok = self._last_cnn_alpha_summary.unsqueeze(1)  # (B, 1, d)
                    seq_w = torch.cat([h_t.unsqueeze(1), cnn_tok, kv_w], dim=1)  # (B, 2+W, d)
                    seq_w_refined = seq_w
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_w_refined = self.pfc(seq_w_refined)
                    h_for_write = seq_w_refined[:, 0, :]
                else:
                    # V16: prepend learnable [WRITE_QUERY] at position 0 if enabled,
                    # so output[0] is dominated by attention (not residual h_t).
                    if self.write_query_emb is not None:
                        Bw = h_t.shape[0]
                        q_tok = self.write_query_emb.view(1, 1, -1).expand(Bw, 1, -1)
                        seq_w = torch.cat([q_tok, h_t.unsqueeze(1), kv_w], dim=1)  # (B, 2+W, d)
                    else:
                        seq_w = torch.cat([h_t.unsqueeze(1), kv_w], dim=1)  # (B, 1+W, d)
                    seq_w_refined = seq_w
                    for _ in range(self.agent_cfg.pfc_iter):
                        seq_w_refined = self.pfc(seq_w_refined)
                    h_for_write = seq_w_refined[:, 0, :]
            else:
                h_for_write = h_t
            # V8_kWTA: sparse top-k on PFC write output before write_head
            if self.agent_cfg.kwta_k > 0:
                h_for_write = kwta_topk(h_for_write, k=self.agent_cfg.kwta_k)
            # V12: optional LayerNorm before write_head — prevents sigmoid saturation
            # by normalizing h_for_write magnitude. nn.Identity() if write_head_ln=False.
            h_for_write = self.write_head_ln(h_for_write)
            h_for_write_list.append(h_for_write)  # (B, d)
            if self.agent_cfg.ar_write_head:
                # V9-AR: GRUCell autoregressive head, hard-quantized output (B, 1)
                raw, ar_h_rnn, _ = self.write_head.step(
                    h_for_write, ar_prev_cell, ar_h_rnn,
                )
                ar_prev_cell = raw
            elif self.agent_cfg.gumbel_write_head:
                # V14: GumbelWriteHead returns already-quantized scalar (B, 1).
                # Skip quantize_STE, skip write_noise (gumbel has its own noise),
                # skip clip (output already bounded).
                raw = self.write_head(h_for_write)
            else:
                raw = self.write_head(h_for_write)  # (B, signal_output_dim)
                # V12: smooth bounded clip on raw to prevent sigmoid saturation
                if self.agent_cfg.write_head_clip > 0:
                    c = self.agent_cfg.write_head_clip
                    raw = c * torch.tanh(raw / c)
                if self.training and self.agent_cfg.write_noise_std > 0.0:
                    raw = raw + torch.randn_like(raw) * self.agent_cfg.write_noise_std
                # V22: stash pre-quantize scratch_alpha for successor function input
                scratch_alpha_raw_list.append(raw)
                if self.vq is not None:
                    # β-1: VQ replaces scalar quantize. Accumulate commit loss + indices.
                    raw, commit_loss, vq_idx = self.vq(raw)
                    vq_commit_losses.append(commit_loss)
                    vq_indices_list.append(vq_idx)
                elif self.agent_cfg.quantize_levels is not None:
                    raw = _quantize_ste(
                        raw,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
            # A3: predict raw (current slot) BEFORE appending, using prior slots + h_t.
            # Skip first step (no prior scratch).
            if self.scratch_self_predictor is not None and j > 0:
                Bw = h_t.shape[0]
                sig_dim = self.agent_cfg.signal_output_dim
                # Build prior scratch padded to max_W
                prior_slots = []
                for k in range(W):
                    if k < len(scratch_list):
                        prior_slots.append(scratch_list[k])
                    else:
                        prior_slots.append(h_t.new_zeros(Bw, sig_dim))
                prior_padded = torch.cat(prior_slots, dim=-1)  # (B, max_W*sig_dim)
                pred_slot = self.scratch_self_predictor(h_t, prior_padded)
                scratch_self_pred_losses.append(
                    F.mse_loss(pred_slot, raw.detach())
                )
            scratch_list.append(raw)
        if self.scratch_self_predictor is not None and scratch_self_pred_losses:
            self._last_scratch_self_pred_loss = torch.stack(
                scratch_self_pred_losses
            ).mean()
        else:
            self._last_scratch_self_pred_loss = None

        # V9-AR: stash AR head mean entropy across W steps for anti-collapse bonus.
        if self.agent_cfg.ar_write_head:
            self._last_ar_entropy = self.write_head.get_mean_entropy()
        else:
            self._last_ar_entropy = None

        # V20 / V22: stash α scratch (post-quantize) as (B, W) for downstream
        # heads (scratch_mod_aux_head, e_pair consistency loss).
        if scratch_list:
            self._scratch_alpha_for_aux = torch.stack(
                scratch_list, dim=1
            ).squeeze(-1)  # (B, W)
        else:
            self._scratch_alpha_for_aux = None
        # V22: stash α scratch (pre-quantize) for SuccessorFunction input.
        # Empty when AR / gumbel / IF / cross_attn paths produce no raw list.
        if scratch_alpha_raw_list:
            self._scratch_alpha_raw = torch.stack(
                scratch_alpha_raw_list, dim=1
            ).squeeze(-1)  # (B, W)
        else:
            self._scratch_alpha_raw = None
        # V20 scratch_mod_aux_head accumulator (only when head is built).
        scratch_aux_block_logits: Optional[List[torch.Tensor]] = (
            [] if self.scratch_mod_aux_head is not None else None
        )
        # V22: virtual β scratch accumulators — always populated when oracle gate +
        # PFC at write are on (cheap; reused by V20 head + V22 losses).
        scratch_beta_raw_blocks: List[torch.Tensor] = []  # each (B, W) pre-quantize
        scratch_beta_q_blocks: List[torch.Tensor] = []    # each (B, W) post-quantize
        # 51_M5: stash h_alpha = mean over W of h_for_write (pre-quantize PFC out).
        # This is the L_cycle target — what scratch should be able to recover.
        if self.agent_cfg.cycle_loss_lambda > 0.0 and h_for_write_list:
            self._last_h_alpha_for_cycle = torch.stack(
                h_for_write_list, dim=1
            ).mean(dim=1)  # (B, d)
        else:
            self._last_h_alpha_for_cycle = None

        # β-1: aggregate VQ commit losses + collect indices for util tracking.
        if self.vq is not None and vq_commit_losses:
            self._last_vq_commit_loss = torch.stack(vq_commit_losses).mean()
            self._last_vq_indices = torch.stack(vq_indices_list, dim=1)  # (B, W)
        else:
            self._last_vq_commit_loss = None
            self._last_vq_indices = None

        # ===== idea2: Self-review iterative refinement =====
        # After write phase, refine the entire scratch_list via ReviewModule
        # for K_review iterations. Output replaces scratch_list.
        if self.review_module is not None and self.agent_cfg.review_iter > 0:
            Br = scratch_list[0].shape[0]
            h_alpha_end = h_A[:, L - 1, :]  # state at end of α scan
            sp_stack = torch.stack(scratch_list, dim=1)  # (B, W, sig_dim)
            for _ in range(self.agent_cfg.review_iter):
                sp_proj = self.review_scratch_proj(sp_stack) + self.review_pos_emb.unsqueeze(0)
                tokens = torch.cat([h_alpha_end.unsqueeze(1), sp_proj], dim=1)  # (B, 1+W, d)
                refined = self.review_module(tokens)  # (B, 1+W, d)
                refined_slots = self.review_write_head(refined[:, 1:, :])  # (B, W, sig_dim)
                # Apply quantize STE to keep scratch in valid range
                if self.agent_cfg.quantize_levels is not None:
                    refined_flat = refined_slots.reshape(
                        Br * W, self.agent_cfg.signal_output_dim
                    )
                    refined_flat = _quantize_ste(
                        refined_flat,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )
                    refined_slots = refined_flat.view(
                        Br, W, self.agent_cfg.signal_output_dim
                    )
                sp_stack = refined_slots
            # Replace scratch_list with refined slots.
            scratch_list = [sp_stack[:, j, :] for j in range(W)]

        # ===== Compare blocks =====
        compare_logits_list: List[torch.Tensor] = []
        hidden_at_compares_list: List[torch.Tensor] = []
        # 51_M5: collect h_sp per compare block for L_cycle (avg later)
        h_sp_per_block: List[torch.Tensor] = []
        # V17: aux mod head logits, one per compare block
        aux_block_logits: List[torch.Tensor] = []
        # Aux loss accumulators (A2 / A5).
        beta_pred_losses: List[torch.Tensor] = []
        compare_pred_losses: List[torch.Tensor] = []
        # COLLAPSE FIX: VICReg variance accumulators
        beta_vicreg_losses: List[torch.Tensor] = []
        compare_vicreg_losses: List[torch.Tensor] = []
        # Pre-flatten scratch (used by A2 / A5 / A4 if active).
        if (self.beta_predictor is not None or self.compare_predictor is not None
                or self.cf_beta_predictor is not None):
            scratch_flat_full = torch.cat(
                [s for s in scratch_list], dim=-1
            )  # (B, W*signal_output_dim)
        # A4 accumulator
        cf_beta_pred_losses: List[torch.Tensor] = []
        for i in range(K):
            rstart, beta_start, compare_step = cfg.compare_block_phases(i)

            # Segment B: read-back, W steps. Inject scratch into signal channel.
            # β-1: when scratch is vectorial (d_z>1), only inject first dim into
            # input channel 0 (lossy by design — full info still flows via PFC).
            seg_B = inputs[:, rstart:beta_start, :].clone()
            for j in range(W):
                if self.vq is not None:
                    seg_B[:, j, 0] = scratch_list[j][:, 0]
                else:
                    seg_B[:, j, 0:self.agent_cfg.signal_output_dim] = scratch_list[j]
            # V6 Q3: skip Mamba on the W=5 readback. Use CNN-only path:
            # multi-scale CNN over W tokens, then mean-pool → h_alpha.
            # Rationale: W=5 too short for Mamba accumulation; CNN's local pattern
            # recognition reads scratch directly, avoiding 1D-thermometer geometry
            # imposed by Mamba.
            if self.agent_cfg.dual_pathway and self.agent_cfg.dp_scratch_skip_mamba:
                cnn_B = self._encode_v6_cnn_only(seg_B, apply_pool=False)  # (B, W, d)
                h_alpha = cnn_B.mean(dim=1)  # (B, d)
                # V10: stash full readback sequence for PFC seq input
                if self.agent_cfg.pfc_see_cnn_seq:
                    self._last_cnn_alpha_seq_per_block.append(cnn_B)
            else:
                # V7+stride: skip adaptive pool for short seg_B (W=5 too short to pool)
                h_B = self._encode_segment(seg_B, apply_pool=False)  # (B, W, d)
                h_alpha = h_B[:, -1, :]  # state at end of read-back
                # V7: stash CNN-only view of seg_B for PFC compare token bypass.
                # _encode_segment(seg_B) above already populated _last_cnn_features
                # with CNN(seg_B) (gradient-active). Take mean over W readback steps.
                if (self.agent_cfg.dual_pathway
                        and not self.agent_cfg.cnn_pfc_only):
                    cnn_alpha_readback = self._last_cnn_features.mean(dim=1)  # (B, d)
                    self._last_cnn_alpha_readback_per_block.append(cnn_alpha_readback)

            # A2: predict β scan latents BEFORE seeing β (uses h_alpha + scratch).
            beta_pred = None
            if self.beta_predictor is not None:
                beta_pred = self.beta_predictor(h_alpha, scratch_flat_full)
                # (B, T_β_pred, d)

            # A4: predict β raw signal BEFORE seeing β, conditioned on (scratch, β_count proxy).
            cf_pred_signal = None
            if self.cf_beta_predictor is not None:
                # Estimate β_count by thresholding peaks in actual β segment.
                # complex_world: peaks ~0.86, gaps ~0.06; threshold 0.5 separates cleanly.
                beta_signal_full = inputs[:, beta_start:compare_step + 1, 0]  # (B, L+1)
                peak_count = (beta_signal_full > 0.5).sum(dim=1).float()  # (B,)
                # Normalize by L (rough density proxy ∈ [0, 1])
                target_n_norm = (peak_count / cfg.L).unsqueeze(-1)  # (B, 1)
                cf_pred_signal = self.cf_beta_predictor(
                    scratch_flat_full, target_n_norm,
                )  # (B, T_cf)

            # Segment C: beta scan + compare step (L+1 timesteps).
            seg_C = inputs[:, beta_start:compare_step + 1, :]
            # Oracle-gate: route the i-th beta-block centers before encode.
            if self.agent_cfg.use_oracle_spike_gate and beta_centers_list is not None:
                if i < len(beta_centers_list):
                    self._current_oracle_centers = beta_centers_list[i]
                else:
                    self._current_oracle_centers = None
            h_C = self._encode_segment(seg_C)  # (B, L+1, d)

            # V6 dual-pathway: capture CNN summary of beta scan (with gradient).
            # V7+stride: when pooled, _last_cnn_features is T_target ≪ L+1, mean over all.
            if self.agent_cfg.dual_pathway:
                if self.agent_cfg.dp_t_target > 0:
                    cnn_beta_summary = self._last_cnn_features.mean(dim=1)
                    if self.agent_cfg.pfc_see_cnn_seq:
                        # V10: stash β full sequence (B, T_target, d)
                        self._last_cnn_beta_seq_per_block.append(self._last_cnn_features)
                else:
                    cnn_beta_summary = self._last_cnn_features[:, :L, :].mean(dim=1)
                    if self.agent_cfg.pfc_see_cnn_seq:
                        self._last_cnn_beta_seq_per_block.append(self._last_cnn_features[:, :L, :])
                self._last_cnn_beta_summary_per_block.append(cnn_beta_summary)

            # V6 BRANCH: cnn_pfc_only replaces h_beta (last token of seg_C =
            # just the compare-step CNN feature, no integration) with
            # cnn_beta_summary (mean CNN over L beta timesteps).
            if self.agent_cfg.cnn_pfc_only:
                h_beta = self._last_cnn_beta_summary_per_block[i]
            else:
                h_beta = h_C[:, -1, :]  # state at compare_step

            # A2 loss: predicted vs actual first T_β steps of β segment.
            if beta_pred is not None:
                T_beta = self.agent_cfg.beta_pred_t_pred
                actual_beta_latents = h_C[:, :T_beta, :]
                # COLLAPSE FIX: detach target so encoder isn't pulled toward
                # collapse via this BYOL-style loss.
                if self.agent_cfg.aux_detach_targets:
                    target_for_loss = actual_beta_latents.detach()
                else:
                    target_for_loss = actual_beta_latents
                beta_pred_losses.append(
                    F.mse_loss(beta_pred, target_for_loss)
                )
                # VICReg variance regularizer on target (anti-collapse).
                if self.agent_cfg.vicreg_lambda > 0.0:
                    beta_vicreg_losses.append(
                        vicreg_variance_loss(
                            actual_beta_latents.reshape(-1, d),
                            gamma=self.agent_cfg.vicreg_gamma,
                        )
                    )

            # A4 loss: predicted vs actual first T_cf raw signal steps of β.
            if cf_pred_signal is not None:
                T_cf = self.agent_cfg.cf_beta_pred_t_pred
                actual_beta_signal = inputs[:, beta_start:beta_start + T_cf, 0]  # (B, T_cf)
                cf_beta_pred_losses.append(
                    F.mse_loss(cf_pred_signal, actual_beta_signal)
                )

            # M4: optional PFC refinement at compare time.
            # V6 NOTE: NO CNN bypass at compare time for cnn_pfc_only mode.
            # V7 (2026-05-04): when dual_pathway and Mamba-on-readback (i.e.,
            # NOT cnn_pfc_only), PFC compare gets 4 tokens:
            #   [h_α (Mamba on seg_B), h_β (Mamba on seg_C),
            #    cnn_α_readback (CNN view of seg_B with scratch),
            #    cnn_β_summary (CNN view of seg_C beta scan)]
            # Both cnn_α_readback and cnn_β_summary are CURRENT-segment derived
            # (not stashed alpha pre-forget). Self-attention lets PFC dynamically
            # weight Mamba vs CNN signals. NOT a leak: alpha info reaches PFC
            # only via scratch (which goes through CNN(seg_B) here).
            v7_compare = (self.agent_cfg.dual_pathway
                          and not self.agent_cfg.cnn_pfc_only
                          and len(self._last_cnn_alpha_readback_per_block) > i)
            if self.pfc is not None:
                # V10: pfc_see_cnn_seq replaces single h_α / h_β tokens with
                # T_target CNN tokens per side (with pos_emb + side_emb), so
                # PFC compare can extract within-segment positional structure.
                # h_α_pfc / h_β_pfc = mean over each side's T tokens after self-attn.
                v10_seq = (self.agent_cfg.pfc_see_cnn_seq
                           and self.agent_cfg.cnn_pfc_only
                           and len(self._last_cnn_alpha_seq_per_block) > i
                           and len(self._last_cnn_beta_seq_per_block) > i)
                if v10_seq:
                    a_seq = self._last_cnn_alpha_seq_per_block[i]  # (B, W, d) for readback
                    b_seq = self._last_cnn_beta_seq_per_block[i]   # (B, T_target, d)
                    # Add side embeddings; CNN pos_emb only on β side (T = T_target).
                    # α readback has W=5 tokens, use scratch_pos_emb (already designed for W).
                    a_seq = a_seq + self.alpha_side_emb.view(1, 1, -1) + self.scratch_pos_emb.unsqueeze(0)
                    # β has T_target tokens: use cnn_pos_emb
                    T_b = b_seq.shape[1]
                    b_pos = self.cnn_pos_emb[:T_b].unsqueeze(0) if T_b <= self.cnn_pos_emb.shape[0] else self.cnn_pos_emb.unsqueeze(0)
                    b_seq = b_seq + self.beta_side_emb.view(1, 1, -1) + b_pos
                    seq_c = torch.cat([a_seq, b_seq], dim=1)  # (B, W + T_b, d)
                    T_a_n = a_seq.shape[1]
                elif self.agent_cfg.pfc_compare_drop_raw_scratch:
                    if v7_compare:
                        cnn_a_rb = self._last_cnn_alpha_readback_per_block[i].unsqueeze(1)
                        cnn_b_sum = self._last_cnn_beta_summary_per_block[i].unsqueeze(1)
                        seq_c = torch.cat([
                            h_alpha.unsqueeze(1), h_beta.unsqueeze(1),
                            cnn_a_rb, cnn_b_sum,
                        ], dim=1)  # (B, 4, d)
                    else:
                        seq_c = torch.stack([h_alpha, h_beta], dim=1)  # (B, 2, d)
                else:
                    sp = torch.stack(scratch_list, dim=1)  # (B, W, signal_dim)
                    kv = self.scratch_proj(sp) + self.scratch_pos_emb.unsqueeze(0)
                    if v7_compare:
                        cnn_a_rb = self._last_cnn_alpha_readback_per_block[i].unsqueeze(1)
                        cnn_b_sum = self._last_cnn_beta_summary_per_block[i].unsqueeze(1)
                        seq_c = torch.cat([
                            h_alpha.unsqueeze(1), h_beta.unsqueeze(1),
                            cnn_a_rb, cnn_b_sum, kv,
                        ], dim=1)
                    else:
                        seq_c = torch.cat([h_alpha.unsqueeze(1), h_beta.unsqueeze(1), kv], dim=1)
                seq_c_refined = seq_c
                # 51_M5: pfc_compare may differ from pfc (MLP variant has 2 instances)
                pfc_compare_module = self.pfc_compare if self.pfc_compare is not None else self.pfc
                for _ in range(self.agent_cfg.pfc_iter):
                    seq_c_refined = pfc_compare_module(seq_c_refined)
                if v10_seq:
                    # V10: mean over each side's tokens after self-attn
                    h_alpha_pfc = seq_c_refined[:, :T_a_n, :].mean(dim=1)
                    h_beta_pfc = seq_c_refined[:, T_a_n:, :].mean(dim=1)
                else:
                    h_alpha_pfc = seq_c_refined[:, 0, :]
                    h_beta_pfc = seq_c_refined[:, 1, :]
                # V8_kWTA: sparse top-k on PFC compare outputs before comparator
                if self.agent_cfg.kwta_k > 0:
                    h_alpha_pfc = kwta_topk(h_alpha_pfc, k=self.agent_cfg.kwta_k)
                    h_beta_pfc = kwta_topk(h_beta_pfc, k=self.agent_cfg.kwta_k)
                # V24R: stash h_alpha_pfc + h_beta_pfc + LRU end-state at α/β
                # for the recurrent refiner. K=1 expected; first block wins.
                if i == 0:
                    self._last_h_alpha_pfc_for_refine = h_alpha_pfc
                    self._last_h_beta_pfc_for_refine = h_beta_pfc
                    # h_A is α LRU sequence; h_C is β LRU sequence (current block).
                    # Take last unpooled timestep as scalar end-state.
                    self._last_lru_h_alpha_for_refine = h_A[:, -1, :]
                    self._last_lru_h_beta_for_refine = h_C[:, -1, :]
                # V25-V22: stash per-block h_alpha_pfc + h_beta_pfc for
                # h_pair_consistency loss (PFC level E pair MSE).
                if not hasattr(self, "_h_alpha_pfc_blocks") or self._h_alpha_pfc_blocks is None:
                    self._h_alpha_pfc_blocks = []
                    self._h_beta_pfc_blocks = []
                self._h_alpha_pfc_blocks.append(h_alpha_pfc)
                self._h_beta_pfc_blocks.append(h_beta_pfc)
                # 51_M5: simple comparator uses (h_sp − h_β) → Linear(d, C)
                # V8: rbf_compare_head also uses diff input but RBF readout
                # NOTE: in 51_M5 naming, the "h_alpha_pfc" of the read-back path
                # IS what user calls "h_sp" (scratch走完forward的 hidden).
                if self.agent_cfg.simple_comparator or self.agent_cfg.rbf_compare_head:
                    h_sp_block = h_alpha_pfc        # rename for clarity (scratch-derived)
                    h_beta_block = h_beta_pfc
                    diff = h_sp_block - h_beta_block  # (B, d)
                    compare_in = diff
                else:
                    compare_in = torch.cat([h_alpha_pfc, h_beta_pfc], dim=-1)
            else:
                if self.agent_cfg.simple_comparator or self.agent_cfg.rbf_compare_head:
                    compare_in = h_alpha - h_beta
                else:
                    compare_in = torch.cat([h_alpha, h_beta], dim=-1)  # (B, 2d)
            if self.gap_head is not None:
                logits = self.gap_head(compare_in)  # (B, 1) scalar gap pred
            else:
                logits = self.compare_head(compare_in)  # (B, 3)

            # V17: Multi-modular aux head — binary classification per modulus
            if self.multi_modular_aux_head is not None:
                aux_block_logits.append(self.multi_modular_aux_head(compare_in))

            # V20 / V22: Build virtual β scratch via the same write_head +
            # scratch_pos_emb path used for α (h_beta_pfc broadcast over W
            # positions, then write_head + clip + quantize STE). Always run
            # when pfc_at_write is on; downstream heads / losses consume it.
            if (self._scratch_alpha_for_aux is not None
                    and self.agent_cfg.pfc_at_write):
                Bb = h_beta_pfc.shape[0]
                if isinstance(self.write_head, SequentialWriteHead):
                    # V25: sequential rollout on h_beta_pfc directly.
                    raw_beta_pre, raw_beta_q = self.write_head(
                        h_beta_pfc,
                        self.agent_cfg.quantize_levels,
                        self.agent_cfg.quantize_range,
                    )  # both (B, W)
                else:
                    pos_tokens = self.scratch_pos_emb.unsqueeze(0).expand(Bb, -1, -1)
                    beta_w_tokens = h_beta_pfc.unsqueeze(1) + pos_tokens  # (B, W, d)
                    raw_beta_pre = self.write_head(beta_w_tokens)  # (B, W, sig_dim=1)
                    if self.agent_cfg.write_head_clip > 0:
                        cclip = self.agent_cfg.write_head_clip
                        raw_beta_pre = cclip * torch.tanh(raw_beta_pre / cclip)
                    raw_beta_pre = raw_beta_pre.squeeze(-1)  # (B, W) pre-quantize
                    if self.agent_cfg.quantize_levels is not None:
                        raw_beta_q = _quantize_ste(
                            raw_beta_pre,
                            self.agent_cfg.quantize_levels,
                            self.agent_cfg.quantize_range,
                        )
                    else:
                        raw_beta_q = raw_beta_pre
                # V22: stash both forms for e_pair consistency + successor loss.
                scratch_beta_raw_blocks.append(raw_beta_pre)
                scratch_beta_q_blocks.append(raw_beta_q)
                # V20 scratch_mod_aux head consumes the quantized diff.
                if (self.scratch_mod_aux_head is not None
                        and scratch_aux_block_logits is not None):
                    scratch_diff_k = self._scratch_alpha_for_aux - raw_beta_q
                    scratch_aux_block_logits.append(
                        self.scratch_mod_aux_head(scratch_diff_k)
                    )

            # A5: ComparePredictor — predict compare_logits from (h_α, h_β, scratch).
            # Self-consistency loss vs detached actual logits (no gradient through
            # compare_head from this aux loss).
            if self.compare_predictor is not None:
                pred_logits = self.compare_predictor(
                    h_alpha, h_beta, scratch_flat_full
                )
                compare_pred_losses.append(
                    F.mse_loss(pred_logits, logits.detach())
                )
                # COLLAPSE FIX: VICReg variance on actual logits — prevents
                # compare_head from collapsing to constant output.
                if self.agent_cfg.vicreg_lambda > 0.0:
                    compare_vicreg_losses.append(
                        vicreg_variance_loss(
                            logits.reshape(-1, logits.shape[-1]),
                            gamma=self.agent_cfg.vicreg_gamma,
                        )
                    )

            compare_logits_list.append(logits)
            hidden_at_compares_list.append(h_beta)
            # 51_M5: store h_sp for cycle loss (post-PFC if PFC active, else h_α)
            if self.agent_cfg.cycle_loss_lambda > 0.0:
                if self.pfc is not None:
                    h_sp_per_block.append(h_alpha_pfc)
                else:
                    h_sp_per_block.append(h_alpha)

        # Aggregate aux losses across K blocks.
        if beta_pred_losses:
            self._last_beta_pred_loss = torch.stack(beta_pred_losses).mean()
        else:
            self._last_beta_pred_loss = None
        if compare_pred_losses:
            self._last_compare_pred_loss = torch.stack(compare_pred_losses).mean()
        else:
            self._last_compare_pred_loss = None
        if cf_beta_pred_losses:
            self._last_cf_beta_pred_loss = torch.stack(cf_beta_pred_losses).mean()
        else:
            self._last_cf_beta_pred_loss = None
        # COLLAPSE FIX: aggregate VICReg losses
        if beta_vicreg_losses:
            self._last_beta_vicreg = torch.stack(beta_vicreg_losses).mean()
        else:
            self._last_beta_vicreg = None
        if compare_vicreg_losses:
            self._last_compare_vicreg = torch.stack(compare_vicreg_losses).mean()
        else:
            self._last_compare_vicreg = None

        # 51_M5: L_cycle = MSE(h_sp_avg, sg(h_alpha))
        if (self.agent_cfg.cycle_loss_lambda > 0.0
                and self._last_h_alpha_for_cycle is not None
                and h_sp_per_block):
            h_sp_avg = torch.stack(h_sp_per_block, dim=1).mean(dim=1)  # (B, d)
            self._last_cycle_loss = F.mse_loss(
                h_sp_avg, self._last_h_alpha_for_cycle.detach()
            )
        else:
            self._last_cycle_loss = None

        # ===== P1 (L1579): Dual-loss imagination architecture =====
        if (self.world_signal_predictor is not None
                and self.agent_cfg.dual_loss_enabled):
            t_p = self.agent_cfg.dual_pred_t_p
            # Step 1: predict T_p future raw signal scalars from end-of-α latent
            h_alpha_end = h_A[:, L - 1, :]                              # (B, d)
            pred_signal = self.world_signal_predictor(h_alpha_end)      # (B, T_p)

            # Step 2: random β block selection
            i_rand = int(torch.randint(0, K, (1,)).item())
            rstart_d, beta_start_d, compare_step_d = cfg.compare_block_phases(i_rand)
            actual_beta_signal = inputs[:, beta_start_d:beta_start_d + t_p, 0]  # (B, T_p)

            # Step 3: L_world = MSE(predicted, actual β signal)
            self._last_dual_world_loss = F.mse_loss(pred_signal, actual_beta_signal)

            # Step 4: build imagined input — append predicted signal to α
            # Use phase markers from positions L..L+T_p-1 of original input.
            imag_template = inputs[:, L:L + t_p, :].clone()             # (B, T_p, 3)
            imag_template[:, :, 0] = pred_signal
            combined = torch.cat([inputs[:, :L, :], imag_template], dim=1)  # (B, L+T_p, 3)

            # Step 5: re-scan combined sequence to get imagined Mamba latents
            # Use the imagined positions L..L+W-1 (right after imagined-future ends?)
            # Actually L1579 says h_wp' = scan(h_world_pred'). The write timesteps
            # in original input are L..L+W-1. After concat, those are at the same
            # positions in combined (since combined[:, :L, :] = α). So h_at_writes
            # = h_wp_full[:, L:L+W, :] — but L..L+W-1 is now BLANK in combined
            # (we only filled L..L+T_p-1 with predictions). Use end-of-imagined.
            # Better interpretation: imagined writes happen at end of imagined scan
            # = positions (L + t_p - W) .. (L + t_p - 1).
            h_wp_full = self._encode_segment(combined)                  # (B, L+T_p, d)
            write_start = max(L, L + t_p - W)
            h_at_imag_writes = h_wp_full[:, write_start:write_start + W, :]  # (B, W, d)

            # Step 6: imagined scratch via shared write logic
            h_sp_imag = self._run_write_loop(h_at_imag_writes, W)       # (B, W, sig_dim)

            # Step 7: actual β path — encode β segment + write to get h_sp_β
            seg_β = inputs[:, beta_start_d:beta_start_d + L + W, :]
            # Bound by episode length to avoid out-of-bounds
            avail_len = inputs.shape[1] - beta_start_d
            if avail_len >= L + W:
                seg_β_full = inputs[:, beta_start_d:beta_start_d + L + W, :]
                h_β_scan = self._encode_segment(seg_β_full)
                h_at_β_writes = h_β_scan[:, L:L + W, :]
                h_sp_β = self._run_write_loop(h_at_β_writes, W)         # (B, W, sig_dim)
                # Step 8: L_sp (detach β-side so β-write isn't perturbed)
                self._last_dual_sp_loss = F.mse_loss(h_sp_imag, h_sp_β.detach())
            else:
                # Episode too short for full β + W write window; skip L_sp
                self._last_dual_sp_loss = h_A.new_zeros(())
        else:
            self._last_dual_world_loss = None
            self._last_dual_sp_loss = None

        compare_logits = torch.stack(compare_logits_list, dim=1)  # (B, K, 3)
        hidden_at_compares = torch.stack(hidden_at_compares_list, dim=1)  # (B,K,d)
        # V17: stash aux mod logits (B, K, n_moduli) for trainer loss computation
        if aux_block_logits:
            self._last_aux_mod_logits = torch.stack(aux_block_logits, dim=1)
        else:
            self._last_aux_mod_logits = None
        # V20: stash scratch_mod_aux logits (B, K, n_mod_scratch)
        if scratch_aux_block_logits:
            self._last_scratch_aux_mod_logits = torch.stack(
                scratch_aux_block_logits, dim=1,
            )
        else:
            self._last_scratch_aux_mod_logits = None
        # V22: stash virtual β scratch tensors (B, K, W) — both pre- and
        # post-quantize. Trainer reads these for E-pair consistency loss
        # (post-quantize) and successor loss (pre-quantize input, post-quantize
        # detached target).
        if scratch_beta_raw_blocks:
            self._last_scratch_beta_K_raw = torch.stack(scratch_beta_raw_blocks, dim=1)
            self._last_scratch_beta_K_q = torch.stack(scratch_beta_q_blocks, dim=1)
        else:
            self._last_scratch_beta_K_raw = None
            self._last_scratch_beta_K_q = None
        # β-1: when scratch is vectorial (d_z>1), return VQ indices (B, W) as
        # the analyzable scratch_pad for downstream metrics. Otherwise legacy.
        if self.vq is not None and self._last_vq_indices is not None:
            scratch_pad = self._last_vq_indices.float()  # (B, W) — discrete indices
        else:
            scratch_pad = torch.stack(scratch_list, dim=1).squeeze(-1)  # (B, W)
        return compare_logits, scratch_pad, hidden_at_compares
