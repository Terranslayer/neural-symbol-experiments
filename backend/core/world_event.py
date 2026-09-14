"""Event-driven world for the active-agent paradigm (Stage 2 Phase 5+).

Unlike scene.py which produces a fixed (T, 3) tensor rolling out every phase,
this module provides a *live* world: agent chooses when/where to attend,
external compare-queries arrive as events, and a "life" contains many such
sessions with persistent agent hidden state + notebook.

Design principles (from 2026-04-24 design session):
  - World is a 1D continuous float array (same-medium basis)
  - Agent observes only a local window of width w; saccades clip at edges
  - Each session: one world + one compare-query target; agent emits an answer
    before a timeout; reward is sparse (+1 correct, -1 wrong, -1 timeout)
  - Session-to-session: new randomized world + new compare-query, but agent's
    hidden state and notebook PERSIST across sessions (that is the life)
  - Compare-query answers within a difficulty bucket are shuffled so adjacent
    sessions never have the same true answer (defeats short-term memorization)

This is the prior-free complement to scene.py. scene.py stays in use for the
supervised experiments (S1-S2P4); world_event.py is for the RL-active-agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from backend.core.scene import (
    ALPHA_EQUAL,
    ALPHA_GREATER,
    ALPHA_LESS,
    _scan_world_complex,
    count_foods,
)


# Notebook values are snapped to 6 discrete levels in [0, 1] so their precision
# is coarser than world observations (which are continuous floats). This is the
# "same medium, coarser precision" design constraint (2026-04-24 decision 3).
NOTE_LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)


def quantize_note_value(v: float) -> float:
    """Snap v to nearest of 6 allowed notebook levels."""
    clipped = max(0.0, min(1.0, float(v)))
    return float(NOTE_LEVELS[int(np.argmin(np.abs(NOTE_LEVELS - clipped)))])


@dataclass
class WorldConfig:
    """Static world parameters for a life."""

    L: int = 50           # world length
    window_w: int = 5     # agent's local observation window
    # n_mid_noise_range and n_distractors_range reuse the S2-P3 complex-world
    # generator; pass None to let the generator pick uniform random values.
    n_mid_noise_range: Tuple[int, int] = (3, 5)
    n_distractors_range: Tuple[int, int] = (0, 2)
    # When auto_scan=True, attn_pos advances by auto_scan_step each tick
    # regardless of action, and saccade actions are ignored. New sessions
    # start with attn_pos=0 so the agent sees the world left-to-right in
    # disjoint window_w chunks.
    auto_scan: bool = False
    auto_scan_step: int = 5
    # Three-phase session timing. Total t_budget = t_observe + t_gap + t_answer.
    # During observe: real world window. During gap: blank window (latent decay).
    # During answer: blank window but agent may emit. If phased=False (default,
    # backward compat) sessions are single-phase and emit is allowed anytime.
    phased: bool = False
    t_observe: int = 50
    t_gap: int = 10
    t_answer: int = 20
    # Timeout penalty. Was -0.5 (milder than wrong-emit -1.0) but we found
    # the agent exploited the mildness + target imbalance by "emit-less on
    # easy cases, timeout on hard cases where greater is needed". Pushing
    # this to -2.0 makes timeout the worst outcome by far.
    timeout_penalty: float = -0.5


@dataclass
class NBucket:
    """One stratum of the N-distribution for pool generation."""
    lo: int
    hi: int                       # inclusive
    weight: float = 1.0           # relative sampling weight


def save_world_pool(
    path, worlds: List[np.ndarray], Ns: List[int], metadata: dict,
) -> None:
    """Persist a world pool to disk as .npz + manifest .json."""
    import json
    path = str(path)
    arr = np.stack(worlds, axis=0).astype(np.float32)    # (pool, L)
    Ns_arr = np.asarray(Ns, dtype=np.int32)
    np.savez_compressed(path, worlds=arr, Ns=Ns_arr)
    meta_path = path.replace(".npz", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def load_world_pool(path) -> Tuple[List[np.ndarray], List[int], dict]:
    """Load a persisted pool. Returns (worlds, Ns, metadata)."""
    import json
    path = str(path)
    data = np.load(path)
    worlds = [arr for arr in data["worlds"]]
    Ns = data["Ns"].tolist()
    meta_path = path.replace(".npz", ".meta.json")
    meta = {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        pass
    return worlds, Ns, meta


def generate_world_pool(
    world_cfg: WorldConfig,
    buckets: List[NBucket],
    pool_size: int,
    rng: np.random.Generator,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Pre-generate a pool of worlds with N drawn from stratified buckets.

    Returns (worlds, Ns): two parallel lists of length pool_size. Callers sample
    with replacement during training; same world may recur many times, but the
    compare-query target is re-randomized each session so answers vary.
    """
    weights = np.array([b.weight for b in buckets], dtype=np.float64)
    weights /= weights.sum()
    worlds: List[np.ndarray] = []
    Ns: List[int] = []
    for i in range(pool_size):
        bidx = int(rng.choice(len(buckets), p=weights))
        b = buckets[bidx]
        N = int(rng.integers(b.lo, b.hi + 1))
        world = _scan_world_complex(N, world_cfg.L, rng)
        worlds.append(world)
        Ns.append(N)
        if verbose and (i + 1) % max(1, pool_size // 10) == 0:
            print(f"  pool gen: {i + 1}/{pool_size}")
    return worlds, Ns


@dataclass
class Session:
    """One world + one compare-query + bookkeeping.

    Attributes
    ----------
    world_signal
        The full 1D array the agent can saccade over (length L).
    alpha_count
        Ground-truth number of food items in the world.
    target_count
        The compare-query value (beta). Agent must emit >, <, = relative to alpha.
    label
        The true comparison class (0=greater, 1=less, 2=equal).
    t_budget
        Max ticks before timeout.
    """

    world_signal: np.ndarray
    alpha_count: int
    target_count: int
    label: int
    t_budget: int


def sample_session(
    world_cfg: WorldConfig,
    n_max: int,
    t_budget: int,
    rng: np.random.Generator,
) -> Session:
    """Build a fresh session: sample N in [1, n_max], build world, pick target."""
    alpha_count = int(rng.integers(1, n_max + 1))
    world_signal = _scan_world_complex(alpha_count, world_cfg.L, rng)
    if rng.random() < 0.3:
        target_count = alpha_count
    else:
        pool = [k for k in range(1, n_max + 1) if k != alpha_count]
        target_count = int(rng.choice(pool))
    if alpha_count > target_count:
        label = ALPHA_GREATER
    elif alpha_count < target_count:
        label = ALPHA_LESS
    else:
        label = ALPHA_EQUAL
    return Session(
        world_signal=world_signal.astype(np.float32),
        alpha_count=alpha_count,
        target_count=target_count,
        label=label,
        t_budget=t_budget,
    )


def sample_session_from_pool(
    pool_worlds: List[np.ndarray],
    pool_Ns: List[int],
    t_budget: int,
    rng: np.random.Generator,
) -> Session:
    """Pick a pre-generated world; pick a fresh target with BALANCED labels.

    The label (>, <, =) is drawn uniformly among those feasible for the pool's
    alpha value: 'greater' requires alpha > 1 (target exists), 'less' requires
    alpha < n_ceiling. Once the label is chosen, target is sampled from the
    matching slice. This defeats the 80%-less imbalance caused by uniform target
    sampling combined with stratified small-alpha pool.

    The world is COPIED so downstream mutations do not corrupt the pool.
    """
    idx = int(rng.integers(0, len(pool_worlds)))
    world_signal = pool_worlds[idx].copy()
    alpha_count = pool_Ns[idx]
    n_ceiling = max(pool_Ns)

    options: List[int] = [ALPHA_EQUAL]
    if alpha_count > 1:
        options.append(ALPHA_GREATER)
    if alpha_count < n_ceiling:
        options.append(ALPHA_LESS)
    label = int(rng.choice(options))

    if label == ALPHA_EQUAL:
        target_count = alpha_count
    elif label == ALPHA_GREATER:
        target_count = int(rng.integers(1, alpha_count))          # [1, alpha-1]
    else:
        target_count = int(rng.integers(alpha_count + 1, n_ceiling + 1))

    return Session(
        world_signal=world_signal,
        alpha_count=alpha_count,
        target_count=target_count,
        label=label,
        t_budget=t_budget,
    )


@dataclass
class LifeState:
    """Running state of a life that spans many sessions.

    Attention position is kept as a float for continuous saccade. Integer
    window slices use int(round(attn_pos)) clipped to valid range.
    """

    attn_pos: float = 0.0                  # continuous position (left edge of window)
    t: int = 0                             # tick counter within current session
    current_session: Optional[Session] = None
    past_pages: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    # Each page: (content_scalars[4], feedback_slot ∈ {-1=unanswered, 0=wrong, 1=right})
    active_page: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    active_page_target_slot: int = 0       # next write target within active_page


class WorldEngine:
    """Stateful world driver: holds the current session, handles saccade,
    emission, session transitions, and notebook updates.

    Agent interacts by calling:
      obs = engine.observe()
      reward, done = engine.step(action)
    """

    def __init__(
        self,
        world_cfg: WorldConfig,
        n_max: int,
        t_budget: int,
        notebook_capacity: int,
        rng: np.random.Generator,
        pool_worlds: Optional[List[np.ndarray]] = None,
        pool_Ns: Optional[List[int]] = None,
    ):
        self.world_cfg = world_cfg
        self.n_max = n_max
        self.t_budget = t_budget
        self.notebook_capacity = notebook_capacity
        self.rng = rng
        # If a pool is provided, _start_new_session samples from it instead of
        # generating on the fly. This avoids per-session Python CPU cost on the
        # hot loop (e.g., large L=1500 worlds generating per tick transition).
        self.pool_worlds = pool_worlds
        self.pool_Ns = pool_Ns
        # Normalizer for target_cue: use pool ceiling if pool-mode, else n_max.
        if pool_Ns:
            self._cue_norm = float(max(pool_Ns))
        else:
            self._cue_norm = float(max(1, n_max))
        self.state = LifeState()
        self._start_new_session()

    # ------------------------------------------------------------------ life

    def _start_new_session(self) -> None:
        # If phased, override t_budget to sum of phase durations.
        effective_budget = self.t_budget
        if self.world_cfg.phased:
            effective_budget = (
                self.world_cfg.t_observe
                + self.world_cfg.t_gap
                + self.world_cfg.t_answer
            )
        if self.pool_worlds is not None and self.pool_Ns is not None:
            self.state.current_session = sample_session_from_pool(
                self.pool_worlds, self.pool_Ns, effective_budget, self.rng
            )
        else:
            self.state.current_session = sample_session(
                self.world_cfg, self.n_max, effective_budget, self.rng
            )
        self.state.t = 0
        # Under auto-scan mode the agent sees the world from the left edge; the
        # env advances attn_pos each tick, so starting at 0 covers the whole
        # world in (L-w)/step+1 ticks. Without auto-scan, start in the middle
        # so the agent can saccade either direction.
        if self.world_cfg.auto_scan:
            self.state.attn_pos = 0.0
        else:
            self.state.attn_pos = float(self.world_cfg.L // 2)
        self.state.active_page = np.zeros(4, dtype=np.float32)
        self.state.active_page_target_slot = 0

    # ------------------------------------------------------------- observe

    def _current_phase(self) -> str:
        """Return 'observe' / 'gap' / 'answer' based on self.state.t."""
        if not self.world_cfg.phased:
            return "observe"
        t = self.state.t
        if t < self.world_cfg.t_observe:
            return "observe"
        if t < self.world_cfg.t_observe + self.world_cfg.t_gap:
            return "gap"
        return "answer"

    def observe(self) -> dict:
        """Return the agent's current observation: local window + phase info.

        Under phased=True, window is only real during the observe phase; during
        gap/answer it is all zeros. A phase_cue scalar tells agent where it is:
        0.0 observe, 0.5 gap, 1.0 answer.
        """
        w = self.world_cfg.window_w
        L = self.world_cfg.L
        phase = self._current_phase()
        if phase == "observe":
            left = int(max(0, min(L - w, round(self.state.attn_pos))))
            window = self.state.current_session.world_signal[left:left + w].copy()
        else:
            window = np.zeros(w, dtype=np.float32)
        phase_cue = {"observe": 0.0, "gap": 0.5, "answer": 1.0}[phase]
        return {
            "window": window,
            "target_cue": float(self.state.current_session.target_count) / self._cue_norm,
            "attn_pos_normalized": self.state.attn_pos / float(max(1, L - w)),
            "active_page": self.state.active_page.copy(),
            "t_remaining_fraction": 1.0 - self.state.t / float(self.state.current_session.t_budget),
            "phase_cue": phase_cue,
        }

    # ---------------------------------------------------------------- step

    def step(self, action: "ActionRecord") -> Tuple[float, bool]:
        """Execute the action, advance time, return (reward, done).

        done=True when the session ends (EMIT or timeout). Upon done, engine
        auto-starts the next session; callers should re-observe after done.
        """
        self.state.t += 1
        reward = 0.0
        done = False

        kind = action.kind
        if kind == "saccade":
            if not self.world_cfg.auto_scan:
                L, w = self.world_cfg.L, self.world_cfg.window_w
                self.state.attn_pos = float(
                    max(0.0, min(float(L - w), self.state.attn_pos + action.saccade_delta))
                )
            # else: auto-scan mode ignores saccade actions entirely.
        elif kind == "think":
            pass  # agent-internal recurrence handled in the policy; world idle
        elif kind == "write":
            # Agent can only write to the 4 content slots (0..3); slot 4 is
            # feedback, engine-controlled. Value is snapped to the 6-level
            # lattice so notebook precision is coarser than world.
            slot = int(action.write_slot) % 4
            self.state.active_page[slot] = quantize_note_value(action.write_value)
            self.state.active_page_target_slot = (slot + 1) % 4
        elif kind == "read_page":
            # No world change; policy will read past_pages[action.page_index]
            # directly from life state on the next observe(). We still consume
            # a tick so reading is not free.
            pass
        elif kind == "emit":
            # Under phased mode, emit is only honored during answer phase;
            # otherwise it wastes a tick (the policy receives no reward).
            phase_now = self._current_phase()
            if self.world_cfg.phased and phase_now != "answer":
                pass  # silently no-op; agent burned a tick
            else:
                correct = int(action.emit_class == self.state.current_session.label)
                reward = 1.0 if correct else -1.0
                feedback = 1.0 if correct else 0.0
                self._commit_active_page(feedback)
                done = True
        elif kind == "noop":
            pass
        else:
            raise ValueError(f"Unknown action kind: {kind!r}")

        # Auto-scan: env advances attn_pos each tick (after action dispatch)
        if self.world_cfg.auto_scan and not done:
            L, w = self.world_cfg.L, self.world_cfg.window_w
            self.state.attn_pos = float(
                min(float(L - w), self.state.attn_pos + self.world_cfg.auto_scan_step)
            )

        if not done and self.state.t >= self.state.current_session.t_budget:
            # Timeout penalty is configurable via WorldConfig. Stronger penalty
            # forces the policy to EMIT (even wrongly) rather than waste the
            # budget silent.
            reward = self.world_cfg.timeout_penalty
            self._commit_active_page(0.0)
            done = True

        if done:
            self._start_new_session()

        return reward, done

    # ------------------------------------------------------------- notebook

    def _commit_active_page(self, feedback_value: float) -> None:
        """Push the just-filled active page onto past_pages, evicting oldest
        if capacity exceeded (FIFO)."""
        page = np.concatenate(
            [self.state.active_page, np.array([feedback_value], dtype=np.float32)]
        )
        self.state.past_pages.append((page, feedback_value))
        if len(self.state.past_pages) > self.notebook_capacity:
            self.state.past_pages.pop(0)

    def read_page(self, idx: int) -> np.ndarray:
        """Fetch a past page by index; returns zeros if out of range."""
        if 0 <= idx < len(self.state.past_pages):
            return self.state.past_pages[idx][0]
        return np.zeros(5, dtype=np.float32)


@dataclass
class ActionRecord:
    """Canonical action description consumed by WorldEngine.step."""

    kind: str                                 # 'saccade'|'think'|'write'|'read_page'|'emit'|'noop'
    saccade_delta: float = 0.0
    write_slot: int = 0
    write_value: float = 0.0
    read_index: int = 0
    emit_class: int = 0                       # 0=greater, 1=less, 2=equal
