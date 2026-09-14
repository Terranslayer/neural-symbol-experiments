"""Tests for backend/core/world_event.py (Stage 2 Phase 5+ active-agent world)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.core.world_event import (
    ActionRecord,
    Session,
    WorldConfig,
    WorldEngine,
    sample_session,
)


@pytest.fixture
def rng():
    return np.random.default_rng(seed=0)


@pytest.fixture
def world_cfg():
    return WorldConfig(L=50, window_w=5)


class TestSession:
    def test_session_built_correctly(self, world_cfg, rng):
        s = sample_session(world_cfg, n_max=10, t_budget=50, rng=rng)
        assert s.world_signal.shape == (50,)
        assert 1 <= s.alpha_count <= 10
        assert 1 <= s.target_count <= 10
        # label consistency
        if s.alpha_count > s.target_count:
            assert s.label == 0
        elif s.alpha_count < s.target_count:
            assert s.label == 1
        else:
            assert s.label == 2

    def test_no_forbidden_prior_leak(self, world_cfg, rng):
        """Observation must NOT expose alpha_count directly to the agent."""
        engine = WorldEngine(world_cfg, n_max=10, t_budget=50,
                             notebook_capacity=4, rng=rng)
        obs = engine.observe()
        # The agent never sees alpha_count. target_cue exposes only target.
        assert "alpha_count" not in obs
        assert "world_signal" not in obs
        assert set(obs.keys()) == {"window", "target_cue", "attn_pos_normalized",
                                    "active_page", "t_remaining_fraction", "phase_cue"}


class TestEngineActions:
    def test_saccade_clips(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        # Try to saccade way past the right edge
        r, d = eng.step(ActionRecord(kind="saccade", saccade_delta=1000.0))
        assert not d
        # Position should clip to L - window_w = 50 - 5 = 45
        assert eng.state.attn_pos == 45.0
        # And way past left
        r, d = eng.step(ActionRecord(kind="saccade", saccade_delta=-1000.0))
        assert eng.state.attn_pos == 0.0

    def test_write_quantizes_to_lattice(self, world_cfg, rng):
        """Values are snapped to 6 levels {0, 0.2, 0.4, 0.6, 0.8, 1.0}."""
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        eng.step(ActionRecord(kind="write", write_slot=2, write_value=0.75))
        # 0.75 rounds to 0.8
        assert eng.state.active_page[2] == pytest.approx(0.8, abs=1e-6)

        eng.step(ActionRecord(kind="write", write_slot=0, write_value=0.09))
        # 0.09 rounds to 0.0
        assert eng.state.active_page[0] == pytest.approx(0.0, abs=1e-6)

    def test_write_cannot_touch_feedback_slot(self, world_cfg, rng):
        """Active page has 4 content slots; slot index wraps so agent cannot
        reach the feedback slot which lives separately in committed pages."""
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        eng.step(ActionRecord(kind="write", write_slot=4, write_value=0.99))
        # slot=4 wraps to slot 0; 0.99 quantizes to 1.0.
        assert eng.state.active_page[0] == pytest.approx(1.0, abs=1e-6)

    def test_emit_correct_gives_plus_one(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        true_label = eng.state.current_session.label
        r, d = eng.step(ActionRecord(kind="emit", emit_class=true_label))
        assert r == 1.0
        assert d is True

    def test_emit_wrong_gives_minus_one(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        true_label = eng.state.current_session.label
        wrong = (true_label + 1) % 3
        r, d = eng.step(ActionRecord(kind="emit", emit_class=wrong))
        assert r == -1.0
        assert d is True

    def test_emit_starts_new_session(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        world0 = eng.state.current_session.world_signal.copy()
        eng.step(ActionRecord(kind="emit", emit_class=0))
        world1 = eng.state.current_session.world_signal
        # Sessions are sampled fresh so worlds should differ
        assert not np.array_equal(world0, world1)
        # Timer reset
        assert eng.state.t == 0

    def test_timeout_penalty_from_cfg(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=3,
                          notebook_capacity=4, rng=rng)
        eng.step(ActionRecord(kind="noop"))
        eng.step(ActionRecord(kind="noop"))
        r, d = eng.step(ActionRecord(kind="noop"))
        assert r == world_cfg.timeout_penalty
        assert d is True


class TestNotebook:
    def test_emit_commits_page_with_feedback(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        # Fill the active page deliberately
        for slot, val in enumerate([0.2, 0.4, 0.6, 0.8]):
            eng.step(ActionRecord(kind="write", write_slot=slot, write_value=val))
        true_label = eng.state.current_session.label
        eng.step(ActionRecord(kind="emit", emit_class=true_label))
        assert len(eng.state.past_pages) == 1
        page, feedback = eng.state.past_pages[0]
        assert page.shape == (5,)
        # Content slots preserved
        np.testing.assert_allclose(page[:4], [0.2, 0.4, 0.6, 0.8])
        # Feedback = 1.0 for correct answer
        assert page[4] == pytest.approx(1.0)
        assert feedback == pytest.approx(1.0)

    def test_wrong_emit_feedback_is_zero(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        true_label = eng.state.current_session.label
        wrong = (true_label + 1) % 3
        eng.step(ActionRecord(kind="emit", emit_class=wrong))
        page, feedback = eng.state.past_pages[0]
        assert page[4] == pytest.approx(0.0)
        assert feedback == pytest.approx(0.0)

    def test_capacity_fifo_eviction(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=3,
                          notebook_capacity=2, rng=rng)
        # Drive 5 timeouts — 5*t_budget = 15 noops will produce exactly 5
        # session completions (engine resets t on done, so counting ticks not
        # sessions).
        sessions_done = 0
        while sessions_done < 5:
            _, done = eng.step(ActionRecord(kind="noop"))
            if done:
                sessions_done += 1
        assert len(eng.state.past_pages) == 2

    def test_read_page_fetches_correctly(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        eng.step(ActionRecord(kind="write", write_slot=0, write_value=0.3))
        true_label = eng.state.current_session.label
        eng.step(ActionRecord(kind="emit", emit_class=true_label))
        page = eng.read_page(0)
        assert page.shape == (5,)
        # 0.3 quantized. Due to float repr (0.3 > 0.2 + 0.1 by machine eps),
        # np.argmin picks 0.4 rather than 0.2.
        assert page[0] == pytest.approx(0.4)

    def test_read_out_of_range_returns_zeros(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=4, rng=rng)
        page = eng.read_page(99)
        np.testing.assert_array_equal(page, np.zeros(5, dtype=np.float32))


class TestPersistence:
    def test_notebook_persists_across_sessions(self, world_cfg, rng):
        eng = WorldEngine(world_cfg, n_max=5, t_budget=100,
                          notebook_capacity=10, rng=rng)
        eng.step(ActionRecord(kind="write", write_slot=0, write_value=0.5))
        eng.step(ActionRecord(kind="emit", emit_class=0))
        first_n_pages = len(eng.state.past_pages)
        # Do the next session
        eng.step(ActionRecord(kind="write", write_slot=1, write_value=0.7))
        eng.step(ActionRecord(kind="emit", emit_class=1))
        assert len(eng.state.past_pages) == first_n_pages + 1
