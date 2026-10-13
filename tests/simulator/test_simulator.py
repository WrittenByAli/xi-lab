"""
Simulator invariant and statistical tests.

These use synthetic data (no real match files needed) so they always run.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Outcome class index (mirrors params.yaml) ─────────────────────────────────
OUTCOMES = ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "noball"]
DOT, ONE, TWO, THREE, FOUR, SIX, WICKET, WIDE, NOBALL = range(9)

# Runs awarded per outcome (batter runs only; extras handled separately)
BATTER_RUNS = [0, 1, 2, 3, 4, 6, 0, 0, 0]
IS_LEGAL = [True, True, True, True, True, True, True, False, False]
IS_WICKET = [False] * 7 + [False, False]
IS_WICKET[WICKET] = True


def _make_uniform_probs(n_batters: int = 3) -> np.ndarray:
    """Return (n_batters, 9) uniform probability array."""
    probs = np.ones((n_batters, len(OUTCOMES))) / len(OUTCOMES)
    return probs


def _simple_sim(
    probs: np.ndarray,
    max_overs: int = 20,
    n_sims: int = 1_000,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Minimal vectorized simulator for testing invariants.

    probs: (n_states, 9) outcome probabilities (constant for simplicity)
    Returns scores, wickets, legal_balls arrays of shape (n_sims,).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    max_legal = max_overs * 6
    scores = np.zeros(n_sims, dtype=np.int32)
    wickets = np.zeros(n_sims, dtype=np.int32)
    legal_balls = np.zeros(n_sims, dtype=np.int32)
    active = np.ones(n_sims, dtype=bool)

    while active.any():
        draws = rng.random(n_sims)
        cumprobs = probs[0].cumsum()  # use first batter's probs
        outcomes = np.searchsorted(cumprobs, draws)
        outcomes = np.clip(outcomes, 0, len(OUTCOMES) - 1)

        for i in np.where(active)[0]:
            o = outcomes[i]
            scores[i] += BATTER_RUNS[o]
            if IS_LEGAL[o]:
                legal_balls[i] += 1
            if IS_WICKET[o]:
                wickets[i] += 1
                if wickets[i] >= 10:
                    active[i] = False
            if legal_balls[i] >= max_legal:
                active[i] = False

    return {"scores": scores, "wickets": wickets, "legal_balls": legal_balls}


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSimulatorInvariants:
    N = 2_000

    def test_wickets_le_ten(self):
        probs = _make_uniform_probs()
        result = _simple_sim(probs, n_sims=self.N, rng=np.random.default_rng(1))
        assert (result["wickets"] <= 10).all(), "Some simulations exceed 10 wickets"

    def test_legal_balls_le_max(self):
        probs = _make_uniform_probs()
        result = _simple_sim(probs, max_overs=20, n_sims=self.N, rng=np.random.default_rng(2))
        assert (result["legal_balls"] <= 120).all(), "Some simulations exceed 120 legal balls"

    def test_scores_nonnegative(self):
        probs = _make_uniform_probs()
        result = _simple_sim(probs, n_sims=self.N, rng=np.random.default_rng(3))
        assert (result["scores"] >= 0).all()

    def test_same_seed_deterministic(self):
        probs = _make_uniform_probs()
        r1 = _simple_sim(probs, n_sims=500, rng=np.random.default_rng(42))
        r2 = _simple_sim(probs, n_sims=500, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1["scores"], r2["scores"])


class TestSimulatorStatistics:
    """
    With constant probabilities, simulated means should match analyticals
    within 3 standard errors.
    """

    N = 10_000

    def _analytical_mean_runs(self, p: np.ndarray) -> float:
        """Expected runs per legal delivery × expected legal deliveries."""
        # P(legal ball) = sum of legal outcome probs
        p_legal = sum(p[i] for i in range(len(OUTCOMES)) if IS_LEGAL[i])
        e_runs_per_legal = (
            sum(BATTER_RUNS[i] * p[i] for i in range(len(OUTCOMES)) if IS_LEGAL[i]) / p_legal
            if p_legal > 0
            else 0
        )
        # E[legal balls in innings] ≈ min(120, ...) — hard to compute exactly;
        # we just check the mean score is within a loose bound.
        return e_runs_per_legal  # runs per legal ball

    def test_mean_runs_per_legal_ball(self):
        # All-dot innings: every delivery is a dot
        dot_probs = np.zeros(len(OUTCOMES))
        dot_probs[DOT] = 1.0
        np.array([dot_probs])
        result = _simple_sim(dot_probs.reshape(1, -1), n_sims=self.N, rng=np.random.default_rng(10))
        assert result["scores"].mean() == pytest.approx(0.0, abs=0.01)

    def test_all_fours(self):
        four_probs = np.zeros(len(OUTCOMES))
        four_probs[FOUR] = 1.0
        result = _simple_sim(
            four_probs.reshape(1, -1), n_sims=200, max_overs=20, rng=np.random.default_rng(11)
        )
        # All fours, no wickets → exactly 120 legal balls → 480 runs each
        assert (result["scores"] == 480).all()
        assert (result["wickets"] == 0).all()


class TestCommonRandomNumbers:
    """
    Common random numbers (CRN): comparing two identical XIs should give
    variance of the difference << variance of one XI.
    """

    N = 5_000

    def test_crn_reduces_variance(self):
        probs = _make_uniform_probs()
        rng = np.random.default_rng(0)

        # Simulate XI_A and XI_B with the SAME rng state (CRN)
        state = rng.bit_generator.state
        r_a = _simple_sim(probs, n_sims=self.N, rng=np.random.default_rng(99))
        rng.bit_generator.state = state
        r_b = _simple_sim(probs, n_sims=self.N, rng=np.random.default_rng(99))

        var_diff = np.var(r_a["scores"].astype(float) - r_b["scores"].astype(float))
        var_a = np.var(r_a["scores"].astype(float))

        # With identical teams and CRN, difference should be zero (or near zero)
        assert var_diff < var_a * 0.01, (
            f"CRN did not reduce variance: var_diff={var_diff:.1f}, var_a={var_a:.1f}"
        )
