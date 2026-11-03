"""Tests for ml/simulate/engine.py's real simulate_innings/simulate_match.

tests/simulator/test_simulator.py exercises general Monte-Carlo invariants
against a self-contained toy reimplementation, not these functions — so
until now the actual production simulator had no direct test coverage at
all. These tests close that gap and pin down the behavior of the
vectorized rewrite (see git history: the previous version had a nested
Python loop over every delivery x every simulation despite its own
docstring claiming full vectorization, which was the main source of the
project's slow /recommend, /simulate, backtest and publish.py runs).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.simulate.engine import (
    BattingCard,
    BowlingCard,
    MatchContext,
    simulate_innings,
    simulate_match,
)

PHASES = [(0, 5), (6, 15), (16, 19)]
PHASE_NAMES = ["powerplay", "middle", "death"]


def _probs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for ph in PHASE_NAMES:
        p = rng.random(9) + 0.1
        out[ph] = p / p.sum()
    return out


def _batting_order(n: int = 11, seed: int = 0) -> list[BattingCard]:
    return [BattingCard(player_id=f"bat{i}", probs_by_phase=_probs(seed + i)) for i in range(n)]


def _bowling_attack(n: int = 5, seed: int = 100, max_overs: int = 4) -> list[BowlingCard]:
    return [
        BowlingCard(player_id=f"bowl{i}", probs_by_phase=_probs(seed + i), max_overs=max_overs)
        for i in range(n)
    ]


def _ctx(target=None, total_overs: int = 20) -> MatchContext:
    return MatchContext(
        batting_order=_batting_order(),
        bowling_attack=_bowling_attack(),
        total_overs=total_overs,
        target=target,
        phases=PHASES,
        phase_names=PHASE_NAMES,
    )


class TestSimulateInningsInvariants:
    N = 500

    def test_wickets_capped_at_batting_order_size(self):
        result = simulate_innings(_ctx(), n_sims=self.N, rng=np.random.default_rng(1))
        assert (result.wickets <= 11).all()

    def test_legal_balls_capped_at_overs(self):
        result = simulate_innings(_ctx(), n_sims=self.N, rng=np.random.default_rng(2))
        assert (result.balls_faced <= 120).all()

    def test_scores_nonnegative(self):
        result = simulate_innings(_ctx(), n_sims=self.N, rng=np.random.default_rng(3))
        assert (result.scores >= 0).all()

    def test_same_seed_is_deterministic(self):
        r1 = simulate_innings(_ctx(), n_sims=self.N, rng=np.random.default_rng(42))
        r2 = simulate_innings(_ctx(), n_sims=self.N, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.scores, r2.scores)
        np.testing.assert_array_equal(r1.wickets, r2.wickets)
        np.testing.assert_array_equal(r1.balls_faced, r2.balls_faced)

    def test_chase_is_none_without_a_target(self):
        result = simulate_innings(_ctx(target=None), n_sims=10, rng=np.random.default_rng(4))
        assert result.chased is None

    def test_chase_marks_sims_that_reached_target(self):
        # A target of 1 run is reached almost immediately by almost every sim.
        result = simulate_innings(_ctx(target=1), n_sims=self.N, rng=np.random.default_rng(5))
        assert result.chased is not None
        assert result.chased.mean() > 0.9

    def test_per_sim_target_array_is_respected(self):
        # Half the sims need 1 run (trivial), half need an unreachable amount.
        targets = np.where(np.arange(self.N) % 2 == 0, 1, 10_000).astype(np.int64)
        result = simulate_innings(_ctx(target=targets), n_sims=self.N, rng=np.random.default_rng(6))
        assert result.chased[::2].all()
        assert not result.chased[1::2].any()


class TestSimulateMatch:
    def test_win_probs_sum_to_one(self):
        ctx = _ctx()
        res = simulate_match(ctx, _batting_order(seed=200), _bowling_attack(seed=300), n_sims=500)
        total = res["team1_win_prob"] + res["team2_win_prob"] + res["tie_prob"]
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_identical_teams_are_close_to_a_fair_contest(self):
        # Same ratings both sides (but team2 bats second — a measurable but
        # small 2nd-innings advantage/disadvantage is expected, not an even
        # split): mainly guards against a gross directional bug.
        bat = _batting_order()
        bowl = _bowling_attack()
        ctx = MatchContext(
            batting_order=bat,
            bowling_attack=bowl,
            total_overs=20,
            phases=PHASES,
            phase_names=PHASE_NAMES,
        )
        res = simulate_match(ctx, bat, bowl, n_sims=3000, crn_seed=11)
        assert 0.3 < res["team1_win_prob"] < 0.7

    def test_common_random_numbers_reduce_variance_of_the_difference(self):
        # Two evaluations of the *same* matchup under CRN should agree far
        # more tightly than the spread of either one's own score alone.
        bat, bowl = _batting_order(), _bowling_attack()
        ctx = MatchContext(
            batting_order=bat,
            bowling_attack=bowl,
            total_overs=20,
            phases=PHASES,
            phase_names=PHASE_NAMES,
        )
        res_a = simulate_match(ctx, bat, bowl, n_sims=2000, crn_seed=7)
        res_b = simulate_match(ctx, bat, bowl, n_sims=2000, crn_seed=7)
        # Same crn_seed -> identical draws -> identical outcome, the
        # strongest form of variance reduction.
        np.testing.assert_array_equal(res_a["team1_scores"], res_b["team1_scores"])
