"""Tests for ml/optimize/selector.py's ILP candidate generation.

Regression coverage for a real bug found while using the live-prediction
pipeline: the ILP objective summed raw bat_rating + bowl_rating, which
are not on comparable scales (bat ~0-170, bowl ~-10 to -110, less
negative being better). That let batting all-rounders' modest bowling
numerically outweigh genuine strike bowlers' weak batting, and concretely
dropped Bumrah and Shami from a recommended India ODI XI. Fixed by
normalizing both to [0,1] before combining (ml.optimize.selector._norm_bat
/_norm_bowl, mirroring ml.simulate.cards's functions of the same name).
"""

from __future__ import annotations

from ml.optimize.selector import PlayerOption, _norm_bat, _norm_bowl, generate_candidate_xis


class TestNormBowl:
    def test_elite_bowler_ranks_above_non_bowler(self):
        # Bumrah-ish rating vs a specialist batter's "never bowls" prior.
        assert _norm_bowl(-12.1) > _norm_bowl(-73.0)

    def test_increases_as_raw_increases(self):
        # Less negative must always normalize higher (better).
        assert _norm_bowl(-10.0) > _norm_bowl(-30.0) > _norm_bowl(-60.0)

    def test_clipped_to_unit_interval(self):
        assert _norm_bowl(-5.0) == 1.0
        assert _norm_bowl(-200.0) == 0.0


def _batter(pid: str, bat: float) -> PlayerOption:
    return PlayerOption(
        player_id=pid,
        name=pid,
        role="pure_bat",
        bat_rating=bat,
        bowl_rating=-50.0,
        is_keeper=False,
    )


def _bowler(pid: str, bowl: float) -> PlayerOption:
    return PlayerOption(
        player_id=pid,
        name=pid,
        role="pure_bowl",
        bat_rating=50.0,
        bowl_rating=bowl,
        is_keeper=False,
    )


def _keeper(pid: str, bat: float) -> PlayerOption:
    return PlayerOption(
        player_id=pid,
        name=pid,
        role="keeper_bat",
        bat_rating=bat,
        bowl_rating=-50.0,
        is_keeper=True,
    )


class TestGenerateCandidateXis:
    def test_does_not_systematically_drop_elite_specialist_bowlers(self):
        # 6 strong batters, 1 keeper, 2 elite specialist bowlers (Bumrah/
        # Shami-ish ratings), 2 weak bench bowlers. Every optimal-ish XI
        # needs the 2 elite bowlers to meet the "5 bowling options"
        # constraint sensibly — if the objective still favored raw
        # bat+bowl sums, the weak-batting elite bowlers would lose out to
        # padding the XI with extra specialist batters instead.
        squad = [
            _keeper("K1", 90.0),
            *[_batter(f"B{i}", 85.0 + i) for i in range(6)],
            _bowler("Elite1", -12.0),
            _bowler("Elite2", -13.0),
            _bowler("Weak1", -60.0),
            _bowler("Weak2", -65.0),
        ]

        candidates = generate_candidate_xis(squad, n_candidates=10, max_overseas=11)
        assert candidates, "expected at least one feasible XI"

        best = candidates[0]
        best_ids = {p.player_id for p in best}
        assert "Elite1" in best_ids
        assert "Elite2" in best_ids


class TestNormBat:
    def test_increases_with_rating(self):
        assert _norm_bat(100.0) > _norm_bat(50.0) > _norm_bat(20.0)
