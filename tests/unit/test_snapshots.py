"""Tests for ml/features/snapshots.py — correctness and no-leakage guarantees."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.features.snapshots import (
    _decay_weights,
    _prepare_balls,
    build_batting_features,
    build_bowling_features,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_balls():
    """Minimal balls DataFrame for unit tests."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "match_id": "m1" if i < 10 else "m2",
                "innings_no": 1,
                "over": i % 6,
                "legal_ball_idx": 0,
                "delivery_seq": i,
                "batter_id": "Bat_A",
                "non_striker_id": "Bat_B",
                "bowler_id": "Bowl_X",
                "runs_batter": 4 if i % 5 == 0 else 1,
                "runs_extras": 0,
                "extras_type": None,
                "is_wide": False,
                "is_noball": False,
                "is_legal": True,
                "wicket_kind": "bowled" if i == 7 else None,
                "player_out_id": "Bat_A" if i == 7 else None,
                "score_before": 0,
                "wkts_before": 0,
                "legal_balls_before": i,
                "target": None,
                "rrr_before": None,
                "outcome_class": "4" if i % 5 == 0 else "1",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_match_dates():
    s = pd.Series(
        {
            "m1": pd.Timestamp("2023-06-15"),
            "m2": pd.Timestamp("2023-09-10"),
        },
        name="date",
    )
    s.index.name = "match_id"
    return s


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDecayWeights:
    def test_zero_days_ago_is_one(self):
        w = _decay_weights(np.array([0.0]), half_life=730)
        assert np.isclose(w[0], 1.0)

    def test_half_life_gives_half(self):
        w = _decay_weights(np.array([730.0]), half_life=730)
        assert np.isclose(w[0], 0.5, atol=1e-6)

    def test_monotonically_decreasing(self):
        days = np.array([0.0, 100.0, 365.0, 730.0, 1460.0])
        w = _decay_weights(days, half_life=730)
        assert all(w[i] > w[i + 1] for i in range(len(w) - 1))


class TestNoLeakage:
    def test_snapshot_uses_only_past_data(self, synthetic_balls, synthetic_match_dates):
        """A snapshot at 2023-08-01 must not see match m2 (2023-09-10)."""
        cutoff = pd.Timestamp("2023-08-01")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        assert prepared is not None
        # Only m1 (June 15) should be present
        assert set(prepared["match_id"].unique()) == {"m1"}

    def test_snapshot_at_exact_date_excludes_it(self, synthetic_balls, synthetic_match_dates):
        """Cutoff is strict: data ON the cutoff date is excluded."""
        cutoff = pd.Timestamp("2023-06-15")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        assert prepared is None  # no data before June 15

    def test_snapshot_after_all_matches(self, synthetic_balls, synthetic_match_dates):
        cutoff = pd.Timestamp("2024-01-01")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        assert set(prepared["match_id"].unique()) == {"m1", "m2"}


class TestBattingFeatures:
    def test_basic_aggregation(self, synthetic_balls, synthetic_match_dates):
        cutoff = pd.Timestamp("2024-01-01")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        legal = prepared[prepared["is_legal"]]
        bat = build_batting_features(legal)

        bat_a = bat[bat["player_id"] == "Bat_A"].iloc[0]
        assert bat_a["bat_balls_raw"] == 20
        assert bat_a["bat_sr_wtd"] > 0
        assert 0 <= bat_a["bat_dot_pct_wtd"] <= 1
        assert bat_a["bat_wkts_wtd"] > 0  # one bowled dismissal

    def test_no_data_returns_empty(self, synthetic_balls, synthetic_match_dates):
        cutoff = pd.Timestamp("2020-01-01")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        assert prepared is None


class TestBowlingFeatures:
    def test_basic_aggregation(self, synthetic_balls, synthetic_match_dates):
        cutoff = pd.Timestamp("2024-01-01")
        prepared = _prepare_balls(synthetic_balls, synthetic_match_dates, cutoff, half_life=730)
        legal = prepared[prepared["is_legal"]]
        bowl = build_bowling_features(legal, prepared)

        bowl_x = bowl[bowl["player_id"] == "Bowl_X"].iloc[0]
        assert bowl_x["bowl_balls_raw"] == 20
        assert bowl_x["bowl_econ_wtd"] > 0
        assert bowl_x["bowl_wkts_wtd"] > 0  # one bowled
