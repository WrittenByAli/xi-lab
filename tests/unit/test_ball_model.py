"""Tests for ml/models/ball/train.py — feature join logic and no-leakage guarantees."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.ball.train import (
    ALL_FEATURES,
    BATTER_SNAPSHOT_FEATURES,
    BOWLER_SNAPSHOT_FEATURES,
    CATEGORICAL_FEATURES,
    asof_join_snapshots,
    encode_categoricals,
    split_by_date,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_balls(n_per_match: int = 5) -> pd.DataFrame:
    """Minimal balls DataFrame with 3 matches across different dates."""
    rows = []
    for match_idx, match_id in enumerate(["m1", "m2", "m3"]):
        for i in range(n_per_match):
            rows.append({
                "match_id": match_id,
                "innings_no": 1,
                "over": i % 6,
                "batter_id": "Bat_A",
                "bowler_id": "Bowl_X",
                "runs_batter": i % 3,
                "outcome_class": ["dot", "1", "2"][i % 3],
                "score_before": i * 2,
                "wkts_before": 0,
                "legal_balls_before": i,
                "target": None,
                "rrr_before": None,
                "is_legal": True,
            })
    return pd.DataFrame(rows)


def _make_matches() -> pd.DataFrame:
    """Three matches at known dates."""
    return pd.DataFrame({
        "match_id": ["m1", "m2", "m3"],
        "date": pd.to_datetime(["2023-03-15", "2023-06-20", "2024-02-10"]),
        "is_international": [1, 0, 1],
    })


def _make_snapshots() -> pd.DataFrame:
    """Monthly snapshots for Bat_A and Bowl_X at a few months."""
    months = pd.to_datetime(["2023-01-01", "2023-03-01", "2023-06-01", "2024-01-01"])
    rows = []
    for month in months:
        # Batter snapshot — values encode the month for easy verification
        month_num = month.month + month.year * 100
        rows.append({
            "player_id": "Bat_A",
            "snapshot_month": month,
            "bat_sr_wtd": float(month_num),
            "bat_avg_wtd": float(month_num) + 0.1,
            "bat_boundary_pct_wtd": 0.15,
            "bat_dot_pct_wtd": 0.35,
            "bat_sr_powerplay_wtd": 130.0,
            "bat_sr_middle_wtd": 120.0,
            "bat_sr_death_wtd": 150.0,
            "form_bat_sr": 140.0,
            "bat_hand": "RHB",
            "bowl_type": None,
        })
        # Bowler snapshot
        rows.append({
            "player_id": "Bowl_X",
            "snapshot_month": month,
            "bat_sr_wtd": None,
            "bat_avg_wtd": None,
            "bat_boundary_pct_wtd": None,
            "bat_dot_pct_wtd": None,
            "bat_sr_powerplay_wtd": None,
            "bat_sr_middle_wtd": None,
            "bat_sr_death_wtd": None,
            "form_bat_sr": None,
            "bat_hand": None,
            "bowl_type": "RF",
            "bowl_econ_wtd": float(month_num),
            "bowl_sr_wtd": float(month_num) + 0.2,
            "bowl_dot_pct_wtd": 0.42,
            "bowl_econ_powerplay_wtd": 7.5,
            "bowl_econ_middle_wtd": 6.2,
            "bowl_econ_death_wtd": 9.1,
            "form_bowl_econ": 7.8,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def balls():
    return _make_balls()


@pytest.fixture
def matches():
    return _make_matches()


@pytest.fixture
def snapshots():
    return _make_snapshots()


# ── As-of join correctness ──────────────────────────────────────────────────


class TestAsOfJoinNoLeakage:
    """The as-of join must use the latest snapshot STRICTLY BEFORE the match date."""

    def test_march_match_uses_latest_prior_snapshot(self, balls, matches, snapshots):
        """Match m1 on 2023-03-15 should use the 2023-03-01 snapshot (latest before match).

        The 2023-03-01 snapshot is strictly before 2023-03-15, so it is valid.
        It should be preferred over the older 2023-01-01 snapshot.
        """
        joined = asof_join_snapshots(balls, snapshots, matches)
        m1_rows = joined[joined["match_id"] == "m1"]

        # 2023-03-01 < 2023-03-15 => valid, and it is the most recent
        expected_bat_sr = 202303.0
        assert all(m1_rows["bat_sr_wtd"] == expected_bat_sr), (
            f"m1 (2023-03-15) got bat_sr_wtd={m1_rows['bat_sr_wtd'].iloc[0]}, "
            f"expected {expected_bat_sr} (from 2023-03-01 snapshot)"
        )

    def test_future_snapshot_not_used(self, balls, matches, snapshots):
        """A snapshot dated AFTER the match date must never be picked."""
        joined = asof_join_snapshots(balls, snapshots, matches)
        m1_rows = joined[joined["match_id"] == "m1"]

        # 2023-06-01 and 2024-01-01 snapshots exist but are after m1 (2023-03-15)
        actual = m1_rows["bat_sr_wtd"].iloc[0]
        assert actual != 202306.0 and actual != 202401.0, (
            f"Leakage: m1 (2023-03-15) used a future snapshot: bat_sr_wtd={actual}"
        )

    def test_june_match_uses_june_snapshot(self, balls, matches, snapshots):
        """Match m2 on 2023-06-20 should use the 2023-06-01 snapshot."""
        joined = asof_join_snapshots(balls, snapshots, matches)
        m2_rows = joined[joined["match_id"] == "m2"]

        # 2023-06-01 < 2023-06-20 => valid, and it is the latest before the match
        expected_bat_sr = 202306.0
        assert all(m2_rows["bat_sr_wtd"] == expected_bat_sr), (
            f"m2 (2023-06-20) got bat_sr_wtd={m2_rows['bat_sr_wtd'].iloc[0]}, "
            f"expected {expected_bat_sr}"
        )

    def test_february_match_uses_january_2024_snapshot(self, balls, matches, snapshots):
        """Match m3 on 2024-02-10 should use the 2024-01-01 snapshot."""
        joined = asof_join_snapshots(balls, snapshots, matches)
        m3_rows = joined[joined["match_id"] == "m3"]

        expected_bat_sr = 202401.0
        assert all(m3_rows["bat_sr_wtd"] == expected_bat_sr), (
            f"m3 (2024-02-10) got bat_sr_wtd={m3_rows['bat_sr_wtd'].iloc[0]}, "
            f"expected {expected_bat_sr}"
        )

    def test_bowler_snapshot_joins_correctly(self, balls, matches, snapshots):
        """Bowler features should also join with the correct as-of logic."""
        joined = asof_join_snapshots(balls, snapshots, matches)
        m2_rows = joined[joined["match_id"] == "m2"]

        # Bowl_X snapshot at 2023-06-01 has bowl_econ_wtd = 202306.0
        expected_bowl_econ = 202306.0
        assert all(m2_rows["bowl_econ_wtd"] == expected_bowl_econ), (
            f"Bowler leakage: m2 got bowl_econ_wtd={m2_rows['bowl_econ_wtd'].iloc[0]}, "
            f"expected {expected_bowl_econ}"
        )


class TestAsOfJoinExactDateExclusion:
    """If snapshot_month == match date, the snapshot must NOT be used."""

    def test_snapshot_on_match_date_excluded(self):
        """A snapshot dated exactly on the match date must be excluded."""
        balls = pd.DataFrame({
            "match_id": ["m_exact"] * 3,
            "innings_no": [1, 1, 1],
            "over": [0, 1, 2],
            "batter_id": ["Bat_A"] * 3,
            "bowler_id": ["Bowl_X"] * 3,
            "runs_batter": [1, 2, 0],
            "outcome_class": ["1", "2", "dot"],
            "score_before": [0, 1, 3],
            "wkts_before": [0, 0, 0],
            "legal_balls_before": [0, 1, 2],
            "target": [None] * 3,
            "rrr_before": [None] * 3,
            "is_legal": [True] * 3,
        })

        matches = pd.DataFrame({
            "match_id": ["m_exact"],
            "date": pd.to_datetime(["2023-06-01"]),
            "is_international": [1],
        })

        # Snapshot on exactly 2023-06-01 should NOT be used (same date as match)
        # Only the 2023-03-01 snapshot should be picked
        snapshots = pd.DataFrame({
            "player_id": ["Bat_A", "Bat_A", "Bowl_X", "Bowl_X"],
            "snapshot_month": pd.to_datetime([
                "2023-03-01", "2023-06-01",  # batter
                "2023-03-01", "2023-06-01",  # bowler
            ]),
            "bat_sr_wtd": [100.0, 999.0, None, None],
            "bat_avg_wtd": [30.0, 999.0, None, None],
            "bat_hand": ["RHB", "RHB", None, None],
            "bowl_type": [None, None, "RF", "RF"],
            "bowl_econ_wtd": [None, None, 7.0, 999.0],
        })

        joined = asof_join_snapshots(balls, snapshots, matches)

        # Should use the March snapshot (bat_sr_wtd=100), not June (999)
        assert all(joined["bat_sr_wtd"] == 100.0), (
            f"Leakage: snapshot on exact match date was used. "
            f"Got bat_sr_wtd={joined['bat_sr_wtd'].iloc[0]}, expected 100.0"
        )
        assert all(joined["bowl_econ_wtd"] == 7.0), (
            f"Bowler leakage: snapshot on exact match date was used. "
            f"Got bowl_econ_wtd={joined['bowl_econ_wtd'].iloc[0]}, expected 7.0"
        )


class TestAsOfJoinMissingSnapshots:
    """When no snapshot exists before a match, features should be NaN."""

    def test_no_prior_snapshot_gives_nan(self):
        """If a player has no snapshot before the match date, features are NaN."""
        balls = pd.DataFrame({
            "match_id": ["m_early"] * 2,
            "innings_no": [1, 1],
            "over": [0, 1],
            "batter_id": ["Bat_New", "Bat_New"],
            "bowler_id": ["Bowl_New", "Bowl_New"],
            "runs_batter": [4, 0],
            "outcome_class": ["4", "dot"],
            "score_before": [0, 4],
            "wkts_before": [0, 0],
            "legal_balls_before": [0, 1],
            "target": [None, None],
            "rrr_before": [None, None],
            "is_legal": [True, True],
        })

        matches = pd.DataFrame({
            "match_id": ["m_early"],
            "date": pd.to_datetime(["2020-01-15"]),
            "is_international": [0],
        })

        # Snapshots only exist from 2023 onward
        snapshots = pd.DataFrame({
            "player_id": ["Bat_New"],
            "snapshot_month": pd.to_datetime(["2023-01-01"]),
            "bat_sr_wtd": [150.0],
            "bat_avg_wtd": [40.0],
            "bat_hand": ["LHB"],
            "bowl_type": [None],
        })

        joined = asof_join_snapshots(balls, snapshots, matches)

        # No snapshot before 2020-01-15 exists, so features must be NaN
        assert joined["bat_sr_wtd"].isna().all(), (
            "Expected NaN for bat_sr_wtd when no prior snapshot exists"
        )


# ── Categorical encoding ────────────────────────────────────────────────────


class TestCategoricalEncoding:
    def test_label_encoding_roundtrip(self):
        df = pd.DataFrame({
            "bat_hand": ["RHB", "LHB", "RHB", None],
            "bowl_type": ["RF", "SLA", "OB", "RF"],
        })
        encoded, encoders = encode_categoricals(df, CATEGORICAL_FEATURES, fit=True)

        assert encoded["bat_hand"].dtype in (np.int32, np.int64, np.intp)
        assert encoded["bowl_type"].dtype in (np.int32, np.int64, np.intp)

        # NaN should be mapped to "unknown"
        assert "unknown" in encoders["bat_hand"].classes_

    def test_unseen_labels_mapped_to_unknown(self):
        df_train = pd.DataFrame({
            "bat_hand": ["RHB", "LHB", "unknown"],
            "bowl_type": ["RF", "SLA", "unknown"],
        })
        _, encoders = encode_categoricals(df_train, CATEGORICAL_FEATURES, fit=True)

        df_test = pd.DataFrame({
            "bat_hand": ["SWITCH_HIT"],  # unseen
            "bowl_type": ["MYSTERY"],    # unseen
        })
        encoded, _ = encode_categoricals(df_test, CATEGORICAL_FEATURES, encoders=encoders, fit=False)

        # Should not raise, unseen mapped to "unknown"
        unknown_bat = encoders["bat_hand"].transform(["unknown"])[0]
        assert encoded["bat_hand"].iloc[0] == unknown_bat


# ── Date-based splitting ─────────────────────────────────────────────────────


class TestSplitByDate:
    def test_splits_are_non_overlapping(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2021-06-01", "2022-11-30", "2023-05-15", "2023-12-31", "2024-03-01",
            ]),
            "value": [1, 2, 3, 4, 5],
        })
        splits = {
            "train_end": "2022-12-31",
            "val_start": "2023-01-01",
            "val_end": "2023-12-31",
            "test_start": "2024-01-01",
        }
        train, val, test = split_by_date(df, splits)

        assert len(train) == 2
        assert len(val) == 2
        assert len(test) == 1

        # No row appears in multiple splits
        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == len(train) + len(val) + len(test)

    def test_boundary_dates_assigned_correctly(self):
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2022-12-31",  # train_end boundary
                "2023-01-01",  # val_start boundary
                "2023-12-31",  # val_end boundary
                "2024-01-01",  # test_start boundary
            ]),
            "value": range(4),
        })
        splits = {
            "train_end": "2022-12-31",
            "val_start": "2023-01-01",
            "val_end": "2023-12-31",
            "test_start": "2024-01-01",
        }
        train, val, test = split_by_date(df, splits)

        assert len(train) == 1  # 2022-12-31
        assert len(val) == 2    # 2023-01-01, 2023-12-31
        assert len(test) == 1   # 2024-01-01
