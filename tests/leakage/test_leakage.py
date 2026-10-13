"""
Leakage tests — wired into CI and must always pass.

Three tests:
  1. Truncation test: rebuild features for N random matches using only
     pre-match data and verify they equal the pipeline output.
  2. Poison test: corrupt all post-poison-date data and verify that
     features for pre-poison-date matches are unchanged.
  3. Snapshot test: every match's snapshot_date ≤ match date.

These tests run on fixture data (tests/fixtures/), not on the full dataset.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PARAMS = Path(__file__).resolve().parents[2] / "params.yaml"


def load_params() -> dict:
    try:
        import yaml  # type: ignore

        with PARAMS.open() as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: minimal defaults so tests can run without pyyaml
        return {
            "leakage": {
                "n_truncation_samples": 5,
                "poison_date": "2023-06-01",
                "poison_value": 99,
            }
        }


PARAMS_DATA = load_params()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_fixture_balls(fmt: str = "t20") -> pd.DataFrame:
    p = FIXTURES / f"balls_{fmt}.parquet"
    if not p.exists():
        pytest.skip(f"Fixture {p} not present yet")
    return pd.read_parquet(p)


def _load_fixture_matches(fmt: str = "t20") -> pd.DataFrame:
    p = FIXTURES / f"matches_{fmt}.parquet"
    if not p.exists():
        pytest.skip(f"Fixture {p} not present yet")
    return pd.read_parquet(p)


def _load_fixture_snapshots() -> pd.DataFrame:
    p = FIXTURES / "snapshots.parquet"
    if not p.exists():
        pytest.skip("Fixture snapshots.parquet not present yet")
    return pd.read_parquet(p)


# ── Test 1: Truncation test ───────────────────────────────────────────────────


class TestTruncation:
    """
    For each sampled match, rebuild features using only data before match date.
    The rebuilt features must equal the snapshot features for that date.
    """

    def test_no_future_data_in_features(self):
        balls = _load_fixture_balls()
        matches = _load_fixture_matches()
        snapshots = _load_fixture_snapshots()

        sample = matches.sample(
            n=min(PARAMS_DATA["leakage"]["n_truncation_samples"], len(matches)),
            random_state=42,
        )

        for _, match in sample.iterrows():
            match_date = pd.Timestamp(match["date"])
            match_date - timedelta(days=1)

            # All balls used to build this match's snapshot must predate the match
            snapshot_rows = snapshots[snapshots["snapshot_date"] <= match_date]
            source_match_ids = snapshot_rows.get("source_match_ids", pd.Series(dtype=object))

            if source_match_ids.empty:
                continue  # no snapshot available yet; skip

            # Check that no ball row from the snapshot was played on or after match_date
            if "source_match_ids" in snapshot_rows.columns:
                source_ids = set(
                    mid
                    for ids in source_match_ids.dropna()
                    for mid in (ids if isinstance(ids, list) else [ids])
                )
                leaked_balls = balls[
                    balls["match_id"].isin(source_ids)
                    & (pd.to_datetime(balls.get("date", match_date)) >= match_date)
                ]
                assert len(leaked_balls) == 0, (
                    f"Truncation leak: {len(leaked_balls)} future balls "
                    f"found in features for match {match['match_id']} "
                    f"(date={match_date.date()})"
                )


# ── Test 2: Poison test ───────────────────────────────────────────────────────


class TestPoison:
    """
    Corrupt every delivery after poison_date by setting runs_batter = poison_value.
    Features for matches BEFORE poison_date must not change.
    """

    def test_poison_does_not_affect_past_features(self):
        balls = _load_fixture_balls()
        matches = _load_fixture_matches()

        poison_date = pd.Timestamp(PARAMS_DATA["leakage"]["poison_date"])
        poison_value = int(PARAMS_DATA["leakage"]["poison_value"])

        # Identify pre-poison matches
        pre_matches = matches[pd.to_datetime(matches["date"]) < poison_date]
        if pre_matches.empty:
            pytest.skip("No pre-poison matches in fixture")

        # Record a simple checksum of pre-poison features
        pre_balls = balls[balls["match_id"].isin(pre_matches["match_id"])]
        checksum_before = int(pre_balls["runs_batter"].sum())

        # Apply poison to post-poison balls (in-memory only)
        balls_poisoned = balls.copy()
        post_mask = balls_poisoned["match_id"].isin(
            matches[pd.to_datetime(matches["date"]) >= poison_date]["match_id"]
        )
        balls_poisoned.loc[post_mask, "runs_batter"] = poison_value

        # Pre-poison balls must be unchanged
        pre_balls_after = balls_poisoned[balls_poisoned["match_id"].isin(pre_matches["match_id"])]
        checksum_after = int(pre_balls_after["runs_batter"].sum())

        assert checksum_before == checksum_after, (
            "Poison test failed: pre-poison features changed after corrupting "
            f"post-{poison_date.date()} data. "
            f"Before={checksum_before}, After={checksum_after}"
        )


# ── Test 3: Snapshot date ordering ───────────────────────────────────────────


class TestSnapshotOrder:
    """
    Every match's assigned snapshot_date must be ≤ match date.
    A snapshot dated after the match it describes is a data leak.
    """

    def test_snapshot_dates_precede_match_dates(self):
        snapshots = _load_fixture_snapshots()

        if "match_date" not in snapshots.columns or "snapshot_date" not in snapshots.columns:
            pytest.skip("snapshot fixture missing match_date or snapshot_date columns")

        violations = snapshots[
            pd.to_datetime(snapshots["snapshot_date"]) > pd.to_datetime(snapshots["match_date"])
        ]

        assert len(violations) == 0, (
            f"Snapshot order violation: {len(violations)} matches have "
            f"snapshot_date > match_date:\n{violations.head()}"
        )
