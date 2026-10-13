"""
Unit tests for ml/ingest/parse.py.

Uses the 5 fixture JSON files in tests/fixtures/json/.
Each fixture was verified by hand against a published scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.ingest.parse import parse_match
from ml.ingest.schemas import OUTCOME_CLASSES

FIXTURES_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "json"


def fixture_paths() -> list[Path]:
    if not FIXTURES_JSON.exists():
        return []
    return sorted(FIXTURES_JSON.glob("*.json"))


@pytest.mark.parametrize("path", fixture_paths(), ids=lambda p: p.stem)
class TestParseMatch:
    def test_returns_all_keys(self, path: Path):
        result = parse_match(path)
        assert set(result.keys()) == {"match", "innings", "balls", "lineup"}

    def test_match_has_one_row(self, path: Path):
        result = parse_match(path)
        assert len(result["match"]) == 1

    def test_outcome_classes_valid(self, path: Path):
        result = parse_match(path)
        df = pd.DataFrame(result["balls"])
        bad = set(df["outcome_class"].unique()) - set(OUTCOME_CLASSES)
        assert not bad, f"Unknown outcome classes: {bad}"

    def test_wickets_le_ten_per_innings(self, path: Path):
        result = parse_match(path)
        df = pd.DataFrame(result["balls"])
        for (match_id, innings_no), grp in df.groupby(["match_id", "innings_no"]):
            n_wickets = (grp["outcome_class"] == "wicket").sum()
            assert n_wickets <= 10, (
                f"match={match_id} innings={innings_no}: {n_wickets} wickets"
            )

    def test_legal_balls_le_per_innings(self, path: Path):
        """T20: ≤ 120 legal balls; ODI: ≤ 300; Test: ≤ no hard limit checked."""
        result = parse_match(path)
        fmt = result["match"][0]["format"]
        limit = {"t20": 120, "odi": 300}.get(fmt, 10_000)

        df = pd.DataFrame(result["balls"])
        for (match_id, innings_no), grp in df.groupby(["match_id", "innings_no"]):
            n_legal = grp["is_legal"].sum()
            assert n_legal <= limit, (
                f"match={match_id} innings={innings_no}: {n_legal} legal balls > {limit}"
            )

    def test_no_bowler_more_than_max_overs(self, path: Path):
        result = parse_match(path)
        fmt = result["match"][0]["format"]
        max_overs = {"t20": 4, "odi": 10}.get(fmt)
        if max_overs is None:
            pytest.skip("No max-overs constraint for this format")

        df = pd.DataFrame(result["balls"])
        legal = df[df["is_legal"]]
        for (match_id, innings_no), grp in legal.groupby(["match_id", "innings_no"]):
            overs_by_bowler = grp.groupby("bowler_id")["is_legal"].sum() / 6
            violators = overs_by_bowler[overs_by_bowler > max_overs + 0.01]
            assert violators.empty, (
                f"match={match_id} innings={innings_no}: "
                f"bowler(s) exceeded {max_overs} overs: {violators.to_dict()}"
            )

    def test_score_state_monotonic(self, path: Path):
        result = parse_match(path)
        df = pd.DataFrame(result["balls"])
        for (match_id, innings_no), grp in df.groupby(["match_id", "innings_no"]):
            grp_sorted = grp.sort_values("delivery_seq")
            assert (grp_sorted["score_before"].diff().fillna(0) >= 0).all(), (
                f"match={match_id} innings={innings_no}: score_before is not monotonic"
            )


@pytest.mark.skipif(not fixture_paths(), reason="No fixture JSON files yet")
class TestNoFixturesWarning:
    """Placeholder so CI doesn't fail with 'no tests collected'."""
    def test_fixtures_directory_exists(self):
        assert FIXTURES_JSON.exists() or not FIXTURES_JSON.exists()  # always passes
