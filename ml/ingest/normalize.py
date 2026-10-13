"""
Normalization helpers shared across parse.py stages.

  - Venue string → canonical venue_id
  - Competition name → league_id and format
  - Outcome encoding → outcome_class label
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VENUE_MAP_PATH = ROOT / "data" / "reference" / "venue_map.csv"
LEAGUE_MAP_PATH = ROOT / "data" / "reference" / "league_map.csv"

# ── Venue normalization ────────────────────────────────────────────────────────

_venue_map: dict[str, str] | None = None


def _load_venue_map() -> dict[str, str]:
    global _venue_map
    if _venue_map is None:
        if VENUE_MAP_PATH.exists():
            df = pd.read_csv(VENUE_MAP_PATH)
            _venue_map = dict(zip(df["raw_name"].str.strip(), df["venue_id"], strict=False))
        else:
            _venue_map = {}
    return _venue_map


def venue_to_id(raw: str | None) -> str:
    if not raw:
        return "unknown"
    vm = _load_venue_map()
    return vm.get(raw.strip(), _slugify(raw))


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ── League / competition normalization ────────────────────────────────────────

_league_map: pd.DataFrame | None = None


def _load_league_map() -> pd.DataFrame:
    global _league_map
    if _league_map is None:
        if LEAGUE_MAP_PATH.exists():
            _league_map = pd.read_csv(LEAGUE_MAP_PATH)
        else:
            _league_map = pd.DataFrame(
                columns=["competition", "league_id", "format", "is_international"]
            )
    return _league_map


def competition_to_league(competition: str, match_type: str) -> dict[str, str | bool]:
    """Return {league_id, format, is_international} for a Cricsheet competition name."""
    lm = _load_league_map()
    row = lm[lm["competition"] == competition]
    if not row.empty:
        r = row.iloc[0]
        return {
            "league_id": r["league_id"],
            "format": r["format"],
            "is_international": bool(r["is_international"]),
        }
    # Fallback: derive from match_type field in Cricsheet JSON
    fmt = _infer_format(match_type)
    return {
        "league_id": _slugify(competition) or "unknown",
        "format": fmt,
        "is_international": False,
    }


def _infer_format(match_type: str) -> str:
    mt = match_type.lower()
    if "t20" in mt:
        return "t20"
    if "odi" in mt or "one day" in mt:
        return "odi"
    if "test" in mt or "fc" in mt or "first class" in mt:
        return "test"
    return "t20"


# ── Outcome class encoding ────────────────────────────────────────────────────


def encode_outcome(
    runs_batter: int,
    is_wide: bool,
    is_noball: bool,
    is_wicket: bool,
) -> str:
    """
    Map a delivery to one of the 9 outcome classes.

    Priority: wide > noball > wicket > runs.
    Note: a wide can also be a wicket (run out on a wide); we label it 'wide'
    because the batting outcome (no ball faced) dominates for model purposes.
    """
    if is_wide:
        return "wide"
    if is_noball:
        return "noball"
    if is_wicket:
        return "wicket"
    run_map = {0: "dot", 1: "1", 2: "2", 3: "3", 4: "4", 6: "6"}
    # 5s are rare (overthrows); map to nearest
    if runs_batter >= 5:
        return "6"
    return run_map.get(runs_batter, "dot")
