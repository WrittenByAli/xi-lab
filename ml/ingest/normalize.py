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
            _league_map = pd.DataFrame(columns=["competition", "league_id", "format"])
    return _league_map


def competition_to_league(competition: str, match_type: str) -> dict[str, str]:
    """Return {league_id, format} for a Cricsheet competition name."""
    lm = _load_league_map()
    row = lm[lm["competition"] == competition]
    if not row.empty:
        r = row.iloc[0]
        return {"league_id": r["league_id"], "format": r["format"]}
    # Fallback: both men's and women's cricket have a long tail of
    # competition names not worth enumerating (women's especially: 300+
    # distinct strings, mostly one per bilateral tour), but league_id is
    # cosmetic (a slug for display) and format comes straight off
    # Cricsheet's own match_type field, so there's nothing to look up.
    return {"league_id": _slugify(competition) or "unknown", "format": _infer_format(match_type)}


# ── International vs. domestic ──────────────────────────────────────────────
#
# A match is international iff both teams are national sides, not whichever
# competition they're playing in — competition-name matching was tried first
# (league_map.csv's `is_international` column, plus a "contains 'women'"
# fallback for the long tail of women's tour names) and was wrong for the
# vast majority of matches: e.g. "ICC Champions Trophy", "NatWest Series"
# and ordinary tour names like "Australia tour of India" never matched any
# known string, so men's ODI showed just 5% international and men's Test
# showed 0% — every Test match is international by definition. Team
# identity is a far more robust signal and needs no enumeration of
# competition names at all: it doesn't matter what a bilateral series or
# a one-off tournament happens to be called, only whether the two sides
# are national teams or domestic/franchise sides.
#
# Built directly from this project's own data (259 unique team names
# across every format and gender currently parsed — see git history for
# how), not a generic country list: it deliberately excludes domestic
# sides that happen to share a sovereign nation's name (a cricket "country"
# and a place name aren't the same thing here). "Barbados" plays only as
# a West Indian regional/franchise side in this data (the sovereign nation
# fields no separate international team — that's "West Indies"), so it's
# left out despite being a real country; "Guernsey"/"Jersey"/"Gibraltar"/
# "St Helena"/"Turks and Caicos Island" are kept in because each fields
# its own ICC-affiliated national side. "Africa XI"/"Asia XI"/"ICC World
# XI" are honorary multi-nation exhibition sides, not national teams, and
# are deliberately excluded.
COUNTRY_TEAMS: frozenset[str] = frozenset(
    {
        "Argentina",
        "Australia",
        "Austria",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Belgium",
        "Belize",
        "Bermuda",
        "Bhutan",
        "Botswana",
        "Brazil",
        "Bulgaria",
        "Cambodia",
        "Cameroon",
        "Canada",
        "Cayman Islands",
        "Chile",
        "China",
        "Cook Islands",
        "Costa Rica",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "England",
        "Estonia",
        "Eswatini",
        "Fiji",
        "Finland",
        "France",
        "Gambia",
        "Germany",
        "Ghana",
        "Gibraltar",
        "Greece",
        "Guernsey",
        "Hong Kong",
        "Hungary",
        "India",
        "Indonesia",
        "Iran",
        "Ireland",
        "Isle of Man",
        "Israel",
        "Italy",
        "Ivory Coast",
        "Japan",
        "Jersey",
        "Kenya",
        "Kuwait",
        "Lesotho",
        "Luxembourg",
        "Malawi",
        "Malaysia",
        "Maldives",
        "Mali",
        "Malta",
        "Mexico",
        "Mongolia",
        "Mozambique",
        "Myanmar",
        "Namibia",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "Norway",
        "Oman",
        "Pakistan",
        "Panama",
        "Papua New Guinea",
        "Peru",
        "Philippines",
        "Portugal",
        "Qatar",
        "Romania",
        "Rwanda",
        "Samoa",
        "Saudi Arabia",
        "Scotland",
        "Serbia",
        "Seychelles",
        "Sierra Leone",
        "Singapore",
        "Slovenia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sri Lanka",
        "St Helena",
        "Suriname",
        "Swaziland",
        "Sweden",
        "Switzerland",
        "Tanzania",
        "Thailand",
        "Timor-Leste",
        "Turkey",
        "Turks and Caicos Island",
        "Uganda",
        "United Arab Emirates",
        "United States of America",
        "Uzbekistan",
        "Vanuatu",
        "West Indies",
        "Zambia",
        "Zimbabwe",
    }
)


def is_international_match(team1: str, team2: str) -> bool:
    """True iff both sides are national teams (see COUNTRY_TEAMS above)."""
    return team1 in COUNTRY_TEAMS and team2 in COUNTRY_TEAMS


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
