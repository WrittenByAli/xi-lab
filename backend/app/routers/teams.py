"""GET /teams — team listings, squad rosters, and venues from match data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.services.selection import load_player_meta, load_ratings

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]

_GENDER_PATTERN = "^(male|female)$"


@lru_cache(maxsize=8)
def _load_lineups(fmt: str, gender: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "interim" / fmt / gender / "lineups.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@lru_cache(maxsize=8)
def _load_matches(fmt: str, gender: str) -> pd.DataFrame | None:
    path = ROOT / "data" / "interim" / fmt / gender / "matches.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@lru_cache(maxsize=1)
def _load_venue_map() -> pd.DataFrame:
    path = ROOT / "data" / "reference" / "venue_map.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["venue_id", "canonical_name", "city", "country"])


def _venue_display_name(venue_id: str, venue_map: pd.DataFrame) -> str:
    row = venue_map[venue_map["venue_id"] == venue_id]
    if not row.empty:
        return str(row.iloc[0]["canonical_name"])
    return venue_id.replace("_", " ").title()


@router.get("")
def list_teams(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
    limit: int = Query(50, le=200),
):
    matches = _load_matches(format, gender)
    lineups = _load_lineups(format, gender)
    if matches is None or lineups is None:
        return []

    cutoff = matches["date"].max() - pd.DateOffset(months=18)
    recent_ids = matches[matches["date"] >= cutoff]["match_id"]
    recent = lineups[lineups["match_id"].isin(recent_ids)]

    team_stats = (
        recent.groupby("team")
        .agg(
            players=("player_id", "nunique"),
            matches=("match_id", "nunique"),
        )
        .sort_values("matches", ascending=False)
        .head(limit)
        .reset_index()
    )

    return [
        {
            "team": row["team"],
            "recent_matches": int(row["matches"]),
            "squad_size": int(row["players"]),
        }
        for _, row in team_stats.iterrows()
    ]


@router.get("/{team}/squad")
def get_squad(
    team: str,
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
    months: int = Query(12, ge=3, le=36),
    all_time: bool = Query(
        False, description="Ignore the recency cutoff and use the team's full match history"
    ),
):
    matches = _load_matches(format, gender)
    lineups = _load_lineups(format, gender)
    if matches is None or lineups is None:
        raise HTTPException(404, f"No {gender} data for format '{format}'")

    if all_time:
        recent_match_ids = matches["match_id"]
    else:
        cutoff = matches["date"].max() - pd.DateOffset(months=months)
        recent_match_ids = matches[matches["date"] >= cutoff]["match_id"]
    recent = lineups[(lineups["match_id"].isin(recent_match_ids)) & (lineups["team"] == team)]

    if recent.empty:
        scope = "match history" if all_time else f"recent {gender} {format} data"
        raise HTTPException(404, f"Team '{team}' not found in {scope}")

    player_ids = recent["player_id"].unique()

    meta = load_player_meta()
    ratings = load_ratings(gender)

    squad = []
    for pid in player_ids:
        player: dict[str, Any] = {"player_id": pid, "name": pid}

        meta_row = meta[meta["player_id"] == pid]
        if not meta_row.empty:
            row = meta_row.iloc[0]
            player["name"] = row.get("name", pid)
            player["bat_hand"] = row.get("bat_hand", "R")
            player["bowl_type"] = row.get("bowl_type", "none")
            player["is_keeper"] = bool(row.get("is_keeper", False))
        else:
            player["bat_hand"] = "R"
            player["bowl_type"] = "none"
            player["is_keeper"] = False

        player["role"] = "pure_bat"
        player["bat_rating"] = 0.5
        player["bowl_rating"] = 0.5

        if ratings is not None:
            r = ratings[ratings["player_id"] == pid]
            if not r.empty:
                latest = r.sort_values("snapshot_month").iloc[-1]
                player["role"] = str(latest.get("role", "pure_bat"))
                player["bat_rating"] = round(float(latest.get("bat_rating", 0.5)), 4)
                player["bowl_rating"] = round(float(latest.get("bowl_rating", 0.5)), 4)

        app_count = int(recent[recent["player_id"] == pid]["match_id"].nunique())
        player["recent_matches"] = app_count
        squad.append(player)

    squad.sort(key=lambda p: int(p.get("recent_matches", 0)), reverse=True)

    return {
        "team": team,
        "format": format,
        "gender": gender,
        "all_time": all_time,
        "squad_size": len(squad),
        "players": squad,
    }


@router.get("/venues")
def list_venues(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
    limit: int = Query(200, le=500),
):
    """Distinct grounds with at least one match on record, most-used first."""
    matches = _load_matches(format, gender)
    if matches is None:
        return []

    venue_map = _load_venue_map()
    counts = matches.groupby("venue_id").size().sort_values(ascending=False).head(limit)

    return [
        {
            "venue_id": venue_id,
            "name": _venue_display_name(venue_id, venue_map),
            "match_count": int(count),
        }
        for venue_id, count in counts.items()
    ]
