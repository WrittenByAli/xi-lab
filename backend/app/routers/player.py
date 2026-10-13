"""GET /player/{player_id} — player profile with phase ratings."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.models import PlayerProfile
from backend.app.services.selection import get_player_profile, load_player_meta

router = APIRouter()


@router.get("/search")
def search_players(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, le=100),
):
    meta = load_player_meta()
    mask = meta["name"].str.contains(q, case=False, na=False)
    results = meta[mask].head(limit)
    return [
        {
            "player_id": row["player_id"],
            "name": row["name"],
            "bat_hand": row.get("bat_hand"),
            "bowl_type": row.get("bowl_type"),
            "is_keeper": bool(row.get("is_keeper", False)),
        }
        for _, row in results.iterrows()
    ]


@router.get("/{player_id}", response_model=PlayerProfile)
def get_player(
    player_id: str,
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
):
    profile = get_player_profile(player_id, format)
    if profile is None:
        raise HTTPException(404, f"Player '{player_id}' not found")
    return profile
