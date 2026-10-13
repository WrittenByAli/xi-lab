"""POST /recommend — best XI from a squad."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.models import (
    PlayerSummary,
    RecommendRequest,
    RecommendResponse,
)
from ml.optimize.selector import PlayerOption, select_xi

router = APIRouter()


def _to_player_option(p) -> PlayerOption:
    return PlayerOption(
        player_id=p.player_id,
        name=p.name,
        role=p.role,
        bat_rating=p.bat_rating,
        bowl_rating=p.bowl_rating,
        is_keeper=p.is_keeper,
        is_overseas=p.is_overseas,
        bat_hand=p.bat_hand,
        bowl_type=p.bowl_type,
    )


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if len(req.squad) < 11:
        raise HTTPException(400, "Squad must have at least 11 players")

    squad = [_to_player_option(p) for p in req.squad]
    result = select_xi(squad, max_overseas=req.max_overseas)

    xi_out = [
        PlayerSummary(
            player_id=p.player_id,
            name=p.name,
            role=p.role,
            bat_rating=round(p.bat_rating, 4),
            bowl_rating=round(p.bowl_rating, 4),
            is_keeper=p.is_keeper,
            is_overseas=p.is_overseas,
        )
        for p in result.xi
    ]

    return RecommendResponse(
        xi=xi_out,
        win_prob=round(result.win_prob, 4),
        batting_order=result.batting_order,
        bowling_plan=result.bowling_plan,
        tie_set=result.tie_set,
        swap_matrix=result.swap_matrix,
    )
