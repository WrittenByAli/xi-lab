"""POST /simulate — win probability + score distributions for two XIs."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from backend.app.schemas.models import (
    ScoreDistribution,
    SimulateRequest,
    SimulateResponse,
)
from backend.app.services.selection import build_score_distribution
from ml.simulate.engine import (
    BattingCard,
    BowlingCard,
    MatchContext,
    simulate_match,
)

router = APIRouter()

_DEFAULT_PROBS = np.array([0.35, 0.25, 0.08, 0.01, 0.12, 0.06, 0.08, 0.03, 0.02])


def _player_to_cards(player) -> tuple[BattingCard, BowlingCard]:
    bat_probs = _DEFAULT_PROBS.copy()
    bat_probs[0] = max(0.05, 0.40 - player.bat_rating * 0.1)
    bat_probs[4] = min(0.25, 0.08 + player.bat_rating * 0.05)
    bat_probs[5] = min(0.15, 0.04 + player.bat_rating * 0.03)
    bat_probs /= bat_probs.sum()

    bowl_probs = _DEFAULT_PROBS.copy()
    bowl_probs[6] = min(0.20, 0.06 + player.bowl_rating * 0.04)
    bowl_probs[0] = min(0.50, 0.30 + player.bowl_rating * 0.05)
    bowl_probs /= bowl_probs.sum()

    phases = {p: bat_probs for p in ["powerplay", "middle", "death"]}
    bowl_phases = {p: bowl_probs for p in ["powerplay", "middle", "death"]}

    return (
        BattingCard(player_id=player.player_id, probs_by_phase=phases),
        BowlingCard(player_id=player.player_id, probs_by_phase=bowl_phases),
    )


@router.post("", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    t1_bat, t1_bowl = [], []
    for p in req.team1:
        bc, blc = _player_to_cards(p)
        t1_bat.append(bc)
        t1_bowl.append(blc)

    t2_bat, t2_bowl = [], []
    for p in req.team2:
        bc, blc = _player_to_cards(p)
        t2_bat.append(bc)
        t2_bowl.append(blc)

    bowlers1 = [
        b
        for b, p in zip(t1_bowl, req.team1, strict=False)
        if p.role in ("pure_bowl", "bowl_allrounder", "bat_allrounder")
    ][:5]
    bowlers2 = [
        b
        for b, p in zip(t2_bowl, req.team2, strict=False)
        if p.role in ("pure_bowl", "bowl_allrounder", "bat_allrounder")
    ][:5]
    if len(bowlers1) < 5:
        bowlers1 = t1_bowl[:5]
    if len(bowlers2) < 5:
        bowlers2 = t2_bowl[:5]

    ctx1 = MatchContext(batting_order=t1_bat, bowling_attack=bowlers2)
    result = simulate_match(ctx1, t2_bat, t1_bowl, n_sims=min(req.n_sims, 50000))

    return SimulateResponse(
        team1_win_prob=round(result["team1_win_prob"], 4),
        team2_win_prob=round(result["team2_win_prob"], 4),
        tie_prob=round(result["tie_prob"], 4),
        team1_scores=ScoreDistribution(**build_score_distribution(result["team1_scores"])),
        team2_scores=ScoreDistribution(**build_score_distribution(result["team2_scores"])),
    )
