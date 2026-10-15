"""POST /compare — paired delta-WP between two XI variants vs same opponent."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from backend.app.schemas.models import CompareRequest, CompareResponse
from ml.simulate.engine import (
    BattingCard,
    BowlingCard,
    MatchContext,
    simulate_innings,
)

router = APIRouter()

_DEFAULT_PROBS = np.array([0.35, 0.25, 0.08, 0.01, 0.12, 0.06, 0.08, 0.03, 0.02])


def _norm_bat(raw: float) -> float:
    return float(np.clip((raw - 25) / 75, 0, 1))


def _norm_bowl(raw: float) -> float:
    return float(np.clip((-raw - 10) / 40, 0, 1))


def _make_cards(player):
    br = _norm_bat(player.bat_rating)
    wr = _norm_bowl(player.bowl_rating)
    bat_probs = _DEFAULT_PROBS.copy()
    bat_probs[0] = max(0.05, 0.40 - br * 0.1)
    bat_probs[4] = min(0.25, 0.08 + br * 0.05)
    bat_probs /= bat_probs.sum()
    bowl_probs = _DEFAULT_PROBS.copy()
    bowl_probs[6] = min(0.20, 0.06 + wr * 0.04)
    bowl_probs /= bowl_probs.sum()
    phases = {p: bat_probs for p in ["powerplay", "middle", "death"]}
    bowl_phases = {p: bowl_probs for p in ["powerplay", "middle", "death"]}
    return (
        BattingCard(player_id=player.player_id, probs_by_phase=phases),
        BowlingCard(player_id=player.player_id, probs_by_phase=bowl_phases),
    )


def _quick_wp(team_bat, opp_bowl, n_sims, seed):
    ctx = MatchContext(batting_order=team_bat, bowling_attack=opp_bowl[:5])
    rng = np.random.default_rng(seed)
    r = simulate_innings(ctx, n_sims=n_sims, rng=rng)
    return r.scores.mean()


@router.post("", response_model=CompareResponse)
def compare(req: CompareRequest):
    n = min(req.n_sims, 20000)
    seed = 42

    a_bat = [_make_cards(p)[0] for p in req.team1_a]
    b_bat = [_make_cards(p)[0] for p in req.team1_b]
    opp_bowl = [_make_cards(p)[1] for p in req.opponent]

    score_a = _quick_wp(a_bat, opp_bowl, n, seed)
    score_b = _quick_wp(b_bat, opp_bowl, n, seed)

    diff = float(score_a - score_b)
    se = float(np.sqrt(score_a + score_b) / np.sqrt(n)) * 2

    return CompareResponse(
        delta_wp=round(diff, 2),
        ci_lower=round(diff - 1.96 * se, 2),
        ci_upper=round(diff + 1.96 * se, 2),
        team_a_wp=round(float(score_a) / 200, 4),
        team_b_wp=round(float(score_b) / 200, 4),
    )
