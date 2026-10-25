"""POST /compare — paired delta-WP between two XI variants vs same opponent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from backend.app.schemas.models import CompareRequest, CompareResponse
from backend.app.services.selection import build_player_feature_row, format_config
from ml.simulate.cards import CardContext, build_cards
from ml.simulate.engine import MatchContext, simulate_innings

router = APIRouter()


def _quick_wp(team_bat, opp_bowl, n_sims, seed):
    ctx = MatchContext(batting_order=team_bat, bowling_attack=opp_bowl[:5])
    rng = np.random.default_rng(seed)
    r = simulate_innings(ctx, n_sims=n_sims, rng=rng)
    return r.scores.mean()


@router.post("", response_model=CompareResponse)
def compare(req: CompareRequest):
    n = min(req.n_sims, 20000)
    seed = 42

    fmt_cfg = format_config(req.format)
    ctx = CardContext(
        fmt=req.format,
        gender=req.gender,
        phase_names=fmt_cfg.get("phase_names", ["powerplay", "middle", "death"]),
        venue_id=req.venue or "unknown",
        pitch_condition=req.pitch_condition,
        max_overs_per_bowler=fmt_cfg.get("max_bowler_overs", 4),
    )

    opp_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.opponent]
    opp_avg = pd.DataFrame(opp_rows).mean(numeric_only=True).to_dict() if opp_rows else {}

    a_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team1_a]
    b_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team1_b]
    a_avg = pd.DataFrame(a_rows).mean(numeric_only=True).to_dict() if a_rows else {}
    b_avg = pd.DataFrame(b_rows).mean(numeric_only=True).to_dict() if b_rows else {}

    a_bat = [build_cards(row, opp_avg, ctx)[0] for row in a_rows]
    b_bat = [build_cards(row, opp_avg, ctx)[0] for row in b_rows]
    opp_bowl_a = [build_cards(row, a_avg, ctx)[1] for row in opp_rows]
    opp_bowl_b = [build_cards(row, b_avg, ctx)[1] for row in opp_rows]

    score_a = _quick_wp(a_bat, opp_bowl_a, n, seed)
    score_b = _quick_wp(b_bat, opp_bowl_b, n, seed)

    diff = float(score_a - score_b)
    se = float(np.sqrt(score_a + score_b) / np.sqrt(n)) * 2

    return CompareResponse(
        delta_wp=round(diff, 2),
        ci_lower=round(diff - 1.96 * se, 2),
        ci_upper=round(diff + 1.96 * se, 2),
        team_a_wp=round(float(score_a) / 200, 4),
        team_b_wp=round(float(score_b) / 200, 4),
    )
