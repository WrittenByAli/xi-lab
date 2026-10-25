"""POST /simulate — win probability + score distributions for two XIs."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from backend.app.schemas.models import (
    ScoreDistribution,
    SimulateRequest,
    SimulateResponse,
)
from backend.app.services.selection import (
    build_player_feature_row,
    build_score_distribution,
    format_config,
)
from ml.simulate.cards import CardContext, build_cards
from ml.simulate.engine import MatchContext, simulate_match

router = APIRouter()


@router.post("", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    fmt_cfg = format_config(req.format)
    ctx = CardContext(
        fmt=req.format,
        gender=req.gender,
        phase_names=fmt_cfg.get("phase_names", ["powerplay", "middle", "death"]),
        venue_id=req.venue or "unknown",
        toss_bat_first=(req.toss_decision == "bat"),
        pitch_condition=req.pitch_condition,
        max_overs_per_bowler=fmt_cfg.get("max_bowler_overs", 4),
    )

    t1_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team1]
    t2_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team2]
    t1_opp_avg = {}  # team2's own average stands in for "the attack team1 faces"
    t2_opp_avg = {}
    if t2_rows:
        t1_opp_avg = pd.DataFrame(t2_rows).mean(numeric_only=True).to_dict()
    if t1_rows:
        t2_opp_avg = pd.DataFrame(t1_rows).mean(numeric_only=True).to_dict()

    t1_bat, t1_bowl = [], []
    for row in t1_rows:
        bc, blc = build_cards(row, t1_opp_avg, ctx)
        t1_bat.append(bc)
        t1_bowl.append(blc)

    t2_bat, t2_bowl = [], []
    for row in t2_rows:
        bc, blc = build_cards(row, t2_opp_avg, ctx)
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

    total_overs = fmt_cfg.get("overs", 20)
    ctx1 = MatchContext(
        batting_order=t1_bat,
        bowling_attack=bowlers2,
        total_overs=total_overs,
        phases=fmt_cfg.get("phases", [[0, 5], [6, 15], [16, 19]]),
        phase_names=ctx.phase_names,
    )
    result = simulate_match(ctx1, t2_bat, t1_bowl, n_sims=min(req.n_sims, 50000))

    return SimulateResponse(
        team1_win_prob=round(result["team1_win_prob"], 4),
        team2_win_prob=round(result["team2_win_prob"], 4),
        tie_prob=round(result["tie_prob"], 4),
        team1_scores=ScoreDistribution(**build_score_distribution(result["team1_scores"])),
        team2_scores=ScoreDistribution(**build_score_distribution(result["team2_scores"])),
    )
