"""POST /compare — paired delta-WP between two XI variants vs same opponent."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter

from backend.app.schemas.models import CompareRequest, CompareResponse
from backend.app.services.selection import build_player_feature_row, format_config
from ml.simulate.cards import CardContext, build_cards
from ml.simulate.engine import MatchContext, simulate_match

router = APIRouter()

_BOWLING_ROLES = ("pure_bowl", "bowl_allrounder", "bat_allrounder")


def _bowling_attack(bowl_cards: list, players: list) -> list:
    bowlers = [b for b, p in zip(bowl_cards, players, strict=False) if p.role in _BOWLING_ROLES][:5]
    return bowlers if len(bowlers) == 5 else bowl_cards[:5]


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
    total_overs = fmt_cfg.get("overs", 20)
    phases = fmt_cfg.get("phases", [[0, 5], [6, 15], [16, 19]])

    opp_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.opponent]
    opp_avg = pd.DataFrame(opp_rows).mean(numeric_only=True).to_dict() if opp_rows else {}

    a_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team1_a]
    b_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.team1_b]
    a_avg = pd.DataFrame(a_rows).mean(numeric_only=True).to_dict() if a_rows else {}
    b_avg = pd.DataFrame(b_rows).mean(numeric_only=True).to_dict() if b_rows else {}

    a_bat = [build_cards(row, opp_avg, ctx)[0] for row in a_rows]
    b_bat = [build_cards(row, opp_avg, ctx)[0] for row in b_rows]
    opp_cards_vs_a = [build_cards(row, a_avg, ctx) for row in opp_rows]
    opp_cards_vs_b = [build_cards(row, b_avg, ctx) for row in opp_rows]
    opp_bat_vs_a = [c[0] for c in opp_cards_vs_a]
    opp_bat_vs_b = [c[0] for c in opp_cards_vs_b]
    opp_bowl_vs_a = _bowling_attack([c[1] for c in opp_cards_vs_a], req.opponent)
    opp_bowl_vs_b = _bowling_attack([c[1] for c in opp_cards_vs_b], req.opponent)

    # Variant A/B each bat first against the same opponent's attack, then the
    # opponent chases — a full match each (not just a projected 1st-innings
    # score), with both sides on the *same* crn_seed so the comparison is
    # paired: the opponent faces identical random draws in both runs, which
    # is what makes a per-simulation win-probability difference statistically
    # valid instead of comparing two independent, noisier estimates.
    ctx_a = MatchContext(
        batting_order=a_bat,
        bowling_attack=opp_bowl_vs_a,
        total_overs=total_overs,
        phases=phases,
        phase_names=ctx.phase_names,
    )
    ctx_b = MatchContext(
        batting_order=b_bat,
        bowling_attack=opp_bowl_vs_b,
        total_overs=total_overs,
        phases=phases,
        phase_names=ctx.phase_names,
    )
    res_a = simulate_match(ctx_a, opp_bat_vs_a, opp_bowl_vs_a, n_sims=n, crn_seed=seed)
    res_b = simulate_match(ctx_b, opp_bat_vs_b, opp_bowl_vs_b, n_sims=n, crn_seed=seed)

    # Per-simulation win indicator (team1 wins iff the opponent's chase
    # falls short), derived from the same score arrays simulate_match uses
    # internally to compute its own aggregate win_prob — lets the CI below
    # use the actual paired variance instead of an aggregate approximation.
    win_a = (res_a["team2_scores"] < res_a["team1_scores"]).astype(float)
    win_b = (res_b["team2_scores"] < res_b["team1_scores"]).astype(float)
    diff_per_sim = win_a - win_b

    wp_a = res_a["team1_win_prob"] + 0.5 * res_a["tie_prob"]
    wp_b = res_b["team1_win_prob"] + 0.5 * res_b["tie_prob"]
    delta = float(diff_per_sim.mean())
    se = float(diff_per_sim.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    return CompareResponse(
        delta_wp=round(delta, 4),
        ci_lower=round(delta - 1.96 * se, 4),
        ci_upper=round(delta + 1.96 * se, 4),
        team_a_wp=round(float(wp_a), 4),
        team_b_wp=round(float(wp_b), 4),
    )
