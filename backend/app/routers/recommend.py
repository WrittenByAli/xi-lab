"""POST /recommend — best XI from a squad.

When `opponent_team` is given, candidate XIs are ranked by actually
simulating them against that opponent's current squad (venue/toss/pitch
aware, via `ml.simulate.cards`) — otherwise falls back to the previous
rating-sum heuristic, so the endpoint still works with no match context.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app.routers.teams import get_squad
from backend.app.schemas.models import (
    PlayerSummary,
    RecommendRequest,
    RecommendResponse,
)
from backend.app.services.selection import (
    build_player_feature_row,
    enrich_with_snapshot,
    format_config,
)
from ml.optimize.selector import PlayerOption, select_xi
from ml.simulate.cards import CardContext, build_cards
from ml.simulate.engine import MatchContext, simulate_match

router = APIRouter()

_BOWLING_ROLES = ("pure_bowl", "bowl_allrounder", "bat_allrounder")

# The ball-by-ball simulator is not vectorized over simulations (each ball is
# a Python-level loop), so it's cheap enough for one bounded /simulate call
# but too slow to call at racing-search's full escalating sim budget (up to
# 8000) across ~30 candidates x 4 rounds x the swap-matrix's own pass. Common
# random numbers (every candidate shares the same seed) make even a modest
# sample size give a stable *relative* ranking, which is all the optimizer
# needs — so every evaluate_fn call here is capped regardless of what the
# racing-search schedule requests.
_OPPONENT_EVAL_SIM_CAP = 25


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


def _build_opponent_evaluate_fn(req: RecommendRequest, opponent_team: str, squad_rows: list[dict]):
    """Build an evaluate_fn(xi, n_sims) -> win_prob that simulates each
    candidate XI against `opponent_team`'s current squad. Returns None
    (caller falls back to the default heuristic) if the opponent can't be
    resolved — an optional feature should degrade, not 500.

    Every squad player's cards are built exactly once up front (LightGBM
    inference is not free) — `evaluate_fn` just looks candidates up by id
    from that precomputed pool rather than rebuilding/re-predicting per
    racing-search round.
    """
    try:
        opp_squad = get_squad(team=opponent_team, format=req.format, gender=req.gender, months=12)
    except HTTPException:
        return None

    opp_players = opp_squad["players"][:11]
    if len(opp_players) < 11:
        return None

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

    squad_avg = pd.DataFrame(squad_rows).mean(numeric_only=True).to_dict() if squad_rows else {}

    opp_rows = [
        enrich_with_snapshot(
            {
                "player_id": p["player_id"],
                "bat_hand": p.get("bat_hand", "R"),
                "bowl_type": p.get("bowl_type", "none"),
                "bat_rating": p.get("bat_rating", 0.5),
                "bowl_rating": p.get("bowl_rating", -50.0),
            },
            p["player_id"],
            req.format,
            req.gender,
        )
        for p in opp_players
    ]
    opp_avg = pd.DataFrame(opp_rows).mean(numeric_only=True).to_dict() if opp_rows else {}

    opp_cards = [build_cards(row, squad_avg, ctx) for row in opp_rows]
    opp_bat_cards = [c[0] for c in opp_cards]
    opp_bowl_cards = [
        c[1]
        for c, p in zip(opp_cards, opp_players, strict=False)
        if p.get("role") in _BOWLING_ROLES
    ][:5]
    if len(opp_bowl_cards) < 5:
        opp_bowl_cards = [c[1] for c in opp_cards][:5]

    # Precompute every squad player's cards once — candidate XIs are just
    # different 11-subsets of the same pool, so each player's probability
    # profile (which depends only on them + the opponent, not the specific
    # candidate) never needs to be rebuilt.
    player_cards = {row["player_id"]: build_cards(row, opp_avg, ctx) for row in squad_rows}

    total_overs = fmt_cfg.get("overs", 20)
    phases = fmt_cfg.get("phases", [[0, 5], [6, 15], [16, 19]])

    def evaluate_fn(xi: list[PlayerOption], n_sims: int = 1000) -> float:
        cards = [player_cards[p.player_id] for p in xi]
        bat_cards = [c[0] for c in cards]
        bowl_cards = [c[1] for c, p in zip(cards, xi, strict=False) if p.role in _BOWLING_ROLES][:5]
        if len(bowl_cards) < 5:
            bowl_cards = [c[1] for c in cards][:5]

        match_ctx = MatchContext(
            batting_order=bat_cards,
            bowling_attack=opp_bowl_cards,
            total_overs=total_overs,
            phases=phases,
            phase_names=ctx.phase_names,
        )
        result = simulate_match(
            match_ctx, opp_bat_cards, bowl_cards, n_sims=min(n_sims, _OPPONENT_EVAL_SIM_CAP)
        )
        return result["team1_win_prob"] + 0.5 * result["tie_prob"]

    return evaluate_fn


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if len(req.squad) < 11:
        raise HTTPException(400, "Squad must have at least 11 players")

    squad = [_to_player_option(p) for p in req.squad]

    evaluate_fn = None
    if req.opponent_team:
        squad_rows = [build_player_feature_row(p, req.format, req.gender) for p in req.squad]
        evaluate_fn = _build_opponent_evaluate_fn(req, req.opponent_team, squad_rows)

    result = select_xi(squad, evaluate_fn=evaluate_fn, max_overseas=req.max_overseas)

    # select_xi's "at least 1 keeper" constraint is infeasible when the
    # squad itself has no one flagged is_keeper — PuLP reports that as
    # LpStatusInfeasible for every candidate, generate_candidate_xis then
    # falls back to a greedy selection that silently skips the keeper slot
    # rather than failing outright (so recommend() always returns *some*
    # XI). That's reasonable degradation, but silently fielding a team
    # with no keeper is worth surfacing rather than hiding.
    warnings = []
    if not any(p.is_keeper for p in squad):
        warnings.append(
            "No recognized wicketkeeper in this squad — the recommended XI has none. "
            "Add a specialist keeper to the squad for a fieldable team."
        )

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
        warnings=warnings,
    )
