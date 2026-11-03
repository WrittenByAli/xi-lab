"""
Generate and publish a match prediction as JSON.

Called by the publish_prediction GitHub Action workflow. When the opponent's
squad can be resolved from match history, the XI is ranked by actually
simulating each candidate against that opponent (venue/toss/pitch-aware, via
`ml.simulate.cards` + `ml.simulate.engine`) — the same evaluate_fn the
`/recommend` API endpoint builds. Otherwise it falls back to the plain
rating-sum heuristic, so the script still works pre-season with no fixture
history for the opponent.

Usage:
    python ml/optimize/publish.py \
        --team India --opponent Australia --venue "MCG" \
        --format t20 --date 2024-03-15 \
        --squad "V Kohli,R Sharma,JJ Bumrah,..." \
        --out live/predictions/India_vs_Australia_2024-03-15.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from backend.app.routers.teams import get_squad
from backend.app.services.selection import (
    build_player_feature_row,
    enrich_with_snapshot,
    format_config,
)
from ml.optimize.selector import PlayerOption, select_xi
from ml.simulate.cards import CardContext, build_cards
from ml.simulate.engine import MatchContext, simulate_match

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

_BOWLING_ROLES = ("pure_bowl", "bowl_allrounder", "bat_allrounder")
_OPPONENT_EVAL_SIM_CAP = 25


def _load_player_meta() -> dict:
    meta = pd.read_csv(ROOT / "data" / "reference" / "player_meta.csv", comment="#")
    return {row["player_id"]: row.to_dict() for _, row in meta.iterrows()}


def _load_ratings(gender: str = "male") -> dict:
    path = ROOT / "data" / "processed" / "models" / "ability" / gender / "ratings.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    latest = df.sort_values("snapshot_month").groupby("player_id").last()
    return latest.to_dict(orient="index")


def _model_hash(fmt: str, gender: str) -> str:
    h = hashlib.sha256()
    for name in (
        f"ability/{gender}/ratings.parquet",
        f"ball/model_{fmt}_{gender}.lgb",
    ):
        path = ROOT / "data" / "processed" / "models" / name
        if path.exists():
            h.update(path.read_bytes()[:4096])
    return h.hexdigest()[:12]


def _build_opponent_evaluate_fn(
    fmt: str,
    gender: str,
    opponent_team: str,
    venue: str | None,
    toss_decision: str | None,
    pitch_condition: str,
    squad_rows: list[dict],
    sim_cap: int = _OPPONENT_EVAL_SIM_CAP,
):
    """Mirrors `backend.app.routers.recommend._build_opponent_evaluate_fn` —
    kept as a standalone copy here because this script runs outside the API
    process (a scheduled GitHub Action), not behind it.

    `sim_cap` defaults to the same cap `/recommend` uses, since that value
    is only ever asked for inside racing search (many candidates, every
    round). The final published win_prob is a single one-off evaluation of
    the winning XI, so `main()` rebuilds this with a much higher `sim_cap`
    for that one call — no reason to publish a number read off 25 trials.
    """
    try:
        opp_squad = get_squad(team=opponent_team, format=fmt, gender=gender, months=12)
    except Exception as exc:  # noqa: BLE001 — degrade to heuristic, never crash the publish job
        print(f"[warn] could not resolve opponent squad for '{opponent_team}': {exc}")
        return None

    opp_players = opp_squad["players"][:11]
    if len(opp_players) < 11:
        print(f"[warn] opponent '{opponent_team}' has < 11 recent players; using heuristic")
        return None

    fmt_cfg = format_config(fmt)
    ctx = CardContext(
        fmt=fmt,
        gender=gender,
        phase_names=fmt_cfg.get("phase_names", ["powerplay", "middle", "death"]),
        venue_id=venue or "unknown",
        toss_bat_first=(toss_decision == "bat"),
        pitch_condition=pitch_condition,
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
            fmt,
            gender,
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

    player_cards = {row["player_id"]: build_cards(row, opp_avg, ctx) for row in squad_rows}

    total_overs = fmt_cfg.get("overs", 20)
    phases = fmt_cfg.get("phases", [[0, 5], [6, 15], [16, 19]])

    def evaluate_fn(xi: list[PlayerOption], n_sims: int = 1000) -> float:
        cards = [player_cards[p.player_id] for p in xi]
        bat_cards = [c[0] for c in cards]
        bowl_cards = [c[1] for c, p in zip(cards, xi, strict=False) if p.role in _BOWLING_ROLES][
            :5
        ]
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
            match_ctx, opp_bat_cards, bowl_cards, n_sims=min(n_sims, sim_cap)
        )
        return result["team1_win_prob"] + 0.5 * result["tie_prob"]

    return evaluate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--gender", choices=["male", "female"], default="male")
    parser.add_argument("--date", required=True)
    parser.add_argument("--squad", required=True)
    parser.add_argument("--toss-decision", choices=["bat", "bowl"], default=None)
    parser.add_argument("--pitch-condition", default="auto")
    parser.add_argument("--max-overseas", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    squad_ids = [s.strip() for s in args.squad.split(",")]
    meta = _load_player_meta()
    ratings = _load_ratings(args.gender)

    squad = []
    for pid in squad_ids:
        info = meta.get(pid, {})
        rating = ratings.get(pid, {})
        squad.append(
            PlayerOption(
                player_id=pid,
                name=info.get("name", pid),
                role=rating.get("role", "pure_bat"),
                bat_rating=float(rating.get("bat_rating", 0.5)),
                bowl_rating=float(rating.get("bowl_rating", -50.0)),
                is_keeper=bool(info.get("is_keeper", False)),
                is_overseas=False,
                bat_hand=str(info.get("bat_hand", "R")),
                bowl_type=str(info.get("bowl_type", "none")),
            )
        )

    squad_rows = [build_player_feature_row(p, args.format, args.gender) for p in squad]
    evaluate_fn = _build_opponent_evaluate_fn(
        args.format,
        args.gender,
        args.opponent,
        args.venue,
        args.toss_decision,
        args.pitch_condition,
        squad_rows,
    )

    result = select_xi(squad, evaluate_fn=evaluate_fn, max_overseas=args.max_overseas)

    # Racing search only ever needs `evaluate_fn` capped (cheap, many
    # candidates x rounds). The winning XI is evaluated exactly once more
    # here, uncapped, so the published win_prob has real simulation fidelity
    # instead of the ~25-trial estimate racing search used internally.
    final_win_prob = result.win_prob
    if evaluate_fn is not None:
        final_evaluate_fn = _build_opponent_evaluate_fn(
            args.format,
            args.gender,
            args.opponent,
            args.venue,
            args.toss_decision,
            args.pitch_condition,
            squad_rows,
            sim_cap=PARAMS["simulator"]["n_sims_default"],
        )
        if final_evaluate_fn is not None:
            final_win_prob = final_evaluate_fn(result.xi, PARAMS["simulator"]["n_sims_default"])

    prediction = {
        "team": args.team,
        "opponent": args.opponent,
        "venue": args.venue,
        "format": args.format,
        "gender": args.gender,
        "match_date": args.date,
        "toss_decision": args.toss_decision,
        "pitch_condition": args.pitch_condition,
        "context_aware": evaluate_fn is not None,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_hash": _model_hash(args.format, args.gender),
        "xi": [
            {
                "player_id": p.player_id,
                "name": p.name,
                "role": p.role,
                "bat_rating": round(p.bat_rating, 4),
                "bowl_rating": round(p.bowl_rating, 4),
            }
            for p in result.xi
        ],
        "win_prob": round(final_win_prob, 4),
        "batting_order": result.batting_order,
        "bowling_plan": result.bowling_plan,
        "tie_set": result.tie_set,
        "swap_matrix": result.swap_matrix,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prediction, indent=2))
    print(f"Prediction written to {out_path}")


if __name__ == "__main__":
    main()
