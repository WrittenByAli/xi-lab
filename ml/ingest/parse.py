"""
Parse Cricsheet JSON files → ball, innings, match, lineup Parquet tables.

Usage:
    python ml/ingest/parse.py                    # parse all formats
    python ml/ingest/parse.py --format t20       # parse T20s only

Each table is validated against its Pandera schema before writing.
Scorecard reconciliation: parsed innings totals must match Cricsheet totals.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from ml.ingest.normalize import competition_to_league, encode_outcome, venue_to_id
from ml.ingest.schemas import (
    BallSchema,
    InningsSchema,
    LineupSchema,
    MatchSchema,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "cricsheet"
INTERIM = ROOT / "data" / "interim"

FORMAT_DIRS = {
    "t20": RAW / "t20s",
    "odi": RAW / "odis",
    "test": RAW / "tests",
}


# ── Single-match parser ───────────────────────────────────────────────────────


def parse_match(path: Path) -> dict[str, list[dict]]:
    """
    Parse one Cricsheet JSON file.

    Returns a dict with keys 'match', 'innings', 'balls', 'lineup'.
    Raises ValueError with the match_id if reconciliation fails.
    """
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    info: dict = raw.get("info", {})
    deliveries_data: list[dict] = raw.get("innings", [])

    # ── Match-level fields ────────────────────────────────────────────────────
    match_id: str = info.get("cricsheet_id") or str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))
    date_str = info.get("dates", ["1900-01-01"])[0]
    competition = info.get("competition", info.get("event", {}).get("name", "unknown"))
    if isinstance(competition, dict):
        competition = competition.get("name", "unknown")
    match_type = info.get("match_type", "T20")

    league_info = competition_to_league(competition, match_type)
    teams: list[str] = info.get("teams", ["", ""])
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})

    winner = outcome.get("winner")
    result_type = "normal"
    if "tie" in outcome:
        result_type = "tie"
    elif "no result" in str(outcome.get("result", "")).lower():
        result_type = "no_result"
    elif outcome.get("method", "").lower() == "dls":
        result_type = "dls"

    match_row = {
        "match_id": match_id,
        "date": pd.Timestamp(date_str),
        "season": str(info.get("season", date_str[:4])),
        "competition": competition,
        "league_id": league_info["league_id"],
        "format": league_info["format"],
        "is_international": league_info["is_international"],
        "venue_id": venue_to_id(info.get("venue")),
        "team1": teams[0] if len(teams) > 0 else "",
        "team2": teams[1] if len(teams) > 1 else "",
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result_type": result_type,
        "is_reduced": bool(outcome.get("method") == "D/L"),
        "has_super_over": any(inn.get("super_over", False) for inn in deliveries_data),
    }

    innings_rows: list[dict] = []
    ball_rows: list[dict] = []
    lineup_rows: list[dict] = []

    for innings_idx, innings_data in enumerate(deliveries_data, start=1):
        batting_team = innings_data.get("team", "")
        bowling_team = next((t for t in teams if t != batting_team), "")
        is_super_over = innings_data.get("super_over", False)
        target_info = innings_data.get("target", {})
        target_runs = target_info.get("runs") if target_info else None

        overs_data: list[dict] = innings_data.get("overs", [])

        # --- Walk deliveries to build ball rows and track state ---------------
        score = 0
        wickets = 0
        legal_balls = 0
        batting_order: list[str] = []  # order of appearance
        bowler_overs: dict[str, float] = {}
        is_keeper_set: set[str] = set()

        innings_ball_rows: list[dict] = []

        for over_data in overs_data:
            over_no: int = over_data.get("over", 0)
            for delivery in over_data.get("deliveries", []):
                batter_id = delivery.get("batter", "")
                non_striker_id = delivery.get("non_striker", "")
                bowler_id = delivery.get("bowler", "")

                if batter_id not in batting_order:
                    batting_order.append(batter_id)
                if non_striker_id not in batting_order:
                    batting_order.append(non_striker_id)

                runs_info = delivery.get("runs", {})
                runs_batter = int(runs_info.get("batter", 0))
                runs_extras = int(runs_info.get("extras", 0))
                extras = delivery.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not is_wide and not is_noball

                wicket_info = delivery.get("wickets", [{}])[0] if delivery.get("wickets") else {}
                is_wicket = bool(wicket_info)
                wicket_kind = wicket_info.get("kind") if is_wicket else None
                player_out_id = wicket_info.get("player_out") if is_wicket else None

                # Keep-track of stumpings → keeper inference
                if wicket_kind == "stumped":
                    for fielder in wicket_info.get("fielders", []):
                        is_keeper_set.add(fielder.get("name", ""))

                rrr = None
                if target_runs and legal_balls < (
                    20 * 6 if league_info["format"] == "t20" else 50 * 6
                ):
                    balls_remaining = (
                        20 * 6 - legal_balls
                        if league_info["format"] == "t20"
                        else 50 * 6 - legal_balls
                    )
                    runs_needed = target_runs - score
                    rrr = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else None

                outcome_class = encode_outcome(runs_batter, is_wide, is_noball, is_wicket)

                innings_ball_rows.append(
                    {
                        "match_id": match_id,
                        "innings_no": innings_idx,
                        "over": over_no,
                        "legal_ball_idx": legal_balls % 6,
                        "delivery_seq": len(innings_ball_rows),
                        "batter_id": batter_id,
                        "non_striker_id": non_striker_id,
                        "bowler_id": bowler_id,
                        "runs_batter": runs_batter,
                        "runs_extras": runs_extras,
                        "extras_type": next(iter(extras.keys()), None) if extras else None,
                        "is_wide": is_wide,
                        "is_noball": is_noball,
                        "is_legal": is_legal,
                        "wicket_kind": wicket_kind,
                        "player_out_id": player_out_id,
                        "score_before": score,
                        "wkts_before": wickets,
                        "legal_balls_before": legal_balls,
                        "target": target_runs,
                        "rrr_before": rrr,
                        "outcome_class": outcome_class,
                    }
                )

                score += runs_batter + runs_extras
                if is_legal:
                    legal_balls += 1
                if is_wicket:
                    wickets += 1
                bowler_overs[bowler_id] = bowler_overs.get(bowler_id, 0) + (
                    1 / 6 if is_legal else 0
                )

        # --- Scorecard reconciliation ----------------------------------------
        cricsheet_score = innings_data.get("runs", innings_data.get("score"))
        if cricsheet_score is not None and score != int(cricsheet_score):
            raise ValueError(
                f"Scorecard mismatch match={match_id} innings={innings_idx}: "
                f"parsed={score} cricsheet={cricsheet_score}"
            )

        innings_rows.append(
            {
                "match_id": match_id,
                "innings_no": innings_idx,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "target_runs": target_runs,
                "is_super_over": is_super_over,
                "total_runs": score,
                "total_wickets": wickets,
                "total_legal_balls": legal_balls,
            }
        )

        ball_rows.extend(innings_ball_rows)

        # --- Lineup rows for this innings -------------------------------------
        for pos, pid in enumerate(batting_order, start=1):
            lineup_rows.append(
                {
                    "match_id": match_id,
                    "team": batting_team,
                    "player_id": pid,
                    "bat_position_actual": pos if pos <= 11 else None,
                    "overs_bowled": None,
                    "is_keeper_inferred": pid in is_keeper_set,
                }
            )
        for pid, overs in bowler_overs.items():
            # Update or append bowling entry
            existing = next(
                (
                    r
                    for r in lineup_rows
                    if r["match_id"] == match_id
                    and r["team"] == bowling_team
                    and r["player_id"] == pid
                ),
                None,
            )
            if existing:
                existing["overs_bowled"] = round(overs, 2)
            else:
                lineup_rows.append(
                    {
                        "match_id": match_id,
                        "team": bowling_team,
                        "player_id": pid,
                        "bat_position_actual": None,
                        "overs_bowled": round(overs, 2),
                        "is_keeper_inferred": False,
                    }
                )

    return {
        "match": [match_row],
        "innings": innings_rows,
        "balls": ball_rows,
        "lineup": lineup_rows,
    }


# ── Batch parser ──────────────────────────────────────────────────────────────


def parse_format(fmt: str) -> None:
    src_dir = FORMAT_DIRS[fmt]
    paths = sorted(src_dir.glob("*.json"))
    if not paths:
        logger.warning("No JSON files found in %s", src_dir)
        return

    logger.info("Parsing %d %s files...", len(paths), fmt)

    all_matches, all_innings, all_balls, all_lineups = [], [], [], []
    errors: list[str] = []

    for path in paths:
        try:
            result = parse_match(path)
            all_matches.extend(result["match"])
            all_innings.extend(result["innings"])
            all_balls.extend(result["balls"])
            all_lineups.extend(result["lineup"])
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        logger.warning("%d parse errors:", len(errors))
        for e in errors[:20]:
            logger.warning("  %s", e)

    out_dir = INTERIM / fmt
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(rows: list[dict], schema: type[pa.DataFrameModel], name: str) -> None:
        df = pd.DataFrame(rows)
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            logger.warning(
                "Schema warnings in %s/%s: %d failures", fmt, name, len(exc.failure_cases)
            )
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        logger.info("  Saved %s/%s: %d rows", fmt, name, len(df))

    _save(all_matches, MatchSchema, "matches")
    _save(all_innings, InningsSchema, "innings")
    _save(all_balls, BallSchema, "balls")
    _save(all_lineups, LineupSchema, "lineups")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Cricsheet JSON → Parquet")
    parser.add_argument(
        "--format",
        choices=list(FORMAT_DIRS.keys()),
        default=None,
        help="Parse only this format (default: all)",
    )
    args = parser.parse_args()

    formats = [args.format] if args.format else list(FORMAT_DIRS.keys())
    for fmt in formats:
        parse_format(fmt)
