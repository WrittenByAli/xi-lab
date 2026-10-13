"""
Pandera schemas for every parsed table.

Each schema is the single source of truth for column names, types and
constraints. Parse functions validate against these before writing Parquet.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import DataFrameModel
from pandera.typing import Series


# â”€â”€ matches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class MatchSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(unique=True, nullable=False)
    date: Series[pa.DateTime] = pa.Field(nullable=False)
    season: Series[str] = pa.Field(nullable=False)
    competition: Series[str] = pa.Field(nullable=False)
    league_id: Series[str] = pa.Field(nullable=False)
    format: Series[str] = pa.Field(isin=["t20", "odi", "test"], nullable=False)
    is_international: Series[bool] = pa.Field(nullable=False)
    venue_id: Series[str] = pa.Field(nullable=False)
    team1: Series[str] = pa.Field(nullable=False)
    team2: Series[str] = pa.Field(nullable=False)
    toss_winner: Series[str] = pa.Field(nullable=True)
    toss_decision: Series[str] = pa.Field(isin=["bat", "field", None], nullable=True)
    winner: Series[str] = pa.Field(nullable=True)
    result_type: Series[str] = pa.Field(
        isin=["normal", "tie", "no_result", "dls"], nullable=False
    )
    is_reduced: Series[bool] = pa.Field(nullable=False)
    has_super_over: Series[bool] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = "filter"


# â”€â”€ innings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class InningsSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(nullable=False)
    innings_no: Series[int] = pa.Field(ge=1, le=4)
    batting_team: Series[str] = pa.Field(nullable=False)
    bowling_team: Series[str] = pa.Field(nullable=False)
    target_runs: Series[int] = pa.Field(ge=0, nullable=True)
    is_super_over: Series[bool] = pa.Field(nullable=False)
    total_runs: Series[int] = pa.Field(ge=0)
    total_wickets: Series[int] = pa.Field(ge=0, le=10)
    total_legal_balls: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = "filter"


# â”€â”€ balls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

OUTCOME_CLASSES = ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "noball"]


class BallSchema(pa.DataFrameModel):
    # Identity
    match_id: Series[str] = pa.Field(nullable=False)
    innings_no: Series[int] = pa.Field(ge=1, le=4)
    over: Series[int] = pa.Field(ge=0, le=49)          # 0-indexed; 0-19 T20, 0-49 ODI
    legal_ball_idx: Series[int] = pa.Field(ge=0, le=5)  # within over, legal deliveries
    delivery_seq: Series[int] = pa.Field(ge=0)          # absolute sequence incl. extras

    # Players
    batter_id: Series[str] = pa.Field(nullable=False)
    non_striker_id: Series[str] = pa.Field(nullable=False)
    bowler_id: Series[str] = pa.Field(nullable=False)

    # Runs
    runs_batter: Series[int] = pa.Field(ge=0, le=6)
    runs_extras: Series[int] = pa.Field(ge=0)
    extras_type: Series[str] = pa.Field(nullable=True)

    # Flags
    is_wide: Series[bool]
    is_noball: Series[bool]
    is_legal: Series[bool]

    # Dismissal
    wicket_kind: Series[str] = pa.Field(nullable=True)
    player_out_id: Series[str] = pa.Field(nullable=True)

    # Pre-delivery state
    score_before: Series[int] = pa.Field(ge=0)
    wkts_before: Series[int] = pa.Field(ge=0, le=9)
    legal_balls_before: Series[int] = pa.Field(ge=0)
    target: Series[int] = pa.Field(ge=0, nullable=True)  # None in 1st innings
    rrr_before: Series[float] = pa.Field(nullable=True)  # can be negative when chasing team is ahead

    # Label
    outcome_class: Series[str] = pa.Field(isin=OUTCOME_CLASSES)

    class Config:
        coerce = True
        strict = "filter"


# â”€â”€ lineups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class LineupSchema(pa.DataFrameModel):
    match_id: Series[str] = pa.Field(nullable=False)
    team: Series[str] = pa.Field(nullable=False)
    player_id: Series[str] = pa.Field(nullable=False)
    bat_position_actual: Series[int] = pa.Field(ge=1, le=11, nullable=True)
    overs_bowled: Series[float] = pa.Field(ge=0, le=50, nullable=True)
    is_keeper_inferred: Series[bool] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = "filter"


# â”€â”€ player_meta â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PlayerMetaSchema(pa.DataFrameModel):
    player_id: Series[str] = pa.Field(unique=True, nullable=False)
    name: Series[str] = pa.Field(nullable=False)
    bat_hand: Series[str] = pa.Field(isin=["R", "L", "unknown"], nullable=False)
    bowl_arm: Series[str] = pa.Field(isin=["R", "L", "none", "unknown"], nullable=False)
    bowl_type: Series[str] = pa.Field(
        isin=["pace", "off", "leg", "left_orth", "left_wrist", "none", "unknown"],
        nullable=False,
    )
    is_keeper: Series[bool] = pa.Field(nullable=False)
    source: Series[str] = pa.Field(nullable=False)  # "manual" | "style_clf" | "unknown"
    verified: Series[bool] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = "filter"


# â”€â”€ venues â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class VenueSchema(pa.DataFrameModel):
    venue_id: Series[str] = pa.Field(unique=True, nullable=False)
    canonical_name: Series[str] = pa.Field(nullable=False)
    city: Series[str] = pa.Field(nullable=True)
    country: Series[str] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = "filter"


