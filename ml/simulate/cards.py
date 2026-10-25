"""
Build per-player BattingCard/BowlingCard outcome-probability profiles.

This is the single place that turns a player + match context (format,
gender, venue, toss, pitch) into the probability vectors the Monte Carlo
simulator (`engine.py`) consumes. It replaces three near-identical
hand-rolled heuristics that used to live separately in
`backend/app/routers/{simulate,compare,recommend}.py`.

Two paths, chosen automatically per (format, gender):

1. **Real model** (`load_ball_model` finds a trained LightGBM file): each
   phase's outcome vector comes from the actual ball-outcome model, fed the
   player's own snapshot features plus `venue_id`/`toss_bat_first` — this is
   the honest channel for a ground/opponent/toss to matter. The opposing
   side's features are either a specific opponent's average (when one was
   selected) or the format's league average.
2. **Rating heuristic fallback** (no trained model for that format/gender —
   e.g. Test, or any combination not yet trained): the previous hand-rolled
   `bat_rating`/`bowl_rating` -> outcome-vector heuristic, unchanged, so
   every format/gender combination still produces *something* simulable.

`pitch.py`'s manual overrides are applied last, on top of either path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.models.ball.train import (
    ALL_FEATURES,
    BATTER_SNAPSHOT_FEATURES,
    BOWLER_SNAPSHOT_FEATURES,
    CATEGORICAL_FEATURES,
)
from ml.simulate.engine import BattingCard, BowlingCard
from ml.simulate.pitch import OUTCOME_CLASSES, apply_pitch_adjustment

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "data" / "processed" / "models" / "ball"
METRICS_DIR = ROOT / "data" / "processed" / "metrics"
SNAPSHOTS_DIR = ROOT / "data" / "processed" / "snapshots"

# Representative phase names the ball model was trained with (hardcoded in
# ball/train.py's feature list regardless of format — a pre-existing
# limitation, not introduced here: ODI's own phase_names in params.yaml
# don't include "middle", so those columns are NaN for ODI rows both at
# train and inference time, same as before).
_TRAINED_PHASES = ["powerplay", "middle", "death"]

_DEFAULT_PROBS = np.array([0.35, 0.25, 0.08, 0.01, 0.12, 0.06, 0.08, 0.03, 0.02])


@dataclass
class BallModelBundle:
    booster: lgb.Booster
    encoders: dict[str, dict[str, int]]
    features: list[str]
    categorical_features: list[str]
    outcome_classes: list[str]


@lru_cache(maxsize=8)
def load_ball_model(fmt: str, gender: str) -> BallModelBundle | None:
    """Load the trained ball-outcome model + its persisted categorical vocab.

    Returns None when no model was trained for this (format, gender) — the
    caller should fall back to the rating heuristic (e.g. Test format).
    """
    model_path = MODELS_DIR / f"model_{fmt}_{gender}.lgb"
    meta_path = METRICS_DIR / f"ball_{fmt}_{gender}.json"
    if not model_path.exists() or not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    booster = lgb.Booster(model_file=str(model_path))
    encoders = {
        col: {val: i for i, val in enumerate(values)}
        for col, values in meta.get("encoders", {}).items()
    }
    return BallModelBundle(
        booster=booster,
        encoders=encoders,
        features=meta.get("features", ALL_FEATURES),
        categorical_features=meta.get("categorical_features", CATEGORICAL_FEATURES),
        outcome_classes=meta.get("outcome_classes", []),
    )


@lru_cache(maxsize=8)
def league_average_snapshot(fmt: str, gender: str) -> dict[str, float] | None:
    """Mean of the latest month's batter/bowler snapshot features.

    Used as the "generic opponent" when no specific opponent is selected.
    """
    path = SNAPSHOTS_DIR / fmt / gender / "snapshots.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=None)
    latest = df[df["snapshot_month"] == df["snapshot_month"].max()]
    cols = [c for c in BATTER_SNAPSHOT_FEATURES + BOWLER_SNAPSHOT_FEATURES if c in latest.columns]
    if not cols:
        return None
    return latest[cols].mean(numeric_only=True).to_dict()


def average_snapshot_rows(rows: list[dict]) -> dict[str, float]:
    """Mean snapshot features across a specific set of players (an opponent XI)."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    cols = [c for c in BATTER_SNAPSHOT_FEATURES + BOWLER_SNAPSHOT_FEATURES if c in df.columns]
    return df[cols].mean(numeric_only=True).to_dict()


def _encode(bundle: BallModelBundle, col: str, value: object) -> int | float:
    """Map a raw categorical value to its training-time class index.

    Falls back to the "unknown" class for values the training vocabulary
    never saw — but some columns (notably `venue_id`: every historical ball
    always has a real venue) never had an "unknown" class fitted at all. In
    that case, silently defaulting to class 0 would pick an arbitrary real
    venue and bias the prediction toward it; np.nan instead lets LightGBM's
    own native missing-value handling take over, which is the honest
    representation of "no venue specified."
    """
    enc = bundle.encoders.get(col, {})
    key = "unknown" if value is None else str(value)
    if key in enc:
        return enc[key]
    if "unknown" in enc:
        return enc["unknown"]
    return np.nan


def _phase_situation(fmt: str, phase_idx: int, n_phases: int) -> dict:
    """Simple, documented situational defaults for a representative ball in
    this phase — not a specific live match state, just enough context for
    the model to place the prediction in roughly the right part of an
    innings. over/legal_balls_before/score_before/wkts_before are a neutral
    ~8 runs/over trajectory with wickets accruing steadily; rrr_before is a
    neutral target rate rather than a specific chase.
    """
    overs_total = {"t20": 20, "odi": 50, "test": 90}.get(fmt, 20)
    over = round((phase_idx + 0.5) / n_phases * overs_total)
    over = min(over, overs_total - 1)
    return {
        "over": over,
        "legal_balls_before": over * 6,
        "score_before": round(over * 8),
        "wkts_before": min(over // 4, 6),
        "rrr_before": 8.0,
        "innings_no": 1,
        "is_international": 1,
        "toss_bat_first": 0,
    }


def _predict_phase_probs(
    bundle: BallModelBundle,
    batter_feats: dict,
    bowler_feats: dict,
    batter_hand: str | None,
    bowler_type: str | None,
    venue_id: str,
    toss_bat_first: bool,
    situation: dict,
) -> np.ndarray:
    row: dict[str, object] = dict(situation)
    row["toss_bat_first"] = int(toss_bat_first)
    for col in BATTER_SNAPSHOT_FEATURES:
        row[col] = batter_feats.get(col, np.nan)
    for col in BOWLER_SNAPSHOT_FEATURES:
        row[col] = bowler_feats.get(col, np.nan)
    row["bat_hand"] = _encode(bundle, "bat_hand", batter_hand)
    row["bowl_type"] = _encode(bundle, "bowl_type", bowler_type)
    row["venue_id"] = _encode(bundle, "venue_id", venue_id)

    frame = pd.DataFrame([row])
    for col in bundle.features:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[bundle.features]

    probs = bundle.booster.predict(frame)[0]
    # Re-order model output (indexed by training-time class order) to the
    # canonical OUTCOME_CLASSES order the simulator expects.
    if bundle.outcome_classes and bundle.outcome_classes != OUTCOME_CLASSES:
        reindex = [bundle.outcome_classes.index(c) for c in OUTCOME_CLASSES]
        probs = probs[reindex]
    return probs


def _norm_bat(raw: float) -> float:
    return float(np.clip((raw - 25) / 75, 0, 1))


def _norm_bowl(raw: float) -> float:
    return float(np.clip((-raw - 10) / 40, 0, 1))


def _heuristic_cards(player: dict, phase_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The original rating-based fallback (bat_rating/bowl_rating -> probs),
    used when no trained ball model exists for this (format, gender).
    """
    br = _norm_bat(float(player.get("bat_rating", 0.5)))
    wr = _norm_bowl(float(player.get("bowl_rating", -0.5)))

    bat_probs = _DEFAULT_PROBS.copy()
    bat_probs[0] = max(0.05, 0.40 - br * 0.1)
    bat_probs[4] = min(0.25, 0.08 + br * 0.05)
    bat_probs[5] = min(0.15, 0.04 + br * 0.03)
    bat_probs /= bat_probs.sum()

    bowl_probs = _DEFAULT_PROBS.copy()
    bowl_probs[6] = min(0.20, 0.06 + wr * 0.04)
    bowl_probs[0] = min(0.50, 0.30 + wr * 0.05)
    bowl_probs /= bowl_probs.sum()

    return bat_probs, bowl_probs


@dataclass
class CardContext:
    """Match context shared by every player's card in one request."""

    fmt: str
    gender: str = "male"
    phase_names: list[str] = field(default_factory=lambda: ["powerplay", "middle", "death"])
    venue_id: str = "unknown"
    toss_bat_first: bool = False
    pitch_condition: str = "auto"
    max_overs_per_bowler: int = 4


def build_cards(
    player: dict,
    opponent_avg: dict | None,
    ctx: CardContext,
) -> tuple[BattingCard, BowlingCard]:
    """Build (BattingCard, BowlingCard) for one player under `ctx`.

    `player` is a dict with this player's latest snapshot features
    (BATTER_SNAPSHOT_FEATURES/BOWLER_SNAPSHOT_FEATURES keys), `bat_hand`,
    `bowl_type`, plus `bat_rating`/`bowl_rating` for the heuristic fallback.
    `opponent_avg` is the average snapshot of the specific opponent XI (or
    the league average, or None to use a neutral default) standing in for
    "the other side" the model needs as its other set of features.
    """
    bundle = load_ball_model(ctx.fmt, ctx.gender)
    opponent = opponent_avg or {}

    bat_phases: dict[str, np.ndarray] = {}
    bowl_phases: dict[str, np.ndarray] = {}

    if bundle is not None:
        n_phases = len(_TRAINED_PHASES)
        for i, pname in enumerate(_TRAINED_PHASES):
            situation = _phase_situation(ctx.fmt, i, n_phases)
            bat_probs = _predict_phase_probs(
                bundle,
                batter_feats=player,
                bowler_feats=opponent,
                batter_hand=player.get("bat_hand"),
                bowler_type=None,
                venue_id=ctx.venue_id,
                toss_bat_first=ctx.toss_bat_first,
                situation=situation,
            )
            bowl_probs = _predict_phase_probs(
                bundle,
                batter_feats=opponent,
                bowler_feats=player,
                batter_hand=None,
                bowler_type=player.get("bowl_type"),
                venue_id=ctx.venue_id,
                toss_bat_first=ctx.toss_bat_first,
                situation=situation,
            )
            bat_probs = apply_pitch_adjustment(bat_probs, ctx.pitch_condition, bowl_type=None)
            bowl_probs = apply_pitch_adjustment(
                bowl_probs, ctx.pitch_condition, bowl_type=player.get("bowl_type")
            )
            bat_phases[pname] = bat_probs
            bowl_phases[pname] = bowl_probs

        # Requested phase_names may differ from the trained phase vocab
        # (e.g. ODI's "consolidation"/"transition") — map every requested
        # phase onto the nearest trained bucket (early/mid/late of innings).
        bat_phases = _remap_phases(bat_phases, ctx.phase_names)
        bowl_phases = _remap_phases(bowl_phases, ctx.phase_names)
    else:
        bat_probs, bowl_probs = _heuristic_cards(player, ctx.phase_names)
        bat_probs = apply_pitch_adjustment(bat_probs, ctx.pitch_condition, bowl_type=None)
        bowl_probs = apply_pitch_adjustment(
            bowl_probs, ctx.pitch_condition, bowl_type=player.get("bowl_type")
        )
        bat_phases = {p: bat_probs for p in ctx.phase_names}
        bowl_phases = {p: bowl_probs for p in ctx.phase_names}

    player_id = player.get("player_id", "")
    return (
        BattingCard(player_id=player_id, probs_by_phase=bat_phases),
        BowlingCard(
            player_id=player_id,
            probs_by_phase=bowl_phases,
            max_overs=ctx.max_overs_per_bowler,
        ),
    )


def _remap_phases(
    trained: dict[str, np.ndarray], requested: list[str]
) -> dict[str, np.ndarray]:
    """Map a 3-bucket (powerplay/middle/death) prediction onto an arbitrary
    requested phase list by position: first phase -> powerplay, last ->
    death, everything in between -> middle. Falls through unchanged when
    `requested` already matches the trained vocab.
    """
    if requested == _TRAINED_PHASES:
        return trained
    n = len(requested)
    out = {}
    for i, name in enumerate(requested):
        if i == 0:
            out[name] = trained["powerplay"]
        elif i == n - 1:
            out[name] = trained["death"]
        else:
            out[name] = trained["middle"]
    return out
