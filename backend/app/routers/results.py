"""GET /results — evaluation artifacts and backtest metrics."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from backend.app.schemas.models import BacktestMetrics

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = ROOT / "data" / "processed" / "metrics"


_GENDER_PATTERN = "^(male|female)$"


def _read_scoped(name: str, fmt: str, gender: str) -> dict | None:
    """Read data/processed/metrics/{name}_{fmt}_{gender}.json, falling back
    to the unqualified {name}.json (the "last run" copy every training
    script also writes) when the specific combination isn't available.
    """
    path = METRICS_DIR / f"{name}_{fmt}_{gender}.json"
    if not path.exists():
        path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@router.get("", response_model=BacktestMetrics)
def get_backtest_results(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
):
    data = _read_scoped("backtest", format, gender)
    if data is None:
        return BacktestMetrics(format=format, test_matches=0)
    return BacktestMetrics(**data)


@router.get("/ability")
def get_ability_metrics(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
):
    data = _read_scoped("ability", format, gender)
    if data is None:
        return {"error": "No ability metrics available. Run the training pipeline first."}
    return data


@router.get("/ball")
def get_ball_metrics(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
):
    data = _read_scoped("ball", format, gender)
    if data is None:
        return {"error": "No ball model metrics available. Run the training pipeline first."}
    return data


@router.get("/policies")
def get_policy_metrics(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
):
    data = _read_scoped("policies", format, gender)
    if data is None:
        return {"error": "No policy metrics available. Run the training pipeline first."}
    return data


@router.get("/all")
def get_all_metrics(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    gender: str = Query("male", pattern=_GENDER_PATTERN),
):
    results = {}
    for name in ("ability", "ball", "backtest", "policies"):
        data = _read_scoped(name, format, gender)
        if data is not None:
            results[name] = data
    return results
