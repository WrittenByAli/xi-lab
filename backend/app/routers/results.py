"""GET /results — evaluation artifacts and backtest metrics."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from backend.app.schemas.models import BacktestMetrics

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
METRICS_DIR = ROOT / "data" / "processed" / "metrics"


@router.get("", response_model=BacktestMetrics)
def get_backtest_results(format: str = Query("t20", pattern="^(t20|odi|test)$")):
    path = METRICS_DIR / "backtest.json"
    if not path.exists():
        return BacktestMetrics(format=format, test_matches=0)

    data = json.loads(path.read_text())
    return BacktestMetrics(**data)


@router.get("/ability")
def get_ability_metrics():
    path = METRICS_DIR / "ability.json"
    if not path.exists():
        return {"error": "No ability metrics available. Run the training pipeline first."}
    return json.loads(path.read_text())


@router.get("/ball")
def get_ball_metrics():
    path = METRICS_DIR / "ball.json"
    if not path.exists():
        return {"error": "No ball model metrics available. Run the training pipeline first."}
    return json.loads(path.read_text())


@router.get("/policies")
def get_policy_metrics():
    path = METRICS_DIR / "policies.json"
    if not path.exists():
        return {"error": "No policy metrics available. Run the training pipeline first."}
    return json.loads(path.read_text())


@router.get("/all")
def get_all_metrics():
    results = {}
    for name in ("ability", "ball", "backtest", "policies"):
        path = METRICS_DIR / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())
    return results
