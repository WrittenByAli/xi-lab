"""GET /live — prediction log with results."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Query

from backend.app.schemas.models import PredictionRecord

router = APIRouter()

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"


@router.get("", response_model=list[PredictionRecord])
def list_predictions(
    format: str = Query("t20", pattern="^(t20|odi|test)$"),
    limit: int = Query(50, le=500),
):
    pred_file = PREDICTIONS_DIR / f"{format}_predictions.json"
    if not pred_file.exists():
        return []

    records = json.loads(pred_file.read_text())
    records.sort(key=lambda r: r.get("date", ""), reverse=True)
    return records[:limit]


@router.get("/stats")
def prediction_stats(format: str = Query("t20", pattern="^(t20|odi|test)$")):
    pred_file = PREDICTIONS_DIR / f"{format}_predictions.json"
    if not pred_file.exists():
        return {"total": 0, "correct": 0, "accuracy": None}

    records = json.loads(pred_file.read_text())
    resolved = [r for r in records if r.get("correct") is not None]
    correct = sum(1 for r in resolved if r["correct"])

    return {
        "total": len(records),
        "resolved": len(resolved),
        "correct": correct,
        "accuracy": round(correct / len(resolved), 4) if resolved else None,
    }
