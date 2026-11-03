"""
Record a completed match's actual result against a published prediction,
closing the loop the `/live` and `/live/stats` endpoints read from.

`publish.py` writes one JSON file per match to `live/predictions/`. That
is the audit trail (tagged GitHub Release, model hash, full XI). The API
instead reads a per-format aggregate list from
`data/processed/predictions/{format}_predictions.json` (`PredictionRecord`
in backend/app/schemas/models.py) — this script is what turns a published
prediction plus a real-world result into one of those records.

Usage:
    python ml/optimize/record_result.py \
        --prediction live/predictions/SouthAfrica_vs_India_2026-02-22.json \
        --winner "South Africa"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"
LIVE_RESULTS_DIR = ROOT / "live" / "results"


def record_result(prediction_path: Path, winner: str) -> dict:
    prediction = json.loads(prediction_path.read_text())
    fmt = prediction["format"]
    team = prediction["team"]

    record = {
        "match_id": f"{team}_vs_{prediction['opponent']}_{prediction['match_date']}",
        "date": prediction["match_date"],
        "team1": team,
        "team2": prediction["opponent"],
        "predicted_wp": prediction["win_prob"],
        "actual_winner": winner,
        "correct": (prediction["win_prob"] > 0.5) == (winner == team),
    }

    # Aggregate, machine-readable — this is what `/live` and `/live/stats`
    # actually read.
    out_path = PREDICTIONS_DIR / f"{fmt}_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    records = [r for r in records if r["match_id"] != record["match_id"]]
    records.append(record)
    out_path.write_text(json.dumps(records, indent=2))

    # Git-committed audit trail, one file per match — the result-side
    # counterpart to the prediction file `publish.py` writes to
    # `live/predictions/`, named the same way so the two are easy to pair
    # up by eye.
    LIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_RESULTS_DIR / prediction_path.name).write_text(
        json.dumps({**record, "prediction_file": prediction_path.name}, indent=2)
    )

    print(f"Recorded result for {record['match_id']} -> {out_path}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--winner", required=True)
    args = parser.parse_args()
    record_result(args.prediction, args.winner)


if __name__ == "__main__":
    main()
