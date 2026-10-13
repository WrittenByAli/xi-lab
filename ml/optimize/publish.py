"""
Generate and publish a match prediction as JSON.

Called by the publish_prediction GitHub Action workflow.

Usage:
    python ml/optimize/publish.py \
        --team IND --opponent AUS --venue "MCG" \
        --format t20 --date 2024-03-15 \
        --squad "V Kohli,R Sharma,JJ Bumrah,..." \
        --out live/predictions/IND_vs_AUS_2024-03-15.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from ml.optimize.selector import PlayerOption, select_xi

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def _load_player_meta() -> dict:
    import pandas as pd
    meta = pd.read_csv(ROOT / "data" / "reference" / "player_meta.csv", comment="#")
    return {row["player_id"]: row.to_dict() for _, row in meta.iterrows()}


def _load_ratings() -> dict:
    import pandas as pd
    path = ROOT / "data" / "processed" / "models" / "ability" / "ratings.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    latest = df.sort_values("snapshot_month").groupby("player_id").last()
    return latest.to_dict(orient="index")


def _model_hash() -> str:
    h = hashlib.sha256()
    for name in ("ability/ratings.parquet", "ball/model.lgb"):
        path = ROOT / "data" / "processed" / "models" / name
        if path.exists():
            h.update(path.read_bytes()[:4096])
    return h.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--squad", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    squad_ids = [s.strip() for s in args.squad.split(",")]
    meta = _load_player_meta()
    ratings = _load_ratings()

    squad = []
    for pid in squad_ids:
        info = meta.get(pid, {})
        rating = ratings.get(pid, {})
        squad.append(PlayerOption(
            player_id=pid,
            name=info.get("name", pid),
            role=rating.get("role", "pure_bat"),
            bat_rating=float(rating.get("bat_rating", 0.5)),
            bowl_rating=float(rating.get("bowl_rating", 0.5)),
            is_keeper=bool(info.get("is_keeper", False)),
            is_overseas=False,
            bat_hand=str(info.get("bat_hand", "R")),
            bowl_type=str(info.get("bowl_type", "none")),
        ))

    result = select_xi(squad)

    prediction = {
        "team": args.team,
        "opponent": args.opponent,
        "venue": args.venue,
        "format": args.format,
        "match_date": args.date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_hash": _model_hash(),
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
        "win_prob": round(result.win_prob, 4),
        "batting_order": result.batting_order,
        "bowling_plan": result.bowling_plan,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prediction, indent=2))
    print(f"Prediction written to {out_path}")


if __name__ == "__main__":
    main()
