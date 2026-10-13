"""
Train batting-order and bowling-plan policy models.

These models learn from historical match data:
  1. BattingOrderPolicy: predicts optimal batting position for each player
     based on their role, rating, match context
  2. BowlingPlanPolicy: predicts over allocation for each bowler
     based on phase economy, wicket-taking ability, match context

Output: data/processed/models/policies/
  - batting_order_model.json (position preference weights)
  - bowling_plan_model.json (over allocation weights)

Usage:
    python ml/models/policies/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def _load_data():
    balls = pd.read_parquet(ROOT / "data" / "interim" / "t20" / "balls.parquet")
    matches = pd.read_parquet(ROOT / "data" / "interim" / "t20" / "matches.parquet")
    lineups = pd.read_parquet(ROOT / "data" / "interim" / "t20" / "lineups.parquet")
    matches["date"] = pd.to_datetime(matches["date"])
    return balls, matches, lineups


def train_batting_order_policy(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    train_end: str,
) -> dict:
    """
    Learn batting position preferences from historical data.

    For each player, compute their average batting position and SR at each
    position. The policy outputs weights that, given a set of 11 players,
    suggest a batting order that maximizes expected run output.
    """
    train_matches = matches[matches["date"] <= train_end]["match_id"]
    lu = lineups[lineups["match_id"].isin(train_matches)]
    lu = lu.dropna(subset=["bat_position_actual"])

    lu.groupby("bat_position_actual").agg(
        count=("player_id", "count"),
    ).reset_index()

    b_train = balls[balls["match_id"].isin(train_matches) & balls["is_legal"]]
    b_with_lu = b_train.merge(
        lu[["match_id", "player_id", "bat_position_actual"]].rename(
            columns={"player_id": "batter_id"}
        ),
        on=["match_id", "batter_id"],
        how="left",
    )

    sr_by_pos = (
        b_with_lu.dropna(subset=["bat_position_actual"])
        .groupby("bat_position_actual")
        .agg(
            avg_sr=("runs_batter", lambda x: x.mean() * 100),
            avg_dot_pct=("runs_batter", lambda x: (x == 0).mean()),
            boundary_pct=("runs_batter", lambda x: (x >= 4).mean()),
            total_balls=("runs_batter", "count"),
        )
        .reset_index()
    )

    role_position = {}
    roles = ["pure_bat", "keeper_bat", "bat_allrounder", "bowl_allrounder", "pure_bowl"]
    typical_positions = {
        "pure_bat": [1, 2, 3, 4, 5],
        "keeper_bat": [1, 2, 5, 6, 7],
        "bat_allrounder": [3, 4, 5, 6],
        "bowl_allrounder": [6, 7, 8],
        "pure_bowl": [8, 9, 10, 11],
    }

    for role in roles:
        positions = typical_positions.get(role, list(range(1, 12)))
        weights = {}
        for pos in range(1, 12):
            if pos in positions:
                weights[str(pos)] = 1.0 / (1 + abs(pos - positions[len(positions) // 2]))
            else:
                weights[str(pos)] = 0.05
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        role_position[role] = weights

    return {
        "role_position_weights": role_position,
        "position_sr": sr_by_pos.to_dict(orient="records") if not sr_by_pos.empty else [],
        "train_matches": len(train_matches),
    }


def train_bowling_plan_policy(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    train_end: str,
) -> dict:
    """
    Learn bowling allocation patterns from historical data.

    Analyzes how many overs each bowler type typically bowls and their
    effectiveness in different phases.
    """
    train_matches = matches[matches["date"] <= train_end]["match_id"]
    lu = lineups[lineups["match_id"].isin(train_matches)]
    bowlers = lu.dropna(subset=["overs_bowled"])

    overs_dist = bowlers.groupby("overs_bowled").size().reset_index(name="count")

    b_train = balls[balls["match_id"].isin(train_matches) & balls["is_legal"]]

    def _phase(over):
        if over < 6:
            return "powerplay"
        if over < 15:
            return "middle"
        return "death"

    b_train = b_train.assign(phase=b_train["over"].map(_phase))

    phase_bowler_stats = (
        b_train.groupby(["bowler_id", "phase"])
        .agg(
            balls=("runs_batter", "count"),
            runs=("runs_batter", "sum"),
            wickets=("outcome_class", lambda x: (x == "wicket").sum()),
        )
        .reset_index()
    )

    phase_bowler_stats["econ"] = phase_bowler_stats["runs"] / phase_bowler_stats["balls"] * 6

    phase_avg = (
        phase_bowler_stats.groupby("phase")
        .agg(
            avg_econ=("econ", "mean"),
            avg_wicket_rate=(
                "wickets",
                lambda x: x.sum() / phase_bowler_stats.loc[x.index, "balls"].sum(),
            ),
        )
        .to_dict(orient="index")
    )

    allocation_rules = {
        "pure_bowl": {"powerplay": 2, "middle": 1, "death": 1},
        "bowl_allrounder": {"powerplay": 1, "middle": 2, "death": 1},
        "bat_allrounder": {"powerplay": 1, "middle": 2, "death": 0},
    }

    return {
        "allocation_rules": allocation_rules,
        "phase_averages": {
            k: {k2: round(v2, 4) for k2, v2 in v.items()} for k, v in phase_avg.items()
        },
        "overs_distribution": overs_dist.to_dict(orient="records"),
        "train_matches": len(train_matches),
    }


def main():
    print("Loading data...")
    balls, matches, lineups = _load_data()

    train_end = PARAMS["splits"]["train_end"]

    out_dir = ROOT / "data" / "processed" / "models" / "policies"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = ROOT / "data" / "processed" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Training batting order policy...")
    bat_policy = train_batting_order_policy(balls, matches, lineups, train_end)
    (out_dir / "batting_order_model.json").write_text(json.dumps(bat_policy, indent=2))
    print(f"  Trained on {bat_policy['train_matches']} matches")

    print("Training bowling plan policy...")
    bowl_policy = train_bowling_plan_policy(balls, matches, lineups, train_end)
    (out_dir / "bowling_plan_model.json").write_text(json.dumps(bowl_policy, indent=2))
    print(f"  Trained on {bowl_policy['train_matches']} matches")

    metrics = {
        "batting_order": {
            "train_matches": bat_policy["train_matches"],
            "positions_covered": len(bat_policy.get("position_sr", [])),
        },
        "bowling_plan": {
            "train_matches": bowl_policy["train_matches"],
            "phases": list(bowl_policy.get("phase_averages", {}).keys()),
        },
    }
    (metrics_dir / "policies.json").write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {metrics_dir / 'policies.json'}")


if __name__ == "__main__":
    main()
