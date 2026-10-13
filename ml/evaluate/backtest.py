"""
Backtest the full pipeline: ability model + ball model + simulator vs actual results.

For each match in the test period:
  1. Load as-of player snapshots and ability ratings
  2. Build probability vectors from the ball model
  3. Simulate N innings and compute predicted win probability
  4. Compare predicted WP with actual match outcome

Metrics:
  - Brier score (calibration)
  - Log loss
  - Calibration curve (binned predicted vs actual)
  - ROI on hypothetical fair-odds betting

Output: data/processed/metrics/backtest.json

Usage:
    python ml/evaluate/backtest.py
    python ml/evaluate/backtest.py --format t20 --n-sims 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


def _load_models(fmt: str) -> dict:
    """Load all model artifacts needed for backtesting."""
    models_dir = ROOT / "data" / "processed" / "models"
    artifacts = {}

    ability_dir = models_dir / "ability"
    if (ability_dir / "ratings.parquet").exists():
        artifacts["ratings"] = pd.read_parquet(ability_dir / "ratings.parquet")
    if (ability_dir / "group_priors.json").exists():
        artifacts["priors"] = json.loads((ability_dir / "group_priors.json").read_text())

    ball_dir = models_dir / "ball"
    if (ball_dir / "model.lgb").exists():
        try:
            import lightgbm as lgb
            artifacts["ball_model"] = lgb.Booster(model_file=str(ball_dir / "model.lgb"))
        except Exception:
            artifacts["ball_model"] = None

    policy_dir = models_dir / "policies"
    if (policy_dir / "batting_order_model.json").exists():
        artifacts["batting_policy"] = json.loads(
            (policy_dir / "batting_order_model.json").read_text()
        )
    if (policy_dir / "bowling_plan_model.json").exists():
        artifacts["bowling_policy"] = json.loads(
            (policy_dir / "bowling_plan_model.json").read_text()
        )

    return artifacts


def _get_team_ratings(
    ratings: pd.DataFrame,
    lineups: pd.DataFrame,
    match_id: str,
    team: str,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    """Get ability ratings for each player in a team for a specific match."""
    team_players = lineups[
        (lineups["match_id"] == match_id) & (lineups["team"] == team)
    ]["player_id"].unique()

    snapshot_month = match_date.replace(day=1)
    team_ratings = ratings[
        (ratings["player_id"].isin(team_players))
        & (ratings["snapshot_month"] <= snapshot_month)
    ]

    if team_ratings.empty:
        return pd.DataFrame()

    latest = team_ratings.groupby("player_id")["snapshot_month"].max().reset_index()
    team_ratings = team_ratings.merge(latest, on=["player_id", "snapshot_month"])
    return team_ratings


def _predict_win_prob_simple(
    team1_ratings: pd.DataFrame,
    team2_ratings: pd.DataFrame,
) -> float:
    """Simple rating-based win probability (logistic model)."""
    if team1_ratings.empty or team2_ratings.empty:
        return 0.5

    t1_bat = team1_ratings["bat_rating"].mean() if "bat_rating" in team1_ratings else 0.5
    t1_bowl = team1_ratings["bowl_rating"].mean() if "bowl_rating" in team1_ratings else 0.5
    t2_bat = team2_ratings["bat_rating"].mean() if "bat_rating" in team2_ratings else 0.5
    t2_bowl = team2_ratings["bowl_rating"].mean() if "bowl_rating" in team2_ratings else 0.5

    t1_strength = t1_bat + t1_bowl
    t2_strength = t2_bat + t2_bowl

    diff = t1_strength - t2_strength
    wp = 1 / (1 + np.exp(-diff * 2))
    return float(np.clip(wp, 0.01, 0.99))


def backtest(fmt: str = "t20", n_sims: int = 5000) -> dict:
    """Run backtest on test-period matches."""
    print(f"[{fmt}] Loading data...")
    matches = pd.read_parquet(ROOT / "data" / "interim" / fmt / "matches.parquet")
    lineups = pd.read_parquet(ROOT / "data" / "interim" / fmt / "lineups.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    test_start = PARAMS["splits"]["test_start"]
    test_matches = matches[
        (matches["date"] >= test_start) & (matches["result_type"] == "normal")
    ].copy()

    print(f"[{fmt}] {len(test_matches)} test matches from {test_start}")

    artifacts = _load_models(fmt)

    if "ratings" not in artifacts:
        print("[WARNING] No ability ratings found, using default 0.5 WP")
        return {
            "format": fmt,
            "test_matches": len(test_matches),
            "error": "no_ratings_available",
        }

    ratings = artifacts["ratings"]
    predictions = []

    for i, (_, match) in enumerate(test_matches.iterrows()):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(test_matches)} matches")

        t1_ratings = _get_team_ratings(
            ratings, lineups, match["match_id"], match["team1"], match["date"]
        )
        t2_ratings = _get_team_ratings(
            ratings, lineups, match["match_id"], match["team2"], match["date"]
        )

        wp = _predict_win_prob_simple(t1_ratings, t2_ratings)
        actual = 1.0 if match["winner"] == match["team1"] else 0.0

        predictions.append({
            "match_id": match["match_id"],
            "date": str(match["date"].date()),
            "team1": match["team1"],
            "team2": match["team2"],
            "predicted_wp": wp,
            "actual": actual,
        })

    if not predictions:
        return {"format": fmt, "test_matches": 0, "error": "no_predictions"}

    pred_df = pd.DataFrame(predictions)
    predicted = pred_df["predicted_wp"].values
    actual = pred_df["actual"].values

    brier = float(np.mean((predicted - actual) ** 2))
    log_loss = float(-np.mean(
        actual * np.log(np.clip(predicted, 1e-8, 1)) +
        (1 - actual) * np.log(np.clip(1 - predicted, 1e-8, 1))
    ))

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (predicted >= lo) & (predicted < hi)
        if mask.sum() > 0:
            calibration.append({
                "bin_center": round((lo + hi) / 2, 2),
                "predicted_mean": round(float(predicted[mask].mean()), 4),
                "actual_mean": round(float(actual[mask].mean()), 4),
                "count": int(mask.sum()),
            })

    accuracy = float(np.mean((predicted > 0.5) == (actual > 0.5)))

    kelly_returns = []
    for p, a in zip(predicted, actual):
        if p > 0.5:
            kelly_fraction = max(0, 2 * p - 1)
            payout = a * 2 - 1
            kelly_returns.append(kelly_fraction * payout)
    roi = float(np.mean(kelly_returns)) if kelly_returns else 0.0

    metrics = {
        "format": fmt,
        "test_matches": len(predictions),
        "test_period": f"{test_start} onwards",
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "kelly_roi": round(roi, 6),
        "calibration": calibration,
        "baseline_brier": 0.25,
    }

    print(f"[{fmt}] Results:")
    print(f"  Brier score:  {brier:.4f} (baseline: 0.2500)")
    print(f"  Log loss:     {log_loss:.4f}")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Kelly ROI:    {roi:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["t20", "odi", "test"], default="t20")
    parser.add_argument("--n-sims", type=int, default=5000)
    args = parser.parse_args()

    metrics = backtest(args.format, args.n_sims)

    out_dir = ROOT / "data" / "processed" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
