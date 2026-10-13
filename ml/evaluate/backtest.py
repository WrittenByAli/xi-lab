"""
Backtest the full pipeline: ability ratings → calibrated WP → actual results.

Two-stage approach:
  1. Fit logistic regression on validation set (team rating diffs → actual outcomes)
  2. Evaluate on held-out test set

Metrics: Brier score, log loss, calibration curve, accuracy, Kelly ROI.

Output: data/processed/metrics/backtest.json
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


def _load_ratings() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "models" / "ability" / "ratings.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _get_team_ratings(
    ratings: pd.DataFrame,
    lineups: pd.DataFrame,
    match_id: str,
    team: str,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    team_players = lineups[(lineups["match_id"] == match_id) & (lineups["team"] == team)][
        "player_id"
    ].unique()

    snapshot_month = match_date.replace(day=1)
    team_ratings = ratings[
        (ratings["player_id"].isin(team_players)) & (ratings["snapshot_month"] <= snapshot_month)
    ]

    if team_ratings.empty:
        return pd.DataFrame()

    latest = team_ratings.groupby("player_id")["snapshot_month"].max().reset_index()
    return team_ratings.merge(latest, on=["player_id", "snapshot_month"])


def _compute_features(t1: pd.DataFrame, t2: pd.DataFrame) -> dict | None:
    """Extract normalized team-level features from individual ratings."""
    if t1.empty or t2.empty:
        return None

    def _team_stats(df):
        bat = df["bat_rating"].dropna()
        bowl = df["bowl_rating"].dropna()
        return {
            "bat_mean": bat.mean() if len(bat) > 0 else 0,
            "bat_top3": bat.nlargest(3).mean()
            if len(bat) >= 3
            else (bat.mean() if len(bat) > 0 else 0),
            "bowl_mean": bowl.mean() if len(bowl) > 0 else 0,
            "bowl_top3": bowl.nsmallest(3).mean()
            if len(bowl) >= 3
            else (bowl.mean() if len(bowl) > 0 else 0),
            "n_players": len(df),
        }

    s1 = _team_stats(t1)
    s2 = _team_stats(t2)

    return {
        "bat_diff": s1["bat_mean"] - s2["bat_mean"],
        "bat_top3_diff": s1["bat_top3"] - s2["bat_top3"],
        "bowl_diff": s1["bowl_mean"] - s2["bowl_mean"],
        "bowl_top3_diff": s1["bowl_top3"] - s2["bowl_top3"],
        "n1": s1["n_players"],
        "n2": s2["n_players"],
    }


def _build_dataset(
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
    ratings: pd.DataFrame,
    date_start: str | None,
    date_end: str | None,
    label: str,
) -> pd.DataFrame:
    """Build features + labels for a date range of matches."""
    mask = matches["result_type"] == "normal"
    if date_start:
        mask = mask & (matches["date"] >= date_start)
    if date_end:
        mask = mask & (matches["date"] <= date_end)

    subset = matches[mask]
    print(f"  [{label}] {len(subset)} matches")

    rows = []
    for _, match in subset.iterrows():
        t1 = _get_team_ratings(ratings, lineups, match["match_id"], match["team1"], match["date"])
        t2 = _get_team_ratings(ratings, lineups, match["match_id"], match["team2"], match["date"])
        feats = _compute_features(t1, t2)
        if feats is None:
            continue

        actual = 1.0 if match["winner"] == match["team1"] else 0.0
        rows.append(
            {
                "match_id": match["match_id"],
                "date": str(match["date"].date()),
                "team1": match["team1"],
                "team2": match["team2"],
                "actual": actual,
                **feats,
            }
        )

    return pd.DataFrame(rows)


def _fit_logistic(
    x: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 500
) -> tuple[np.ndarray, float]:
    """Fit logistic regression with gradient descent (no sklearn dependency for this)."""
    n_features = x.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = x @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        error = p - y
        grad_w = (x.T @ error) / len(y) + 0.001 * w
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def _predict_logistic(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = x @ w + b
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def backtest(fmt: str = "t20") -> dict:
    print(f"[{fmt}] Loading data...")
    matches = pd.read_parquet(ROOT / "data" / "interim" / fmt / "matches.parquet")
    lineups = pd.read_parquet(ROOT / "data" / "interim" / fmt / "lineups.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    ratings = _load_ratings()
    if ratings is None:
        return {"format": fmt, "test_matches": 0, "error": "no_ratings_available"}

    splits = PARAMS["splits"]
    val_start = splits["val_start"]
    val_end = splits["val_end"]
    test_start = splits["test_start"]

    print(f"[{fmt}] Building validation set ({val_start} to {val_end})...")
    val_df = _build_dataset(matches, lineups, ratings, val_start, val_end, "val")

    print(f"[{fmt}] Building test set ({test_start} onwards)...")
    test_df = _build_dataset(matches, lineups, ratings, test_start, None, "test")

    if val_df.empty or test_df.empty:
        return {"format": fmt, "test_matches": 0, "error": "insufficient_data"}

    feature_cols = ["bat_diff", "bat_top3_diff", "bowl_diff", "bowl_top3_diff"]
    x_val = val_df[feature_cols].values.astype(float)
    y_val = val_df["actual"].values.astype(float)
    x_test = test_df[feature_cols].values.astype(float)
    y_test = test_df["actual"].values.astype(float)

    # Z-score normalize using validation stats
    mu = x_val.mean(axis=0)
    sigma = x_val.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    x_val_norm = (x_val - mu) / sigma
    x_test_norm = (x_test - mu) / sigma

    print(f"[{fmt}] Fitting logistic model on {len(val_df)} validation matches...")
    w, b = _fit_logistic(x_val_norm, y_val, lr=0.1, epochs=2000)

    print(
        f"[{fmt}] Feature weights: {dict(zip(feature_cols, [round(float(x), 4) for x in w], strict=False))}"
    )
    print(f"[{fmt}] Bias: {b:.4f}")

    # Evaluate on test
    predicted = _predict_logistic(x_test_norm, w, b)
    predicted = np.clip(predicted, 0.01, 0.99)
    actual = y_test

    brier = float(np.mean((predicted - actual) ** 2))
    log_loss = float(-np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))
    accuracy = float(np.mean((predicted > 0.5) == (actual > 0.5)))

    # Calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (predicted >= lo) & (predicted < hi)
        if mask.sum() > 0:
            calibration.append(
                {
                    "bin_center": round((lo + hi) / 2, 2),
                    "predicted_mean": round(float(predicted[mask].mean()), 4),
                    "actual_mean": round(float(actual[mask].mean()), 4),
                    "count": int(mask.sum()),
                }
            )

    # Kelly ROI
    kelly_returns = []
    for p, a in zip(predicted, actual, strict=False):
        if p > 0.5:
            kelly_fraction = max(0, 2 * p - 1)
            payout = a * 2 - 1
            kelly_returns.append(kelly_fraction * payout)
    roi = float(np.mean(kelly_returns)) if kelly_returns else 0.0

    metrics = {
        "format": fmt,
        "test_matches": len(test_df),
        "val_matches": len(val_df),
        "test_period": f"{test_start} onwards",
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "kelly_roi": round(roi, 6),
        "calibration": calibration,
        "baseline_brier": 0.25,
        "feature_weights": dict(zip(feature_cols, [round(float(x), 4) for x in w], strict=False)),
        "bias": round(float(b), 4),
        "normalization": {"mean": mu.tolist(), "std": sigma.tolist()},
    }

    print(f"\n[{fmt}] Test Results:")
    print(f"  Brier score:  {brier:.4f} (baseline: 0.2500)")
    print(f"  Log loss:     {log_loss:.4f} (baseline: 0.6931)")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Kelly ROI:    {roi:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["t20", "odi", "test"], default="t20")
    args = parser.parse_args()

    metrics = backtest(args.format)

    out_dir = ROOT / "data" / "processed" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{args.format}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main()
