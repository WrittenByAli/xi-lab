"""
Empirical Bayes shrinkage ability model.

For each player snapshot, produces a batting and bowling rating that shrinks
noisy small-sample estimates toward the player's role x format group mean.

    rating = (n * player_mean + prior_n * group_mean) / (n + prior_n)

where n = effective weighted balls faced/bowled and prior_n = min_balls_for_rating.

Outputs
-------
- data/processed/models/ability/ratings.parquet
    player_id, snapshot_month, format, role, bat_rating, bat_uncertainty,
    bowl_rating, bowl_uncertainty
- data/processed/models/ability/group_priors.json
    Mean and variance of each metric per (role, format) group.
- data/processed/metrics/ability.json
    Correlation between predicted and actual performance on the validation set.

Usage
-----
    python ml/models/ability/train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

def classify_role(
    bat_balls: float,
    bowl_balls: float,
    is_keeper: bool,
) -> str:
    """Classify a player into one of five roles based on ball volumes.

    Thresholds (on weighted balls):
      - keeper_bat: is_keeper and bat_balls >= 30
      - pure_bat: bat >= 60 and bowl < 30
      - pure_bowl: bowl >= 60 and bat < 30
      - bat_allrounder: bat >= 60 and bowl >= 30
      - bowl_allrounder: bowl >= 60 and bat >= 30
      - fallback: if bat > bowl -> pure_bat else pure_bowl
    """
    bat_balls = bat_balls if not np.isnan(bat_balls) else 0.0
    bowl_balls = bowl_balls if not np.isnan(bowl_balls) else 0.0

    if is_keeper and bat_balls >= 30:
        return "keeper_bat"

    if bat_balls >= 60 and bowl_balls < 30:
        return "pure_bat"
    if bowl_balls >= 60 and bat_balls < 30:
        return "pure_bowl"
    if bat_balls >= 60 and bowl_balls >= 30:
        return "bat_allrounder"
    if bowl_balls >= 60 and bat_balls >= 30:
        return "bowl_allrounder"

    # Fallback for low-volume players
    if bat_balls >= bowl_balls:
        return "pure_bat"
    return "pure_bowl"


# ---------------------------------------------------------------------------
# Shrinkage helpers
# ---------------------------------------------------------------------------

BAT_METRICS = ["bat_sr_wtd", "bat_avg_wtd"]
BOWL_METRICS = ["bowl_econ_wtd", "bowl_sr_wtd"]


def compute_group_priors(df: pd.DataFrame) -> dict:
    """Compute mean and variance of each metric per (role, format) group.

    Returns a nested dict: {role: {format: {metric: {mean, var, count}}}}
    """
    priors: dict = {}
    for (role, fmt), grp in df.groupby(["role", "format"]):
        if role not in priors:
            priors[role] = {}
        metrics = {}
        for col in BAT_METRICS + BOWL_METRICS:
            vals = grp[col].dropna()
            metrics[col] = {
                "mean": float(vals.mean()) if len(vals) > 0 else 0.0,
                "var": float(vals.var()) if len(vals) > 1 else 1.0,
                "count": int(len(vals)),
            }
        priors[role][fmt] = metrics
    return priors


def shrink(
    player_mean: float,
    n: float,
    group_mean: float,
    prior_n: float,
) -> tuple[float, float]:
    """Apply Empirical Bayes shrinkage.

    Returns (shrunk_rating, uncertainty).
    Uncertainty = prior_n / (n + prior_n), i.e. the weight placed on the prior.
    """
    if np.isnan(player_mean) or n <= 0:
        return group_mean, 1.0
    total = n + prior_n
    rating = (n * player_mean + prior_n * group_mean) / total
    uncertainty = prior_n / total
    return rating, uncertainty


def apply_shrinkage(df: pd.DataFrame, priors: dict, prior_n: float) -> pd.DataFrame:
    """Apply shrinkage to all rows and return the ratings table."""
    bat_ratings = np.full(len(df), np.nan)
    bat_uncerts = np.full(len(df), np.nan)
    bowl_ratings = np.full(len(df), np.nan)
    bowl_uncerts = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        role = row["role"]
        fmt = row["format"]
        group = priors.get(role, {}).get(fmt, {})

        # --- Batting: combine SR and avg into a single rating ---
        sr_prior = group.get("bat_sr_wtd", {}).get("mean", 0.0)
        avg_prior = group.get("bat_avg_wtd", {}).get("mean", 0.0)

        bat_n = row.get("bat_balls_wtd", 0.0)
        bat_n = bat_n if not np.isnan(bat_n) else 0.0

        sr_val = row.get("bat_sr_wtd", np.nan)
        avg_val = row.get("bat_avg_wtd", np.nan)

        sr_shrunk, sr_unc = shrink(sr_val, bat_n, sr_prior, prior_n)
        avg_shrunk, avg_unc = shrink(avg_val, bat_n, avg_prior, prior_n)

        # Composite batting rating: geometric-style blend of SR and avg
        bat_ratings[i] = sr_shrunk * 0.6 + avg_shrunk * 0.4
        bat_uncerts[i] = (sr_unc + avg_unc) / 2.0

        # --- Bowling: combine economy and SR ---
        econ_prior = group.get("bowl_econ_wtd", {}).get("mean", 0.0)
        bowl_sr_prior = group.get("bowl_sr_wtd", {}).get("mean", 0.0)

        bowl_n = row.get("bowl_balls_wtd", 0.0)
        bowl_n = bowl_n if not np.isnan(bowl_n) else 0.0

        econ_val = row.get("bowl_econ_wtd", np.nan)
        bowl_sr_val = row.get("bowl_sr_wtd", np.nan)

        econ_shrunk, econ_unc = shrink(econ_val, bowl_n, econ_prior, prior_n)
        bowl_sr_shrunk, bowl_sr_unc = shrink(bowl_sr_val, bowl_n, bowl_sr_prior, prior_n)

        # Bowling rating: lower is better for both econ and SR, so invert
        # Use negative sum so higher rating = better bowler
        bowl_ratings[i] = -(econ_shrunk * 0.5 + bowl_sr_shrunk * 0.5)
        bowl_uncerts[i] = (econ_unc + bowl_sr_unc) / 2.0

    return df.assign(
        bat_rating=bat_ratings,
        bat_uncertainty=bat_uncerts,
        bowl_rating=bowl_ratings,
        bowl_uncertainty=bowl_uncerts,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_validation(
    train_ratings: pd.DataFrame,
    val_snapshots: pd.DataFrame,
    prior_n: float,
    priors: dict,
) -> dict:
    """Evaluate the ability model on the validation set.

    For each player in the validation period, we use the most recent train-period
    rating as the prediction and correlate with actual validation-period stats.
    """
    # Get the last rating per player from the training period
    last_train = (
        train_ratings
        .sort_values("snapshot_month")
        .groupby(["player_id", "format"])
        .last()
        .reset_index()
    )

    # Classify and apply shrinkage to val snapshots so we have "actual" values
    val_agg = (
        val_snapshots
        .sort_values("snapshot_month")
        .groupby(["player_id", "format"])
        .last()
        .reset_index()
    )

    # Merge predicted (from train) with actual (from val)
    merged = last_train[["player_id", "format", "bat_rating", "bowl_rating"]].merge(
        val_agg[["player_id", "format", "bat_sr_wtd", "bat_avg_wtd",
                 "bowl_econ_wtd", "bowl_sr_wtd"]],
        on=["player_id", "format"],
        how="inner",
        suffixes=("_pred", "_actual"),
    )

    metrics = {"n_players": len(merged)}

    # Batting correlation: predicted bat_rating vs actual SR
    mask_bat = merged["bat_sr_wtd"].notna()
    if mask_bat.sum() > 5:
        corr_bat_sr = float(merged.loc[mask_bat, "bat_rating"].corr(
            merged.loc[mask_bat, "bat_sr_wtd"]
        ))
        metrics["bat_rating_vs_sr_corr"] = round(corr_bat_sr, 4)
    else:
        metrics["bat_rating_vs_sr_corr"] = None

    mask_avg = merged["bat_avg_wtd"].notna()
    if mask_avg.sum() > 5:
        corr_bat_avg = float(merged.loc[mask_avg, "bat_rating"].corr(
            merged.loc[mask_avg, "bat_avg_wtd"]
        ))
        metrics["bat_rating_vs_avg_corr"] = round(corr_bat_avg, 4)
    else:
        metrics["bat_rating_vs_avg_corr"] = None

    # Bowling correlation: predicted bowl_rating vs actual economy (inverted)
    mask_econ = merged["bowl_econ_wtd"].notna()
    if mask_econ.sum() > 5:
        corr_bowl_econ = float(merged.loc[mask_econ, "bowl_rating"].corr(
            -merged.loc[mask_econ, "bowl_econ_wtd"]
        ))
        metrics["bowl_rating_vs_econ_corr"] = round(corr_bowl_econ, 4)
    else:
        metrics["bowl_rating_vs_econ_corr"] = None

    mask_bsr = merged["bowl_sr_wtd"].notna()
    if mask_bsr.sum() > 5:
        corr_bowl_sr = float(merged.loc[mask_bsr, "bowl_rating"].corr(
            -merged.loc[mask_bsr, "bowl_sr_wtd"]
        ))
        metrics["bowl_rating_vs_sr_corr"] = round(corr_bowl_sr, 4)
    else:
        metrics["bowl_rating_vs_sr_corr"] = None

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Empirical Bayes shrinkage ability model."
    )
    parser.add_argument(
        "--format", choices=["t20", "odi", "test"], default="t20",
        help="Cricket format to train on (default: t20).",
    )
    args = parser.parse_args()
    fmt = args.format

    # ── Config ────────────────────────────────────────────────────────────
    splits = PARAMS["splits"]
    ability = PARAMS["ability"]
    train_end = pd.Timestamp(splits["train_end"])
    val_start = pd.Timestamp(splits["val_start"])
    val_end = pd.Timestamp(splits["val_end"])
    prior_n = float(ability["min_balls_for_rating"])

    # ── Load snapshots ────────────────────────────────────────────────────
    snap_path = ROOT / "data" / "processed" / "snapshots" / fmt / "snapshots.parquet"
    print(f"Loading snapshots from {snap_path}")
    snapshots = pd.read_parquet(snap_path)
    snapshots["snapshot_month"] = pd.to_datetime(snapshots["snapshot_month"])

    # ── Split ─────────────────────────────────────────────────────────────
    train_df = snapshots[snapshots["snapshot_month"] <= train_end].copy()
    val_df = snapshots[
        (snapshots["snapshot_month"] >= val_start)
        & (snapshots["snapshot_month"] <= val_end)
    ].copy()

    print(f"Train rows: {len(train_df):,}  |  Val rows: {len(val_df):,}")

    # ── Classify roles ────────────────────────────────────────────────────
    for df in (train_df, val_df):
        df["role"] = [
            classify_role(
                row.get("bat_balls_wtd", 0.0),
                row.get("bowl_balls_wtd", 0.0),
                bool(row.get("is_keeper", False)),
            )
            for _, row in df.iterrows()
        ]

    # ── Compute group priors on training data ─────────────────────────────
    priors = compute_group_priors(train_df)
    print(f"Groups: {sum(len(v) for v in priors.values())} (role x format)")

    # ── Apply shrinkage ───────────────────────────────────────────────────
    train_ratings = apply_shrinkage(train_df, priors, prior_n)

    # ── Select output columns ─────────────────────────────────────────────
    out_cols = [
        "player_id", "snapshot_month", "format", "role",
        "bat_rating", "bat_uncertainty", "bowl_rating", "bowl_uncertainty",
    ]
    ratings_out = train_ratings[out_cols].copy()

    # ── Also produce ratings for val set (using train priors) ─────────────
    val_ratings = apply_shrinkage(val_df, priors, prior_n)
    all_ratings = pd.concat(
        [ratings_out, val_ratings[out_cols]],
        ignore_index=True,
    )

    # ── Save model artifacts ──────────────────────────────────────────────
    model_dir = ROOT / "data" / "processed" / "models" / "ability"
    model_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = model_dir / "ratings.parquet"
    all_ratings.to_parquet(ratings_path, index=False, engine="pyarrow")
    print(f"Ratings written to {ratings_path}  ({len(all_ratings):,} rows)")

    priors_path = model_dir / "group_priors.json"
    with open(priors_path, "w") as f:
        json.dump(priors, f, indent=2)
    print(f"Group priors written to {priors_path}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = evaluate_on_validation(train_ratings, val_df, prior_n, priors)
    metrics["format"] = fmt
    metrics["prior_n"] = prior_n
    metrics["train_end"] = str(train_end.date())
    metrics["val_start"] = str(val_start.date())
    metrics["val_end"] = str(val_end.date())

    metrics_dir = ROOT / "data" / "processed" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "ability.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {metrics_path}")
    print(f"  bat_rating_vs_sr_corr:  {metrics.get('bat_rating_vs_sr_corr')}")
    print(f"  bat_rating_vs_avg_corr: {metrics.get('bat_rating_vs_avg_corr')}")
    print(f"  bowl_rating_vs_econ_corr: {metrics.get('bowl_rating_vs_econ_corr')}")
    print(f"  bowl_rating_vs_sr_corr:   {metrics.get('bowl_rating_vs_sr_corr')}")


if __name__ == "__main__":
    main()
