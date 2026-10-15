"""
Train a LightGBM multiclass ball-outcome model.

For each legal delivery, predicts one of 9 outcome classes:
  dot, 1, 2, 3, 4, 6, wicket, wide, noball

Features come from match-situation columns (over, score, wickets, etc.)
plus batter and bowler snapshot features joined as-of the match date
(most recent snapshot_month strictly before the match date).

Usage:
    python ml/models/ball/train.py
    python ml/models/ball/train.py --format t20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[3]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

# ── Feature lists ─────────────────────────────────────────────────────────────

SITUATION_FEATURES = [
    "over",
    "legal_balls_before",
    "score_before",
    "wkts_before",
    "rrr_before",
    "innings_no",
    "is_international",
    "venue_id",
    "toss_bat_first",
]

BATTER_SNAPSHOT_FEATURES = [
    "bat_sr_wtd",
    "bat_avg_wtd",
    "bat_boundary_pct_wtd",
    "bat_dot_pct_wtd",
    "bat_sr_powerplay_wtd",
    "bat_sr_middle_wtd",
    "bat_sr_death_wtd",
    "form_bat_sr",
]

BOWLER_SNAPSHOT_FEATURES = [
    "bowl_econ_wtd",
    "bowl_sr_wtd",
    "bowl_dot_pct_wtd",
    "bowl_econ_powerplay_wtd",
    "bowl_econ_middle_wtd",
    "bowl_econ_death_wtd",
    "form_bowl_econ",
]

CATEGORICAL_FEATURES = [
    "bat_hand",
    "bowl_type",
    "venue_id",
]

ALL_FEATURES = (
    SITUATION_FEATURES + BATTER_SNAPSHOT_FEATURES + BOWLER_SNAPSHOT_FEATURES + CATEGORICAL_FEATURES
)


# ── As-of snapshot join ──────────────────────────────────────────────────────


def asof_join_snapshots(
    balls: pd.DataFrame,
    snapshots: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Join batter and bowler snapshots to each ball row using an as-of join.

    For each ball, the snapshot used is the one from the latest snapshot_month
    that is strictly BEFORE the match date.  This prevents any data leakage.

    Parameters
    ----------
    balls : DataFrame
        Ball-by-ball data with match_id, batter_id, bowler_id columns.
    snapshots : DataFrame
        Monthly player snapshots with snapshot_month, player_id, and feature columns.
    matches : DataFrame
        Match metadata with match_id, date, venue_id, league_id, is_international.

    Returns
    -------
    DataFrame
        Balls enriched with batter/bowler snapshot features and match metadata.
    """
    # Attach match date and metadata to each ball
    match_cols = ["match_id", "date", "is_international", "venue_id", "toss_decision"]
    available_match_cols = [c for c in match_cols if c in matches.columns]
    balls = balls.merge(matches[available_match_cols], on="match_id", how="left")
    balls["date"] = pd.to_datetime(balls["date"])

    # Derive binary toss feature: did the batting team choose to bat first?
    if "toss_decision" in balls.columns:
        balls["toss_bat_first"] = (balls["toss_decision"] == "bat").astype(int)
    else:
        balls["toss_bat_first"] = 0

    # Ensure snapshot_month is datetime
    snaps = snapshots.copy()
    snaps["snapshot_month"] = pd.to_datetime(snaps["snapshot_month"])

    # -- Batter snapshot join --
    batter_cols = ["player_id", "snapshot_month"] + [
        c for c in BATTER_SNAPSHOT_FEATURES + CATEGORICAL_FEATURES if c in snaps.columns
    ]
    batter_snaps = snaps[batter_cols].drop_duplicates()
    batter_snaps = batter_snaps.rename(columns={"player_id": "batter_id"})

    # For the as-of join, we need both sides sorted by the join key
    balls = balls.sort_values("date").reset_index(drop=True)
    batter_snaps = batter_snaps.sort_values("snapshot_month")

    # Perform as-of merge: for each ball, find the latest snapshot_month < date
    # pd.merge_asof matches the last row in batter_snaps where snapshot_month <= date
    # We want strictly less than, so we use allow_exact_matches=False
    # But merge_asof requires the "by" column, so we do per-batter
    batter_snaps_renamed = batter_snaps.rename(columns={"snapshot_month": "date"})

    balls = pd.merge_asof(
        balls,
        batter_snaps_renamed,
        on="date",
        by="batter_id",
        direction="backward",
        allow_exact_matches=False,
    )

    # -- Bowler snapshot join --
    bowler_feature_cols = [c for c in BOWLER_SNAPSHOT_FEATURES if c in snaps.columns]
    # bowl_type may also be in snapshots — grab it for bowler if not already present
    bowler_extra_cats = [c for c in ["bowl_type"] if c in snaps.columns and c not in balls.columns]
    bowler_cols = ["player_id", "snapshot_month"] + bowler_feature_cols + bowler_extra_cats
    bowler_snaps = snaps[bowler_cols].drop_duplicates()
    bowler_snaps = bowler_snaps.rename(columns={"player_id": "bowler_id"})
    # Prefix bowler snapshot columns that would collide (bowl_type already distinct)
    bowler_snaps = bowler_snaps.sort_values("snapshot_month")
    bowler_snaps_renamed = bowler_snaps.rename(columns={"snapshot_month": "date"})

    balls = pd.merge_asof(
        balls,
        bowler_snaps_renamed,
        on="date",
        by="bowler_id",
        direction="backward",
        allow_exact_matches=False,
    )

    return balls


# ── Label encoding ───────────────────────────────────────────────────────────


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns in-place, returning fitted encoders."""
    if encoders is None:
        encoders = {}
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "unknown"
        # Fill NaN with "unknown" so LabelEncoder doesn't fail
        df[col] = df[col].fillna("unknown").astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels by mapping them to "unknown"
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x, k=known: x if x in k else "unknown")
            df[col] = le.transform(df[col])
    return df, encoders


# ── Split helpers ────────────────────────────────────────────────────────────


def split_by_date(
    df: pd.DataFrame,
    splits: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train / val / test by match date."""
    train_end = pd.Timestamp(splits["train_end"])
    val_start = pd.Timestamp(splits["val_start"])
    val_end = pd.Timestamp(splits["val_end"])
    test_start = pd.Timestamp(splits["test_start"])

    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test = df[df["date"] >= test_start]

    return train, val, test


# ── Training ─────────────────────────────────────────────────────────────────


def train_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    target_col: str,
    params_cfg: dict,
) -> lgb.Booster:
    """Train a LightGBM multiclass model with early stopping."""
    outcome_classes = params_cfg["outcome_classes"]
    n_classes = len(outcome_classes)

    lgb_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "num_leaves": params_cfg["num_leaves"],
        "min_child_samples": params_cfg["min_child_samples"],
        "learning_rate": params_cfg["learning_rate"],
        "subsample": params_cfg["subsample"],
        "colsample_bytree": params_cfg["colsample_bytree"],
        "seed": params_cfg["seed"],
        "verbose": -1,
    }

    train_ds = lgb.Dataset(train[features], label=train[target_col])
    val_ds = lgb.Dataset(val[features], label=val[target_col], reference=train_ds)

    callbacks: list = [
        lgb.early_stopping(params_cfg["early_stopping_rounds"]),
        lgb.log_evaluation(100),
    ]

    model = lgb.train(
        lgb_params,
        train_ds,
        num_boost_round=params_cfg["n_estimators"],
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=callbacks,
    )

    return model


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: lgb.Booster,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    outcome_classes: list[str],
    split_name: str,
) -> dict:
    """Compute metrics for a given split."""
    y_true = df[target_col].values
    y_pred_proba = model.predict(df[features])
    y_pred = np.argmax(y_pred_proba, axis=1)

    ll = float(log_loss(y_true, y_pred_proba, labels=list(range(len(outcome_classes)))))
    acc = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(outcome_classes))),
        zero_division=0,
    )

    per_class = {}
    for i, cls_name in enumerate(outcome_classes):
        per_class[cls_name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # Feature importance (top 20)
    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    feat_imp = sorted(zip(feat_names, importance.tolist(), strict=False), key=lambda x: -x[1])[:20]

    return {
        "split": split_name,
        "log_loss": round(ll, 6),
        "accuracy": round(acc, 4),
        "n_samples": len(df),
        "per_class": per_class,
        "feature_importance_top20": [{"feature": f, "gain": round(g, 2)} for f, g in feat_imp],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM ball-outcome model")
    parser.add_argument(
        "--format",
        default="t20",
        choices=["t20", "odi", "test"],
        help="Cricket format (default: t20)",
    )
    args = parser.parse_args()

    fmt = args.format
    ball_cfg = PARAMS["ball_model"]
    split_cfg = PARAMS["splits"]
    outcome_classes = ball_cfg["outcome_classes"]

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[{fmt}] Loading data...")
    balls_path = ROOT / "data" / "interim" / fmt / "balls.parquet"
    matches_path = ROOT / "data" / "interim" / fmt / "matches.parquet"
    snapshots_path = ROOT / "data" / "processed" / "snapshots" / fmt / "snapshots.parquet"

    balls = pd.read_parquet(balls_path)
    matches = pd.read_parquet(matches_path)
    snapshots = pd.read_parquet(snapshots_path)

    print(f"  Balls: {len(balls):,} rows")
    print(f"  Matches: {len(matches):,} rows")
    print(f"  Snapshots: {len(snapshots):,} rows")

    # ── Join snapshots to balls ──────────────────────────────────────────────
    print("Joining snapshots to balls (as-of join)...")
    df = asof_join_snapshots(balls, snapshots, matches)

    # ── Encode target ────────────────────────────────────────────────────────
    class_to_idx = {cls: i for i, cls in enumerate(outcome_classes)}
    df["target"] = df["outcome_class"].map(class_to_idx)

    # Drop rows with unmapped outcome classes
    n_before = len(df)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped:,} rows with unmapped outcome_class")

    # ── Ensure is_international exists ───────────────────────────────────────
    if "is_international" not in df.columns:
        df["is_international"] = 0

    # ── Encode categoricals ──────────────────────────────────────────────────
    print("Encoding categorical features...")
    available_cats = [c for c in CATEGORICAL_FEATURES if True]
    df, encoders = encode_categoricals(df, available_cats, fit=True)

    # ── Fill NaN in numeric features ─────────────────────────────────────────
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    # LightGBM handles NaN natively, so no imputation needed

    # ── Split ────────────────────────────────────────────────────────────────
    print("Splitting by date...")
    train, val, test = split_by_date(df, split_cfg)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    if len(train) == 0 or len(val) == 0:
        raise ValueError("Train or validation set is empty. Check date splits.")

    # ── Train ────────────────────────────────────────────────────────────────
    print("Training LightGBM model...")
    model = train_model(train, val, ALL_FEATURES, "target", ball_cfg)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("Evaluating...")
    metrics = {}
    metrics["val"] = evaluate_model(model, val, ALL_FEATURES, "target", outcome_classes, "val")
    if len(test) > 0:
        metrics["test"] = evaluate_model(
            model, test, ALL_FEATURES, "target", outcome_classes, "test"
        )

    print(
        f"  Val log_loss: {metrics['val']['log_loss']:.4f}  accuracy: {metrics['val']['accuracy']:.4f}"
    )
    if "test" in metrics:
        print(
            f"  Test log_loss: {metrics['test']['log_loss']:.4f}  accuracy: {metrics['test']['accuracy']:.4f}"
        )

    # ── Save model ───────────────────────────────────────────────────────────
    model_dir = ROOT / "data" / "processed" / "models" / "ball"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.lgb"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics_dir = ROOT / "data" / "processed" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "ball.json"

    # Also store the label encoders mapping and feature list for reproducibility
    metrics["outcome_classes"] = outcome_classes
    metrics["features"] = ALL_FEATURES
    metrics["encoders"] = {col: list(le.classes_) for col, le in encoders.items()}

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
