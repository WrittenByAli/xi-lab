"""
Coverage report: which players are missing from player_meta.csv.

Shows the top uncovered players by balls involved, so you know exactly
who to add next to reach the 85%+ coverage target.

Usage:
    python ml/ingest/coverage.py               # T20 only
    python ml/ingest/coverage.py --format odi
    python ml/ingest/coverage.py --top 50      # show top 50 missing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
REFERENCE = ROOT / "data" / "reference"


def run(fmt: str = "t20", top_n: int = 100) -> None:
    balls_path = INTERIM / fmt / "balls.parquet"
    meta_path = REFERENCE / "player_meta.csv"

    if not balls_path.exists():
        print(f"[ERROR] {balls_path} not found. Run parse.py first.")
        return

    balls = pd.read_parquet(balls_path, columns=["batter_id", "bowler_id", "is_legal"])
    meta = pd.read_csv(meta_path, comment="#")

    # Count balls per player (faced + bowled)
    faced = balls[balls["is_legal"]].groupby("batter_id").size().rename("balls_faced")
    bowled = balls[balls["is_legal"]].groupby("bowler_id").size().rename("balls_bowled")
    involvement = (
        pd.concat([faced, bowled], axis=1)
        .fillna(0)
        .assign(total=lambda df: df["balls_faced"] + df["balls_bowled"])
        .sort_values("total", ascending=False)
    )

    total_balls = involvement["total"].sum()
    covered_ids = set(meta["player_id"].dropna())

    involvement["covered"] = involvement.index.isin(covered_ids)
    covered_balls = involvement.loc[involvement["covered"], "total"].sum()

    print(f"\n{'=' * 60}")
    print(f"  player_meta coverage report  ({fmt.upper()})")
    print(f"{'=' * 60}")
    print(f"  Total players in data : {len(involvement):,}")
    print(f"  Covered in meta       : {involvement['covered'].sum():,}")
    print(f"  Ball coverage         : {covered_balls / total_balls * 100:.1f}%")
    print(f"{'=' * 60}\n")

    missing = involvement[~involvement["covered"]].head(top_n)
    if missing.empty:
        print("  All top players are covered!")
        return

    print(f"  Top {top_n} uncovered players (add these to player_meta.csv):\n")
    print(f"  {'Rank':<5} {'player_id':<40} {'balls_faced':>12} {'balls_bowled':>12} {'total':>8}")
    print(f"  {'-' * 5} {'-' * 40} {'-' * 12} {'-' * 12} {'-' * 8}")
    for rank, (pid, row) in enumerate(missing.iterrows(), 1):
        print(
            f"  {rank:<5} {pid:<40} "
            f"{int(row['balls_faced']):>12,} "
            f"{int(row['balls_bowled']):>12,} "
            f"{int(row['total']):>8,}"
        )

    # Also write to a file for easy reference
    out = REFERENCE / f"coverage_missing_{fmt}.csv"
    missing.reset_index().rename(columns={"index": "player_id"}).to_csv(out, index=False)
    print(f"\n  Full list written to: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="player_meta coverage report")
    parser.add_argument("--format", default="t20", choices=["t20", "odi", "test"])
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    run(fmt=args.format, top_n=args.top)
