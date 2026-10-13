"""
Build monthly as-of player snapshots from ball-by-ball data.

Each snapshot row describes what we *know* about a player as of the 1st of a
given month, using only data strictly before that date (no leakage).

Output: data/processed/snapshots/{format}/snapshots.parquet
  One row per (player_id, snapshot_month) with batting, bowling, and fielding
  aggregates weighted by exponential recency decay.

Usage:
    python ml/features/snapshots.py             # all available formats
    python ml/features/snapshots.py --format t20
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())

BOWLER_DISMISSALS = frozenset(
    ["bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"]
)


def _load_data(fmt: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "data" / "interim" / fmt
    balls = pd.read_parquet(base / "balls.parquet")
    matches = pd.read_parquet(base / "matches.parquet")
    innings = pd.read_parquet(base / "innings.parquet")
    lineups = pd.read_parquet(base / "lineups.parquet")
    matches["date"] = pd.to_datetime(matches["date"])
    return balls, matches, innings, lineups


def _load_meta() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "reference" / "player_meta.csv", comment="#")


def _month_range(dates: pd.Series) -> pd.DatetimeIndex:
    first = dates.min().replace(day=1)
    last = dates.max().replace(day=1) + pd.offsets.MonthBegin(1)
    return pd.date_range(first, last, freq="MS")


def _decay_weights(days_ago: np.ndarray, half_life: float) -> np.ndarray:
    return np.exp(-np.log(2) * days_ago / half_life)


def _prepare_balls(
    balls: pd.DataFrame,
    match_dates: pd.Series,
    cutoff: pd.Timestamp,
    half_life: float,
) -> pd.DataFrame | None:
    """Merge dates, filter before cutoff, compute decay weights and pre-multiplied columns."""
    b = balls.merge(match_dates.rename("date"), left_on="match_id", right_index=True)
    b = b[b["date"] < cutoff]
    if b.empty:
        return None

    days_ago = (cutoff - b["date"]).dt.days.values.astype(float)
    w = _decay_weights(days_ago, half_life)

    b = b.assign(
        _w=w,
        _w_runs=w * b["runs_batter"].values,
        _w_dot=w * (b["runs_batter"].values == 0).astype(float),
        _w_four=w * (b["runs_batter"].values == 4).astype(float),
        _w_six=w * (b["runs_batter"].values == 6).astype(float),
        _w_runs_conceded=w * (b["runs_batter"].values + b["runs_extras"].values),
        _w_bowl_dot=w * ((b["runs_batter"].values + b["runs_extras"].values) == 0).astype(float),
        _is_bowler_wkt=(
            b["wicket_kind"].isin(BOWLER_DISMISSALS).values
            & (b["player_out_id"] == b["batter_id"]).values
        ),
        _is_bowler_wkt_any=b["wicket_kind"].isin(BOWLER_DISMISSALS).values,
    )
    b["_w_bat_wkt"] = w * b["_is_bowler_wkt"].astype(float)
    b["_w_bowl_wkt"] = w * b["_is_bowler_wkt_any"].astype(float)

    return b


def build_batting_features(legal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batting stats. Input is pre-prepared legal balls with _w columns."""
    agg = legal.groupby("batter_id", sort=False).agg(
        bat_balls_raw=("_w", "count"),
        bat_balls_wtd=("_w", "sum"),
        bat_runs_wtd=("_w_runs", "sum"),
        bat_dots_wtd=("_w_dot", "sum"),
        bat_fours_wtd=("_w_four", "sum"),
        bat_sixes_wtd=("_w_six", "sum"),
        bat_wkts_wtd=("_w_bat_wkt", "sum"),
    )

    bw = agg["bat_balls_wtd"].values
    agg["bat_sr_wtd"] = np.where(bw > 0, agg["bat_runs_wtd"].values / bw * 100, np.nan)
    agg["bat_dot_pct_wtd"] = np.where(bw > 0, agg["bat_dots_wtd"].values / bw, np.nan)
    agg["bat_boundary_pct_wtd"] = np.where(
        bw > 0, (agg["bat_fours_wtd"].values + agg["bat_sixes_wtd"].values) / bw, np.nan
    )
    wk = agg["bat_wkts_wtd"].values
    agg["bat_avg_wtd"] = np.where(wk > 0, agg["bat_runs_wtd"].values / wk, np.nan)

    agg.index.name = "player_id"
    return agg.reset_index()


def build_bowling_features(legal: pd.DataFrame, all_balls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bowling stats. legal = legal balls; all_balls includes extras."""
    agg = legal.groupby("bowler_id", sort=False).agg(
        bowl_balls_raw=("_w", "count"),
        bowl_balls_wtd=("_w", "sum"),
        bowl_runs_wtd=("_w_runs_conceded", "sum"),
        bowl_dots_wtd=("_w_bowl_dot", "sum"),
        bowl_fours_wtd=("_w_four", "sum"),
        bowl_sixes_wtd=("_w_six", "sum"),
        bowl_wkts_wtd=("_w_bowl_wkt", "sum"),
    )

    extras = all_balls[all_balls["is_wide"] | all_balls["is_noball"]]
    extras_agg = extras.groupby("bowler_id", sort=False)["_w"].sum().rename("bowl_extras_wtd")
    agg = agg.join(extras_agg, how="left")
    agg["bowl_extras_wtd"] = agg["bowl_extras_wtd"].fillna(0.0)

    bw = agg["bowl_balls_wtd"].values
    agg["bowl_econ_wtd"] = np.where(bw > 0, agg["bowl_runs_wtd"].values / bw * 6, np.nan)
    wk = agg["bowl_wkts_wtd"].values
    agg["bowl_sr_wtd"] = np.where(wk > 0, bw / wk, np.nan)
    agg["bowl_dot_pct_wtd"] = np.where(bw > 0, agg["bowl_dots_wtd"].values / bw, np.nan)

    agg.index.name = "player_id"
    return agg.reset_index()


def build_phase_features(
    legal: pd.DataFrame,
    phases: list[list[int]],
    phase_names: list[str],
) -> pd.DataFrame:
    """Batting SR and bowling economy split by game phase."""
    over_vals = legal["over"].values

    phase_col = np.full(len(legal), phase_names[-1], dtype=object)
    for rng, name in zip(phases, phase_names, strict=False):
        mask = (over_vals >= rng[0]) & (over_vals <= rng[1])
        phase_col[mask] = name

    legal = legal.assign(_phase=phase_col)

    parts = []
    for pname in phase_names:
        ph = legal[legal["_phase"] == pname]
        if ph.empty:
            continue

        bat = ph.groupby("batter_id", sort=False).agg(
            balls=("_w", "sum"),
            runs=("_w_runs", "sum"),
        )
        bat[f"bat_sr_{pname}_wtd"] = np.where(
            bat["balls"].values > 0, bat["runs"].values / bat["balls"].values * 100, np.nan
        )
        bat = bat.rename(columns={"balls": f"bat_balls_{pname}_wtd"})[
            [f"bat_sr_{pname}_wtd", f"bat_balls_{pname}_wtd"]
        ]
        bat.index.name = "player_id"
        parts.append(bat)

        bowl = ph.groupby("bowler_id", sort=False).agg(
            balls=("_w", "sum"),
            runs=("_w_runs_conceded", "sum"),
        )
        bowl[f"bowl_econ_{pname}_wtd"] = np.where(
            bowl["balls"].values > 0, bowl["runs"].values / bowl["balls"].values * 6, np.nan
        )
        bowl = bowl.rename(columns={"balls": f"bowl_balls_{pname}_wtd"})[
            [f"bowl_econ_{pname}_wtd", f"bowl_balls_{pname}_wtd"]
        ]
        bowl.index.name = "player_id"
        parts.append(bowl)

    if not parts:
        return pd.DataFrame()

    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, how="outer")
    return result.reset_index()


def build_recent_form(
    balls: pd.DataFrame,
    match_dates: pd.Series,
    cutoff: pd.Timestamp,
    window_days: int = 90,
) -> pd.DataFrame:
    """Simple unweighted stats over the last N days — a 'form' signal."""
    b = balls.merge(match_dates.rename("date"), left_on="match_id", right_index=True)
    start = cutoff - pd.Timedelta(days=window_days)
    b = b[(b["date"] >= start) & (b["date"] < cutoff)]
    if b.empty:
        return pd.DataFrame()

    legal = b[b["is_legal"]]
    runs_batter = legal["runs_batter"].values
    runs_extras = legal["runs_extras"].values

    bat = legal.groupby("batter_id", sort=False).agg(
        form_bat_balls=("runs_batter", "count"),
        form_bat_runs=("runs_batter", "sum"),
    )
    bat["form_bat_sr"] = np.where(
        bat["form_bat_balls"].values > 0,
        bat["form_bat_runs"].values / bat["form_bat_balls"].values * 100,
        np.nan,
    )
    bat.index.name = "player_id"

    legal = legal.assign(_rc=runs_batter + runs_extras)
    bowl = legal.groupby("bowler_id", sort=False).agg(
        form_bowl_balls=("_rc", "count"),
        form_bowl_runs=("_rc", "sum"),
    )
    bowl["form_bowl_econ"] = np.where(
        bowl["form_bowl_balls"].values > 0,
        bowl["form_bowl_runs"].values / bowl["form_bowl_balls"].values * 6,
        np.nan,
    )
    bowl.index.name = "player_id"

    return bat.reset_index().merge(bowl.reset_index(), on="player_id", how="outer")


def build_snapshots_for_format(fmt: str) -> pd.DataFrame:
    """Build all monthly snapshots for one format."""
    print(f"[{fmt}] Loading data...")
    balls, matches, innings, lineups = _load_data(fmt)
    meta = _load_meta()

    match_dates = matches.set_index("match_id")["date"]
    half_life = PARAMS["ability"]["rating_half_life_days"]

    fmt_params = PARAMS["formats"].get(fmt, PARAMS["formats"]["t20"])
    phases = fmt_params.get("phases", [[0, 5], [6, 15], [16, 19]])
    phase_names = fmt_params.get("phase_names", ["powerplay", "middle", "death"])

    months = _month_range(matches["date"])
    print(f"[{fmt}] {len(months)} monthly snapshots from {months[0].date()} to {months[-1].date()}")

    all_snapshots = []
    for i, cutoff in enumerate(months):
        if i % 12 == 0:
            print(f"  {cutoff.date()} ({i + 1}/{len(months)})")

        prepared = _prepare_balls(balls, match_dates, cutoff, half_life)
        if prepared is None:
            continue

        legal = prepared[prepared["is_legal"]]

        bat = build_batting_features(legal)
        bowl = build_bowling_features(legal, prepared)
        phase = build_phase_features(legal, phases, phase_names)
        form = build_recent_form(balls, match_dates, cutoff)

        # Union all player_ids from batting and bowling
        all_pids = set()
        if not bat.empty:
            all_pids.update(bat["player_id"])
        if not bowl.empty:
            all_pids.update(bowl["player_id"])
        snap = pd.DataFrame({"player_id": sorted(all_pids)})

        if not bat.empty:
            snap = snap.merge(bat, on="player_id", how="left")
        if not bowl.empty:
            snap = snap.merge(bowl, on="player_id", how="left")
        if not phase.empty:
            snap = snap.merge(phase, on="player_id", how="left")
        if not form.empty:
            snap = snap.merge(form, on="player_id", how="left")

        meta_cols = ["player_id", "bat_hand", "bowl_type", "is_keeper"]
        snap = snap.merge(meta[meta_cols], on="player_id", how="left")

        snap.insert(0, "snapshot_month", cutoff)
        snap.insert(1, "format", fmt)
        all_snapshots.append(snap)

    result = pd.concat(all_snapshots, ignore_index=True)
    print(
        f"[{fmt}] {len(result):,} total snapshot rows ({result['player_id'].nunique():,} unique players)"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["t20", "odi", "test"], default=None)
    args = parser.parse_args()

    formats = []
    for fmt in ("t20", "odi", "test"):
        if args.format and fmt != args.format:
            continue
        data_dir = ROOT / "data" / "interim" / fmt
        if (data_dir / "balls.parquet").exists():
            formats.append(fmt)

    if not formats:
        print("No parsed data found. Run parse.py first.")
        return

    out_dir = ROOT / "data" / "processed" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        df = build_snapshots_for_format(fmt)
        fmt_dir = out_dir / fmt
        fmt_dir.mkdir(exist_ok=True)
        out_path = fmt_dir / "snapshots.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"[{fmt}] Written to {out_path}")


if __name__ == "__main__":
    main()
