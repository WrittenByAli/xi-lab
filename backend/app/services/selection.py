"""Service layer for XI selection and simulation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


@lru_cache(maxsize=1)
def load_ratings() -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "models" / "ability" / "ratings.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@lru_cache(maxsize=1)
def load_player_meta() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "reference" / "player_meta.csv", comment="#")


@lru_cache(maxsize=1)
def load_snapshots(fmt: str = "t20") -> pd.DataFrame | None:
    path = ROOT / "data" / "processed" / "snapshots" / fmt / "snapshots.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@lru_cache(maxsize=1)
def load_backtest_metrics() -> dict | None:
    path = ROOT / "data" / "processed" / "metrics" / "backtest.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def get_player_profile(player_id: str, fmt: str = "t20") -> dict | None:
    meta = load_player_meta()
    row = meta[meta["player_id"] == player_id]
    if row.empty:
        return None

    info = row.iloc[0].to_dict()
    profile = {
        "player_id": player_id,
        "name": info.get("name", player_id),
        "bat_hand": info.get("bat_hand"),
        "bowl_type": info.get("bowl_type"),
        "is_keeper": bool(info.get("is_keeper", False)),
        "role": "pure_bat",
    }

    ratings = load_ratings()
    if ratings is not None:
        r = ratings[ratings["player_id"] == player_id]
        if not r.empty:
            latest = r.sort_values("snapshot_month").iloc[-1]
            profile["bat_rating"] = float(latest.get("bat_rating", 0.5))
            profile["bowl_rating"] = float(latest.get("bowl_rating", 0.5))
            profile["role"] = str(latest.get("role", "pure_bat"))

    snaps = load_snapshots(fmt)
    if snaps is not None:
        s = snaps[snaps["player_id"] == player_id]
        if not s.empty:
            latest = s.sort_values("snapshot_month").iloc[-1]
            profile["bat_sr"] = _safe_float(latest.get("bat_sr_wtd"))
            profile["bat_avg"] = _safe_float(latest.get("bat_avg_wtd"))
            profile["bowl_econ"] = _safe_float(latest.get("bowl_econ_wtd"))
            profile["bowl_sr"] = _safe_float(latest.get("bowl_sr_wtd"))
            profile["total_balls_bat"] = int(latest.get("bat_balls_raw", 0))
            profile["total_balls_bowl"] = int(latest.get("bowl_balls_raw", 0))

            phase_ratings = []
            for pname in ["powerplay", "middle", "death"]:
                pr = {"phase": pname}
                sr_key = f"bat_sr_{pname}_wtd"
                econ_key = f"bowl_econ_{pname}_wtd"
                balls_key = f"bat_balls_{pname}_wtd"
                pr["bat_sr"] = _safe_float(latest.get(sr_key))
                pr["bowl_econ"] = _safe_float(latest.get(econ_key))
                pr["balls"] = int(latest.get(balls_key, 0))
                phase_ratings.append(pr)
            profile["phase_ratings"] = phase_ratings

    return profile


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def build_score_distribution(scores: np.ndarray) -> dict:
    return {
        "mean": round(float(scores.mean()), 2),
        "std": round(float(scores.std()), 2),
        "median": round(float(np.median(scores)), 2),
        "q10": round(float(np.percentile(scores, 10)), 2),
        "q90": round(float(np.percentile(scores, 90)), 2),
    }
