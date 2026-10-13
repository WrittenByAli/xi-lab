"""
Look up a player's Cricsheet ID from people.csv by name.

Usage:
    python ml/ingest/lookup_player.py "Babar Azam"
    python ml/ingest/lookup_player.py "Shaheen"         # partial match
    python ml/ingest/lookup_player.py --team Pakistan   # list all from a team
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PEOPLE_CSV = ROOT / "data" / "reference" / "people.csv"


def load_people() -> pd.DataFrame:
    if not PEOPLE_CSV.exists():
        print(f"[ERROR] {PEOPLE_CSV} not found. Run download.py first.")
        raise SystemExit(1)
    df = pd.read_csv(PEOPLE_CSV, low_memory=False)
    # Cricsheet people.csv columns vary; normalize
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def lookup(query: str) -> None:
    df = load_people()
    name_col = next((c for c in df.columns if "name" in c), df.columns[0])
    id_col = next((c for c in df.columns if "identifier" in c or c == "key"), None)

    mask = df[name_col].str.contains(query, case=False, na=False)
    results = df[mask][[name_col, id_col] if id_col else [name_col]].head(20)

    if results.empty:
        print(f"No player found matching '{query}'")
    else:
        print(results.to_string(index=False))


def list_team(team: str) -> None:
    df = load_people()
    team_col = next((c for c in df.columns if "team" in c or "country" in c), None)
    if not team_col:
        print("[WARN] No team column found in people.csv")
        return
    mask = df[team_col].str.contains(team, case=False, na=False)
    print(df[mask].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Look up Cricsheet player IDs")
    parser.add_argument("name", nargs="?", help="Player name (partial match OK)")
    parser.add_argument("--team", help="List all players from this team")
    args = parser.parse_args()

    if args.team:
        list_team(args.team)
    elif args.name:
        lookup(args.name)
    else:
        parser.print_help()
