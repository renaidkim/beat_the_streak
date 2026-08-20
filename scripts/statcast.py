"""Fetch season-level Statcast leaderboards from Baseball Savant's public
CSV export endpoint (baseballsavant.mlb.com/leaderboard/custom), cached
to disk like every other fetch in this project.

player_id in these leaderboards is the same MLB Stats API person id used
everywhere else in this codebase, so no id-mapping step is needed.

Statcast leaderboards are season-level aggregates (not counting stats),
so unlike career_avg etc. there's no clean way to sum across multiple
prior seasons into a "career" rate without weighting by batted-ball-event
counts the leaderboard doesn't expose here. Callers use a single lag
instead: the immediately preceding completed season's profile as the
"entering this season" feature. That's also arguably more appropriate for
physical/skill metrics like exit velocity or sprint speed, which
meaningfully drift year to year (aging, swing changes) in a way a
multi-year blend would smear together.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "statcast"
BASE = "https://baseballsavant.mlb.com/leaderboard/custom"

BATTER_SELECTIONS = [
    "xba", "xwoba", "xslg", "exit_velocity_avg", "barrel_batted_rate",
    "hard_hit_percent", "sprint_speed", "whiff_percent",
]
PITCHER_SELECTIONS = [
    "xwoba", "xba", "exit_velocity_avg", "barrel_batted_rate",
    "hard_hit_percent", "whiff_percent", "fastball_avg_speed", "k_percent",
    "bb_percent",
]

session = requests.Session()


def _cached_csv(year: int, player_type: str, selections: list[str]) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{year}-{player_type}-{','.join(selections)}".encode()).hexdigest()
    path = CACHE_DIR / f"{key}.csv"
    if path.exists():
        return path.read_text(encoding="utf-8-sig")
    resp = session.get(
        BASE,
        params={
            "year": year,
            "type": player_type,
            "filter": "",
            "min": 1,
            "selections": ",".join(selections),
            "chart": "false",
            "x": selections[0],
            "y": selections[0],
            "r": "no",
            "chartType": "beeswarm",
            "csv": "true",
        },
        timeout=30,
    )
    resp.raise_for_status()
    # A season-level qualified-batter/pitcher leaderboard should have
    # hundreds of rows; a handful means a truncated or otherwise bad
    # response (seen intermittently through the proxy) -- fail loudly
    # instead of silently caching garbage that would poison every run
    # after it.
    row_count = resp.text.count("\n")
    if row_count < 50:
        raise RuntimeError(
            f"Statcast leaderboard for {year}/{player_type} only had "
            f"{row_count} lines -- looks truncated, not caching. Response: "
            f"{resp.text[:300]!r}"
        )
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def fetch_leaderboard(
    year: int, player_type: str, selections: list[str]
) -> dict[int, dict[str, float]]:
    """{player_id: {selection_name: value or None}} for one season."""
    text = _cached_csv(year, player_type, selections)
    reader = csv.DictReader(io.StringIO(text))
    result: dict[int, dict[str, float]] = {}
    for row in reader:
        pid = int(row["player_id"])
        feats: dict[str, float] = {}
        for col in selections:
            v = (row.get(col) or "").strip()
            feats[col] = float(v) if v else None
        result[pid] = feats
    return result


if __name__ == "__main__":
    lb = fetch_leaderboard(2023, "batter", BATTER_SELECTIONS)
    print(f"{len(lb)} batters fetched for 2023")
    sys.exit(0)
