"""Recompute data/park_factors.json from the MLB Stats API.

Methodology: for each team, take the ratio of its own hitters' batting
average at home vs. on the road, averaged over the last 3 completed
seasons. Using the same team's hitters home vs. away cancels out that
team's own offensive quality, leaving park as the main remaining driver
of the gap (schedule-strength imbalance between a team's home and road
slates is a smaller, uncontrolled confound). Averaging 3 seasons instead
of 1 smooths out year-to-year sampling noise enough to match published,
more rigorous (opponent-adjusted, multi-year) park factors: Coors Field
and Seattle land at the expected extremes, and most parks cluster within
a few percent of neutral.

This is a deliberately simple proxy, not Statcast-grade park factors
(no split by handedness, batted-ball type, or weather) — good enough to
weight a heuristic hit-probability model, not for a research paper.
Rerun this once or twice a season; park effects don't shift week to week.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import requests

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
SEASONS_BACK = 3
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "park_factors.json"

# Keep the computed factor within a sane band so one noisy season (or a
# team with a short current-park history) can't produce an outlier the
# ranker would over-trust.
MIN_FACTOR = 0.85
MAX_FACTOR = 1.30


def latest_completed_season() -> int:
    import datetime

    today = datetime.date.today()
    # The current season isn't "completed" for this purpose until well
    # after it ends; treat anything from this year as in-progress.
    return today.year - 1


def fetch_team_park_factors(session: requests.Session) -> dict[int, dict]:
    resp = session.get(
        f"{MLB_STATS_API_BASE}/teams", params={"sportId": 1, "activeStatus": "Yes"}, timeout=15
    )
    resp.raise_for_status()
    teams = [
        (t["id"], t["name"])
        for t in resp.json()["teams"]
        if t.get("league", {}).get("id") in (103, 104)
    ]

    end_season = latest_completed_season()
    seasons = range(end_season - SEASONS_BACK + 1, end_season + 1)

    table: dict[int, dict] = {}
    for team_id, name in teams:
        ratios = []
        for year in seasons:
            ratio = _home_away_avg_ratio(session, team_id, year)
            if ratio is not None:
                ratios.append(ratio)
        if not ratios:
            continue
        factor = min(max(statistics.mean(ratios), MIN_FACTOR), MAX_FACTOR)
        table[team_id] = {
            "name": name,
            "park_factor": round(factor, 3),
            "seasons_used": len(ratios),
        }
    return table


def _home_away_avg_ratio(
    session: requests.Session, team_id: int, year: int
) -> float | None:
    try:
        home = session.get(
            f"{MLB_STATS_API_BASE}/teams/{team_id}/stats",
            params={"stats": "statSplits", "group": "hitting", "season": year, "sitCodes": "h"},
            timeout=15,
        ).json()["stats"][0]["splits"][0]["stat"]
        away = session.get(
            f"{MLB_STATS_API_BASE}/teams/{team_id}/stats",
            params={"stats": "statSplits", "group": "hitting", "season": year, "sitCodes": "a"},
            timeout=15,
        ).json()["stats"][0]["splits"][0]["stat"]
        return (home["hits"] / home["atBats"]) / (away["hits"] / away["atBats"])
    except (KeyError, IndexError, ZeroDivisionError):
        return None


def main() -> int:
    table = fetch_team_park_factors(requests.Session())
    OUTPUT_PATH.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(table)} team park factors to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
