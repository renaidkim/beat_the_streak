"""Build a hit-probability model from scratch with a broad feature set,
trained on multiple past seasons and validated on a season it never saw.

Requires requirements-analysis.txt (pandas/numpy/scikit-learn).

This is a separate, independent pipeline from fit_hit_probability_model.py
(which stays as the record of that earlier, smaller-feature-set attempt).
Differences from that one, all deliberate:

1. **Multi-season training, true out-of-time test.** That script trained
   and evaluated within a single season (2025), split by date. This one
   trains on 2023-2025 and evaluates purely on 2026 -- a season the model
   never saw a single row of during fitting or feature selection. That's
   a meaningfully stronger generalization test: a date-split within one
   season can still let the model implicitly pick up on that season's
   specific park/weather/league conditions.
2. **A broad, systematically-built feature set** (~20+ features) instead
   of 6 hand-picked ones: full career slash line (avg/obp/slg/iso/babip),
   career walk and strikeout rates, career power, a pitcher's career ERA/
   WHIP/K9/BB9/HR9 (not just season-to-date average-against), age,
   handedness for both sides, rest days, month of season, in addition to
   the platoon/park/order/home features the other script already found.
   Deliberately not pre-filtered by hand -- feature importance (see
   below) is what decides what mattered, not a guess in advance.
3. **Several real model families compared**, not just logistic regression
   vs. one gradient-boosting config: L2-regularized logistic regression,
   random forest, gradient boosting, and a small neural net (MLP) -- all
   via scikit-learn, no new runtime dependency.
4. **Model-derived explainability**, not hand-written threshold text.
   Permutation importance (sklearn.inspection) on the true holdout
   measures the actual drop in held-out performance when a feature is
   shuffled -- a more honest signal than impurity-based importance,
   which inflates high-cardinality/continuous features and gets
   confused by correlated ones.
5. **The currently-shipped model is scored on the same 2026 holdout too**,
   for a fair, apples-to-apples comparison -- it had previously only ever
   been validated with a within-season date split, never on a genuinely
   separate season.

Career-stat aggregation: yearByYear hitting/pitching splits give raw
counting stats per season (hits, at-bats, total bases, walks, strikeouts,
outs, earned runs, etc.) -- rates are recomputed from *summed* counts
across all prior seasons, not averaged from each season's own rate, so a
230-PA rookie season doesn't get weighted the same as a 650-PA everyday
season.

Rerun this periodically (e.g. once the season being used as the test set
completes, so it can become part of the next training window) --
schedule.py has SEASON constants at the top of main() for this reason.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = "https://statsapi.mlb.com/api/v1"

TRAIN_SEASONS = [2023, 2024, 2025]
TEST_SEASON = 2026
N_BATTERS_PER_SEASON = 100
CACHE_DIR = REPO_ROOT / ".cache" / "ml_model"
PARK_FACTORS_PATH = REPO_ROOT / "data" / "park_factors.json"

DEFAULT_BATTING_ORDER = 5.0

session = requests.Session()


def cached_get(url: str, params: dict) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    path.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------- fetch ---


def fetch_season_raw(season: int, end_date: str | None = None) -> dict:
    """Everything needed to build rows for one season: schedule, sampled
    batters' game logs, needed pitchers' game logs, boxscores for batting
    order. Does NOT fetch career (yearByYear) stats -- those are fetched
    once, globally, across the union of players from every season (see
    fetch_career_stats), since yearByYear already returns full history
    regardless of which season you ask from.
    """
    print(f"[{season}] Fetching schedule...")
    schedule = cached_get(
        f"{BASE}/schedule",
        params={
            "sportId": 1,
            "startDate": f"{season}-03-01",
            "endDate": end_date or f"{season}-10-01",
            "hydrate": "probablePitcher,team",
            "gameType": "R",
        },
    )
    final_games = [
        g for d in schedule["dates"] for g in d["games"] if g["status"]["detailedState"] == "Final"
    ]
    print(f"[{season}]   {len(final_games)} final games")

    print(f"[{season}] Fetching top {N_BATTERS_PER_SEASON} batters by PA...")
    leaders = cached_get(
        f"{BASE}/stats/leaders",
        params={
            "leaderCategories": "plateAppearances",
            "season": season,
            "sportId": 1,
            "limit": N_BATTERS_PER_SEASON,
        },
    )
    batter_ids = [e["person"]["id"] for e in leaders["leagueLeaders"][0]["leaders"]]

    print(f"[{season}] Fetching batter game logs...")
    batter_logs = {}
    batter_team_ids = set()
    for i, bid in enumerate(batter_ids):
        gl = cached_get(
            f"{BASE}/people/{bid}/stats",
            params={"stats": "gameLog", "group": "hitting", "season": season},
        )
        batter_logs[bid] = gl.get("stats", [{}])[0].get("splits", [])
        for s in batter_logs[bid]:
            batter_team_ids.add(s["team"]["id"])
        if (i + 1) % 50 == 0:
            print(f"[{season}]   {i + 1}/{len(batter_ids)}")

    print(f"[{season}] Determining opposing starters needed...")
    schedule_map: dict[tuple[str, int], int] = {}
    needed_pitcher_ids: set[int] = set()
    for g in final_games:
        date = g["officialDate"]
        for side, opp in (("home", "away"), ("away", "home")):
            team_id = g["teams"][side]["team"]["id"]
            pp = g["teams"][opp].get("probablePitcher")
            if pp and (date, team_id) not in schedule_map:
                schedule_map[(date, team_id)] = pp["id"]
            if pp and team_id in batter_team_ids:
                needed_pitcher_ids.add(pp["id"])
    print(f"[{season}]   {len(needed_pitcher_ids)} distinct pitchers")

    print(f"[{season}] Fetching pitcher game logs...")
    pitcher_logs = {}
    for i, pid in enumerate(sorted(needed_pitcher_ids)):
        gl = cached_get(
            f"{BASE}/people/{pid}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": season},
        )
        pitcher_logs[pid] = gl.get("stats", [{}])[0].get("splits", [])
        if (i + 1) % 80 == 0:
            print(f"[{season}]   {i + 1}/{len(needed_pitcher_ids)}")

    print(f"[{season}] Fetching pitcher handedness...")
    pitcher_hand = {}
    pids = sorted(needed_pitcher_ids)
    for i in range(0, len(pids), 50):
        chunk = pids[i : i + 50]
        resp = cached_get(
            f"{BASE}/people",
            params={"personIds": ",".join(str(p) for p in chunk), "hydrate": "pitchHand"},
        )
        for person in resp.get("people", []):
            pitcher_hand[person["id"]] = person.get("pitchHand", {}).get("code", "R")

    print(f"[{season}] Fetching boxscores for batting order ({len(final_games)} games)...")
    batting_order: dict[tuple[str, int, int], int] = {}
    for i, g in enumerate(final_games):
        box = cached_get(f"{BASE}/game/{g['gamePk']}/boxscore", params={})
        date = g["officialDate"]
        for side in ("home", "away"):
            team_id = g["teams"][side]["team"]["id"]
            for pdata in box.get("teams", {}).get(side, {}).get("players", {}).values():
                order = pdata.get("battingOrder")
                if order is not None:
                    order = int(order)
                    if order % 100 == 0:
                        batting_order[(date, team_id, pdata["person"]["id"])] = order // 100
        if (i + 1) % 400 == 0:
            print(f"[{season}]   {i + 1}/{len(final_games)}")

    return {
        "batter_ids": batter_ids,
        "batter_logs": batter_logs,
        "pitcher_logs": pitcher_logs,
        "pitcher_hand": pitcher_hand,
        "schedule_map": schedule_map,
        "batting_order": batting_order,
    }


def fetch_career_stats(person_ids: set[int], group: str) -> dict[int, list[dict]]:
    """yearByYear stats (all seasons, full career) for a set of person ids,
    batched. group is "hitting" or "pitching".
    """
    result: dict[int, list[dict]] = {}
    ids = sorted(person_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        resp = cached_get(
            f"{BASE}/people",
            params={
                "personIds": ",".join(str(p) for p in chunk),
                "hydrate": f"stats(group=[{group}],type=[yearByYear])",
            },
        )
        for person in resp.get("people", []):
            for stat_group in person.get("stats", []):
                if stat_group.get("type", {}).get("displayName") == "yearByYear":
                    result[person["id"]] = stat_group.get("splits", [])
    return result


def main() -> int:
    import datetime

    t0 = time.time()
    raw_by_season = {}
    for season in TRAIN_SEASONS:
        raw_by_season[season] = fetch_season_raw(season)

    test_end = datetime.date.today().isoformat()
    raw_by_season[TEST_SEASON] = fetch_season_raw(TEST_SEASON, end_date=test_end)

    all_batter_ids = set()
    all_pitcher_ids = set()
    for raw in raw_by_season.values():
        all_batter_ids.update(raw["batter_ids"])
        all_pitcher_ids.update(raw["pitcher_logs"].keys())

    print(f"\nFetching career hitting stats for {len(all_batter_ids)} batters...")
    batter_career = fetch_career_stats(all_batter_ids, "hitting")
    print(f"Fetching career pitching stats for {len(all_pitcher_ids)} pitchers...")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")

    combined = {
        "raw_by_season": raw_by_season,
        "batter_career": batter_career,
        "pitcher_career": pitcher_career,
    }
    out_path = CACHE_DIR / "_combined_raw.json"
    # Keys must be strings for JSON; season ints and tuple keys need care.
    # Simpler: pickle instead of JSON for this one aggregate blob.
    import pickle

    with open(CACHE_DIR / "_combined_raw.pkl", "wb") as f:
        pickle.dump(combined, f)
    print(f"\nWrote combined raw data to {CACHE_DIR / '_combined_raw.pkl'}")
    print(f"Total fetch time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
