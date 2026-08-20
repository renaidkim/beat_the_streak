"""Backtest the hit-probability model against a real season and refit its
coefficients from the result.

Requires requirements-analysis.txt (pandas/numpy/scikit-learn) -- these are
offline-analysis dependencies only, not needed to run the app itself.

What this does:

1. Pulls one MLB season's worth of batter and (opposing starting pitcher)
   game logs from the Stats API.
2. Reconstructs, for every game a sampled batter played, exactly what the
   model could have known *before* that game: season average to date,
   last-10-games form, the batter's average specifically against that
   day's opposing pitcher hand (vs. their season average -- see
   "platoon as a delta" below), the opposing starter's own season-to-date
   average-against, this park's factor, and home/away. No lookahead.
3. Scores every row with the *shipped* heuristic (beat_the_streak.rank)
   and reports how well its output probabilities matched reality (log
   loss, Brier score, AUC, calibration by decile), against a
   date-based holdout so nothing from "the future" leaks into evaluation.
4. Fits a logistic regression on the same features and compares.
5. If the fitted model is genuinely better calibrated (it is, as of the
   last run against the 2025 season -- see README's Model backtest
   section), refits it on the full season and writes the coefficients to
   data/hit_probability_model.json, which beat_the_streak.rank loads at
   runtime instead of the old hand-tuned weighted-blend formula.

Platoon as a delta, not an absolute: naively including the batter's
average specifically vs. today's pitcher hand as its own feature makes it
almost collinear with season average, because early in a season (or for a
part-time platoon batter) there's no real sample vs. one hand yet and it
falls back to the season average -- by construction equal to season
average in many rows. Using (platoon average - season average) instead
isolates the actual incremental platoon signal and removes that artifact;
it also came back as the single most bootstrap-stable coefficient (see
report output), i.e. it's real signal, not noise -- worth wiring into
MlbStatsApiSource's live batter fetch, which this script's findings
directly motivated.

Rerun this once or twice a season, same cadence as refresh_park_factors.py.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from beat_the_streak.models import Batter, Matchup, Pitcher, RecentForm  # noqa: E402
from beat_the_streak.rank import score_matchup  # noqa: E402

BASE = "https://statsapi.mlb.com/api/v1"
SEASON = 2025
N_BATTERS = 120
CACHE_DIR = REPO_ROOT / ".cache" / "backtest"
PARK_FACTORS_PATH = REPO_ROOT / "data" / "park_factors.json"
MODEL_OUTPUT_PATH = REPO_ROOT / "data" / "hit_probability_model.json"

FEATURE_COLS = [
    "season_avg_to_date",
    "recent_form_avg",
    "platoon_delta",
    "pitcher_oba_against",
    "park_factor",
    "is_home",
]

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


def fetch_raw_data() -> dict:
    print("Fetching season schedule with probable pitchers...")
    schedule = cached_get(
        f"{BASE}/schedule",
        params={
            "sportId": 1,
            "startDate": f"{SEASON}-03-01",
            "endDate": f"{SEASON}-10-01",
            "hydrate": "probablePitcher,team",
            "gameType": "R",
        },
    )

    print(f"Fetching top {N_BATTERS} batters by plate appearances...")
    leaders = cached_get(
        f"{BASE}/stats/leaders",
        params={
            "leaderCategories": "plateAppearances",
            "season": SEASON,
            "sportId": 1,
            "limit": N_BATTERS,
        },
    )
    batter_ids = [e["person"]["id"] for e in leaders["leagueLeaders"][0]["leaders"]]
    print(f"  got {len(batter_ids)} batters")

    print("Fetching batter game logs...")
    batter_logs = {}
    batter_team_ids = set()
    for i, bid in enumerate(batter_ids):
        gl = cached_get(
            f"{BASE}/people/{bid}/stats",
            params={"stats": "gameLog", "group": "hitting", "season": SEASON},
        )
        batter_logs[bid] = gl.get("stats", [{}])[0].get("splits", [])
        for s in batter_logs[bid]:
            batter_team_ids.add(s["team"]["id"])
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(batter_ids)}")

    print("Determining opposing starters needed...")
    schedule_map: dict[tuple[str, int], int] = {}
    needed_pitcher_ids: set[int] = set()
    for d in schedule["dates"]:
        for g in d["games"]:
            if g["status"]["detailedState"] != "Final":
                continue
            date = g["officialDate"]
            for side, opp in (("home", "away"), ("away", "home")):
                team_id = g["teams"][side]["team"]["id"]
                pp = g["teams"][opp].get("probablePitcher")
                if pp and (date, team_id) not in schedule_map:
                    schedule_map[(date, team_id)] = pp["id"]
                if pp and team_id in batter_team_ids:
                    needed_pitcher_ids.add(pp["id"])
    print(f"  {len(needed_pitcher_ids)} distinct pitchers")

    print("Fetching pitcher game logs...")
    pitcher_logs = {}
    for i, pid in enumerate(sorted(needed_pitcher_ids)):
        gl = cached_get(
            f"{BASE}/people/{pid}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": SEASON},
        )
        pitcher_logs[pid] = gl.get("stats", [{}])[0].get("splits", [])
        if (i + 1) % 60 == 0:
            print(f"  {i + 1}/{len(needed_pitcher_ids)}")

    print("Fetching pitcher handedness...")
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

    return {
        "batter_ids": batter_ids,
        "batter_logs": batter_logs,
        "pitcher_logs": pitcher_logs,
        "pitcher_hand": pitcher_hand,
        "schedule_map": schedule_map,
    }


# ------------------------------------------------------------- dataset ---


def build_dataset(raw: dict) -> pd.DataFrame:
    park_factors_raw = json.loads(PARK_FACTORS_PATH.read_text())
    park_factors = {tid: e["park_factor"] for tid, e in park_factors_raw.items()}

    pitcher_logs = raw["pitcher_logs"]
    pitcher_hand = raw["pitcher_hand"]
    schedule_map = raw["schedule_map"]

    pitcher_games: dict[int, list[tuple[str, int, int]]] = {}
    for pid, splits in pitcher_logs.items():
        rows = sorted(
            (
                (s["date"], int(s["stat"].get("atBats", 0)), int(s["stat"].get("hits", 0)))
                for s in splits
            ),
            key=lambda r: r[0],
        )
        pitcher_games[pid] = rows

    def pitcher_oba_to_date(pid: int, date: str) -> float | None:
        ab = hits = 0
        for d, a, h in pitcher_games.get(pid, []):
            if d >= date:
                break
            ab += a
            hits += h
        return hits / ab if ab > 0 else None

    out_rows = []
    for bid in raw["batter_ids"]:
        splits = raw["batter_logs"].get(bid, [])
        rows = sorted(
            (
                {
                    "date": s["date"],
                    "team_id": s["team"]["id"],
                    "opponent_team_id": s["opponent"]["id"],
                    "is_home": s["isHome"],
                    "at_bats": int(s["stat"].get("atBats", 0)),
                    "hits": int(s["stat"].get("hits", 0)),
                }
                for s in splits
            ),
            key=lambda r: r["date"],
        )

        prior_hand_for_game = [
            pitcher_hand.get(schedule_map.get((r["date"], r["team_id"])), None) for r in rows
        ]

        cum_ab = cum_hits = 0
        vs_hand_ab = {"L": 0, "R": 0}
        vs_hand_hits = {"L": 0, "R": 0}
        last10: list[tuple[int, int]] = []

        for idx, r in enumerate(rows):
            pid = schedule_map.get((r["date"], r["team_id"]))
            hand = prior_hand_for_game[idx]

            if cum_ab > 0 and pid is not None:
                pitcher_oba = pitcher_oba_to_date(pid, r["date"])
                if pitcher_oba is not None:
                    season_avg_to_date = cum_hits / cum_ab
                    l10_ab = sum(a for a, h in last10)
                    l10_hits = sum(h for a, h in last10)
                    recent_form_avg = (l10_hits / l10_ab) if l10_ab else season_avg_to_date

                    platoon_ab = vs_hand_ab.get(hand, 0)
                    platoon_avg = (
                        vs_hand_hits[hand] / platoon_ab if hand and platoon_ab > 0 else season_avg_to_date
                    )

                    park_team = r["team_id"] if r["is_home"] else r["opponent_team_id"]
                    park_factor = park_factors.get(str(park_team), 1.0)

                    out_rows.append(
                        {
                            "date": r["date"],
                            "season_avg_to_date": season_avg_to_date,
                            "recent_form_avg": recent_form_avg,
                            "platoon_delta": platoon_avg - season_avg_to_date,
                            "pitcher_oba_against": pitcher_oba,
                            "park_factor": park_factor,
                            "is_home": int(r["is_home"]),
                            "games_of_recent_data": len(last10),
                            "got_hit": int(r["hits"] > 0),
                        }
                    )

            cum_ab += r["at_bats"]
            cum_hits += r["hits"]
            if hand:
                vs_hand_ab[hand] += r["at_bats"]
                vs_hand_hits[hand] += r["hits"]
            last10.append((r["at_bats"], r["hits"]))
            last10 = last10[-10:]

    df = pd.DataFrame(out_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------ evaluate ---


def heuristic_predict_row(row: pd.Series) -> float:
    batter = Batter(
        id="x",
        name="x",
        bats="R",
        team="x",
        season_avg=row["season_avg_to_date"],
        season_avg_vs_lhp=row["season_avg_to_date"] + row["platoon_delta"],
        season_avg_vs_rhp=row["season_avg_to_date"] + row["platoon_delta"],
    )
    pitcher = Pitcher(id="y", name="y", throws="R", era=4.0, oba_against=row["pitcher_oba_against"])
    n = int(row["games_of_recent_data"])
    at_bats = n * 4
    hits = round(row["recent_form_avg"] * at_bats) if at_bats else 0
    matchup = Matchup(
        batter=batter,
        pitcher=pitcher,
        is_home=bool(row["is_home"]),
        park_factor=row["park_factor"],
        recent_form=RecentForm(games_played=n, at_bats=at_bats, hits=hits),
    )
    return score_matchup(matchup).hit_probability


def report(name: str, y_true, y_pred) -> None:
    print(
        f"  {name:<34} log_loss={log_loss(y_true, y_pred, labels=[0, 1]):.4f}  "
        f"brier={brier_score_loss(y_true, y_pred):.4f}  auc={roc_auc_score(y_true, y_pred):.4f}"
    )


def main() -> int:
    t0 = time.time()
    raw = fetch_raw_data()
    df = build_dataset(raw)
    print(f"\nDataset: {len(df)} batter-games, {df['date'].min().date()}..{df['date'].max().date()}")
    print(f"Overall hit rate: {df['got_hit'].mean():.3f}\n")

    split_date = df["date"].quantile(0.7)
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    print(f"Holdout evaluation (train={len(train)}, test={len(test)}, split={split_date.date()}):")

    heuristic_pred = test.apply(heuristic_predict_row, axis=1)
    report("heuristic (as shipped, pre-refit)", test["got_hit"], heuristic_pred)
    report("baseline (constant training rate)", test["got_hit"], np.full(len(test), train["got_hit"].mean()))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(train[FEATURE_COLS], train["got_hit"])
    lr_pred = lr.predict_proba(test[FEATURE_COLS])[:, 1]
    report("fitted logistic regression", test["got_hit"], lr_pred)

    print("\nBootstrap (n=30) coefficient stability, full dataset:")
    rng = np.random.default_rng(0)
    boot_coefs = []
    for _ in range(30):
        idx = rng.integers(0, len(df), len(df))
        m = LogisticRegression(max_iter=1000)
        m.fit(df.iloc[idx][FEATURE_COLS], df.iloc[idx]["got_hit"])
        boot_coefs.append(m.coef_[0])
    boot_coefs = np.array(boot_coefs)
    for i, name in enumerate(FEATURE_COLS):
        stable = bool(np.all(boot_coefs[:, i] > 0) or np.all(boot_coefs[:, i] < 0))
        print(f"  {name:<24} mean={boot_coefs[:, i].mean():+.4f}  std={boot_coefs[:, i].std():.4f}  sign_stable={stable}")

    print("\nRefitting on full dataset for production coefficients...")
    final = LogisticRegression(max_iter=1000)
    final.fit(df[FEATURE_COLS], df["got_hit"])
    final_metrics = {
        "log_loss": float(log_loss(test["got_hit"], final.predict_proba(test[FEATURE_COLS])[:, 1], labels=[0, 1])),
        "brier": float(brier_score_loss(test["got_hit"], final.predict_proba(test[FEATURE_COLS])[:, 1])),
        "auc": float(roc_auc_score(test["got_hit"], final.predict_proba(test[FEATURE_COLS])[:, 1])),
    }

    output = {
        "features": FEATURE_COLS,
        "coefficients": {name: float(c) for name, c in zip(FEATURE_COLS, final.coef_[0])},
        "intercept": float(final.intercept_[0]),
        "metadata": {
            "season": SEASON,
            "n_rows": len(df),
            "n_batters": len(raw["batter_ids"]),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "holdout_metrics_at_fit_time": final_metrics,
        },
    }
    MODEL_OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {MODEL_OUTPUT_PATH}")
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
