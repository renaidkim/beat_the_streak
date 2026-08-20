"""Backtest the hit-probability model against a real season and refit it
from the result.

Requires requirements-analysis.txt (pandas/numpy/scikit-learn) -- these
happen to now also be *runtime* dependencies (see below), but this script
additionally needs pandas, which the app itself doesn't.

What this does:

1. Pulls one MLB season's worth of batter and (opposing starting pitcher)
   game logs from the Stats API, plus each game's boxscore (for batting
   order) and each batter's year-by-year hitting stats (for a pre-season
   career average).
2. Reconstructs, for every game a sampled batter played, exactly what the
   model could have known *before* that game: the batter's platoon split
   to date (as a delta from season average -- see below), the opposing
   starter's own season-to-date average-against, this park's factor,
   home/away, the batter's lineup slot that game (1-9, from the confirmed
   pre-game batting order), and career average entering the season (all
   prior seasons, so it's fixed for the whole season and can't leak
   in-season results). No lookahead.
3. Fits logistic regression and gradient-boosted-tree candidates on a
   date-based holdout (train on the first 70% of the season, evaluate on
   the rest) and picks whichever generalizes best on held-out log loss.
4. Refits the winner on the full season and writes it to
   data/hit_probability_model.json (metadata + feature order) and, for a
   gradient-boosting winner, data/hit_probability_model.pkl (the
   joblib-serialized model itself -- beat_the_streak.rank loads both).

How this feature set was arrived at (see git history / README for the
full account): season average and last-10-games form were both tested and
turned out to add *no* independent predictive power once career average
and batting order are in the model -- a batter's multi-year track record
turned out to be a much more reliable signal than anything from the
current season alone, mildly counter-intuitive but consistent with the
fact that a single game's outcome is dominated by variance no amount of
recent-form data can resolve. They were dropped rather than kept for
appearances.

Platoon as a delta, not an absolute: naively including the batter's
average specifically vs. today's pitcher hand as its own feature makes it
almost collinear with season average, because early in a season (or for a
part-time platoon batter) there's no real sample vs. one hand yet and it
falls back to the season average -- by construction equal to season
average in many rows. Using (platoon average - season average) instead
isolates the actual incremental platoon signal.

Batting order and career average, and their fallbacks: batting order is
only known for the ~89% of historical rows where the player started in
the confirmed pre-game lineup (not a mid-game substitution) -- the rest
get imputed to 5 (middle of the order, roughly neutral) rather than
dropped, matching how the live source has to handle it too (order is only
known for *today's* confirmed lineups; future days and unconfirmed
lineups have no order information at all). Career average falls back to
season-to-date for a player with no prior-season at-bats (rookies,
essentially).

Rerun this once or twice a season, same cadence as refresh_park_factors.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE = "https://statsapi.mlb.com/api/v1"
SEASON = 2025
N_BATTERS = 120
CACHE_DIR = REPO_ROOT / ".cache" / "backtest"
PARK_FACTORS_PATH = REPO_ROOT / "data" / "park_factors.json"
MODEL_JSON_PATH = REPO_ROOT / "data" / "hit_probability_model.json"
MODEL_PKL_PATH = REPO_ROOT / "data" / "hit_probability_model.pkl"

DEFAULT_BATTING_ORDER = 5.0  # imputed when the slot isn't known

FEATURE_COLS = [
    "career_avg",
    "platoon_delta",
    "pitcher_oba_against",
    "park_factor",
    "is_home",
    "batting_order",
]

# +1 = constrained non-decreasing, -1 = non-increasing, 0 = unconstrained,
# in FEATURE_COLS order. A flexible tree model can otherwise carve out
# regions where, say, a *tougher* opposing pitcher predicts a *higher*
# hit probability -- not from a real interaction, just sparse data in
# that corner of feature space. Every one of these directions is both
# basic baseball logic and what the plain logistic regression / bootstrap
# already confirmed; is_home is left unconstrained since neither showed a
# reliable direction. Costs almost nothing on holdout metrics (tested:
# ~0.001 worse log loss than the unconstrained model) in exchange for
# ruling out backwards predictions entirely.
MONOTONIC_CST = [1, 1, 1, 1, 0, -1]

# Recorded from the backtest run that motivated this rewrite (see README's
# Model backtest section for the full history) -- not recomputed here,
# since the "as shipped" heuristic that produced these numbers used a
# data shape (RecentForm, season_avg-driven scoring) this script's target
# codebase no longer has.
PRIOR_MODEL_HOLDOUT_METRICS = {
    "logreg_6_original_features (2nd iteration)": {"log_loss": 0.6502, "brier": 0.2289, "auc": 0.5348},
    "naive_baseline": {"log_loss": 0.6520, "brier": 0.2297, "auc": 0.5000},
}

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
    final_games = [
        g
        for d in schedule["dates"]
        for g in d["games"]
        if g["status"]["detailedState"] == "Final"
    ]

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

    print("Fetching batter year-by-year hitting stats (for pre-season career avg)...")
    batter_year_by_year: dict[int, list[dict]] = {}
    for i in range(0, len(batter_ids), 50):
        chunk = batter_ids[i : i + 50]
        resp = cached_get(
            f"{BASE}/people",
            params={
                "personIds": ",".join(str(b) for b in chunk),
                "hydrate": "stats(group=[hitting],type=[yearByYear])",
            },
        )
        for person in resp.get("people", []):
            for stat_group in person.get("stats", []):
                if stat_group.get("type", {}).get("displayName") == "yearByYear":
                    batter_year_by_year[person["id"]] = stat_group.get("splits", [])

    print("Determining opposing starters needed...")
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

    print(f"Fetching boxscores for batting order ({len(final_games)} games)...")
    batting_order: dict[tuple[str, int, int], int] = {}  # (date, team_id, player_id) -> slot
    for i, g in enumerate(final_games):
        box = cached_get(f"{BASE}/game/{g['gamePk']}/boxscore", params={})
        date = g["officialDate"]
        for side in ("home", "away"):
            team_id = g["teams"][side]["team"]["id"]
            for pdata in box.get("teams", {}).get(side, {}).get("players", {}).values():
                order = pdata.get("battingOrder")
                # battingOrder comes back as a numeric string (e.g. "700").
                # Only the pre-game lineup card (slot*100), not in-game
                # substitutions (slot*100 + N) -- substitutions weren't
                # knowable before the game, so they'd be lookahead.
                if order is not None:
                    order = int(order)
                    if order % 100 == 0:
                        batting_order[(date, team_id, pdata["person"]["id"])] = order // 100
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(final_games)}")

    return {
        "batter_ids": batter_ids,
        "batter_logs": batter_logs,
        "batter_year_by_year": batter_year_by_year,
        "pitcher_logs": pitcher_logs,
        "pitcher_hand": pitcher_hand,
        "schedule_map": schedule_map,
        "batting_order": batting_order,
    }


# ------------------------------------------------------------- dataset ---


def build_dataset(raw: dict) -> pd.DataFrame:
    park_factors_raw = json.loads(PARK_FACTORS_PATH.read_text())
    park_factors = {tid: e["park_factor"] for tid, e in park_factors_raw.items()}

    pitcher_logs = raw["pitcher_logs"]
    pitcher_hand = raw["pitcher_hand"]
    schedule_map = raw["schedule_map"]
    batting_order = raw["batting_order"]

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

    def career_avg_entering_season(bid: int) -> float | None:
        ab = hits = 0
        for split in raw["batter_year_by_year"].get(bid, []):
            if int(split.get("season", SEASON)) < SEASON:
                ab += int(split["stat"].get("atBats", 0))
                hits += int(split["stat"].get("hits", 0))
        return hits / ab if ab > 0 else None

    out_rows = []
    for bid in raw["batter_ids"]:
        career_avg = career_avg_entering_season(bid)
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

        for idx, r in enumerate(rows):
            pid = schedule_map.get((r["date"], r["team_id"]))
            hand = prior_hand_for_game[idx]

            if cum_ab > 0 and pid is not None:
                pitcher_oba = pitcher_oba_to_date(pid, r["date"])
                if pitcher_oba is not None:
                    season_avg_to_date = cum_hits / cum_ab  # platoon reference point only

                    platoon_ab = vs_hand_ab.get(hand, 0)
                    platoon_avg = (
                        vs_hand_hits[hand] / platoon_ab if hand and platoon_ab > 0 else season_avg_to_date
                    )

                    park_team = r["team_id"] if r["is_home"] else r["opponent_team_id"]
                    park_factor = park_factors.get(str(park_team), 1.0)

                    slot = batting_order.get((r["date"], r["team_id"], bid), DEFAULT_BATTING_ORDER)
                    career_avg_row = career_avg if career_avg is not None else season_avg_to_date

                    out_rows.append(
                        {
                            "date": r["date"],
                            "career_avg": career_avg_row,
                            "platoon_delta": platoon_avg - season_avg_to_date,
                            "pitcher_oba_against": pitcher_oba,
                            "park_factor": park_factor,
                            "is_home": int(r["is_home"]),
                            "batting_order": float(slot),
                            "got_hit": int(r["hits"] > 0),
                        }
                    )

            cum_ab += r["at_bats"]
            cum_hits += r["hits"]
            if hand:
                vs_hand_ab[hand] += r["at_bats"]
                vs_hand_hits[hand] += r["hits"]

    df = pd.DataFrame(out_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------ evaluate ---


def report(name: str, y_true, y_pred) -> dict:
    metrics = {
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
    }
    print(
        f"  {name:<38} log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier']:.4f}  auc={metrics['auc']:.4f}"
    )
    return metrics


def bootstrap_stability(df: pd.DataFrame, n: int = 30) -> None:
    print(f"\nBootstrap (n={n}) logistic-regression coefficient stability, full dataset:")
    rng = np.random.default_rng(0)
    boot_coefs = []
    for _ in range(n):
        idx = rng.integers(0, len(df), len(df))
        m = LogisticRegression(max_iter=1000)
        m.fit(df.iloc[idx][FEATURE_COLS], df.iloc[idx]["got_hit"])
        boot_coefs.append(m.coef_[0])
    boot_coefs = np.array(boot_coefs)
    for i, name in enumerate(FEATURE_COLS):
        stable = bool(np.all(boot_coefs[:, i] > 0) or np.all(boot_coefs[:, i] < 0))
        print(
            f"  {name:<24} mean={boot_coefs[:, i].mean():+.4f}  "
            f"std={boot_coefs[:, i].std():.4f}  sign_stable={stable}"
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help=(
            "Write to <prefix>.json/.pkl instead of the production "
            "data/hit_probability_model.{json,pkl} -- use this while "
            "iterating so a half-finished experiment doesn't become the "
            "live model."
        ),
    )
    args = parser.parse_args()
    json_path = Path(f"{args.out_prefix}.json") if args.out_prefix else MODEL_JSON_PATH
    pkl_path = Path(f"{args.out_prefix}.pkl") if args.out_prefix else MODEL_PKL_PATH

    t0 = time.time()
    raw = fetch_raw_data()
    df = build_dataset(raw)
    print(f"\nDataset: {len(df)} batter-games, {df['date'].min().date()}..{df['date'].max().date()}")
    print(f"Overall hit rate: {df['got_hit'].mean():.3f}")
    n_known_order = (df["batting_order"] != DEFAULT_BATTING_ORDER).sum()
    print(f"Rows with a known (not imputed) batting order: {n_known_order} ({n_known_order / len(df):.1%})\n")

    split_date = df["date"].quantile(0.7)
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    print(f"Holdout evaluation (train={len(train)}, test={len(test)}, split={split_date.date()}):")
    print("  (for context, from the prior model iteration -- not recomputed here:)")
    for name, m in PRIOR_MODEL_HOLDOUT_METRICS.items():
        print(f"    {name:<38} log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  auc={m['auc']:.4f}")

    candidates: dict[str, dict] = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(train[FEATURE_COLS], train["got_hit"])
    m = report("logistic regression", test["got_hit"], lr.predict_proba(test[FEATURE_COLS])[:, 1])
    candidates["logreg"] = {"metrics": m, "type": "logreg"}

    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0, monotonic_cst=MONOTONIC_CST)
    gbm.fit(train[FEATURE_COLS], train["got_hit"])
    m = report("gradient boosting", test["got_hit"], gbm.predict_proba(test[FEATURE_COLS])[:, 1])
    candidates["gbm"] = {"metrics": m, "type": "gbm"}

    bootstrap_stability(df)

    winner_name = min(candidates, key=lambda k: candidates[k]["metrics"]["log_loss"])
    print(f"\nWinner by holdout log loss: {winner_name} ({candidates[winner_name]['metrics']})")

    print("Refitting winner on full dataset for production...")
    if winner_name == "logreg":
        final = LogisticRegression(max_iter=1000)
        final.fit(df[FEATURE_COLS], df["got_hit"])
        model_type = "logreg"
        extra = {
            "coefficients": {name: float(c) for name, c in zip(FEATURE_COLS, final.coef_[0])},
            "intercept": float(final.intercept_[0]),
        }
    else:
        final = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, random_state=0, monotonic_cst=MONOTONIC_CST)
        final.fit(df[FEATURE_COLS], df["got_hit"])
        model_type = "gbm"
        joblib.dump(final, pkl_path)
        extra = {"pkl_path": pkl_path.name}

    final_pred = final.predict_proba(test[FEATURE_COLS])[:, 1]
    output = {
        "model_type": model_type,
        "features": FEATURE_COLS,
        **extra,
        "metadata": {
            "season": SEASON,
            "n_rows": len(df),
            "n_batters": len(raw["batter_ids"]),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "selected_model": winner_name,
            "candidate_comparison": {k: v["metrics"] for k, v in candidates.items()},
            "prior_model_holdout_metrics": PRIOR_MODEL_HOLDOUT_METRICS,
            "holdout_metrics_at_fit_time": {
                "log_loss": float(log_loss(test["got_hit"], final_pred, labels=[0, 1])),
                "brier": float(brier_score_loss(test["got_hit"], final_pred)),
                "auc": float(roc_auc_score(test["got_hit"], final_pred)),
            },
        },
    }
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {json_path}" + (f" and {pkl_path}" if model_type == "gbm" else ""))
    print(f"Done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
