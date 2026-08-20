"""Point-in-time version of the opponent-bullpen-quality test.

test_bullpen.py used the *prior completed season's* bullpen ERA (MLB
Stats API's byDateRange stat type silently ignores sitCodes, so a true
mid-season split isn't available that way -- verified directly). That
version showed a promising but not fully conclusive signal (91% of
bootstrap draws favored it, 95% CI [-0.00097, +0.00019] just barely
touched zero) -- big enough, and hobbled by staleness badly enough
(bullpens churn a lot more within a season than a hitter's underlying
skill does), that it's worth the extra work to get a cleaner answer.

This version reconstructs a true point-in-time bullpen ERA per team per
date from boxscores already being fetched for batting order (see
fit_ml_model.py's bullpen_appearances -- gamesStarted==0 in a boxscore
pitching line marks a relief appearance) -- no new network calls, just
extra parsing of data already cached on disk. For each row, sums that
team's relief outs/earned runs from every game strictly before that
row's date, within the same season -- true no-lookahead, no staleness.

Run after train_ml_model.py has populated the cache.
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

import datetime

from fit_ml_model import fetch_season_raw  # noqa: E402
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

LEAGUE_AVG_BULLPEN_ERA = 4.30


def build_bullpen_index(seasons: list[int]) -> dict[int, dict[int, tuple[list[str], list[int], list[int]]]]:
    """{season: {team_id: (sorted_dates, cumulative_outs, cumulative_er)}}"""
    index: dict[int, dict[int, tuple[list[str], list[int], list[int]]]] = {}
    for season in seasons:
        end_date = datetime.date.today().isoformat() if season == TEST_SEASON else None
        raw = fetch_season_raw(season, end_date=end_date)
        by_team: dict[int, list[tuple[str, int, int]]] = {}
        for team_id, date, outs, er in raw["bullpen_appearances"]:
            by_team.setdefault(team_id, []).append((date, outs, er))
        team_index = {}
        for team_id, appearances in by_team.items():
            appearances.sort(key=lambda a: a[0])
            dates = [a[0] for a in appearances]
            cum_outs, cum_er = [], []
            running_outs = running_er = 0
            for _, outs, er in appearances:
                running_outs += outs
                running_er += er
                cum_outs.append(running_outs)
                cum_er.append(running_er)
            team_index[team_id] = (dates, cum_outs, cum_er)
        index[season] = team_index
    return index


def bullpen_era_before(index, season: int, team_id: int, date: str) -> float:
    team_index = index.get(season, {})
    if team_id not in team_index:
        return LEAGUE_AVG_BULLPEN_ERA
    dates, cum_outs, cum_er = team_index[team_id]
    pos = bisect.bisect_left(dates, date)  # strictly before `date`
    if pos == 0:
        return LEAGUE_AVG_BULLPEN_ERA
    outs, er = cum_outs[pos - 1], cum_er[pos - 1]
    return er * 27 / outs if outs > 0 else LEAGUE_AVG_BULLPEN_ERA


def report(name: str, y_true, y_pred) -> dict:
    metrics = {
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
    }
    print(
        f"  {name:<32} log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier']:.4f}  auc={metrics['auc']:.4f}"
    )
    return metrics


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_train, y_test = train_df["got_hit"], test_df["got_hit"]

    all_seasons = list(TRAIN_SEASONS) + [TEST_SEASON]
    print(f"\nReconstructing point-in-time bullpen ERA from cached boxscores for {all_seasons}...")
    index = build_bullpen_index(all_seasons)

    for df in (train_df, test_df):
        df["opp_bullpen_era_pit"] = [
            bullpen_era_before(index, season, int(opp), date)
            for season, opp, date in zip(df["season"], df["opponent_team_id"], df["date"].dt.strftime("%Y-%m-%d"))
        ]

    print(f"\nSample opp_bullpen_era_pit values: {train_df['opp_bullpen_era_pit'].describe()}")

    print("\n--- Baseline: shipped model's 12 features, refit here for a fair comparison ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(train_df[FEATURE_COLS], y_train)
    pred_base = baseline.predict_proba(test_df[FEATURE_COLS])[:, 1]
    report("baseline (12 features)", y_test, pred_base)

    print("\n--- + opponent bullpen ERA (true point-in-time) ---")
    cols = FEATURE_COLS + ["opp_bullpen_era_pit"]
    cst = MONOTONIC_CST + [1]
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    pred_bp = model.predict_proba(test_df[cols])[:, 1]
    report("+ opp_bullpen_era_pit", y_test, pred_bp)
    perm = permutation_importance(
        model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    idx = cols.index("opp_bullpen_era_pit")
    print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")

    import numpy as np
    y = np.asarray(y_test)
    rng = np.random.default_rng(0)
    n = len(y)
    diffs = np.empty(2000)
    for i in range(2000):
        bidx = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y[bidx], pred_bp[bidx], labels=[0, 1])
            - log_loss(y[bidx], pred_base[bidx], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"\nBootstrap 95% CI on (log_loss[+bullpen] - log_loss[baseline]): "
        f"[{lo:+.5f}, {hi:+.5f}] (point estimate {diffs.mean():+.5f})"
    )
    print(f"Fraction of bootstrap draws where +bullpen is better: {(diffs < 0).mean():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
