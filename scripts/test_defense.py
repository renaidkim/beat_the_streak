"""Does opposing team defensive quality help predict hit probability?

Team-level Outs Above Average (OAA) from Baseball Savant -- a modern,
range-based defensive metric (unlike fielding percentage, which only
penalizes errors on plays actually attempted and says nothing about
plays never reached). Higher OAA = better defense = should lower the
batter's odds of a ball in play going for a hit.

Important caveat, flagged *before* running this, given what just
happened with the bullpen-quality test: OAA is a season-aggregate stat
with no point-in-time reconstruction available the way bullpen ERA was
(it needs play-by-play difficulty-adjusted tracking data, not just
box-score putouts/errors) -- so this can only test the same kind of
prior-completed-season lag proxy that looked promising for bullpen
quality and then reversed under a proper point-in-time test. A team's
defensive personnel is generally more stable within a season than
bullpen usage patterns are, but the same "this might just be proxying
for overall team quality" risk applies, maybe even more so. Held to a
higher bar than usual as a result: a bootstrap CI that still touches
zero should be read as "no," not "promising," given that prior.

Run after train_ml_model.py has populated the cache.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

CACHE_DIR = REPO_ROOT.parent / ".cache" / "defense"
session = requests.Session()


def fetch_team_oaa(year: int) -> dict[int, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{year}.csv"
    if path.exists():
        text = path.read_text(encoding="utf-8-sig")
    else:
        resp = session.get(
            "https://baseballsavant.mlb.com/leaderboard/outs_above_average",
            params={
                "type": "Fielding_Team", "startYear": year, "endYear": year,
                "split": "no", "team": "", "range": "year", "min": 1,
                "pos": "", "roles": "", "viz": "hide", "csv": "true",
            },
            timeout=30,
        )
        resp.raise_for_status()
        if resp.text.count("\n") < 20:
            raise RuntimeError(f"OAA leaderboard for {year} looks truncated: {resp.text[:300]!r}")
        path.write_text(resp.text, encoding="utf-8")
        text = resp.text
    reader = csv.DictReader(io.StringIO(text))
    return {int(row["team_id"]): float(row["outs_above_average"]) for row in reader}


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

    lag_years = {s - 1 for s in TRAIN_SEASONS} | {TEST_SEASON - 1}
    print(f"\nFetching team OAA (prior season) for lag years {sorted(lag_years)}...")
    oaa = {y: fetch_team_oaa(y) for y in lag_years}
    league_avg = {y: sum(v.values()) / len(v) for y, v in oaa.items()}

    for df in (train_df, test_df):
        df["opp_oaa_prior_season"] = [
            oaa.get(season - 1, {}).get(int(opp), league_avg.get(season - 1, 0.0))
            for season, opp in zip(df["season"], df["opponent_team_id"])
        ]

    print("\n--- Baseline: shipped model's 12 features, refit here for a fair comparison ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(train_df[FEATURE_COLS], y_train)
    pred_base = baseline.predict_proba(test_df[FEATURE_COLS])[:, 1]
    report("baseline (12 features)", y_test, pred_base)

    print("\n--- + opponent defense (OAA, prior season) ---")
    cols = FEATURE_COLS + ["opp_oaa_prior_season"]
    cst = MONOTONIC_CST + [-1]  # higher opponent OAA (better defense) shouldn't raise P(hit)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    pred_def = model.predict_proba(test_df[cols])[:, 1]
    report("+ opp_oaa_prior_season", y_test, pred_def)
    perm = permutation_importance(
        model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    idx = cols.index("opp_oaa_prior_season")
    print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")

    y = np.asarray(y_test)
    rng = np.random.default_rng(0)
    n = len(y)
    diffs = np.empty(2000)
    for i in range(2000):
        idx2 = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y[idx2], pred_def[idx2], labels=[0, 1])
            - log_loss(y[idx2], pred_base[idx2], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"\nBootstrap 95% CI on (log_loss[+defense] - log_loss[baseline]): "
        f"[{lo:+.5f}, {hi:+.5f}] (point estimate {diffs.mean():+.5f})"
    )
    print(f"Fraction of bootstrap draws where +defense is better: {(diffs < 0).mean():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
