"""Do Statcast/Baseball Savant features (exit velocity, barrel rate,
expected stats, sprint speed, etc.) improve on the shipped 12-feature
model?

Same true-out-of-time methodology as everything else in this project:
train on 2023-2025, test purely on 2026. Statcast leaderboards are
season-level, so each row uses the *prior* completed season's profile
for both the batter and the opposing pitcher (see statcast.py's
docstring for why a single lag, not a multi-year blend). A batter or
pitcher missing from the prior season's leaderboard (rookie, or debuted
partway through last season) falls back to that leaderboard's own mean
for each column, computed from whatever data was actually fetched.

Tests candidates two ways: individually (avoids one candidate's signal
getting masked by a correlated partner -- xba/xwoba/xslg are all
measuring similar things) and all together (to see the real, redundancy-
aware permutation importance ranking once they're all in one model, the
same way the original 24-feature broad set was pruned to 12).

Run after train_ml_model.py has populated its cache (reuses build_dataset
and the cached MLB Stats API fetches; only the Statcast leaderboard
fetch itself is new network traffic, ~8 requests total).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from statcast import BATTER_SELECTIONS, PITCHER_SELECTIONS, fetch_leaderboard  # noqa: E402
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

# Renamed to avoid colliding with existing feature names (e.g. pitcher
# already has its own xwoba-flavored oba_against).
BATTER_COLS = {s: f"sc_bat_{s}" for s in BATTER_SELECTIONS}
PITCHER_COLS = {s: f"sc_pit_{s}" for s in PITCHER_SELECTIONS}

# Domain-asserted monotonic direction for each Statcast candidate, same
# convention as MONOTONIC_CST in train_ml_model.py: 1 = P(hit) shouldn't
# decrease as the feature increases, -1 = shouldn't increase, 0 = no
# asserted direction.
BATTER_CST = {
    "sc_bat_xba": 1, "sc_bat_xwoba": 1, "sc_bat_xslg": 1,
    "sc_bat_exit_velocity_avg": 1, "sc_bat_barrel_batted_rate": 1,
    "sc_bat_hard_hit_percent": 1, "sc_bat_sprint_speed": 1,
    "sc_bat_whiff_percent": -1,
}
PITCHER_CST = {
    "sc_pit_xwoba": 1, "sc_pit_xba": 1, "sc_pit_exit_velocity_avg": 1,
    "sc_pit_barrel_batted_rate": 1, "sc_pit_hard_hit_percent": 1,
    "sc_pit_whiff_percent": -1, "sc_pit_fastball_avg_speed": -1,
    "sc_pit_k_percent": -1, "sc_pit_bb_percent": 0,
}
ALL_CANDIDATE_CST = {**BATTER_CST, **PITCHER_CST}
CANDIDATES = list(ALL_CANDIDATE_CST.keys())


def _attach_statcast(df, lb_years: set[int]) -> None:
    batter_lb = {y: fetch_leaderboard(y, "batter", BATTER_SELECTIONS) for y in lb_years}
    pitcher_lb = {y: fetch_leaderboard(y, "pitcher", PITCHER_SELECTIONS) for y in lb_years}

    # Fallback = mean of whatever was actually fetched for that column,
    # pooled across all lag years -- a reasonable "league average" when a
    # player is missing from the prior season's leaderboard.
    batter_fallback = {
        sel: float(np.mean([v[sel] for lb in batter_lb.values() for v in lb.values() if v[sel] is not None]))
        for sel in BATTER_SELECTIONS
    }
    pitcher_fallback = {
        sel: float(np.mean([v[sel] for lb in pitcher_lb.values() for v in lb.values() if v[sel] is not None]))
        for sel in PITCHER_SELECTIONS
    }

    for sel, col in BATTER_COLS.items():
        df[col] = [
            (batter_lb.get(season - 1, {}).get(int(bid), {}) or {}).get(sel) or batter_fallback[sel]
            for season, bid in zip(df["season"], df["batter_id"])
        ]
    for sel, col in PITCHER_COLS.items():
        df[col] = [
            (pitcher_lb.get(season - 1, {}).get(int(pid), {}) or {}).get(sel) or pitcher_fallback[sel]
            for season, pid in zip(df["season"], df["pitcher_id"])
        ]


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


def fit_gbm(train_df, test_df, cols, cst, y_train):
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    return model


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_train, y_test = train_df["got_hit"], test_df["got_hit"]

    lag_years = {s - 1 for s in TRAIN_SEASONS} | {TEST_SEASON - 1}
    print(f"\nFetching Statcast leaderboards for lag years {sorted(lag_years)}...")
    _attach_statcast(train_df, lag_years)
    _attach_statcast(test_df, lag_years)

    print("\n--- Baseline: shipped model's 12 features, refit here for a fair comparison ---")
    baseline = fit_gbm(train_df, test_df, FEATURE_COLS, MONOTONIC_CST, y_train)
    report("baseline (12 features)", y_test, baseline.predict_proba(test_df[FEATURE_COLS])[:, 1])

    print("\n--- Each Statcast candidate added one at a time ---")
    for candidate in CANDIDATES:
        cols = FEATURE_COLS + [candidate]
        cst = MONOTONIC_CST + [ALL_CANDIDATE_CST[candidate]]
        model = fit_gbm(train_df, test_df, cols, cst, y_train)
        pred = model.predict_proba(test_df[cols])[:, 1]
        report(f"+ {candidate}", y_test, pred)
        perm = permutation_importance(
            model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
        )
        idx = cols.index(candidate)
        print(
            f"      permutation importance: {perm.importances_mean[idx]:+.5f} "
            f"(+/- {perm.importances_std[idx]:.5f})"
        )

    print("\n--- All Statcast candidates together ---")
    cols = FEATURE_COLS + CANDIDATES
    cst = MONOTONIC_CST + [ALL_CANDIDATE_CST[c] for c in CANDIDATES]
    combined = fit_gbm(train_df, test_df, cols, cst, y_train)
    report("baseline + all Statcast candidates", y_test, combined.predict_proba(test_df[cols])[:, 1])
    perm = permutation_importance(
        combined, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    order = np.argsort(perm.importances_mean)[::-1]
    print("  Permutation importance, all features in the combined model:")
    for i in order:
        print(f"    {cols[i]:<28} {perm.importances_mean[i]:+.5f}  (+/- {perm.importances_std[i]:.5f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
