"""Does recent form actually add nothing, or was that just true for the
10-game window tested once in the old single-season logistic-regression
pipeline (see README's "Model backtest", round one)?

Re-tests the question properly: same multi-season train (2023-2025) /
true-2026-holdout methodology as the shipped model, four window
definitions (5 games, 10 games, 5 calendar days, 10 calendar days, each
expressed as a delta vs. season-to-date average -- see
train_ml_model.py's build_dataset for why deltas, not absolutes), each
added one at a time to the shipped model's 12 features with a monotonic
constraint (higher recent form shouldn't lower P(hit)) so a fair
candidate can't win by exploiting a backwards-prediction artifact the
same way random_forest was rejected for in the main pipeline.

Run after train_ml_model.py has populated the cache (reads through the
same cached fetch functions, no new network calls).
"""

from __future__ import annotations

import sys
from pathlib import Path

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

CANDIDATES = [
    "recent_form_5g_delta",
    "recent_form_10g_delta",
    "recent_form_5d_delta",
    "recent_form_10d_delta",
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


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_train, y_test = train_df["got_hit"], test_df["got_hit"]

    print("\n--- Baseline: shipped model's 12 features, refit here for a fair comparison ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(train_df[FEATURE_COLS], y_train)
    report("baseline (12 features)", y_test, baseline.predict_proba(test_df[FEATURE_COLS])[:, 1])

    print("\n--- Each recent-form candidate added one at a time ---")
    for candidate in CANDIDATES:
        cols = FEATURE_COLS + [candidate]
        cst = MONOTONIC_CST + [1]  # higher recent form shouldn't lower P(hit)
        model = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
            monotonic_cst=cst,
        )
        model.fit(train_df[cols], y_train)
        pred = model.predict_proba(test_df[cols])[:, 1]
        report(f"+ {candidate}", y_test, pred)

        perm = permutation_importance(
            model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
        )
        idx = cols.index(candidate)
        print(
            f"      permutation importance of {candidate}: "
            f"{perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
