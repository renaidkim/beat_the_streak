"""Does post-hoc probability calibration improve the shipped model?

Calibration (isotonic regression or Platt/sigmoid scaling) is a
monotonic transform of the model's raw output, fit on held-out folds of
the training data -- it can't change which matchups rank highest (AUC
is invariant to any strictly monotonic transform), only how well the
predicted numbers themselves match reality (log loss, Brier). Cheap to
test: no new features, no new data, just a different way to turn the
same model's raw scores into probabilities.

Uses sklearn's CalibratedClassifierCV with cv=5 so the calibration
mapping is fit on out-of-fold predictions during training, never on the
2026 test set itself -- avoids any leakage into the number being
reported.

Run after train_ml_model.py has populated the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
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


def bootstrap_ci(y_test, pred_a, pred_b, label: str, n_boot: int = 2000, seed: int = 0) -> None:
    y = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y[idx], pred_a[idx], labels=[0, 1])
            - log_loss(y[idx], pred_b[idx], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"  [{label}] bootstrap 95% CI on log_loss diff: [{lo:+.5f}, {hi:+.5f}] "
        f"(point estimate {diffs.mean():+.5f}); fraction favoring A: {(diffs < 0).mean():.3f}"
    )


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_train, y_test = train_df["got_hit"], test_df["got_hit"]
    X_train, X_test = train_df[FEATURE_COLS], test_df[FEATURE_COLS]

    print("\n--- Uncalibrated baseline (shipped model, refit here) ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(X_train, y_train)
    pred_uncal = baseline.predict_proba(X_test)[:, 1]
    report("uncalibrated", y_test, pred_uncal)

    for method in ("isotonic", "sigmoid"):
        print(f"\n--- Calibrated ({method}, cv=5 on training data) ---")
        base = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
            monotonic_cst=MONOTONIC_CST,
        )
        calibrated = CalibratedClassifierCV(base, method=method, cv=5)
        calibrated.fit(X_train, y_train)
        pred_cal = calibrated.predict_proba(X_test)[:, 1]
        report(f"calibrated ({method})", y_test, pred_cal)
        bootstrap_ci(y_test, pred_cal, pred_uncal, f"{method} vs uncalibrated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
