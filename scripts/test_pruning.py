"""Does dropping the weakest features from the shipped 12-feature model
improve holdout performance, or does removing anything just make things
(marginally) worse?

Same true-out-of-time methodology as everything else: train on
2023-2025, test purely on 2026. Refits the current 12-feature model here
(rather than loading data/hit_probability_model.pkl) so the printed
permutation importance is fresh and directly comparable to the pruned
candidates below it -- all fit the same way, same random_state, same
run. Backward elimination: each candidate drops the next-weakest
feature(s) by that fresh importance ranking, weakest first.

Run after train_ml_model.py has populated the cache (reuses
build_dataset, no new network calls).
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

from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

FEATURE_TO_CST = dict(zip(FEATURE_COLS, MONOTONIC_CST))


def report(name: str, y_true, y_pred) -> dict:
    metrics = {
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
    }
    print(
        f"  {name:<40} log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier']:.4f}  auc={metrics['auc']:.4f}"
    )
    return metrics


def fit_and_eval(name, cols, train_df, test_df, y_train, y_test):
    cst = [FEATURE_TO_CST[c] for c in cols]
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    pred = model.predict_proba(test_df[cols])[:, 1]
    metrics = report(name, y_test, pred)
    return model, metrics, pred


def bootstrap_log_loss_diff(y_test, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> None:
    """Paired bootstrap on the holdout rows: is model A's log loss really
    lower than model B's, or is the point-estimate gap explainable by
    sampling noise at this holdout's size? Resamples rows (not
    predictions) with replacement so the same row's two predictions stay
    paired each draw.
    """
    y = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ll_a = log_loss(y[idx], pred_a[idx], labels=[0, 1])
        ll_b = log_loss(y[idx], pred_b[idx], labels=[0, 1])
        diffs[i] = ll_a - ll_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"  Bootstrap 95% CI on (log_loss[A] - log_loss[B]): [{lo:+.5f}, {hi:+.5f}] "
        f"(point estimate {diffs.mean():+.5f}) -- CI spanning zero means the "
        f"difference isn't distinguishable from noise at this holdout size."
    )


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_train, y_test = train_df["got_hit"], test_df["got_hit"]

    print("\n--- Full 12-feature model (refit here), with fresh permutation importance ---")
    full_model, full_metrics, full_pred = fit_and_eval(
        "full (12 features)", FEATURE_COLS, train_df, test_df, y_train, y_test
    )
    perm = permutation_importance(
        full_model, test_df[FEATURE_COLS], y_test, n_repeats=15, random_state=0,
        scoring="neg_log_loss",
    )
    ranked = sorted(zip(FEATURE_COLS, perm.importances_mean, perm.importances_std), key=lambda t: -t[1])
    for name, mean, std in ranked:
        print(f"    {name:<24} {mean:+.5f}  (+/- {std:.5f})")

    # Backward elimination: drop the next-weakest feature(s), weakest
    # first, based on the ranking just computed.
    weakest_first = [name for name, _, _ in ranked[::-1]]
    print(f"\nWeakest-to-strongest order: {weakest_first}")

    print("\n--- Backward elimination ---")
    results = {"full (12 features)": full_metrics}
    preds = {"full (12 features)": full_pred}
    for n_drop in range(1, 6):
        dropped = weakest_first[:n_drop]
        cols = [c for c in FEATURE_COLS if c not in dropped]
        label = f"drop {n_drop} ({', '.join(dropped)})"
        _, metrics, pred = fit_and_eval(label, cols, train_df, test_df, y_train, y_test)
        results[label] = metrics
        preds[label] = pred

    best = min(results, key=lambda k: results[k]["log_loss"])
    print(f"\nBest by holdout log loss: {best} ({results[best]})")

    print(f"\n--- Is '{best}' actually better than the full model, or just noise? ---")
    bootstrap_log_loss_diff(y_test, preds[best], preds["full (12 features)"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
