"""Should bvp_delta be discounted when it's backed by a tiny sample?

Spotted live: a batter who went 2-for-2 against a pitcher shows up with
a reason like "has hit this pitcher well historically (+0.726 vs. own
season avg)" -- a real number, but from a sample far too small to
support the word "historically." The shipped feature (round four) used
the raw delta regardless of at-bat count, on the reasoning that a
flexible tree model + permutation importance should be trusted to down-
weight noisy small samples on their own. That's worth checking
empirically rather than asserting -- this script tests whether an
explicitly sample-size-aware version of bvp_delta beats the raw one
that's currently shipped.

Two ways to make a feature "trust" sample size, both tested:
1. Hard threshold: zero out bvp_delta entirely below N at-bats (the
   more literal reading of "only consider it with enough sample size").
2. Shrinkage (empirical-Bayes style): multiply by ab/(ab+C), so small
   samples get pulled toward 0 continuously rather than an all-or-
   nothing cutoff at some arbitrary N. Standard technique for exactly
   this problem in sabermetrics (it's how "regressed" stats like
   Marcel projections work).

Each candidate REPLACES bvp_delta in the 13-feature set (not added
alongside it -- they're different encodings of the same underlying
signal, so a straight swap is the fair comparison, not a combined model
that would just create collinearity).

Run after train_ml_model.py has populated the cache (bvp_ab_prior is
now kept in build_dataset's output -- no new network calls).
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

BVP_IDX = FEATURE_COLS.index("bvp_delta")


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


def fit_and_predict(train_df, test_df, feature_col_for_bvp):
    cols = list(FEATURE_COLS)
    cols[BVP_IDX] = feature_col_for_bvp
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    model.fit(train_df[cols], train_df["got_hit"])
    pred = model.predict_proba(test_df[cols])[:, 1]
    return model, pred, cols


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
        f"      [{label}] bootstrap 95% CI on log_loss diff: [{lo:+.5f}, {hi:+.5f}] "
        f"(point estimate {diffs.mean():+.5f}); fraction favoring candidate: {(diffs < 0).mean():.3f}"
    )


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"]

    print(f"\nbvp_ab_prior distribution (train, {len(train_df)} rows):")
    for k in [0, 1, 2, 3, 5, 10, 20]:
        n = (train_df["bvp_ab_prior"] >= k).sum()
        print(f"  >= {k:>2} prior AB vs. that exact pitcher: {n:>6} rows ({n / len(train_df):.1%})")

    for df in (train_df, test_df):
        for k in (3, 5, 10):
            df[f"bvp_delta_min{k}"] = np.where(df["bvp_ab_prior"] >= k, df["bvp_delta"], 0.0)
        for c in (5, 10, 20):
            df[f"bvp_delta_shrunk_c{c}"] = df["bvp_delta"] * df["bvp_ab_prior"] / (df["bvp_ab_prior"] + c)

    print("\n--- Baseline: shipped model, raw bvp_delta (no sample-size awareness) ---")
    _, pred_base, _ = fit_and_predict(train_df, test_df, "bvp_delta")
    base_metrics = report("raw bvp_delta (shipped)", y_test, pred_base)

    print("\n--- Hard-threshold candidates (zero below N at-bats) ---")
    best_name, best_pred, best_ll = None, None, base_metrics["log_loss"]
    for k in (3, 5, 10):
        col = f"bvp_delta_min{k}"
        model, pred, cols = fit_and_predict(train_df, test_df, col)
        m = report(col, y_test, pred)
        perm = permutation_importance(model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss")
        idx = cols.index(col)
        print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")
        bootstrap_ci(y_test, pred, pred_base, f"{col} vs raw")
        if m["log_loss"] < best_ll:
            best_name, best_pred, best_ll = col, pred, m["log_loss"]

    print("\n--- Shrinkage candidates (ab/(ab+C) weighting) ---")
    for c in (5, 10, 20):
        col = f"bvp_delta_shrunk_c{c}"
        model, pred, cols = fit_and_predict(train_df, test_df, col)
        m = report(col, y_test, pred)
        perm = permutation_importance(model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss")
        idx = cols.index(col)
        print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")
        bootstrap_ci(y_test, pred, pred_base, f"{col} vs raw")
        if m["log_loss"] < best_ll:
            best_name, best_pred, best_ll = col, pred, m["log_loss"]

    print(f"\nBest candidate by holdout log loss: {best_name or 'raw bvp_delta (shipped) -- nothing beat it'}")
    if best_name:
        bootstrap_ci(y_test, best_pred, pred_base, f"{best_name} vs raw, final check")

    return 0


if __name__ == "__main__":
    sys.exit(main())
