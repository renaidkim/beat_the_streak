"""Does this season's cumulative batting average -- as distinct from
"recent form" over 5-10 games -- add predictive power beyond career
average?

This is a different question from the recent-form test
(scripts/test_recent_form.py): a 5-10 game window is a tiny, high-
variance sample (a "hot streak" is often just a few lucky bloopers), but
a batter's cumulative average over an entire season-to-date (hundreds of
at-bats by midseason) is a real, statistically credible sample size --
categorically different reliability. The original "season average adds
nothing beyond career average" finding (README's "Model backtest",
round one/two) was from the *old* single-season logistic regression
pipeline, on 2025 data only, and was never re-tested in the current
multi-season/monotonic-GBM/permutation-importance pipeline this project
has since moved to. Worth actually re-checking rather than assuming the
old finding still holds.

Tests season_avg_delta = season-to-date average minus career average
(same delta reparametrization as every other feature here) -- directly
answers the motivating question: does a .300 career hitter batting .230
this season actually deserve a lower prediction than his career average
alone would suggest?

season_avg_to_date is already computed and stored by build_dataset()
(kept there specifically for scoring the old shipped model), so this
needs no new data fetching at all.

Run after train_ml_model.py has populated the cache.
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

    for df in (train_df, test_df):
        df["season_avg_delta"] = df["season_avg_to_date"] - df["career_avg"]

    print(f"\nseason_avg_delta stats (train): {train_df['season_avg_delta'].describe()}")
    slumping = (train_df["season_avg_delta"] <= -0.05).sum()
    print(f"Rows with season_avg_delta <= -0.050 (meaningfully underperforming career avg): {slumping}")

    print("\n--- Baseline: shipped model's 13 features, refit here for a fair comparison ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(train_df[FEATURE_COLS], y_train)
    pred_base = baseline.predict_proba(test_df[FEATURE_COLS])[:, 1]
    report("baseline (13 features)", y_test, pred_base)

    print("\n--- + season_avg_delta (season-to-date minus career average) ---")
    cols = FEATURE_COLS + ["season_avg_delta"]
    cst = MONOTONIC_CST + [1]  # playing better than career avg this season shouldn't lower P(hit)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    pred_sa = model.predict_proba(test_df[cols])[:, 1]
    report("+ season_avg_delta", y_test, pred_sa)
    perm = permutation_importance(
        model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    idx = cols.index("season_avg_delta")
    print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")

    y = np.asarray(y_test)
    rng = np.random.default_rng(0)
    n = len(y)
    diffs = np.empty(2000)
    for i in range(2000):
        idx2 = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y[idx2], pred_sa[idx2], labels=[0, 1])
            - log_loss(y[idx2], pred_base[idx2], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(
        f"\nBootstrap 95% CI on (log_loss[+season_avg_delta] - log_loss[baseline]): "
        f"[{lo:+.5f}, {hi:+.5f}] (point estimate {diffs.mean():+.5f})"
    )
    print(f"Fraction of bootstrap draws where +season_avg_delta is better: {(diffs < 0).mean():.3f}")

    # Direct point-check: the user's exact motivating scenario -- a
    # .300 career hitter batting .230 this season. Does the model
    # actually penalize this, and would it survive a monotonic
    # constraint (i.e. is the direction consistent, not just present)?
    print("\n--- Point-check: .300 career hitter, season_avg_delta swept ---")
    base_row = {c: test_df[c].median() for c in FEATURE_COLS}
    base_row["career_avg"] = 0.300
    for delta in [0.05, 0.0, -0.03, -0.05, -0.07, -0.10, -0.15]:
        row = dict(base_row, season_avg_delta=delta)
        x = [[row[c] for c in cols]]
        p = model.predict_proba(x)[0][1]
        season_avg = 0.300 + delta
        print(f"  season_avg_delta={delta:+.2f} (batting ~.{round(season_avg*1000):03d} this year): P(hit)={p:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
