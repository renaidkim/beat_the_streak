"""Systematic follow-up to the bvp_delta sample-size fix: are OTHER
fractional features in the shipped model also skewed by small
underlying samples?

Checked and ruled safe by construction: `park_factor` is a precomputed
multi-year aggregate across an entire park (thousands of games), not
computed per-matchup, so it can't have this problem. `batting_order`,
`bats_L`, `pitcher_throws_L`, `batter_age` aren't rate/fraction features
at all.

Everything else IS a rate with a real, variable-size denominator:
- pitcher_oba_against: season-to-date average against, can be based on
  as few as 3 batters faced early in a pitcher's season (train_ml_model
  now exposes this AB count as pitcher_oba_ab).
- career_avg / career_obp / career_k_rate / career_bb_rate: career rates
  entering the season, smallest is 55 career at-bats (a brief rookie
  callup) -- exposed as career_avg_ab / career_rate_pa.
- pitcher_era_career / pitcher_k9_career: career rates entering the
  season, smallest is 18 outs (6 innings) of career pitching -- exposed
  as pitcher_career_outs.

Distribution check (train, 37317 rows) found real skew in all of them:
24% of rows have <100 AB behind pitcher_oba_against, 14% have <500
AB/PA behind the career batting rates, 16% have <300 outs (100 IP)
behind the career pitching rates. Worth testing, same rigor as
bvp_delta.

Unlike bvp_delta (already a delta from a baseline, so "no information"
naturally means 0), these are absolute rates -- shrinking them means
blending toward league average, not zero. Each candidate REPLACES its
raw feature in the shipped 13-feature set (not added alongside) using
empirical-Bayes shrinkage: shrunk = raw * w + league_avg * (1 - w),
where w = n / (n + C) and n is the underlying sample size. Several C
(prior strength) values tested per feature.

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

LEAGUE_AVG_AVG = 0.245
LEAGUE_AVG_OBP = 0.320
LEAGUE_AVG_K_RATE = 0.22
LEAGUE_AVG_BB_RATE = 0.08
LEAGUE_AVG_PITCHER_ERA = 4.30
LEAGUE_AVG_PITCHER_K9 = 8.5

# feature name -> (sample-size column, league-average prior, C values to try)
TARGETS = {
    "pitcher_oba_against": ("pitcher_oba_ab", LEAGUE_AVG_AVG, [25, 50, 100]),
    "career_avg": ("career_avg_ab", LEAGUE_AVG_AVG, [100, 300, 600]),
    "career_obp": ("career_avg_ab", LEAGUE_AVG_OBP, [100, 300, 600]),
    "career_k_rate": ("career_rate_pa", LEAGUE_AVG_K_RATE, [100, 300, 600]),
    "career_bb_rate": ("career_rate_pa", LEAGUE_AVG_BB_RATE, [100, 300, 600]),
    "pitcher_era_career": ("pitcher_career_outs", LEAGUE_AVG_PITCHER_ERA, [90, 300, 600]),
    "pitcher_k9_career": ("pitcher_career_outs", LEAGUE_AVG_PITCHER_K9, [90, 300, 600]),
}


def report(name: str, y_true, y_pred) -> dict:
    metrics = {
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_pred)),
    }
    print(
        f"    {name:<38} log_loss={metrics['log_loss']:.4f}  "
        f"brier={metrics['brier']:.4f}  auc={metrics['auc']:.4f}"
    )
    return metrics


def fit_and_predict(train_df, test_df, feature, replacement_col):
    cols = [replacement_col if c == feature else c for c in FEATURE_COLS]
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    model.fit(train_df[cols], train_df["got_hit"])
    pred = model.predict_proba(test_df[cols])[:, 1]
    return model, pred, cols


def bootstrap_ci(y_test, pred_a, pred_b, label: str, n_boot: int = 1500, seed: int = 0) -> float:
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
    frac = (diffs < 0).mean()
    print(
        f"      [{label}] bootstrap 95% CI: [{lo:+.5f}, {hi:+.5f}] "
        f"(point {diffs.mean():+.5f}); fraction favoring candidate: {frac:.3f}"
    )
    return frac


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"]

    summary = []
    for feature, (n_col, prior, c_values) in TARGETS.items():
        print(f"\n=== {feature} (shrink toward {prior}, sample size = {n_col}) ===")
        _, pred_base, _ = fit_and_predict(train_df, test_df, feature, feature)
        base_metrics = report(f"raw {feature} (shipped)", y_test, pred_base)

        best_c, best_frac, best_ll = None, 0.0, base_metrics["log_loss"]
        for c in c_values:
            col = f"{feature}_shrunk_c{c}"
            for df in (train_df, test_df):
                w = df[n_col] / (df[n_col] + c)
                df[col] = df[feature] * w + prior * (1 - w)
            model, pred, cols = fit_and_predict(train_df, test_df, feature, col)
            m = report(f"shrunk C={c}", y_test, pred)
            perm = permutation_importance(model, test_df[cols], y_test, n_repeats=10, random_state=0, scoring="neg_log_loss")
            idx = cols.index(col)
            print(f"      permutation importance: {perm.importances_mean[idx]:+.5f}")
            frac = bootstrap_ci(y_test, pred, pred_base, f"C={c} vs raw")
            if m["log_loss"] < best_ll:
                best_c, best_frac, best_ll = c, frac, m["log_loss"]

        if best_c is not None:
            print(f"  --> best for {feature}: C={best_c} (log_loss {base_metrics['log_loss']:.4f} -> {best_ll:.4f}, {best_frac:.1%} bootstrap)")
            summary.append((feature, best_c, base_metrics["log_loss"], best_ll, best_frac))
        else:
            print(f"  --> nothing beat raw {feature}")
            summary.append((feature, None, base_metrics["log_loss"], base_metrics["log_loss"], 0.5))

    print("\n" + "=" * 70)
    print("SUMMARY (best shrinkage candidate per feature, if any beat raw):")
    for feature, c, base_ll, best_ll, frac in summary:
        verdict = f"C={c}, {frac:.1%} bootstrap" if c is not None else "no improvement"
        print(f"  {feature:<22} {base_ll:.4f} -> {best_ll:.4f}  ({verdict})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
