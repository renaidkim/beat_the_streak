"""Is career_obp pulling its weight, or is it redundant with career_avg +
career_bb_rate?

Prompted by a reasonable mechanical objection: OBP counts walks/HBP as
"successes", but Beat the Streak only cares about hits -- a batter who
walks a lot isn't more likely to get a *hit*. career_obp survived the
round-five shrinkage audit (81% bootstrap favoring shrinkage vs. raw
OBP), but that only tested shrinkage strength, never asked whether OBP
itself belongs in the feature set at all. Two red flags noticed after
that: career_obp's permutation importance is the weakest of all 13
features (+0.00006, barely above bats_L), and it's highly correlated
with the two features that already separately exist in the model
(career_avg r=0.72, career_bb_rate r=0.65) -- exactly what "redundant,
not informative" would look like, since OBP is mechanically just a
blend of hits and walks.

Tests: drop career_obp entirely (12 features) vs. the shipped 13, on the
true 2026 holdout, via the same paired-bootstrap methodology used for
every other feature change this session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)


def fit_predict(train_df, test_df, cols, monotonic):
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=monotonic,
    )
    model.fit(train_df[cols], train_df["got_hit"])
    return model, model.predict_proba(test_df[cols])[:, 1]


def main() -> int:
    print("Building datasets...")
    train_df = build_dataset(TRAIN_SEASONS)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"].to_numpy()

    idx_obp = FEATURE_COLS.index("career_obp")
    cols_dropped = [c for c in FEATURE_COLS if c != "career_obp"]
    mono_dropped = [m for i, m in enumerate(MONOTONIC_CST) if i != idx_obp]

    _, pred_full = fit_predict(train_df, test_df, FEATURE_COLS, MONOTONIC_CST)
    _, pred_dropped = fit_predict(train_df, test_df, cols_dropped, mono_dropped)

    ll_full = log_loss(y_test, pred_full, labels=[0, 1])
    ll_dropped = log_loss(y_test, pred_dropped, labels=[0, 1])
    print(f"\nshipped (13 features, incl. career_obp)  log_loss={ll_full:.5f}")
    print(f"career_obp dropped (12 features)          log_loss={ll_dropped:.5f}")

    rng = np.random.default_rng(0)
    n = len(y_test)
    n_boot = 2000
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y_test[idx], pred_dropped[idx], labels=[0, 1])
            - log_loss(y_test[idx], pred_full[idx], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = (diffs < 0).mean()
    print(f"\nbootstrap 95% CI on (dropped - shipped) log_loss: [{lo:+.5f}, {hi:+.5f}]")
    print(f"fraction of draws favoring dropping career_obp: {frac:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
