"""Combined validation before shipping recency weighting.

train_ml_model.py's batter_career_rates()/pitcher_career_rates() were just
edited so career_k_rate/pitcher_era_career/pitcher_k9_career are computed
with recency weighting baked in (decay=0.5/0.85/0.85 respectively -- see
scripts/test_recency_weighting.py for the per-feature validation). This
script checks the *combined* effect against the currently shipped
production model (data/hit_probability_model.pkl, the shrinkage-only gbm),
via a paired bootstrap on the true 2026 holdout, since features that each
individually help are not guaranteed to help identically when combined.

The shipped model was trained on FLAT (non-recency-weighted) versions of
those 3 columns, so its predictions here are computed on a separate
test_df_flat that reconstructs the old flat values (decay=1.0 is
mathematically identical to no weighting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from fit_ml_model import fetch_career_stats  # noqa: E402
from test_recency_weighting import (  # noqa: E402
    recency_weighted_hitting,
    recency_weighted_pitching,
)
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    LEAGUE_AVG_AVG,
    LEAGUE_AVG_OBP,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)
from sklearn.metrics import log_loss

MODEL_PKL = REPO_ROOT.parent / "data" / "hit_probability_model.pkl"


def main() -> int:
    print("Building test set (true out-of-time holdout, recency-weighted):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"].to_numpy()

    print("Building train set (needed only to know which batters/pitchers to fetch career splits for)")
    train_df = build_dataset(TRAIN_SEASONS)

    all_batter_ids = set(train_df["batter_id"]) | set(test_df["batter_id"])
    all_pitcher_ids = set(train_df["pitcher_id"]) | set(test_df["pitcher_id"])
    batter_career = fetch_career_stats(all_batter_ids, "hitting")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")

    print("Reconstructing FLAT (pre-recency-weighting) columns for the shipped model...")
    test_df_flat = test_df.copy()
    k_rate_flat, era_flat, k9_flat = [], [], []
    cache_h: dict[tuple[int, int], dict | None] = {}
    cache_p: dict[tuple[int, int], dict | None] = {}
    for bid, pid, season in zip(test_df["batter_id"], test_df["pitcher_id"], test_df["season"]):
        hk = (bid, season)
        if hk not in cache_h:
            cache_h[hk] = recency_weighted_hitting(batter_career.get(bid, []), season, 1.0)
        rh = cache_h[hk]
        k_rate_flat.append(rh["k_rate"] if rh else 0.22)

        pk = (pid, season)
        if pk not in cache_p:
            cache_p[pk] = recency_weighted_pitching(pitcher_career.get(pid, []), season, 1.0)
        rp = cache_p[pk]
        era_flat.append(rp["era"] if rp else 4.30)
        k9_flat.append(rp["k9"] if rp else 8.5)

    test_df_flat["career_k_rate"] = k_rate_flat
    test_df_flat["pitcher_era_career"] = era_flat
    test_df_flat["pitcher_k9_career"] = k9_flat

    print("Loading shipped production model:", MODEL_PKL)
    shipped_model = joblib.load(MODEL_PKL)
    pred_shipped = shipped_model.predict_proba(test_df_flat[FEATURE_COLS])[:, 1]

    print("Training new candidate (recency-weighted features baked in) on 2023-2025...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from train_ml_model import MONOTONIC_CST

    candidate = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    candidate.fit(train_df[FEATURE_COLS], train_df["got_hit"])
    pred_candidate = candidate.predict_proba(test_df[FEATURE_COLS])[:, 1]

    ll_shipped = log_loss(y_test, pred_shipped, labels=[0, 1])
    ll_candidate = log_loss(y_test, pred_candidate, labels=[0, 1])
    print(f"\nshipped (flat k_rate/era/k9)      log_loss={ll_shipped:.5f}")
    print(f"candidate (recency-weighted)      log_loss={ll_candidate:.5f}")

    rng = np.random.default_rng(0)
    n = len(y_test)
    n_boot = 2000
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (
            log_loss(y_test[idx], pred_candidate[idx], labels=[0, 1])
            - log_loss(y_test[idx], pred_shipped[idx], labels=[0, 1])
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = (diffs < 0).mean()
    print(f"\nbootstrap 95% CI on (candidate - shipped) log_loss: [{lo:+.5f}, {hi:+.5f}]")
    print(f"fraction of draws favoring candidate: {frac:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
