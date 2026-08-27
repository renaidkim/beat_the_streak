"""The three recency-weighted features (career_k_rate, pitcher_era_career,
pitcher_k9_career) each individually beat the shipped shrinkage-only model
in isolation (test_recency_weighting.py: 82.7%/69.8%/63.6% bootstrap), but
all three combined is a coin flip against shipped (47.0%,
test_recency_weighting_ship.py) -- worse than any single one alone. That
means they don't compose additively; something about combining them
cancels or overfits. This sweeps all 7 non-empty subsets of the 3 features
to find out which combination (if any) actually holds up combined, each
scored against the same shipped-model baseline via paired bootstrap.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from fit_ml_model import fetch_career_stats  # noqa: E402
from test_recency_weighting import (  # noqa: E402
    recency_weighted_hitting,
    recency_weighted_pitching,
)
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

MODEL_PKL = REPO_ROOT.parent / "data" / "hit_probability_model.pkl"

FEATURES = ["career_k_rate", "pitcher_era_career", "pitcher_k9_career"]
DECAYS = {"career_k_rate": 0.5, "pitcher_era_career": 0.85, "pitcher_k9_career": 0.85}
KIND = {"career_k_rate": "hitting", "pitcher_era_career": "pitching", "pitcher_k9_career": "pitching"}
KEY = {"career_k_rate": "k_rate", "pitcher_era_career": "era", "pitcher_k9_career": "k9"}
PRIOR = {"career_k_rate": 0.22, "pitcher_era_career": 4.30, "pitcher_k9_career": 8.5}


def build_col(df, feature, decay, batter_career, pitcher_career, cache):
    kind = KIND[feature]
    key = KEY[feature]
    prior = PRIOR[feature]
    values = []
    for bid, pid, season in zip(df["batter_id"], df["pitcher_id"], df["season"]):
        pid_key = bid if kind == "hitting" else pid
        cache_key = (feature, decay, pid_key, season)
        if cache_key not in cache:
            if kind == "hitting":
                cache[cache_key] = recency_weighted_hitting(batter_career.get(bid, []), season, decay)
            else:
                cache[cache_key] = recency_weighted_pitching(pitcher_career.get(pid, []), season, decay)
        rates = cache[cache_key]
        values.append(rates[key] if rates else prior)
    return values


def main() -> int:
    print("Building train/test sets (current build_dataset -- recency-weighted already baked in)...")
    train_df = build_dataset(TRAIN_SEASONS)
    test_df = build_dataset([TEST_SEASON])
    y_train = train_df["got_hit"]
    y_test = test_df["got_hit"].to_numpy()

    all_batter_ids = set(train_df["batter_id"]) | set(test_df["batter_id"])
    all_pitcher_ids = set(train_df["pitcher_id"]) | set(test_df["pitcher_id"])
    batter_career = fetch_career_stats(all_batter_ids, "hitting")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")
    cache: dict = {}

    # Flat (decay=1.0) columns for each feature, both train and test --
    # these reconstruct the shipped model's original inputs.
    flat_train = {f: build_col(train_df, f, 1.0, batter_career, pitcher_career, cache) for f in FEATURES}
    flat_test = {f: build_col(test_df, f, 1.0, batter_career, pitcher_career, cache) for f in FEATURES}
    # Weighted (best decay) columns -- these match what build_dataset now produces,
    # but recomputed explicitly here so we can mix-and-match per combo.
    weighted_train = {f: build_col(train_df, f, DECAYS[f], batter_career, pitcher_career, cache) for f in FEATURES}
    weighted_test = {f: build_col(test_df, f, DECAYS[f], batter_career, pitcher_career, cache) for f in FEATURES}

    print("Loading shipped production model...")
    shipped_model = joblib.load(MODEL_PKL)
    test_df_flat_all = test_df.copy()
    for f in FEATURES:
        test_df_flat_all[f] = flat_test[f]
    pred_shipped = shipped_model.predict_proba(test_df_flat_all[FEATURE_COLS])[:, 1]
    ll_shipped = log_loss(y_test, pred_shipped, labels=[0, 1])
    print(f"shipped (all flat) log_loss={ll_shipped:.5f}\n")

    rng_seed = 0
    results = []
    for r in range(1, len(FEATURES) + 1):
        for combo in itertools.combinations(FEATURES, r):
            train_variant = train_df.copy()
            test_variant = test_df.copy()
            for f in FEATURES:
                if f in combo:
                    train_variant[f] = weighted_train[f]
                    test_variant[f] = weighted_test[f]
                else:
                    train_variant[f] = flat_train[f]
                    test_variant[f] = flat_test[f]

            model = HistGradientBoostingClassifier(
                max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
                monotonic_cst=MONOTONIC_CST,
            )
            model.fit(train_variant[FEATURE_COLS], y_train)
            pred = model.predict_proba(test_variant[FEATURE_COLS])[:, 1]
            ll = log_loss(y_test, pred, labels=[0, 1])

            rng = np.random.default_rng(rng_seed)
            n = len(y_test)
            n_boot = 1500
            diffs = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.integers(0, n, n)
                diffs[i] = (
                    log_loss(y_test[idx], pred[idx], labels=[0, 1])
                    - log_loss(y_test[idx], pred_shipped[idx], labels=[0, 1])
                )
            frac = (diffs < 0).mean()
            label = "+".join(combo)
            print(f"  {label:<55} log_loss={ll:.5f}  frac_favoring={frac:.3f}")
            results.append((combo, ll, frac))

    print("\n" + "=" * 70)
    print("Sorted by fraction favoring candidate (desc):")
    for combo, ll, frac in sorted(results, key=lambda t: -t[2]):
        print(f"  {'+'.join(combo):<55} log_loss={ll:.5f}  frac={frac:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
