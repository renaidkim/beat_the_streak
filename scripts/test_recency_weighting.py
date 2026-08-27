"""Does weighting recent seasons more heavily than old ones -- not just
distrusting small samples -- improve the career-rate features?

Motivated directly by the sample-size audit (scripts/
test_sample_size_audit.py): shrinkage toward league average helped
career_avg/career_obp/pitcher_oba_against, but measurably HURT
career_k_rate/career_bb_rate/pitcher_era_career/pitcher_k9_career.
Shrinkage only asks "how much total sample backs this rate" -- it
weights a player's rookie season and last season equally as long as the
total at-bat count is the same. Recency weighting asks a different
question: is a stat from 5 years ago as informative about *today's*
skill as one from last year? For pitchers especially, "stuff" and role
change year to year in ways plain sample-size correction can't see.

Implementation: instead of summing raw counts across all prior seasons
equally, each prior season's counts are weighted by decay^(seasons_ago)
before summing (weighted_ab = sum(ab_i * decay^age_i), rate =
weighted_hits/weighted_ab) -- the standard approach real projection
systems (Marcel, etc.) use, and sounder than averaging each season's own
rate (which would let a 50-PA rookie season vote equally against a
600-PA everyday season).

Tests recency-weighted versions of all four career-rate features
(including the two, career_avg/career_obp, that already ship with
sample-size shrinkage -- recency weighting is additive/different, not a
replacement, so worth checking on top of the shipped correction too),
each swapped in for whatever is currently shipped for that feature.

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

from fit_ml_model import fetch_career_stats  # noqa: E402
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    LEAGUE_AVG_AVG,
    LEAGUE_AVG_OBP,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

LEAGUE_AVG_K_RATE = 0.22
LEAGUE_AVG_BB_RATE = 0.08
LEAGUE_AVG_PITCHER_ERA = 4.30
LEAGUE_AVG_PITCHER_K9 = 8.5


def recency_weighted_hitting(splits: list[dict], season_year: int, decay: float) -> dict | None:
    ab = hits = bb = hbp = sf = so = pa = 0.0
    for s in splits:
        season = int(s.get("season", season_year))
        if season >= season_year:
            continue
        age = season_year - season - 1  # 0 for last season, 1 for two seasons ago, etc.
        w = decay**age
        st = s["stat"]
        ab += w * int(st.get("atBats", 0))
        hits += w * int(st.get("hits", 0))
        bb += w * int(st.get("baseOnBalls", 0))
        hbp += w * int(st.get("hitByPitch", 0))
        sf += w * int(st.get("sacFlies", 0))
        so += w * int(st.get("strikeOuts", 0))
        pa += w * int(st.get("plateAppearances", 0))
    if ab == 0:
        return None
    avg = hits / ab
    obp_den = ab + bb + hbp + sf
    return {
        "avg": avg,
        "obp": (hits + bb + hbp) / obp_den if obp_den > 0 else avg,
        "k_rate": so / pa if pa > 0 else LEAGUE_AVG_K_RATE,
        "bb_rate": bb / pa if pa > 0 else LEAGUE_AVG_BB_RATE,
    }


def recency_weighted_pitching(splits: list[dict], season_year: int, decay: float) -> dict | None:
    outs = er = k = 0.0
    for s in splits:
        season = int(s.get("season", season_year))
        if season >= season_year:
            continue
        age = season_year - season - 1
        w = decay**age
        st = s["stat"]
        outs += w * int(st.get("outs", 0))
        er += w * int(st.get("earnedRuns", 0))
        k += w * int(st.get("strikeOuts", 0))
    if outs == 0:
        return None
    return {"era": er * 27 / outs, "k9": k * 27 / outs}


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


def bootstrap_frac(y_test, pred_a, pred_b, label: str, n_boot: int = 1500, seed: int = 0) -> float:
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
    print(f"      [{label}] bootstrap 95% CI: [{lo:+.5f}, {hi:+.5f}]; fraction favoring candidate: {frac:.3f}")
    return frac


def main() -> int:
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print("Building test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"]

    all_seasons = list(TRAIN_SEASONS) + [TEST_SEASON]
    all_batter_ids = set(train_df["batter_id"]) | set(test_df["batter_id"])
    all_pitcher_ids = set(train_df["pitcher_id"]) | set(test_df["pitcher_id"])
    print(f"\nFetching career splits for {len(all_batter_ids)} batters, {len(all_pitcher_ids)} pitchers...")
    batter_career = fetch_career_stats(all_batter_ids, "hitting")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")

    decays = [0.5, 0.7, 0.85]
    targets = {
        "career_avg": ("hitting", "avg", LEAGUE_AVG_AVG),
        "career_obp": ("hitting", "obp", LEAGUE_AVG_OBP),
        "career_k_rate": ("hitting", "k_rate", LEAGUE_AVG_K_RATE),
        "career_bb_rate": ("hitting", "bb_rate", LEAGUE_AVG_BB_RATE),
        "pitcher_era_career": ("pitching", "era", LEAGUE_AVG_PITCHER_ERA),
        "pitcher_k9_career": ("pitching", "k9", LEAGUE_AVG_PITCHER_K9),
    }

    summary = []
    for feature, (kind, key, prior) in targets.items():
        print(f"\n=== {feature} (recency-weighted, decay tested) ===")
        _, pred_base, _ = fit_and_predict(train_df, test_df, feature, feature)
        base_metrics = report(f"shipped {feature}", y_test, pred_base)

        best_d, best_frac, best_ll = None, 0.0, base_metrics["log_loss"]
        for decay in decays:
            col = f"{feature}_decay{decay}"
            cache: dict[tuple[int, int], dict | None] = {}
            for df in (train_df, test_df):
                values = []
                for bid, pid, season in zip(df["batter_id"], df["pitcher_id"], df["season"]):
                    pid_key = bid if kind == "hitting" else pid
                    cache_key = (pid_key, season)
                    if cache_key not in cache:
                        if kind == "hitting":
                            cache[cache_key] = recency_weighted_hitting(batter_career.get(bid, []), season, decay)
                        else:
                            cache[cache_key] = recency_weighted_pitching(pitcher_career.get(pid, []), season, decay)
                    rates = cache[cache_key]
                    values.append(rates[key] if rates else prior)
                df[col] = values
            model, pred, cols = fit_and_predict(train_df, test_df, feature, col)
            m = report(f"decay={decay}", y_test, pred)
            perm = permutation_importance(model, test_df[cols], y_test, n_repeats=10, random_state=0, scoring="neg_log_loss")
            idx = cols.index(col)
            print(f"      permutation importance: {perm.importances_mean[idx]:+.5f}")
            frac = bootstrap_frac(y_test, pred, pred_base, f"decay={decay} vs shipped")
            if m["log_loss"] < best_ll:
                best_d, best_frac, best_ll = decay, frac, m["log_loss"]

        if best_d is not None:
            print(f"  --> best for {feature}: decay={best_d} ({base_metrics['log_loss']:.4f} -> {best_ll:.4f}, {best_frac:.1%})")
            summary.append((feature, best_d, base_metrics["log_loss"], best_ll, best_frac))
        else:
            print(f"  --> nothing beat shipped {feature}")
            summary.append((feature, None, base_metrics["log_loss"], base_metrics["log_loss"], 0.5))

    print("\n" + "=" * 70)
    print("SUMMARY:")
    for feature, decay, base_ll, best_ll, frac in summary:
        verdict = f"decay={decay}, {frac:.1%} bootstrap" if decay is not None else "no improvement"
        print(f"  {feature:<22} {base_ll:.4f} -> {best_ll:.4f}  ({verdict})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
