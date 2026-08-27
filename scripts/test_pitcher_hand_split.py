"""Does a pitcher's career OBA specifically against the current batter's
handedness improve on the model's current pitcher features?

Earlier attempt was blocked: sitCodes (vl/vr platoon filters) are
silently ignored by stats=byDateRange and stats=yearByYear -- verified
by comparing filtered vs. unfiltered responses. Re-investigated and
found the actual fix: stats=statSplits DOES honor sitCodes, but only
when given an explicit season= parameter (one API call per
pitcher-season, not one call for full career like yearByYear). This
script fetches exactly those (pitcher_id, season) pairs that matter --
only seasons a pitcher actually has yearByYear data for, reusing the
season list from the career fetch already cached by fit_ml_model.py --
and manually sums the per-season vl/vr counts into a point-in-time
"entering season Y" split, the same architecture used for every other
career stat in this pipeline.

The new feature is a DELTA, not an absolute: how much worse/better this
specific pitcher does against the batter's actual handedness compared
to his own overall career average, following the same
avoid-collinearity-via-deltas reasoning already used for platoon_delta
and the recent_form_*_delta features (round one/two: absolute versions
collapsed onto season/career average and created artificial
collinearity). Switch hitters are resolved to their effective side
against THIS specific pitcher (opposite of the pitcher's own throwing
hand -- that's the entire point of switch-hitting).

Tests the raw delta, several shrinkage strengths (this is a fractional
feature with small-sample potential exactly like every other rate
feature audited this session), and reports permutation importance +
bootstrap vs. the shipped model.

RESULT: rejected. Every shrinkage strength tested (None, C=10/25/50/100)
underperformed the shipped model -- best was C=50 at 34.4% of a
2000-draw bootstrap favoring, still well below break-even, and the
fraction-favoring didn't move monotonically with C (0.118, 0.216,
0.163, 0.344, 0.179) -- the signature of shrinking toward a real
relationship is a clean gradient as C increases; this bouncing around
is what shrinking noise toward zero looks like instead. Consistent with
a well-known sabermetric finding: pitcher-specific platoon splits are
extremely noisy at the individual level and need enormous sample sizes
(triple digits of PA against the less-common-for-that-pitcher hand,
often more than a full early-to-mid career provides) before they
separate from noise -- serious projection systems heavily regress
individual pitcher platoon splits toward the league-average platoon
effect for exactly this reason. The league-average version of this
effect is arguably already captured by bats_L/pitcher_throws_L
(same-hand vs. opposite-hand, at least directionally); what this
feature tried to add -- *this specific pitcher's* deviation from that
average -- just isn't reliably measurable from the sample sizes
available here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

from fit_ml_model import BASE, cached_get, fetch_career_stats  # noqa: E402
from train_ml_model import (  # noqa: E402
    FEATURE_COLS,
    MONOTONIC_CST,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_dataset,
)

LEAGUE_AVG_AVG = 0.245


def fetch_hand_splits_by_season(pitcher_ids: set[int], seasons_needed: dict[int, set[int]]) -> dict[tuple[int, int], dict[str, dict]]:
    """{(pitcher_id, season): {"vl": {"ab":.., "hits":..}, "vr": {...}}}
    for exactly the (pitcher, season) pairs in seasons_needed.
    """
    result: dict[tuple[int, int], dict[str, dict]] = {}
    pairs = [(pid, s) for pid in pitcher_ids for s in seasons_needed.get(pid, set())]
    print(f"Fetching statSplits for {len(pairs)} (pitcher, season) pairs...")
    for i, (pid, season) in enumerate(pairs):
        resp = cached_get(
            f"{BASE}/people/{pid}/stats",
            params={"stats": "statSplits", "sitCodes": "vl,vr", "group": "pitching", "season": season},
        )
        splits = {}
        for stat_group in resp.get("stats", []):
            for split in stat_group.get("splits", []):
                code = split.get("split", {}).get("code")
                if code in ("vl", "vr"):
                    st = split["stat"]
                    splits[code] = {"ab": int(st.get("atBats", 0)), "hits": int(st.get("hits", 0))}
        if splits:
            result[(pid, season)] = splits
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(pairs)}")
    return result


def career_hand_split_entering_season(pid: int, season_year: int, seasons_available: set[int], hand_splits: dict) -> dict | None:
    ab_l = h_l = ab_r = h_r = 0
    for s in seasons_available:
        if s >= season_year:
            continue
        splits = hand_splits.get((pid, s))
        if not splits:
            continue
        vl, vr = splits.get("vl"), splits.get("vr")
        if vl:
            ab_l += vl["ab"]
            h_l += vl["hits"]
        if vr:
            ab_r += vr["ab"]
            h_r += vr["hits"]
    if ab_l + ab_r == 0:
        return None
    return {"ab_l": ab_l, "h_l": h_l, "ab_r": ab_r, "h_r": h_r}


def effective_batter_hand(bats: str, pitcher_throws: str) -> str:
    if bats == "S":
        return "R" if pitcher_throws == "L" else "L"
    return bats


def shrink(raw: float, n: int, prior: float, c: float) -> float:
    w = n / (n + c)
    return raw * w + prior * (1 - w)


def build_agg_cache(df, hand_splits, seasons_by_pitcher) -> dict[tuple[int, int], dict | None]:
    cache: dict[tuple[int, int], dict | None] = {}
    for pid, season in zip(df["pitcher_id"], df["season"]):
        key = (pid, season)
        if key not in cache:
            cache[key] = career_hand_split_entering_season(pid, season, seasons_by_pitcher.get(pid, set()), hand_splits)
    return cache


def build_delta_column(df, agg_cache, shrink_c: float | None) -> list[float]:
    values = []
    for pid, season, bats, throws in zip(df["pitcher_id"], df["season"], df["bats"], df["pitcher_throws"]):
        agg = agg_cache[(pid, season)]
        if agg is None:
            values.append(0.0)
            continue
        ab_l, h_l, ab_r, h_r = agg["ab_l"], agg["h_l"], agg["ab_r"], agg["h_r"]
        ab_total, h_total = ab_l + ab_r, h_l + h_r
        overall = h_total / ab_total if ab_total > 0 else LEAGUE_AVG_AVG
        hand = effective_batter_hand(bats, throws)
        ab_hand, h_hand = (ab_l, h_l) if hand == "L" else (ab_r, h_r)
        raw_vs_hand = h_hand / ab_hand if ab_hand > 0 else overall
        if shrink_c is not None:
            raw_vs_hand = shrink(raw_vs_hand, ab_hand, overall, shrink_c)
        values.append(raw_vs_hand - overall)
    return values


def fit_predict(train_df, test_df, cols, monotonic):
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=monotonic,
    )
    model.fit(train_df[cols], train_df["got_hit"])
    return model, model.predict_proba(test_df[cols])[:, 1]


def bootstrap_frac(y_test, pred_a, pred_b, n_boot=2000, seed=0) -> tuple[float, float, float]:
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
    return lo, hi, (diffs < 0).mean()


def main() -> int:
    print("Building datasets (need batter bats / pitcher throws per row)...")
    train_df = build_dataset(TRAIN_SEASONS)
    test_df = build_dataset([TEST_SEASON])
    y_test = test_df["got_hit"].to_numpy()

    # build_dataset's out_rows don't carry raw bats/throws strings (only
    # bats_L/bats_S/pitcher_throws_L flags) -- reconstruct them, they're
    # a deterministic function of those flags.
    for df in (train_df, test_df):
        df["bats"] = np.where(df["bats_S"] == 1.0, "S", np.where(df["bats_L"] == 1.0, "L", "R"))
        df["pitcher_throws"] = np.where(df["pitcher_throws_L"] == 1.0, "L", "R")

    all_pitcher_ids = set(train_df["pitcher_id"]) | set(test_df["pitcher_id"])
    print(f"Fetching career pitching splits for {len(all_pitcher_ids)} pitchers (for season lists)...")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")
    seasons_by_pitcher = {
        pid: {int(s.get("season", 0)) for s in splits} for pid, splits in pitcher_career.items()
    }

    hand_splits = fetch_hand_splits_by_season(all_pitcher_ids, seasons_by_pitcher)
    print(f"Got hand-split data for {len(hand_splits)} (pitcher, season) pairs")

    print("Building per-(pitcher, season) aggregate cache...")
    train_agg_cache = build_agg_cache(train_df, hand_splits, seasons_by_pitcher)
    test_agg_cache = build_agg_cache(test_df, hand_splits, seasons_by_pitcher)

    candidates = [None, 10, 25, 50, 100]
    baseline_cols = FEATURE_COLS
    baseline_mono = MONOTONIC_CST
    _, pred_base = fit_predict(train_df, test_df, baseline_cols, baseline_mono)
    ll_base = log_loss(y_test, pred_base, labels=[0, 1])
    print(f"\nshipped (13 features)  log_loss={ll_base:.5f}\n")

    for c in candidates:
        col = f"pitcher_hand_delta_c{c}"
        train_df[col] = build_delta_column(train_df, train_agg_cache, c)
        test_df[col] = build_delta_column(test_df, test_agg_cache, c)
        cols = baseline_cols + [col]
        mono = baseline_mono + [1]
        model, pred = fit_predict(train_df, test_df, cols, mono)
        ll = log_loss(y_test, pred, labels=[0, 1])
        lo, hi, frac = bootstrap_frac(y_test, pred, pred_base)
        nonzero = (train_df[col] != 0.0).mean()
        print(
            f"  shrink_c={str(c):<5} log_loss={ll:.5f}  CI=[{lo:+.5f},{hi:+.5f}]  "
            f"frac_favoring={frac:.3f}  nonzero_frac={nonzero:.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
