"""Does opposing bullpen quality help predict hit probability?

The model currently only encodes info about the probable *starter*, but
a batter's 2nd/3rd/4th plate appearances in a game are often against
relievers, not the starter -- team bullpen quality is a real gap in the
feature set.

Caveat, tested honestly: MLB Stats API's team `byDateRange` stat type
silently ignores `sitCodes` (verified directly -- a relief-only query
for a full season returns identical numbers to the whole-staff query),
so there's no reliable way to get a true point-in-time (mid-season,
as-of-date) bullpen split from this API. Only full *completed*-season
splits respect sitCodes=rp correctly. So, same as the Statcast test,
this uses the *prior* completed season's bullpen ERA as a lag-1 proxy
-- with a bigger honesty caveat than Statcast had, since bullpens
churn within a season (trades, injuries, callups) much more than a
hitter's exit velocity does. If even this stale proxy shows a real
signal, that's suggestive enough to justify the much bigger fetch
(every relief pitcher's game logs, to reconstruct a true point-in-time
number) as a follow-up. If it shows nothing, that's reasonably strong
evidence the fresher version wouldn't be worth chasing either.

Run after train_ml_model.py has populated the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
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

CACHE_DIR = REPO_ROOT.parent / ".cache" / "bullpen"
BASE = "https://statsapi.mlb.com/api/v1"
session = requests.Session()


def fetch_bullpen_era(season: int) -> dict[int, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{season}.json"
    if path.exists():
        import json
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    resp = session.get(
        f"{BASE}/teams/stats",
        params={"stats": "statSplits", "group": "pitching", "season": season, "sportId": 1, "sitCodes": "rp"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    result = {}
    for s in data.get("stats", []):
        for sp in s.get("splits", []):
            team_id = sp.get("team", {}).get("id")
            era = sp.get("stat", {}).get("era")
            if team_id is not None and era is not None:
                result[team_id] = float(era)
    import json
    path.write_text(json.dumps(result))
    return result


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

    if "opponent_team_id" not in train_df.columns:
        print("ERROR: build_dataset doesn't expose opponent_team_id -- see comment in train_ml_model.py")
        return 1

    lag_years = {s - 1 for s in TRAIN_SEASONS} | {TEST_SEASON - 1}
    print(f"\nFetching team bullpen ERA (relief only) for lag years {sorted(lag_years)}...")
    bullpen = {y: fetch_bullpen_era(y) for y in lag_years}
    league_avg = {y: sum(v.values()) / len(v) for y, v in bullpen.items()}

    for df in (train_df, test_df):
        df["opp_bullpen_era_prior_season"] = [
            bullpen.get(season - 1, {}).get(int(opp), league_avg.get(season - 1, 4.3))
            for season, opp in zip(df["season"], df["opponent_team_id"])
        ]

    print("\n--- Baseline: shipped model's 12 features, refit here for a fair comparison ---")
    baseline = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    baseline.fit(train_df[FEATURE_COLS], y_train)
    report("baseline (12 features)", y_test, baseline.predict_proba(test_df[FEATURE_COLS])[:, 1])

    print("\n--- + opponent bullpen ERA (prior season) ---")
    cols = FEATURE_COLS + ["opp_bullpen_era_prior_season"]
    cst = MONOTONIC_CST + [1]  # higher bullpen ERA (worse bullpen) shouldn't lower P(hit)
    model = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=cst,
    )
    model.fit(train_df[cols], y_train)
    pred = model.predict_proba(test_df[cols])[:, 1]
    report("+ opp_bullpen_era_prior_season", y_test, pred)
    perm = permutation_importance(
        model, test_df[cols], y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    idx = cols.index("opp_bullpen_era_prior_season")
    print(f"      permutation importance: {perm.importances_mean[idx]:+.5f} (+/- {perm.importances_std[idx]:.5f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
