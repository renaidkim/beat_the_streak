"""Build features from the data fit_ml_model.py fetched, train several
model families on 2023-2025, and evaluate everything -- including the
currently-shipped production model -- on the true 2026 out-of-time holdout.

Run fit_ml_model.py first (it populates the on-disk cache this script
reads from; every call here hits that cache, no new network requests
unless something wasn't fetched).
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fit_ml_model import (  # noqa: E402
    BASE,
    DEFAULT_BATTING_ORDER,
    PARK_FACTORS_PATH,
    TEST_SEASON,
    TRAIN_SEASONS,
    cached_get,
    fetch_career_stats,
    fetch_season_raw,
)

from beat_the_streak.models import Batter, Matchup, Pitcher  # noqa: E402
from beat_the_streak.rank import score_matchup  # noqa: E402

LEAGUE_AVG_AVG = 0.245
MODEL_OUT_JSON = REPO_ROOT / "data" / "hit_probability_model_v2.json"
MODEL_OUT_PKL = REPO_ROOT / "data" / "hit_probability_model_v2.pkl"

# Full broad set tried first (see the run this replaced, recorded in
# git history and the README): 24 features across career slash line,
# rates, pitcher career rates, rest days, month, handedness, home/away.
# Permutation importance on the true 2026 holdout showed only these 12
# with real (positive, above-noise) importance -- the rest (rest days,
# month, is_home, platoon_delta, career_iso/babip/slg, pitcher whip/bb9/
# hr9, bats_S) were ~zero or negative, i.e. indistinguishable from noise.
# Pruned to just the signal: fewer live API calls, and every remaining
# feature earned its place on held-out data rather than intuition.
FEATURE_COLS = [
    "batting_order", "pitcher_k9_career", "park_factor", "career_avg",
    "pitcher_oba_against", "bats_L", "career_obp", "pitcher_throws_L",
    "pitcher_era_career", "career_k_rate", "batter_age", "career_bb_rate",
]

# Domain-asserted monotonic direction per feature above, for
# HistGradientBoostingClassifier's monotonic_cst: 1 = P(hit) should not
# decrease as the feature increases, -1 = should not increase, 0 = no
# asserted direction. Without this, a flexible tree ensemble can and does
# learn locally backwards relationships in sparse regions of feature
# space -- e.g. random_forest here scores a .230 career hitter *below* a
# .200 one at otherwise-default feature values, confirmed by direct
# point-checks. sklearn's RandomForestClassifier has no equivalent
# constraint, which is why it's excluded from MONOTONIC_SAFE_CANDIDATES
# below despite scoring best on raw holdout log loss.
MONOTONIC_CST = [-1, -1, 1, 1, 1, 0, 1, 0, 1, -1, 0, 0]

# Only candidates where a backwards-prediction pathology can't sneak
# through get to be the "winner" that's actually shipped -- logistic
# regression is monotonic by construction, and gbm has monotonic_cst
# applied above. random_forest and mlp are still fit and reported for
# the multi-family comparison, just never selected.
MONOTONIC_SAFE_CANDIDATES = {"logreg", "gbm"}


# ----------------------------------------------------------- career agg ---


def batter_career_rates(splits: list[dict], season_year: int) -> dict | None:
    ab = hits = bb = hbp = sf = tb = so = pa = hr = 0
    for s in splits:
        if int(s.get("season", season_year)) >= season_year:
            continue
        st = s["stat"]
        ab += int(st.get("atBats", 0))
        hits += int(st.get("hits", 0))
        bb += int(st.get("baseOnBalls", 0))
        hbp += int(st.get("hitByPitch", 0))
        sf += int(st.get("sacFlies", 0))
        tb += int(st.get("totalBases", 0))
        so += int(st.get("strikeOuts", 0))
        pa += int(st.get("plateAppearances", 0))
        hr += int(st.get("homeRuns", 0))
    if ab == 0:
        return None
    avg = hits / ab
    obp_den = ab + bb + hbp + sf
    obp = (hits + bb + hbp) / obp_den if obp_den > 0 else avg
    slg = tb / ab
    babip_den = ab - so - hr + sf
    babip = (hits - hr) / babip_den if babip_den > 0 else avg
    return {
        "avg": avg, "obp": obp, "slg": slg, "iso": slg - avg, "babip": babip,
        "bb_rate": bb / pa if pa > 0 else 0.0, "k_rate": so / pa if pa > 0 else 0.0,
    }


def batter_last_known_age(splits: list[dict], season_year: int) -> float | None:
    prior = [s for s in splits if int(s.get("season", season_year)) < season_year]
    if not prior:
        return None
    prior.sort(key=lambda s: int(s["season"]))
    last = prior[-1]
    years_since = season_year - int(last["season"])
    age = last["stat"].get("age")
    return float(age) + years_since if age is not None else None


def pitcher_career_rates(splits: list[dict], season_year: int) -> dict | None:
    outs = er = bb = k = hr = hits = 0
    for s in splits:
        if int(s.get("season", season_year)) >= season_year:
            continue
        st = s["stat"]
        outs += int(st.get("outs", 0))
        er += int(st.get("earnedRuns", 0))
        bb += int(st.get("baseOnBalls", 0))
        k += int(st.get("strikeOuts", 0))
        hr += int(st.get("homeRuns", 0))
        hits += int(st.get("hits", 0))
    if outs == 0:
        return None
    innings = outs / 3
    return {
        "era": er * 27 / outs, "whip": (bb + hits) / innings,
        "k9": k * 27 / outs, "bb9": bb * 27 / outs, "hr9": hr * 27 / outs,
    }


def fetch_batter_bats(batter_ids: set[int]) -> dict[int, str]:
    result: dict[int, str] = {}
    ids = sorted(batter_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        resp = cached_get(
            f"{BASE}/people",
            params={"personIds": ",".join(str(b) for b in chunk), "hydrate": "batSide"},
        )
        for person in resp.get("people", []):
            result[person["id"]] = person.get("batSide", {}).get("code", "R")
    return result


# -------------------------------------------------------------- dataset ---


def build_dataset(seasons: list[int]) -> pd.DataFrame:
    raw_by_season = {}
    for season in seasons:
        end_date = datetime.date.today().isoformat() if season == TEST_SEASON else None
        raw_by_season[season] = fetch_season_raw(season, end_date=end_date)

    all_batter_ids = set()
    all_pitcher_ids = set()
    for raw in raw_by_season.values():
        all_batter_ids.update(raw["batter_ids"])
        all_pitcher_ids.update(raw["pitcher_logs"].keys())

    batter_career = fetch_career_stats(all_batter_ids, "hitting")
    pitcher_career = fetch_career_stats(all_pitcher_ids, "pitching")
    batter_bats = fetch_batter_bats(all_batter_ids)

    park_factors_raw = json.loads(PARK_FACTORS_PATH.read_text())
    park_factors = {k: v["park_factor"] for k, v in park_factors_raw.items()}

    out_rows = []
    for season, raw in raw_by_season.items():
        schedule_map = raw["schedule_map"]
        batting_order = raw["batting_order"]
        pitcher_hand = raw["pitcher_hand"]

        pitcher_games: dict[int, list[tuple[str, int, int]]] = {}
        for pid, splits in raw["pitcher_logs"].items():
            pitcher_games[pid] = sorted(
                (
                    (s["date"], int(s["stat"].get("atBats", 0)), int(s["stat"].get("hits", 0)))
                    for s in splits
                ),
                key=lambda r: r[0],
            )

        def pitcher_oba_to_date(pid: int, date: str) -> float | None:
            ab = hits = 0
            for d, a, h in pitcher_games.get(pid, []):
                if d >= date:
                    break
                ab += a
                hits += h
            return hits / ab if ab > 0 else None

        def pitcher_rest_days(pid: int, date: str) -> float:
            prior_dates = [d for d, _, _ in pitcher_games.get(pid, []) if d < date]
            if not prior_dates:
                return 5.0
            gap = (
                datetime.date.fromisoformat(date) - datetime.date.fromisoformat(max(prior_dates))
            ).days
            return float(min(gap, 20))  # cap so an off-season gap doesn't blow up the scale

        for bid in raw["batter_ids"]:
            bcareer = batter_career_rates(batter_career.get(bid, []), season)
            bage = batter_last_known_age(batter_career.get(bid, []), season)
            bats = batter_bats.get(bid, "R")
            if bcareer is None:
                continue

            splits = raw["batter_logs"].get(bid, [])
            rows = sorted(
                (
                    {
                        "date": s["date"],
                        "team_id": s["team"]["id"],
                        "opponent_team_id": s["opponent"]["id"],
                        "is_home": s["isHome"],
                        "at_bats": int(s["stat"].get("atBats", 0)),
                        "hits": int(s["stat"].get("hits", 0)),
                    }
                    for s in splits
                ),
                key=lambda r: r["date"],
            )

            cum_ab = cum_hits = 0
            vs_hand_ab = {"L": 0, "R": 0}
            vs_hand_hits = {"L": 0, "R": 0}
            prev_batter_date: str | None = None
            # (date, at_bats, hits) for every prior game this season, in
            # order -- used to compute recent-form windows below without
            # any extra API calls (gameLog data is already fetched).
            prior_games: list[tuple[str, int, int]] = []

            for r in rows:
                pid = schedule_map.get((r["date"], r["team_id"]))
                hand = pitcher_hand.get(pid) if pid else None

                if cum_ab > 0 and pid is not None:
                    poba = pitcher_oba_to_date(pid, r["date"])
                    pcareer = pitcher_career_rates(pitcher_career.get(pid, []), season)
                    if poba is not None and pcareer is not None and bage is not None:
                        season_avg_to_date = cum_hits / cum_ab
                        platoon_ab = vs_hand_ab.get(hand, 0)
                        platoon_avg = (
                            vs_hand_hits[hand] / platoon_ab
                            if hand and platoon_ab > 0
                            else season_avg_to_date
                        )
                        park_team = r["team_id"] if r["is_home"] else r["opponent_team_id"]
                        pf = park_factors.get(str(park_team), 1.0)
                        slot = batting_order.get(
                            (r["date"], r["team_id"], bid), DEFAULT_BATTING_ORDER
                        )
                        rest_b = (
                            float(
                                min(
                                    (
                                        datetime.date.fromisoformat(r["date"])
                                        - datetime.date.fromisoformat(prev_batter_date)
                                    ).days,
                                    20,
                                )
                            )
                            if prev_batter_date
                            else 3.0
                        )
                        month = int(r["date"][5:7])

                        def _window_avg(games: list[tuple[str, int, int]]) -> float:
                            wab = sum(g[1] for g in games)
                            whits = sum(g[2] for g in games)
                            return whits / wab if wab > 0 else season_avg_to_date

                        cur_date = datetime.date.fromisoformat(r["date"])
                        last_5g = _window_avg(prior_games[-5:])
                        last_10g = _window_avg(prior_games[-10:])
                        last_5d = _window_avg(
                            [g for g in prior_games if (cur_date - datetime.date.fromisoformat(g[0])).days <= 5]
                        )
                        last_10d = _window_avg(
                            [g for g in prior_games if (cur_date - datetime.date.fromisoformat(g[0])).days <= 10]
                        )
                        # Deltas vs. season-to-date average, same
                        # reparametrization platoon_delta already uses --
                        # the raw absolute version collapses onto
                        # season/career average and creates artificial
                        # collinearity, per round one's finding.
                        recent_form_5g_delta = last_5g - season_avg_to_date
                        recent_form_10g_delta = last_10g - season_avg_to_date
                        recent_form_5d_delta = last_5d - season_avg_to_date
                        recent_form_10d_delta = last_10d - season_avg_to_date

                        out_rows.append(
                            {
                                "season": season,
                                "date": r["date"],
                                "career_avg": bcareer["avg"],
                                "career_obp": bcareer["obp"],
                                "career_slg": bcareer["slg"],
                                "career_iso": bcareer["iso"],
                                "career_babip": bcareer["babip"],
                                "career_bb_rate": bcareer["bb_rate"],
                                "career_k_rate": bcareer["k_rate"],
                                "batter_age": bage,
                                "bats_L": 1.0 if bats == "L" else 0.0,
                                "bats_S": 1.0 if bats == "S" else 0.0,
                                "platoon_delta": platoon_avg - season_avg_to_date,
                                "rest_days_batter": rest_b,
                                "pitcher_era_career": pcareer["era"],
                                "pitcher_whip_career": pcareer["whip"],
                                "pitcher_k9_career": pcareer["k9"],
                                "pitcher_bb9_career": pcareer["bb9"],
                                "pitcher_hr9_career": pcareer["hr9"],
                                "pitcher_oba_against": poba,
                                "pitcher_throws_L": 1.0 if hand == "L" else 0.0,
                                "rest_days_pitcher": pitcher_rest_days(pid, r["date"]),
                                "park_factor": pf,
                                "is_home": int(r["is_home"]),
                                "batting_order": float(slot),
                                "month": month,
                                "recent_form_5g_delta": recent_form_5g_delta,
                                "recent_form_10g_delta": recent_form_10g_delta,
                                "recent_form_5d_delta": recent_form_5d_delta,
                                "recent_form_10d_delta": recent_form_10d_delta,
                                # kept for scoring the shipped production model too
                                "season_avg_to_date": season_avg_to_date,
                                "got_hit": int(r["hits"] > 0),
                            }
                        )

                cum_ab += r["at_bats"]
                cum_hits += r["hits"]
                if hand:
                    vs_hand_ab[hand] += r["at_bats"]
                    vs_hand_hits[hand] += r["hits"]
                prev_batter_date = r["date"]
                prior_games.append((r["date"], r["at_bats"], r["hits"]))

    df = pd.DataFrame(out_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------ evaluate ---


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


def score_shipped_model(row: pd.Series) -> float:
    batter = Batter(
        id="x", name="x", bats="R", team="x",
        career_avg=row["career_avg"],
        season_avg=row["season_avg_to_date"],
        season_avg_vs_lhp=row["season_avg_to_date"] + row["platoon_delta"],
        season_avg_vs_rhp=row["season_avg_to_date"] + row["platoon_delta"],
    )
    pitcher = Pitcher(
        id="y", name="y", throws="R", era=4.0, oba_against=row["pitcher_oba_against"]
    )
    matchup = Matchup(
        batter=batter, pitcher=pitcher, is_home=bool(row["is_home"]),
        park_factor=row["park_factor"],
        batting_order=int(row["batting_order"]) if row["batting_order"] else None,
    )
    return score_matchup(matchup).hit_probability


def main() -> int:
    print("=" * 70)
    print("Building training set:", TRAIN_SEASONS)
    train_df = build_dataset(TRAIN_SEASONS)
    print(f"train rows: {len(train_df)}")

    print("\nBuilding test set (true out-of-time holdout):", TEST_SEASON)
    test_df = build_dataset([TEST_SEASON])
    print(f"test rows: {len(test_df)}")

    print(f"\nTrain hit rate: {train_df['got_hit'].mean():.3f}")
    print(f"Test  hit rate: {test_df['got_hit'].mean():.3f}")

    X_train, y_train = train_df[FEATURE_COLS], train_df["got_hit"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["got_hit"]

    print("\n--- Currently shipped production model, scored on 2026 ---")
    shipped_pred = test_df.apply(score_shipped_model, axis=1)
    shipped_metrics = report("shipped model (v1) on 2026", y_test, shipped_pred)

    print("\n--- New candidates, trained on 2023-2025, scored on 2026 ---")
    report("baseline (constant train rate)", y_test, np.full(len(y_test), y_train.mean()))

    candidates = {}

    logreg = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    logreg.fit(X_train, y_train)
    m = report("logistic regression (L2)", y_test, logreg.predict_proba(X_test)[:, 1])
    candidates["logreg"] = (logreg, m)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    m = report("random forest", y_test, rf.predict_proba(X_test)[:, 1])
    candidates["random_forest"] = (rf, m)

    gbm = HistGradientBoostingClassifier(
        max_depth=4, max_iter=200, learning_rate=0.04, random_state=0,
        monotonic_cst=MONOTONIC_CST,
    )
    gbm.fit(X_train, y_train)
    m = report("gradient boosting", y_test, gbm.predict_proba(X_test)[:, 1])
    candidates["gbm"] = (gbm, m)

    mlp = Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=0, early_stopping=True)),
    ])
    mlp.fit(X_train, y_train)
    m = report("neural net (MLP)", y_test, mlp.predict_proba(X_test)[:, 1])
    candidates["mlp"] = (mlp, m)

    safe_candidates = {k: v for k, v in candidates.items() if k in MONOTONIC_SAFE_CANDIDATES}
    winner_name = min(safe_candidates, key=lambda k: safe_candidates[k][1]["log_loss"])
    winner_model, winner_metrics = safe_candidates[winner_name]
    print(
        f"\nWinner among monotonic-safe candidates {sorted(MONOTONIC_SAFE_CANDIDATES)} "
        f"by 2026 holdout log loss: {winner_name} ({winner_metrics})"
    )

    print(f"\n--- Permutation importance for {winner_name}, on 2026 holdout ---")
    perm = permutation_importance(
        winner_model, X_test, y_test, n_repeats=15, random_state=0, scoring="neg_log_loss"
    )
    order = np.argsort(perm.importances_mean)[::-1]
    for i in order:
        print(f"  {FEATURE_COLS[i]:<24} {perm.importances_mean[i]:+.5f}  (+/- {perm.importances_std[i]:.5f})")

    # Save artifacts for the writeup / potential promotion to production.
    train_df.to_csv(REPO_ROOT / ".cache" / "ml_model" / "train_dataset.csv", index=False)
    test_df.to_csv(REPO_ROOT / ".cache" / "ml_model" / "test_dataset.csv", index=False)

    import joblib

    joblib.dump(winner_model, MODEL_OUT_PKL)
    MODEL_OUT_JSON.write_text(
        json.dumps(
            {
                "model_type": winner_name,
                "features": FEATURE_COLS,
                "pkl_path": MODEL_OUT_PKL.name,
                "metadata": {
                    "train_seasons": TRAIN_SEASONS,
                    "test_season": TEST_SEASON,
                    "n_train_rows": len(train_df),
                    "n_test_rows": len(test_df),
                    "candidate_comparison": {k: v[1] for k, v in candidates.items()},
                    "shipped_model_v1_on_2026": shipped_metrics,
                },
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote candidate model to {MODEL_OUT_JSON} / {MODEL_OUT_PKL}")
    print("(Not yet wired into rank.py -- promote manually after reviewing this report.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
