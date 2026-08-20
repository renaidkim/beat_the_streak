"""Score matchups by estimated probability of getting a hit that game, and
pick the best options for a Beat the Streak entry.

The model (data/hit_probability_model.json / .pkl) predicts P(batter gets
>=1 hit in this game) from 12 features: the batter's slot in the batting
order, the opposing pitcher's career strikeouts-per-9 and career ERA
(both entering this season, i.e. prior seasons only) and season-to-date
average against, park factor, the batter's career average/OBP/strikeout
rate/walk rate (career, entering this season), age, and both players'
handedness.

These 12 were chosen by permutation importance out of a broader ~24-
feature set (scripts/train_ml_model.py) that also tried platoon split,
home/away, rest days since last game, month of season, and several more
career rate stats for both sides -- all came back with ~zero or negative
importance on a true out-of-time holdout and were cut. The model is
trained on 2023-2025 and validated on 2026, a season it never saw during
fitting or feature selection -- see the README's "Model backtest"
section for the full history, including the earlier 6-feature version
this replaced.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib

from .features import LEAGUE_AVG_AVG, LEAGUE_AVG_OBP, LEAGUE_AVG_PITCHER_K9, BatterFeatures, build_features
from .models import Matchup, Pick

MODEL_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "hit_probability_model.json"
MODEL_PKL_PATH = Path(__file__).resolve().parents[2] / "data" / "hit_probability_model.pkl"

# Safety bounds on the final probability: a model can extrapolate poorly
# for combinations of features more extreme than anything in the training
# data (e.g. a career year at Coors Field against a rookie's worst
# possible matchup). This keeps the output within a range that's
# plausible for a single game regardless of what the model itself outputs.
MIN_HIT_PROB = 0.05
MAX_HIT_PROB = 0.95


def _load_predictor():
    meta = json.loads(MODEL_JSON_PATH.read_text())
    features = meta["features"]

    if meta["model_type"] == "logreg":
        coefficients = meta["coefficients"]
        intercept = meta["intercept"]

        def predict(row: dict) -> float:
            linear = intercept + sum(coefficients[name] * row[name] for name in features)
            return 1.0 / (1.0 + math.exp(-linear))

    else:
        # Any other model_type is a scikit-learn classifier with
        # predict_proba, serialized to the .pkl -- gradient boosting,
        # random forest, whatever scripts/train_ml_model.py picked as the
        # winner by holdout log loss. model_type is just a label at this
        # point; the object itself decides behavior.
        sk_model = joblib.load(MODEL_PKL_PATH)

        def predict(row: dict) -> float:
            x = [[row[name] for name in features]]
            # sklearn warns that a plain array has no column names to
            # check against the ones it was fitted with -- order is
            # already guaranteed correct here (features comes straight
            # from the same JSON the model was trained against), so
            # there's nothing this warning would catch.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return float(sk_model.predict_proba(x)[0][1])

    return predict


_predict = _load_predictor()


def score_matchup(matchup: Matchup) -> Pick:
    features = build_features(matchup)
    row = {
        "career_avg": features.career_avg,
        "career_obp": features.career_obp,
        "career_k_rate": features.career_k_rate,
        "career_bb_rate": features.career_bb_rate,
        "batter_age": features.batter_age,
        "bats_L": features.bats_L,
        "pitcher_oba_against": features.pitcher_oba_against,
        "pitcher_era_career": features.pitcher_era_career,
        "pitcher_k9_career": features.pitcher_k9_career,
        "pitcher_throws_L": features.pitcher_throws_L,
        "park_factor": features.park_factor,
        "batting_order": features.batting_order,
    }
    hit_probability = _predict(row)
    hit_probability = min(max(hit_probability, MIN_HIT_PROB), MAX_HIT_PROB)

    reasons = _explain(features, matchup.pitcher.confirmed)
    return Pick(matchup=matchup, hit_probability=hit_probability, reasons=reasons)


def _explain(features: BatterFeatures, pitcher_confirmed: bool) -> list[str]:
    """Human-readable reasons behind the score. Purely descriptive -- these
    thresholds don't feed the model, they just surface which of the
    model's own inputs stand out for this matchup. Only features the
    model actually uses are described here (see build_features/rank.py's
    docstring) -- nothing gets a reason string it isn't backed by.
    """
    reasons: list[str] = []

    if not pitcher_confirmed:
        reasons.append("opposing starter not yet announced -- using a league-average assumption")

    if features.career_avg - LEAGUE_AVG_AVG >= 0.020:
        reasons.append(f"strong career hitter (.{round(features.career_avg * 1000):03d} career avg)")
    elif LEAGUE_AVG_AVG - features.career_avg >= 0.020:
        reasons.append(f"light-hitting career average (.{round(features.career_avg * 1000):03d})")

    if features.career_obp - LEAGUE_AVG_OBP >= 0.020:
        reasons.append(f"strong on-base skill (.{round(features.career_obp * 1000):03d} career OBP)")
    elif LEAGUE_AVG_OBP - features.career_obp >= 0.020:
        reasons.append(f"below-average on-base skill (.{round(features.career_obp * 1000):03d} career OBP)")

    if features.pitcher_oba_against - LEAGUE_AVG_AVG >= 0.015:
        reasons.append(f"contact-prone pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")
    elif LEAGUE_AVG_AVG - features.pitcher_oba_against >= 0.015:
        reasons.append(f"tough pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")

    if features.pitcher_k9_career - LEAGUE_AVG_PITCHER_K9 >= 1.5:
        reasons.append(f"high-strikeout pitcher ({features.pitcher_k9_career:.1f} career K/9)")
    elif LEAGUE_AVG_PITCHER_K9 - features.pitcher_k9_career >= 1.5:
        reasons.append(f"low-strikeout, contact-oriented pitcher ({features.pitcher_k9_career:.1f} career K/9)")

    if features.park_factor >= 1.05:
        reasons.append(f"hitter-friendly park (factor {features.park_factor:.2f})")
    elif features.park_factor <= 0.95:
        reasons.append(f"pitcher-friendly park (factor {features.park_factor:.2f})")

    if features.batting_order <= 2:
        reasons.append(f"batting near the top of the order ({int(features.batting_order)})")
    elif features.batting_order >= 8:
        reasons.append(f"batting near the bottom of the order ({int(features.batting_order)})")

    return reasons


def rank_matchups(matchups: list[Matchup]) -> list[Pick]:
    picks = [score_matchup(m) for m in matchups]
    picks.sort(key=lambda p: p.hit_probability, reverse=True)
    return picks


@dataclass(frozen=True)
class PickSet:
    picks: list[Pick]
    skipped_for_correlation: list[Pick]


def pick_top(
    ranked_picks: list[Pick], n: int = 2, avoid_same_game: bool = False
) -> PickSet:
    """Select the top `n` picks.

    `avoid_same_game=True` skips a pick if it shares an opposing pitcher
    with one already selected. Off by default -- MLB's actual "double
    down" rule requires *both* picks to get a hit for the day to count
    (and rewards it by advancing the streak by 2), not just one. For a
    both-must-succeed bet, positive correlation between the two outcomes
    *raises* the joint success probability rather than lowering it: two
    batters facing the same pitcher rise and fall together with that
    pitcher's performance (the same logic daily-fantasy players use to
    "stack" hitters against a bad starter), so P(both hit) can run
    several points above what the same two marginal probabilities would
    give if independent. Avoiding that correlation only makes sense for
    an at-least-one-of-two bet, which isn't how this game scores a double
    down. Left in as an opt-in for anyone who wants to diversify anyway.
    """
    selected: list[Pick] = []
    skipped: list[Pick] = []
    used_pitcher_ids: set[str] = set()

    for pick in ranked_picks:
        if len(selected) >= n:
            break
        pitcher_id = pick.matchup.pitcher.id
        if avoid_same_game and pitcher_id in used_pitcher_ids:
            skipped.append(pick)
            continue
        selected.append(pick)
        used_pitcher_ids.add(pitcher_id)

    # If avoiding correlation left us short (e.g. a tiny slate), fill the
    # rest from the highest-probability remaining picks regardless.
    if len(selected) < n:
        for pick in ranked_picks:
            if len(selected) >= n:
                break
            if pick not in selected:
                selected.append(pick)

    return PickSet(picks=selected, skipped_for_correlation=skipped)
