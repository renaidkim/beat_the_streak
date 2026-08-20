"""Score matchups by estimated probability of getting a hit that game, and
pick the best options for a Beat the Streak entry.

The model predicts P(batter gets >=1 hit in this game) from six features:
career batting average entering the season, platoon delta (average vs.
today's opposing pitcher hand, minus in-season overall average), the
opposing pitcher's own season-to-date average-against, park factor,
home/away, and the batter's slot in the batting order (1-9). Coefficients
(or, for the current gradient-boosting winner, a serialized model) live
in data/hit_probability_model.json / .pkl, refreshed by
scripts/fit_hit_probability_model.py the same way data/park_factors.json
is -- see that script's docstring and the README's "Model backtest"
section for the full history and reasoning, including why season average
and last-10-games form were tested and dropped (no independent predictive
power once career average and batting order are in the model) and why
platoon is expressed as a delta rather than an absolute average.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib

from .features import LEAGUE_AVG_AVG, BatterFeatures, build_features
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

    if meta["model_type"] == "gbm":
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

    elif meta["model_type"] == "logreg":
        coefficients = meta["coefficients"]
        intercept = meta["intercept"]

        def predict(row: dict) -> float:
            linear = intercept + sum(coefficients[name] * row[name] for name in features)
            return 1.0 / (1.0 + math.exp(-linear))

    else:
        raise ValueError(f"unknown model_type {meta['model_type']!r} in {MODEL_JSON_PATH}")

    return predict


_predict = _load_predictor()


def score_matchup(matchup: Matchup) -> Pick:
    features = build_features(matchup)
    row = {
        "career_avg": features.career_avg,
        "platoon_delta": features.platoon_delta,
        "pitcher_oba_against": features.pitcher_oba_against,
        "park_factor": features.park_factor,
        "is_home": 1.0 if features.is_home else 0.0,
        "batting_order": features.batting_order,
    }
    hit_probability = _predict(row)
    hit_probability = min(max(hit_probability, MIN_HIT_PROB), MAX_HIT_PROB)

    reasons = _explain(features, matchup.pitcher.confirmed)
    return Pick(matchup=matchup, hit_probability=hit_probability, reasons=reasons)


def _explain(features: BatterFeatures, pitcher_confirmed: bool) -> list[str]:
    """Human-readable reasons behind the score. Purely descriptive -- these
    thresholds don't feed the model, they just surface which signals stand
    out for this matchup.
    """
    reasons: list[str] = []

    if not pitcher_confirmed:
        reasons.append("opposing starter not yet announced -- using a league-average assumption")

    if features.career_avg - LEAGUE_AVG_AVG >= 0.020:
        reasons.append(f"strong career hitter (.{round(features.career_avg * 1000):03d} career avg)")
    elif LEAGUE_AVG_AVG - features.career_avg >= 0.020:
        reasons.append(f"light-hitting career average (.{round(features.career_avg * 1000):03d})")

    if features.platoon_delta >= 0.020:
        reasons.append(f"favorable platoon split ({features.platoon_delta:+.3f} vs. season avg)")
    elif features.platoon_delta <= -0.020:
        reasons.append(f"unfavorable platoon split ({features.platoon_delta:+.3f} vs. season avg)")

    if features.pitcher_oba_against - LEAGUE_AVG_AVG >= 0.015:
        reasons.append(f"contact-prone pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")
    elif LEAGUE_AVG_AVG - features.pitcher_oba_against >= 0.015:
        reasons.append(f"tough pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")

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
