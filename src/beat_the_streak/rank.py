"""Score matchups by estimated probability of getting a hit that game, and
pick the best options for a Beat the Streak entry.

The model is a logistic regression fit directly on real 2025-season
outcomes (see scripts/fit_hit_probability_model.py and the README's
"Model backtest" section) predicting P(batter gets >=1 hit in this game)
from: season average to date, last-10-games form, platoon delta (average
vs. today's opposing pitcher hand, minus season average), the opposing
pitcher's own season-to-date average-against, park factor, and home/away.
Coefficients live in data/hit_probability_model.json, refreshed by that
script the same way data/park_factors.json is.

This replaced an earlier hand-tuned weighted-blend heuristic after
backtesting showed that heuristic was overconfident enough to score worse
than a naive "always predict the league average" baseline on proper
scoring rules (log loss, Brier), despite ranking picks in roughly the
right order (AUC modestly above chance). The fitted model is both better
calibrated and simpler: it predicts the per-game probability directly,
with no need for a separate per-at-bat-to-per-game conversion step.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .features import LEAGUE_AVG_AVG, BatterFeatures, build_features
from .models import Matchup, Pick

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "hit_probability_model.json"

# Safety bounds on the final probability: a linear model can extrapolate
# poorly for combinations of features more extreme than anything in the
# training data (e.g. a career year at Coors Field against a rookie's
# worst possible matchup). Sigmoid already bounds to (0, 1); this keeps
# the output within a range that's plausible for a single game.
MIN_HIT_PROB = 0.05
MAX_HIT_PROB = 0.95


def _load_model() -> dict:
    raw = json.loads(MODEL_PATH.read_text())
    return raw


_MODEL = _load_model()
_FEATURES: list[str] = _MODEL["features"]
_COEFFICIENTS: dict[str, float] = _MODEL["coefficients"]
_INTERCEPT: float = _MODEL["intercept"]


def score_matchup(matchup: Matchup) -> Pick:
    features = build_features(matchup)
    platoon_delta = features.platoon_avg - features.season_avg

    row = {
        "season_avg_to_date": features.season_avg,
        "recent_form_avg": features.recent_form_avg,
        "platoon_delta": platoon_delta,
        "pitcher_oba_against": features.pitcher_oba_against,
        "park_factor": features.park_factor,
        "is_home": 1.0 if features.is_home else 0.0,
    }
    linear = _INTERCEPT + sum(_COEFFICIENTS[name] * row[name] for name in _FEATURES)
    hit_probability = 1.0 / (1.0 + math.exp(-linear))
    hit_probability = min(max(hit_probability, MIN_HIT_PROB), MAX_HIT_PROB)

    reasons = _explain(features, platoon_delta, matchup.pitcher.confirmed)
    return Pick(matchup=matchup, hit_probability=hit_probability, reasons=reasons)


def _explain(features: BatterFeatures, platoon_delta: float, pitcher_confirmed: bool) -> list[str]:
    """Human-readable reasons behind the score. Purely descriptive -- these
    thresholds don't feed the model, they just surface which signals stand
    out for this matchup.
    """
    reasons: list[str] = []

    if not pitcher_confirmed:
        reasons.append("opposing starter not yet announced -- using a league-average assumption")

    if features.recent_form_avg - features.season_avg >= 0.040:
        reasons.append(
            f"hot recent form (.{round(features.recent_form_avg * 1000):03d} "
            f"last {features.games_of_recent_data})"
        )
    elif features.season_avg - features.recent_form_avg >= 0.040:
        reasons.append(
            f"cold recent form (.{round(features.recent_form_avg * 1000):03d} "
            f"last {features.games_of_recent_data})"
        )

    if platoon_delta >= 0.020:
        reasons.append(f"favorable platoon split ({platoon_delta:+.3f} vs. season avg)")
    elif platoon_delta <= -0.020:
        reasons.append(f"unfavorable platoon split ({platoon_delta:+.3f} vs. season avg)")

    if features.pitcher_oba_against - LEAGUE_AVG_AVG >= 0.015:
        reasons.append(f"contact-prone pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")
    elif LEAGUE_AVG_AVG - features.pitcher_oba_against >= 0.015:
        reasons.append(f"tough pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")

    if features.park_factor >= 1.05:
        reasons.append(f"hitter-friendly park (factor {features.park_factor:.2f})")
    elif features.park_factor <= 0.95:
        reasons.append(f"pitcher-friendly park (factor {features.park_factor:.2f})")

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
    ranked_picks: list[Pick], n: int = 2, avoid_same_game: bool = True
) -> PickSet:
    """Select the top `n` picks, optionally avoiding two batters who share
    the same opposing pitcher (their outcomes are correlated: one bad
    pitching performance sinks both picks at once).
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
