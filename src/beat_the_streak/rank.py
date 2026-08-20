"""Score matchups by estimated probability of getting a hit that game, and
pick the best options for a Beat the Streak entry.

The estimate has two stages:

1. Blend season average, recent form, the platoon (vs-hand) split, and the
   opposing pitcher's average-against into a single per-at-bat hit
   probability, then adjust for park.
2. Convert that per-at-bat probability into a per-game probability using
   the standard "at least one hit in N at-bats" formula:
   P(hit in game) = 1 - (1 - p_per_ab) ** expected_at_bats.

This is a transparent heuristic, not a fitted model. It's a reasonable
starting point and a natural place to later swap in a model trained on
historical outcomes without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import LEAGUE_AVG_AVG, BatterFeatures, build_features
from .models import Matchup, Pick

EXPECTED_AT_BATS_PER_GAME = 4.0
MIN_RECENT_GAMES_FOR_FULL_WEIGHT = 5

WEIGHTS = {
    "recent_form": 0.35,
    "season_avg": 0.25,
    "platoon": 0.20,
    "pitcher": 0.20,
}

# Per-at-bat probability floor/ceiling to keep the game-level estimate sane
# even with small samples or unusual inputs.
MIN_PER_AB_PROB = 0.05
MAX_PER_AB_PROB = 0.45


def score_matchup(matchup: Matchup) -> Pick:
    features = build_features(matchup)
    per_ab_prob, reasons = _per_ab_hit_probability(features)
    per_ab_prob *= features.park_factor
    per_ab_prob = min(max(per_ab_prob, MIN_PER_AB_PROB), MAX_PER_AB_PROB)

    if features.park_factor >= 1.05:
        reasons.append(f"hitter-friendly park (factor {features.park_factor:.2f})")
    elif features.park_factor <= 0.95:
        reasons.append(f"pitcher-friendly park (factor {features.park_factor:.2f})")

    game_prob = 1 - (1 - per_ab_prob) ** EXPECTED_AT_BATS_PER_GAME
    return Pick(matchup=matchup, hit_probability=game_prob, reasons=reasons)


def _per_ab_hit_probability(features: BatterFeatures) -> tuple[float, list[str]]:
    reasons: list[str] = []
    weights = dict(WEIGHTS)

    # Down-weight recent form when the sample is thin (early season, call-up,
    # etc.) and redistribute that weight onto season average.
    if features.games_of_recent_data < MIN_RECENT_GAMES_FOR_FULL_WEIGHT:
        shrink = features.games_of_recent_data / MIN_RECENT_GAMES_FOR_FULL_WEIGHT
        recovered = weights["recent_form"] * (1 - shrink)
        weights["recent_form"] *= shrink
        weights["season_avg"] += recovered

    per_ab_prob = (
        weights["recent_form"] * features.recent_form_avg
        + weights["season_avg"] * features.season_avg
        + weights["platoon"] * features.platoon_avg
        + weights["pitcher"] * features.pitcher_oba_against
    )

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

    if features.platoon_avg - features.season_avg >= 0.020:
        reasons.append(f"favorable platoon split (.{round(features.platoon_avg * 1000):03d})")

    if features.pitcher_oba_against - LEAGUE_AVG_AVG >= 0.015:
        reasons.append(f"contact-prone pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")
    elif LEAGUE_AVG_AVG - features.pitcher_oba_against >= 0.015:
        reasons.append(f"tough pitcher (.{round(features.pitcher_oba_against * 1000):03d} against)")

    return per_ab_prob, reasons


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
