"""Turn a raw Matchup into the handful of signals the ranker consumes."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Matchup

LEAGUE_AVG_AVG = 0.245  # rough MLB league-average batting average, used as a prior
RECENT_FORM_WINDOW = 10  # games


@dataclass(frozen=True)
class BatterFeatures:
    season_avg: float
    recent_form_avg: float  # AVG over the last RECENT_FORM_WINDOW games
    games_of_recent_data: int
    platoon_avg: float  # season AVG vs today's pitcher's throwing hand
    pitcher_oba_against: float
    park_factor: float
    is_home: bool


def build_features(matchup: Matchup) -> BatterFeatures:
    form = matchup.recent_form
    recent_form_avg = form.avg if form.avg is not None else matchup.batter.season_avg

    platoon_avg = (
        matchup.batter.season_avg_vs_lhp
        if matchup.pitcher.throws == "L"
        else matchup.batter.season_avg_vs_rhp
    )

    return BatterFeatures(
        season_avg=matchup.batter.season_avg,
        recent_form_avg=recent_form_avg,
        games_of_recent_data=form.games_played,
        platoon_avg=platoon_avg,
        pitcher_oba_against=matchup.pitcher.oba_against,
        park_factor=matchup.park_factor,
        is_home=matchup.is_home,
    )
