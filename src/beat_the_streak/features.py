"""Turn a raw Matchup into the handful of signals the ranker consumes."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Matchup

LEAGUE_AVG_AVG = 0.245  # rough MLB league-average batting average, used as a prior

# Imputed when a matchup's batting order slot isn't known (future days,
# or today's lineup not posted yet) -- middle of the order, roughly
# neutral. Matches the fallback used when the model was backtested (see
# scripts/fit_hit_probability_model.py), so live behavior on an unknown
# slot matches what was actually validated.
DEFAULT_BATTING_ORDER = 5.0


@dataclass(frozen=True)
class BatterFeatures:
    career_avg: float
    platoon_delta: float  # today's-pitcher-hand average minus in-season overall average
    pitcher_oba_against: float
    park_factor: float
    is_home: bool
    batting_order: float  # 1-9, or DEFAULT_BATTING_ORDER when unknown


def build_features(matchup: Matchup) -> BatterFeatures:
    platoon_avg = (
        matchup.batter.season_avg_vs_lhp
        if matchup.pitcher.throws == "L"
        else matchup.batter.season_avg_vs_rhp
    )

    return BatterFeatures(
        career_avg=matchup.batter.career_avg,
        platoon_delta=platoon_avg - matchup.batter.season_avg,
        pitcher_oba_against=matchup.pitcher.oba_against,
        park_factor=matchup.park_factor,
        is_home=matchup.is_home,
        batting_order=float(matchup.batting_order)
        if matchup.batting_order is not None
        else DEFAULT_BATTING_ORDER,
    )
