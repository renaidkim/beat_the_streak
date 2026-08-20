"""Turn a raw Matchup into the handful of signals the ranker consumes."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Matchup

# Rough MLB league averages, used as priors/fallbacks and as the
# reference points _explain() compares a matchup against.
LEAGUE_AVG_AVG = 0.245
LEAGUE_AVG_OBP = 0.320
LEAGUE_AVG_PITCHER_K9 = 8.5

# Imputed when a matchup's batting order slot isn't known (future days,
# or today's lineup not posted yet) -- middle of the order, roughly
# neutral. Matches the fallback used when the model was trained (see
# scripts/train_ml_model.py), so live behavior on an unknown slot
# matches what was actually validated.
DEFAULT_BATTING_ORDER = 5.0


@dataclass(frozen=True)
class BatterFeatures:
    career_avg: float
    career_obp: float
    career_k_rate: float
    career_bb_rate: float
    batter_age: float
    bats_L: float  # 1.0 if the batter hits left-handed, else 0.0
    pitcher_oba_against: float  # opposing pitcher's season-to-date average against
    pitcher_era_career: float
    pitcher_k9_career: float
    pitcher_throws_L: float  # 1.0 if the opposing pitcher throws left-handed, else 0.0
    park_factor: float
    batting_order: float  # 1-9, or DEFAULT_BATTING_ORDER when unknown


def build_features(matchup: Matchup) -> BatterFeatures:
    return BatterFeatures(
        career_avg=matchup.batter.career_avg,
        career_obp=matchup.batter.career_obp,
        career_k_rate=matchup.batter.career_k_rate,
        career_bb_rate=matchup.batter.career_bb_rate,
        batter_age=matchup.batter.age,
        bats_L=1.0 if matchup.batter.bats == "L" else 0.0,
        pitcher_oba_against=matchup.pitcher.oba_against,
        pitcher_era_career=matchup.pitcher.era_career,
        pitcher_k9_career=matchup.pitcher.k9_career,
        pitcher_throws_L=1.0 if matchup.pitcher.throws == "L" else 0.0,
        park_factor=matchup.park_factor,
        batting_order=float(matchup.batting_order)
        if matchup.batting_order is not None
        else DEFAULT_BATTING_ORDER,
    )
