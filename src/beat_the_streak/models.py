"""Core data types shared across the data, feature, and ranking layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pitcher:
    id: str
    name: str
    throws: str  # "L" or "R"
    era: float
    oba_against: float  # opponent batting average against, all batters
    # Career rates entering this season (prior seasons only, no lookahead
    # -- see beat_the_streak.data._career_pitching_rates_entering_season).
    # Defaults are rough league averages, used only when a caller doesn't
    # have real career data (e.g. hand-built test fixtures).
    era_career: float = 4.30
    k9_career: float = 8.5
    # False for a same-team placeholder used when a game's starter hasn't
    # been announced yet (common a few days out) -- see MlbStatsApiSource.
    confirmed: bool = True


@dataclass(frozen=True)
class Batter:
    id: str
    name: str
    bats: str  # "L", "R", or "S" (switch)
    team: str
    career_avg: float  # career batting average entering this season (no in-season data)
    # In-season average vs. each pitcher hand. Not a model feature on its
    # own -- kept only for backward-compatible fixtures/display; the
    # shipped model no longer uses platoon split (see rank.py docstring).
    season_avg: float
    season_avg_vs_lhp: float
    season_avg_vs_rhp: float
    # Career on-base/strikeout/walk rate and age, all entering this season
    # (no lookahead). Defaults are rough league averages.
    career_obp: float = 0.320
    career_k_rate: float = 0.22
    career_bb_rate: float = 0.08
    age: float = 27.0


@dataclass(frozen=True)
class Matchup:
    """One batter's context for a single day's slate."""

    batter: Batter
    pitcher: Pitcher
    is_home: bool
    park_factor: float  # 1.0 = neutral, >1 favors hitters
    # 1-9, from the confirmed pre-game batting order. None when unknown
    # (future days, or today's lineup not posted yet) -- rank.py imputes
    # a neutral default rather than requiring this.
    batting_order: int | None = None


@dataclass(frozen=True)
class Pick:
    matchup: Matchup
    hit_probability: float
    reasons: list[str]
