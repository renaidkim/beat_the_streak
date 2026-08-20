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
    # In-season average vs. each pitcher hand. Not model features on their
    # own -- see rank.py -- used only as the reference point for platoon
    # delta (today's-hand average minus in-season overall average).
    season_avg: float
    season_avg_vs_lhp: float
    season_avg_vs_rhp: float


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
