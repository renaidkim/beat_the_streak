"""Core data types shared across the data, feature, and ranking layers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pitcher:
    id: str
    name: str
    throws: str  # "L" or "R"
    era: float
    oba_against: float  # opponent batting average against, all batters


@dataclass(frozen=True)
class GameLog:
    """One game's batting line for a single batter."""

    date: str  # ISO date
    at_bats: int
    hits: int

    @property
    def got_a_hit(self) -> bool:
        return self.hits > 0


@dataclass(frozen=True)
class Batter:
    id: str
    name: str
    bats: str  # "L", "R", or "S" (switch)
    team: str
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
    recent_logs: list[GameLog] = field(default_factory=list)


@dataclass(frozen=True)
class Pick:
    matchup: Matchup
    hit_probability: float
    reasons: list[str]
