from __future__ import annotations

from beat_the_streak.models import Batter, GameLog, Matchup, Pitcher


def make_batter(**overrides) -> Batter:
    defaults = dict(
        id="b1",
        name="Test Batter",
        bats="R",
        team="Test Team",
        season_avg=0.260,
        season_avg_vs_lhp=0.260,
        season_avg_vs_rhp=0.260,
    )
    defaults.update(overrides)
    return Batter(**defaults)


def make_pitcher(**overrides) -> Pitcher:
    defaults = dict(
        id="p1", name="Test Pitcher", throws="R", era=4.00, oba_against=0.250
    )
    defaults.update(overrides)
    return Pitcher(**defaults)


def make_logs(hit_pattern: list[int], at_bats: int = 4) -> list[GameLog]:
    return [
        GameLog(date=f"2026-08-{i + 1:02d}", at_bats=at_bats, hits=hits)
        for i, hits in enumerate(hit_pattern)
    ]


def make_matchup(**overrides) -> Matchup:
    defaults = dict(
        batter=make_batter(),
        pitcher=make_pitcher(),
        is_home=True,
        park_factor=1.0,
        recent_logs=make_logs([1, 1, 0, 1, 0, 1, 1, 0, 1, 1]),
    )
    defaults.update(overrides)
    return Matchup(**defaults)
