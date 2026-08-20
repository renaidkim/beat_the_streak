from __future__ import annotations

from beat_the_streak.models import Batter, Matchup, Pitcher, RecentForm


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


def make_recent_form(hit_pattern: list[int], at_bats_per_game: int = 4) -> RecentForm:
    """Build a RecentForm aggregate from a per-game hit pattern, e.g.
    [1, 0, 1] = 3 games, one hit in games 1 and 3, none in game 2.
    """
    return RecentForm(
        games_played=len(hit_pattern),
        at_bats=len(hit_pattern) * at_bats_per_game,
        hits=sum(hit_pattern),
    )


def make_matchup(**overrides) -> Matchup:
    defaults = dict(
        batter=make_batter(),
        pitcher=make_pitcher(),
        is_home=True,
        park_factor=1.0,
        recent_form=make_recent_form([1, 1, 0, 1, 0, 1, 1, 0, 1, 1]),
    )
    defaults.update(overrides)
    return Matchup(**defaults)
