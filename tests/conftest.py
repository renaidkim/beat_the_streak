from __future__ import annotations

from beat_the_streak.models import Batter, Matchup, Pitcher


def make_batter(**overrides) -> Batter:
    defaults = dict(
        id="b1",
        name="Test Batter",
        bats="R",
        team="Test Team",
        career_avg=0.260,
        season_avg=0.260,
        season_avg_vs_lhp=0.260,
        season_avg_vs_rhp=0.260,
        career_obp=0.320,
        career_k_rate=0.22,
        career_bb_rate=0.08,
        age=27.0,
    )
    defaults.update(overrides)
    return Batter(**defaults)


def make_pitcher(**overrides) -> Pitcher:
    defaults = dict(
        id="p1",
        name="Test Pitcher",
        throws="R",
        era=4.00,
        oba_against=0.250,
        era_career=4.00,
        k9_career=8.5,
    )
    defaults.update(overrides)
    return Pitcher(**defaults)


def make_matchup(**overrides) -> Matchup:
    defaults = dict(
        batter=make_batter(),
        pitcher=make_pitcher(),
        is_home=True,
        park_factor=1.0,
        batting_order=5,
    )
    defaults.update(overrides)
    return Matchup(**defaults)
