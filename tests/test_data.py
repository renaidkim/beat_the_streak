from __future__ import annotations

from beat_the_streak.data import _placeholder_pitcher
from beat_the_streak.features import LEAGUE_AVG_AVG


def test_placeholder_pitcher_is_unconfirmed_and_league_average():
    pitcher = _placeholder_pitcher("tbd:12345:away")
    assert pitcher.confirmed is False
    assert pitcher.oba_against == LEAGUE_AVG_AVG
    assert pitcher.id == "tbd:12345:away"
