from __future__ import annotations

from beat_the_streak.data import MlbStatsApiSource, _career_avg_entering_season, _placeholder_pitcher
from beat_the_streak.features import LEAGUE_AVG_AVG


def test_placeholder_pitcher_is_unconfirmed_and_league_average():
    pitcher = _placeholder_pitcher("tbd:12345:away")
    assert pitcher.confirmed is False
    assert pitcher.oba_against == LEAGUE_AVG_AVG
    assert pitcher.id == "tbd:12345:away"


def test_career_avg_excludes_current_season():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [
                    {"season": "2023", "stat": {"atBats": 400, "hits": 100}},
                    {"season": "2024", "stat": {"atBats": 500, "hits": 150}},
                    {"season": "2025", "stat": {"atBats": 300, "hits": 120}},  # current season, excluded
                ],
            }
        ]
    }
    # (100 + 150) / (400 + 500), not including the 2025 row
    assert _career_avg_entering_season(person, 2025) == (100 + 150) / (400 + 500)


def test_career_avg_none_when_no_prior_seasons():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [{"season": "2025", "stat": {"atBats": 50, "hits": 12}}],
            }
        ]
    }
    assert _career_avg_entering_season(person, 2025) is None


def test_starting_batters_uses_boxscore_batting_order_as_slot():
    source = MlbStatsApiSource()
    boxscore = {"teams": {"home": {"battingOrder": [111, 222, 333]}}}
    lineup = source._get_starting_batters(boxscore, "home", team_id=1)
    assert lineup == {111: 1, 222: 2, 333: 3}
