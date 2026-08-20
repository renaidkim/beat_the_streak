from __future__ import annotations

from beat_the_streak.data import (
    MlbStatsApiSource,
    _career_hitting_rates_entering_season,
    _career_pitching_rates_entering_season,
    _game_has_started,
    _placeholder_pitcher,
)
from beat_the_streak.features import LEAGUE_AVG_AVG


def test_placeholder_pitcher_is_unconfirmed_and_league_average():
    pitcher = _placeholder_pitcher("tbd:12345:away")
    assert pitcher.confirmed is False
    assert pitcher.oba_against == LEAGUE_AVG_AVG
    assert pitcher.id == "tbd:12345:away"


def test_career_hitting_rates_excludes_current_season():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [
                    {
                        "season": "2023",
                        "stat": {"atBats": 400, "hits": 100, "baseOnBalls": 40, "plateAppearances": 450, "strikeOuts": 80},
                    },
                    {
                        "season": "2024",
                        "stat": {"atBats": 500, "hits": 150, "baseOnBalls": 60, "plateAppearances": 570, "strikeOuts": 100},
                    },
                    {  # current season, excluded
                        "season": "2025",
                        "stat": {"atBats": 300, "hits": 120, "baseOnBalls": 30, "plateAppearances": 340, "strikeOuts": 50},
                    },
                ],
            }
        ]
    }
    rates = _career_hitting_rates_entering_season(person, 2025)
    # (100 + 150) / (400 + 500), not including the 2025 row
    assert rates["avg"] == (100 + 150) / (400 + 500)
    assert rates["k_rate"] == (80 + 100) / (450 + 570)
    assert rates["bb_rate"] == (40 + 60) / (450 + 570)


def test_career_hitting_rates_none_when_no_prior_seasons():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [{"season": "2025", "stat": {"atBats": 50, "hits": 12}}],
            }
        ]
    }
    assert _career_hitting_rates_entering_season(person, 2025) is None


def test_career_pitching_rates_excludes_current_season():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [
                    {"season": "2023", "stat": {"outs": 540, "earnedRuns": 70, "strikeOuts": 160}},
                    {"season": "2025", "stat": {"outs": 300, "earnedRuns": 40, "strikeOuts": 90}},  # current, excluded
                ],
            }
        ]
    }
    rates = _career_pitching_rates_entering_season(person, 2025)
    assert rates["era"] == 70 * 27 / 540
    assert rates["k9"] == 160 * 27 / 540


def test_career_pitching_rates_none_when_no_prior_seasons():
    person = {
        "stats": [
            {
                "type": {"displayName": "yearByYear"},
                "splits": [{"season": "2025", "stat": {"outs": 100, "earnedRuns": 10, "strikeOuts": 30}}],
            }
        ]
    }
    assert _career_pitching_rates_entering_season(person, 2025) is None


def test_game_has_started_false_for_preview():
    assert _game_has_started({"status": {"abstractGameState": "Preview"}}) is False


def test_game_has_started_true_for_live_and_final():
    assert _game_has_started({"status": {"abstractGameState": "Live"}}) is True
    assert _game_has_started({"status": {"abstractGameState": "Final"}}) is True


def test_starting_batters_uses_boxscore_batting_order_as_slot():
    source = MlbStatsApiSource()
    boxscore = {"teams": {"home": {"battingOrder": [111, 222, 333]}}}
    lineup = source._get_starting_batters(boxscore, "home", team_id=1)
    assert lineup == {111: 1, 222: 2, 333: 3}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def get(self, *args, **kwargs):
        return _FakeResponse(self._payload)


def test_bvp_delta_excludes_season_none_aggregate_row():
    # The vsPlayer endpoint's response includes a season=None row that's
    # already the sum of the per-season rows below it (verified directly
    # against the live API) -- summing everything would double-count.
    payload = {
        "stats": [
            {
                "splits": [
                    {"season": None, "stat": {"atBats": 6, "hits": 2}},  # aggregate, excluded
                    {"season": "2018", "stat": {"atBats": 2, "hits": 1}},
                    {"season": "2019", "stat": {"atBats": 1, "hits": 1}},
                    {"season": "2024", "stat": {"atBats": 3, "hits": 0}},
                ]
            }
        ]
    }
    source = MlbStatsApiSource(session=_FakeSession(payload))
    delta = source._get_bvp_delta(batter_id=1, pitcher_id=2, season_avg=0.250)
    # (1 + 1 + 0) / (2 + 1 + 3) - 0.250, not (2+1+1+0)/(6+2+1+3)
    assert delta == (1 + 1 + 0) / (2 + 1 + 3) - 0.250


def test_bvp_delta_zero_when_no_prior_meeting():
    payload = {"stats": [{"splits": []}]}
    source = MlbStatsApiSource(session=_FakeSession(payload))
    delta = source._get_bvp_delta(batter_id=1, pitcher_id=2, season_avg=0.250)
    assert delta == 0.0
