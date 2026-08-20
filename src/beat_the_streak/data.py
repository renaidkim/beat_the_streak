"""Data sources for a day's slate.

DataSource is the interface the rest of the app depends on. Two
implementations are provided:

- MlbStatsApiSource: hits the public MLB Stats API (statsapi.mlb.com).
  Validated against live traffic. Uses the boxscore's confirmed batting
  order when it's been posted, falls back to the full active roster when
  it hasn't, and batches player-stat lookups (chunked `/people?personIds=`
  calls) instead of firing one request per batter per game.
- FixtureSource: reads a static JSON fixture from disk. Useful for
  development, tests, and demos without network access.

Swap sources via `--source` on the CLI; both return the same `Matchup`
objects so nothing downstream needs to know which one was used.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from .features import RECENT_FORM_WINDOW
from .models import Batter, Matchup, Pitcher, RecentForm

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"

# How many ids to pack into one batched /people request. The API accepted
# 80 comma-separated ids without complaint in testing; 50 leaves headroom.
PEOPLE_BATCH_SIZE = 50

PARK_FACTORS_PATH = Path(__file__).resolve().parents[2] / "data" / "park_factors.json"


class DataSource(ABC):
    """Everything the ranking pipeline needs for one day's slate."""

    @abstractmethod
    def get_matchups(self, date: str) -> list[Matchup]:
        """Return one Matchup per starting batter expected to play on `date`."""
        raise NotImplementedError


class MlbStatsApiSource(DataSource):
    """Live source backed by the public MLB Stats API."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._roster_cache: dict[int, list[int]] = {}

    def get_matchups(self, date: str) -> list[Matchup]:
        games = self._get_schedule_with_probables(date)

        # Pass 1: work out who's playing and who they're facing, without
        # yet fetching per-player stats, so those can be batched afterward
        # instead of fired one request at a time.
        game_matchups: list[dict[str, Any]] = []
        pitcher_ids: set[int] = set()
        batter_ids: set[int] = set()

        for game in games:
            park_factor = self._get_park_factor(game["teams"]["home"]["team"]["id"])
            boxscore = self._get_boxscore(game["gamePk"])
            for side, opponent_side in (("home", "away"), ("away", "home")):
                team = game["teams"][side]["team"]
                opp_pitcher_info = game["teams"][opponent_side].get("probablePitcher")
                if not opp_pitcher_info:
                    continue
                pitcher_id = opp_pitcher_info["id"]
                lineup = self._get_starting_batters(boxscore, side, team["id"])
                pitcher_ids.add(pitcher_id)
                batter_ids.update(lineup)
                game_matchups.append(
                    {
                        "pitcher_id": pitcher_id,
                        "batter_ids": lineup,
                        "is_home": side == "home",
                        "park_factor": park_factor,
                    }
                )

        batters = self._get_batters(batter_ids)
        pitchers = self._get_pitchers(pitcher_ids)

        matchups: list[Matchup] = []
        for gm in game_matchups:
            pitcher = pitchers.get(gm["pitcher_id"])
            if pitcher is None:
                continue
            for batter_id in gm["batter_ids"]:
                entry = batters.get(batter_id)
                if entry is None:
                    continue
                batter, recent_form = entry
                matchups.append(
                    Matchup(
                        batter=batter,
                        pitcher=pitcher,
                        is_home=gm["is_home"],
                        park_factor=gm["park_factor"],
                        recent_form=recent_form,
                    )
                )
        return matchups

    def _get_schedule_with_probables(self, date: str) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/schedule",
            params={"sportId": 1, "date": date, "hydrate": "probablePitcher,team,venue"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        games: list[dict[str, Any]] = []
        for date_entry in payload.get("dates", []):
            games.extend(date_entry.get("games", []))
        return games

    def _get_boxscore(self, game_pk: int) -> dict[str, Any]:
        resp = self._session.get(f"{MLB_STATS_API_BASE}/game/{game_pk}/boxscore", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _get_starting_batters(
        self, boxscore: dict[str, Any], side: str, team_id: int
    ) -> list[int]:
        batting_order = boxscore.get("teams", {}).get(side, {}).get("battingOrder") or []
        if batting_order:
            return list(batting_order)
        # Lineup not posted yet (common well before game time): fall back
        # to every position player on the active roster rather than
        # skipping the team's batters entirely.
        return self._get_roster_batters(team_id)

    def _get_roster_batters(self, team_id: int) -> list[int]:
        if team_id in self._roster_cache:
            return self._roster_cache[team_id]
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
            params={"rosterType": "active"},
            timeout=15,
        )
        resp.raise_for_status()
        roster = resp.json().get("roster", [])
        batters = [
            entry["person"]["id"]
            for entry in roster
            if entry.get("position", {}).get("abbreviation") != "P"
        ]
        self._roster_cache[team_id] = batters
        return batters

    def _get_batters(self, batter_ids: set[int]) -> dict[int, tuple[Batter, RecentForm]]:
        result: dict[int, tuple[Batter, RecentForm]] = {}
        for chunk in _chunks(sorted(batter_ids), PEOPLE_BATCH_SIZE):
            resp = self._session.get(
                f"{MLB_STATS_API_BASE}/people",
                params={
                    "personIds": ",".join(str(i) for i in chunk),
                    "hydrate": (
                        "currentTeam,"
                        f"stats(group=[hitting],type=[season,lastXGames],limit={RECENT_FORM_WINDOW})"
                    ),
                },
                timeout=20,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                season_stat = _stat_split(person, "season")
                recent_stat = _stat_split(person, "lastXGames")
                season_avg = float(season_stat.get("avg") or ".250")
                batter = Batter(
                    id=str(person["id"]),
                    name=person["fullName"],
                    bats=person.get("batSide", {}).get("code", "R"),
                    team=person.get("currentTeam", {}).get("name", ""),
                    season_avg=season_avg,
                    # No vs-pitcher-hand split endpoint wired up yet; fall
                    # back to season average until that's added.
                    season_avg_vs_lhp=season_avg,
                    season_avg_vs_rhp=season_avg,
                )
                recent_form = RecentForm(
                    games_played=int(recent_stat.get("gamesPlayed", 0)),
                    at_bats=int(recent_stat.get("atBats", 0)),
                    hits=int(recent_stat.get("hits", 0)),
                )
                result[person["id"]] = (batter, recent_form)
        return result

    def _get_pitchers(self, pitcher_ids: set[int]) -> dict[int, Pitcher]:
        result: dict[int, Pitcher] = {}
        for chunk in _chunks(sorted(pitcher_ids), PEOPLE_BATCH_SIZE):
            resp = self._session.get(
                f"{MLB_STATS_API_BASE}/people",
                params={
                    "personIds": ",".join(str(i) for i in chunk),
                    "hydrate": "stats(group=[pitching],type=[season])",
                },
                timeout=20,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                stat = _stat_split(person, "season")
                result[person["id"]] = Pitcher(
                    id=str(person["id"]),
                    name=person["fullName"],
                    throws=person.get("pitchHand", {}).get("code", "R"),
                    era=float(stat.get("era") or 4.50),
                    oba_against=float(stat.get("avg") or ".250"),
                )
        return result

    def _get_park_factor(self, home_team_id: int) -> float:
        return _load_park_factors().get(str(home_team_id), 1.0)


@lru_cache(maxsize=1)
def _load_park_factors() -> dict[str, float]:
    """team id (str) -> multiplicative park factor, from data/park_factors.json.

    Precomputed by scripts/refresh_park_factors.py rather than fetched
    live: park effects move slowly (see that script's docstring for
    methodology), so there's no need to hit the network for this on every
    run. Missing file or missing team id both fall back to neutral (1.0).
    """
    if not PARK_FACTORS_PATH.exists():
        return {}
    raw = json.loads(PARK_FACTORS_PATH.read_text())
    return {team_id: entry["park_factor"] for team_id, entry in raw.items()}


def _stat_split(person: dict[str, Any], type_display_name: str) -> dict[str, Any]:
    for stat_group in person.get("stats", []):
        if stat_group.get("type", {}).get("displayName") == type_display_name:
            splits = stat_group.get("splits", [])
            if splits:
                return splits[0].get("stat", {})
    return {}


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class FixtureSource(DataSource):
    """Loads a slate from a local JSON fixture. No network required."""

    def __init__(self, fixture_path: str | Path) -> None:
        self._fixture_path = Path(fixture_path)

    def get_matchups(self, date: str) -> list[Matchup]:
        payload = json.loads(self._fixture_path.read_text())
        if payload.get("date") != date:
            raise ValueError(
                f"fixture {self._fixture_path} is for {payload.get('date')!r}, "
                f"not {date!r}"
            )
        matchups: list[Matchup] = []
        for entry in payload["matchups"]:
            batter = Batter(**entry["batter"])
            pitcher = Pitcher(**entry["pitcher"])
            recent_form = RecentForm(**entry["recent_form"])
            matchups.append(
                Matchup(
                    batter=batter,
                    pitcher=pitcher,
                    is_home=entry["is_home"],
                    park_factor=entry["park_factor"],
                    recent_form=recent_form,
                )
            )
        return matchups
