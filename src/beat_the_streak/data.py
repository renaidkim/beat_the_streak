"""Data sources for a day's slate.

DataSource is the interface the rest of the app depends on. Two
implementations are provided:

- MlbStatsApiSource: hits the public MLB Stats API (statsapi.mlb.com).
  This is the real, live source but requires outbound network access to
  statsapi.mlb.com, which is not reachable from every environment (it is
  blocked, for example, by this sandbox's egress policy). Run it wherever
  you actually have network access.
- FixtureSource: reads a static JSON fixture from disk. Useful for
  development, tests, and demos without network access.

Swap sources via `--source` on the CLI; both return the same `Matchup`
objects so nothing downstream needs to know which one was used.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

from .models import Batter, GameLog, Matchup, Pitcher

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_RECENT_GAMES = 15


class DataSource(ABC):
    """Everything the ranking pipeline needs for one day's slate."""

    @abstractmethod
    def get_matchups(self, date: str) -> list[Matchup]:
        """Return one Matchup per starting batter expected to play on `date`."""
        raise NotImplementedError


class MlbStatsApiSource(DataSource):
    """Live source backed by the public MLB Stats API.

    Requires network access to statsapi.mlb.com. If that host is blocked
    (as it is in this sandbox), construct and use FixtureSource instead.
    """

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def get_matchups(self, date: str) -> list[Matchup]:
        games = self._get_schedule_with_probables(date)
        matchups: list[Matchup] = []
        for game in games:
            for side, opponent_side in (("home", "away"), ("away", "home")):
                team = game["teams"][side]["team"]
                opp_pitcher_info = game["teams"][opponent_side].get("probablePitcher")
                if not opp_pitcher_info:
                    continue
                pitcher = self._get_pitcher(opp_pitcher_info["id"])
                park_factor = self._get_park_factor(game["venue"]["id"])
                for batter_id in self._get_probable_lineup(team["id"]):
                    batter = self._get_batter(batter_id)
                    logs = self._get_recent_game_logs(batter_id)
                    matchups.append(
                        Matchup(
                            batter=batter,
                            pitcher=pitcher,
                            is_home=(side == "home"),
                            park_factor=park_factor,
                            recent_logs=logs,
                        )
                    )
        return matchups

    def _get_schedule_with_probables(self, date: str) -> list[dict[str, Any]]:
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/schedule",
            params={
                "sportId": 1,
                "date": date,
                "hydrate": "probablePitcher,team,venue",
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        games: list[dict[str, Any]] = []
        for date_entry in payload.get("dates", []):
            games.extend(date_entry.get("games", []))
        return games

    def _get_pitcher(self, pitcher_id: int) -> Pitcher:
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/people/{pitcher_id}",
            params={"hydrate": "stats(group=[pitching],type=[season])"},
            timeout=15,
        )
        resp.raise_for_status()
        person = resp.json()["people"][0]
        stat_line = _first_stat_split(person)
        return Pitcher(
            id=str(pitcher_id),
            name=person["fullName"],
            throws=person.get("pitchHand", {}).get("code", "R"),
            era=float(stat_line.get("era", 4.50)),
            oba_against=float(stat_line.get("avg", ".250")),
        )

    def _get_batter(self, batter_id: int) -> Batter:
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/people/{batter_id}",
            params={"hydrate": "stats(group=[hitting],type=[season])"},
            timeout=15,
        )
        resp.raise_for_status()
        person = resp.json()["people"][0]
        stat_line = _first_stat_split(person)
        season_avg = float(stat_line.get("avg", ".250"))
        return Batter(
            id=str(batter_id),
            name=person["fullName"],
            bats=person.get("batSide", {}).get("code", "R"),
            team=person.get("currentTeam", {}).get("name", ""),
            season_avg=season_avg,
            # The season endpoint doesn't split by pitcher hand; a
            # dedicated splits call would be needed for real vs-hand
            # averages. Fall back to season average until that's wired up.
            season_avg_vs_lhp=season_avg,
            season_avg_vs_rhp=season_avg,
        )

    def _get_recent_game_logs(
        self, batter_id: int, limit: int = DEFAULT_RECENT_GAMES
    ) -> list[GameLog]:
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/people/{batter_id}/stats",
            params={"stats": "gameLog", "group": "hitting"},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        splits = payload.get("stats", [{}])[0].get("splits", [])
        logs = [
            GameLog(
                date=split["date"],
                at_bats=int(split["stat"].get("atBats", 0)),
                hits=int(split["stat"].get("hits", 0)),
            )
            for split in splits
        ]
        return logs[-limit:]

    def _get_probable_lineup(self, team_id: int) -> list[int]:
        # The Stats API doesn't publish a confirmed lineup far in advance;
        # a real implementation would poll the boxscore endpoint closer to
        # game time and fall back to the last known lineup otherwise.
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/teams/{team_id}/roster",
            params={"rosterType": "active"},
            timeout=15,
        )
        resp.raise_for_status()
        roster = resp.json().get("roster", [])
        return [
            entry["person"]["id"]
            for entry in roster
            if entry.get("position", {}).get("abbreviation") != "P"
        ]

    def _get_park_factor(self, venue_id: int) -> float:
        # Park factors aren't exposed directly by this API; plug in a
        # lookup table (e.g. from Baseball Savant) here. Neutral until then.
        return 1.0


def _first_stat_split(person: dict[str, Any]) -> dict[str, Any]:
    stats = person.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return {}
    return stats[0]["splits"][0].get("stat", {})


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
            logs = [GameLog(**log) for log in entry["recent_logs"]]
            matchups.append(
                Matchup(
                    batter=batter,
                    pitcher=pitcher,
                    is_home=entry["is_home"],
                    park_factor=entry["park_factor"],
                    recent_logs=logs,
                )
            )
        return matchups
