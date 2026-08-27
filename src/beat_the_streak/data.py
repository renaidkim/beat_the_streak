"""Data sources for a day's slate.

DataSource is the interface the rest of the app depends on. Two
implementations are provided:

- MlbStatsApiSource: hits the public MLB Stats API (statsapi.mlb.com).
  Validated against live traffic. Uses the boxscore's confirmed batting
  order when it's been posted (also the source of each batter's lineup
  slot), falls back to the full active roster when it hasn't, and
  batches player-stat lookups (chunked `/people?personIds=` calls)
  instead of firing one request per batter per game.
- FixtureSource: reads a static JSON fixture from disk. Useful for
  development, tests, and demos without network access.

Swap sources via `--source` on the CLI; both return the same `Matchup`
objects so nothing downstream needs to know which one was used.
"""

from __future__ import annotations

import datetime
import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .features import LEAGUE_AVG_AVG, LEAGUE_AVG_OBP, LEAGUE_AVG_PITCHER_K9
from .models import Batter, Matchup, Pitcher

MLB_STATS_API_BASE = "https://statsapi.mlb.com/api/v1"

# MLB's schedule is organized by "baseball day," which follows US Eastern
# time (where the league office is), not UTC or wherever this process
# happens to run -- a 10pm ET West Coast game is still "today" even
# though it's past midnight UTC. Matters for deciding whether a boxscore
# is worth fetching at all (see is_today below) and for the site
# generator's day labels.
EASTERN = ZoneInfo("America/New_York")

# Rough league-average ERA, used only for the TBD-starter placeholder
# below -- oba_against there comes from LEAGUE_AVG_AVG instead, since
# that's what actually feeds the model.
LEAGUE_AVG_ERA = 4.30

# Fallbacks for a batter with no prior-season yearByYear data (a true
# rookie) -- rare, only hit when career_k_rate/career_bb_rate can't be
# computed from real counting stats.
LEAGUE_AVG_K_RATE = 0.22
LEAGUE_AVG_BB_RATE = 0.08

# Minimum prior at-bats vs. the exact opposing pitcher before bvp_delta
# is trusted rather than treated as 0 (no information) -- matches
# scripts/train_ml_model.py's MIN_BVP_AB, the value the shipped model
# was actually trained and validated with (see that file's comment for
# the empirical justification). Keep these two in sync.
MIN_BVP_AB = 3

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
        self._bvp_cache: dict[tuple[int, int], float] = {}

    def get_matchups(self, date: str) -> list[Matchup]:
        games = self._get_schedule_with_probables(date)
        season_year = int(date[:4])

        # Pass 1: work out who's playing (and their lineup slot, if known)
        # and who they're facing, without yet fetching per-player stats,
        # so those can be batched afterward instead of fired one at a time.
        game_matchups: list[dict[str, Any]] = []
        pitcher_ids: set[int] = set()
        batter_ids: set[int] = set()

        # Confirmed lineups only ever exist for games happening today;
        # skip the boxscore call entirely for future dates rather than
        # firing a request that's guaranteed to come back empty.
        is_today = date == datetime.datetime.now(EASTERN).date().isoformat()

        for game in games:
            park_factor = self._get_park_factor(game["teams"]["home"]["team"]["id"])
            boxscore = self._get_boxscore(game["gamePk"]) if is_today else {}
            game_started = _game_has_started(game)
            for side, opponent_side in (("home", "away"), ("away", "home")):
                team = game["teams"][side]["team"]
                opp_pitcher_info = game["teams"][opponent_side].get("probablePitcher")
                lineup = self._get_starting_batters(boxscore, side, team["id"])
                batter_ids.update(lineup.keys())
                if opp_pitcher_info:
                    pitcher_id: int | str = opp_pitcher_info["id"]
                    pitcher_ids.add(pitcher_id)
                else:
                    # Starter not announced yet (routine a few days out --
                    # see the schedule endpoint's typical lead time).
                    # Score with a neutral placeholder instead of dropping
                    # the whole team's batters for the day.
                    pitcher_id = f"tbd:{game['gamePk']}:{opponent_side}"
                game_matchups.append(
                    {
                        "pitcher_id": pitcher_id,
                        "lineup": lineup,  # batter_id -> slot (1-9) or None
                        "is_home": side == "home",
                        "park_factor": park_factor,
                        "game_started": game_started,
                    }
                )

        batters = self._get_batters(batter_ids, season_year, date)
        pitchers = self._get_pitchers(pitcher_ids, season_year)

        matchups: list[Matchup] = []
        for gm in game_matchups:
            pitcher_id = gm["pitcher_id"]
            if isinstance(pitcher_id, str):
                pitcher = _placeholder_pitcher(pitcher_id)
            else:
                pitcher = pitchers.get(pitcher_id)
            if pitcher is None:
                continue
            for batter_id, slot in gm["lineup"].items():
                batter = batters.get(batter_id)
                if batter is None:
                    continue
                bvp_delta, bvp_ab = (
                    self._get_bvp_delta(batter_id, pitcher_id, batter.season_avg)
                    if isinstance(pitcher_id, int)
                    else (0.0, 0)  # TBD placeholder pitcher -- no specific pitcher to have history against
                )
                matchups.append(
                    Matchup(
                        batter=batter,
                        pitcher=pitcher,
                        is_home=gm["is_home"],
                        park_factor=gm["park_factor"],
                        batting_order=slot,
                        bvp_delta=bvp_delta,
                        bvp_ab=bvp_ab,
                        game_started=gm["game_started"],
                    )
                )
        return matchups

    def _get_bvp_delta(self, batter_id: int, pitcher_id: int, season_avg: float) -> tuple[float, int]:
        """(delta, at_bats): this batter's history against this specific
        pitcher, delta vs. their own season-to-date average, and the
        at-bat count behind it. MLB's vsPlayer stat type needs one
        request per (batter, pitcher) pair -- no batching -- so this is
        only feasible called once per matchup during a live slate build
        (a few hundred requests/day), not for backtesting training data
        (tens of thousands of pairs); see scripts/train_ml_model.py's
        _compute_bvp_history for the cheaper, truncated-window proxy
        used there instead.

        The response includes a season=None aggregate row *in addition
        to* the per-season breakdown (verified directly against the live
        API) -- summing everything would double-count, so only the
        dated, per-season splits are summed here.

        delta is 0.0 (no information) below MIN_BVP_AB at-bats, even
        though the real at_bats count is still returned -- e.g. a
        2-for-2 shouldn't move the model or be described as "historical"
        performance. See MIN_BVP_AB's comment for why this specific
        threshold (empirically the best of several tested, not just a
        round number).
        """
        cache_key = (batter_id, pitcher_id)
        if cache_key in self._bvp_cache:
            return self._bvp_cache[cache_key]
        resp = self._session.get(
            f"{MLB_STATS_API_BASE}/people/{batter_id}/stats",
            params={
                "stats": "vsPlayer",
                "opposingPlayerId": pitcher_id,
                "group": "hitting",
                "sportId": 1,
            },
            timeout=15,
        )
        resp.raise_for_status()
        ab = hits = 0
        for stat_group in resp.json().get("stats", []):
            for split in stat_group.get("splits", []):
                if split.get("season") is None:
                    continue  # career-aggregate row, already covered below
                stat = split.get("stat", {})
                ab += int(stat.get("atBats", 0))
                hits += int(stat.get("hits", 0))
        delta = (hits / ab - season_avg) if ab >= MIN_BVP_AB else 0.0
        result = (delta, ab)
        self._bvp_cache[cache_key] = result
        return result

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
    ) -> dict[int, int | None]:
        # The boxscore's battingOrder is a flat, ordered list of the 9
        # starters' ids -- list position (1-indexed) is the lineup slot.
        order_list = boxscore.get("teams", {}).get(side, {}).get("battingOrder") or []
        if order_list:
            return {batter_id: slot for slot, batter_id in enumerate(order_list, start=1)}
        # Lineup not posted yet (common well before game time): fall back
        # to every position player on the active roster, with no known
        # slot, rather than skipping the team's batters entirely.
        return {bid: None for bid in self._get_roster_batters(team_id)}

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

    def _get_batters(
        self, batter_ids: set[int], season_year: int, as_of_date: str
    ) -> dict[int, Batter]:
        as_of = datetime.date.fromisoformat(as_of_date)
        result: dict[int, Batter] = {}
        for chunk in _chunks(sorted(batter_ids), PEOPLE_BATCH_SIZE):
            resp = self._session.get(
                f"{MLB_STATS_API_BASE}/people",
                params={
                    "personIds": ",".join(str(i) for i in chunk),
                    "hydrate": (
                        "currentTeam,"
                        "stats(group=[hitting],type=[season,statSplits,yearByYear],"
                        "sitCodes=[vl,vr])"
                    ),
                },
                timeout=20,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                season_stat = _stat_split(person, "season")
                platoon_splits = _platoon_splits(person)
                season_avg = float(season_stat.get("avg") or ".250")
                career = _career_hitting_rates_entering_season(person, season_year)
                birth_date = person.get("birthDate")
                age = (
                    (as_of - datetime.date.fromisoformat(birth_date)).days / 365.25
                    if birth_date
                    else 27.0
                )
                result[person["id"]] = Batter(
                    id=str(person["id"]),
                    name=person["fullName"],
                    bats=person.get("batSide", {}).get("code", "R"),
                    team=person.get("currentTeam", {}).get("name", ""),
                    career_avg=career["avg"] if career else season_avg,
                    season_avg=season_avg,
                    # Falls back to season average when a split has too few
                    # at-bats to report an avg (early season, part-time
                    # platoon batter facing an unfamiliar hand, etc.).
                    season_avg_vs_lhp=float(platoon_splits.get("vl", {}).get("avg") or season_avg),
                    season_avg_vs_rhp=float(platoon_splits.get("vr", {}).get("avg") or season_avg),
                    career_obp=career["obp"] if career else float(season_stat.get("obp") or LEAGUE_AVG_OBP),
                    career_k_rate=career["k_rate"] if career else LEAGUE_AVG_K_RATE,
                    career_bb_rate=career["bb_rate"] if career else LEAGUE_AVG_BB_RATE,
                    age=age,
                )
        return result

    def _get_pitchers(self, pitcher_ids: set[int], season_year: int) -> dict[int, Pitcher]:
        result: dict[int, Pitcher] = {}
        for chunk in _chunks(sorted(pitcher_ids), PEOPLE_BATCH_SIZE):
            resp = self._session.get(
                f"{MLB_STATS_API_BASE}/people",
                params={
                    "personIds": ",".join(str(i) for i in chunk),
                    "hydrate": "stats(group=[pitching],type=[season,yearByYear])",
                },
                timeout=20,
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                stat = _stat_split(person, "season")
                season_era = float(stat.get("era") or 4.50)
                career = _career_pitching_rates_entering_season(person, season_year)
                result[person["id"]] = Pitcher(
                    id=str(person["id"]),
                    name=person["fullName"],
                    throws=person.get("pitchHand", {}).get("code", "R"),
                    era=season_era,
                    oba_against=float(stat.get("avg") or ".250"),
                    era_career=career["era"] if career else season_era,
                    k9_career=(
                        career["k9"]
                        if career
                        else float(stat.get("strikeoutsPer9Inn") or LEAGUE_AVG_PITCHER_K9)
                    ),
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


def _game_has_started(game: dict[str, Any]) -> bool:
    """True once a game is live or final -- i.e. no longer "Preview"
    (scheduled/pre-game/warmup). Beat the Streak only allows picking a
    player before their game begins, so live and final are equally
    unselectable; only the not-yet-started distinction matters here.
    """
    return game.get("status", {}).get("abstractGameState") != "Preview"


def _stat_split(person: dict[str, Any], type_display_name: str) -> dict[str, Any]:
    for stat_group in person.get("stats", []):
        if stat_group.get("type", {}).get("displayName") == type_display_name:
            splits = stat_group.get("splits", [])
            if splits:
                return splits[0].get("stat", {})
    return {}


def _platoon_splits(person: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """{"vl": stat, "vr": stat} for a person's statSplits stat group, keyed
    by the split's own "vl"/"vr" code rather than list position.
    """
    result: dict[str, dict[str, Any]] = {}
    for stat_group in person.get("stats", []):
        if stat_group.get("type", {}).get("displayName") != "statSplits":
            continue
        for split in stat_group.get("splits", []):
            code = split.get("split", {}).get("code")
            if code in ("vl", "vr"):
                result[code] = split.get("stat", {})
    return result


def _career_hitting_rates_entering_season(
    person: dict[str, Any], season_year: int
) -> dict[str, float] | None:
    """Career avg/obp/k_rate/bb_rate summed from counting stats across
    every season strictly before season_year, from the person's
    yearByYear stat group. Rates are recomputed from summed counts, not
    averaged from each season's own rate, so a 230-PA rookie season
    doesn't get weighted the same as a 650-PA everyday one. None if
    there's no prior-season data (a rookie) -- callers fall back to
    season-level figures in that case.
    """
    ab = hits = bb = hbp = sf = so = pa = 0
    for stat_group in person.get("stats", []):
        if stat_group.get("type", {}).get("displayName") != "yearByYear":
            continue
        for split in stat_group.get("splits", []):
            season = int(split.get("season", season_year))
            if season >= season_year:
                continue
            st = split["stat"]
            ab += int(st.get("atBats", 0))
            hits += int(st.get("hits", 0))
            bb += int(st.get("baseOnBalls", 0))
            hbp += int(st.get("hitByPitch", 0))
            sf += int(st.get("sacFlies", 0))
            so += int(st.get("strikeOuts", 0))
            pa += int(st.get("plateAppearances", 0))
    if ab == 0:
        return None
    obp_den = ab + bb + hbp + sf
    avg = hits / ab
    return {
        "avg": avg,
        "obp": (hits + bb + hbp) / obp_den if obp_den > 0 else avg,
        "k_rate": so / pa if pa > 0 else LEAGUE_AVG_K_RATE,
        "bb_rate": bb / pa if pa > 0 else LEAGUE_AVG_BB_RATE,
    }


def _career_pitching_rates_entering_season(
    person: dict[str, Any], season_year: int
) -> dict[str, float] | None:
    """Career ERA and K/9 summed from counting stats across every season
    strictly before season_year, from the person's yearByYear stat group.
    None if there's no prior MLB season (a rookie) -- callers fall back
    to this season's own rates in that case.
    """
    outs = er = k = 0
    for stat_group in person.get("stats", []):
        if stat_group.get("type", {}).get("displayName") != "yearByYear":
            continue
        for split in stat_group.get("splits", []):
            season = int(split.get("season", season_year))
            if season >= season_year:
                continue
            st = split["stat"]
            outs += int(st.get("outs", 0))
            er += int(st.get("earnedRuns", 0))
            k += int(st.get("strikeOuts", 0))
    if outs == 0:
        return None
    return {"era": er * 27 / outs, "k9": k * 27 / outs}


def _placeholder_pitcher(tbd_id: str) -> Pitcher:
    """Stand-in for a game whose starter hasn't been announced yet.

    Neutral by construction: league-average figures for every stat that
    actually feeds the model, and confirmed=False so rank.py can call
    this out in its explanation rather than silently presenting it as
    real information.
    """
    return Pitcher(
        id=tbd_id,
        name="TBD",
        throws="R",
        era=LEAGUE_AVG_ERA,
        oba_against=LEAGUE_AVG_AVG,
        era_career=LEAGUE_AVG_ERA,
        k9_career=LEAGUE_AVG_PITCHER_K9,
        confirmed=False,
    )


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
            matchups.append(
                Matchup(
                    batter=batter,
                    pitcher=pitcher,
                    is_home=entry["is_home"],
                    park_factor=entry["park_factor"],
                    batting_order=entry.get("batting_order"),
                    bvp_delta=entry.get("bvp_delta", 0.0),
                    bvp_ab=entry.get("bvp_ab", 0),
                    game_started=entry.get("game_started", False),
                )
            )
        return matchups
