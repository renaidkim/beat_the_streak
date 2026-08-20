# Beat the Streak

Ranks MLB batters by estimated probability of getting a hit on a given day,
for [MLB.com's Beat the Streak](https://www.mlb.com/apps/beat-the-streak)
game (pick 1-2 batters/day who you think will record a hit).

## How it works

1. **Data layer** (`beat_the_streak/data.py`) — pulls one `Matchup` per
   starting batter for the day: the batter, the opposing probable pitcher,
   home/away, park factor, and the batter's recent form.
   - `MlbStatsApiSource` hits the live public MLB Stats API
     (`statsapi.mlb.com`), validated against live traffic. It uses each
     game's confirmed batting order once the boxscore posts it, and falls
     back to the full active roster for games where the lineup isn't out
     yet — so a slate with only a few confirmed lineups still produces
     predictions for everyone else instead of skipping them. See
     [Performance](#performance) for how it keeps request volume down.
   - `FixtureSource` reads a static JSON file instead, for offline dev,
     tests, and demos (`data/sample_slate_2026-08-19.json`).
2. **Features** (`beat_the_streak/features.py`) — turns a `Matchup` into a
   handful of signals: season average, last-10-games form, the platoon
   (vs left/right) split, the opposing pitcher's average-against, park
   factor, home/away.
3. **Ranking** (`beat_the_streak/rank.py`) — blends those signals into a
   per-at-bat hit probability, then converts to a per-game probability via
   `1 - (1 - p)^at_bats` (the standard "at least one hit in N at-bats"
   formula). Thin recent-game samples are shrunk toward season average so
   one hot game doesn't dominate. Picking the top N also avoids selecting
   two batters who face the *same* pitcher, since their outcomes are
   correlated — one shutdown pitching performance sinks both picks at once.
4. **CLI** (`beat_the_streak/cli.py`) — prints the ranked slate and the
   recommended picks.

This is a transparent heuristic, not a fitted model — a deliberate starting
point. Swapping `score_matchup` for a model trained on historical
outcomes (logistic regression / gradient boosting over these same features)
is the natural next step once there's a results log to train on.

## Park factor: does it actually matter?

Checked empirically before trusting it (`scripts/refresh_park_factors.py`):
for each team, the ratio of that team's own hitters' batting average at
home vs. on the road, averaged over the last 3 completed seasons (using
the same team's hitters home vs. away cancels out their own quality,
isolating the park). Result, from `data/park_factors.json`:

| Park | Factor |
| :--- | :--- |
| Colorado Rockies (Coors Field) | 1.245 |
| Kansas City Royals | 1.096 |
| Boston Red Sox | 1.082 |
| *(24 more teams)* | 0.96 – 1.08 |
| Seattle Mariners | 0.915 |

Verdict: yes, it's worth including, but the effect is concentrated in a
handful of extreme parks rather than spread evenly. Coors Field alone is
one of the largest single factors anywhere in the model — a +24.5%
per-at-bat boost dwarfs most form/matchup differences — and it shows up
immediately in live output: every top pick on a day the Rockies play at
home is a Coors Field batter. Seattle's marine layer suppresses hits by
about 8.5%. For the other ~28 teams the gap from neutral is mostly 1-8%,
smaller than a lot of the other signals in the model, so don't expect park
to be the deciding factor most days — just the occasional day it's the
whole story.

This 3-year home/away ratio is a deliberately simple proxy (no split by
handedness or batted-ball type, doesn't control for the mix of opponents
each team's home vs. road slate happened to include) — good enough to
weight a heuristic model, not research-grade. Re-run the script once or
twice a season; park effects don't shift week to week.

## Performance

`MlbStatsApiSource` batches instead of doing one request per batter:
roster/boxscore calls are made once per game, and player stats (season
average + last-10-games form, or season pitching line) are fetched via
chunked `/people?personIds=...` calls covering up to 50 players each,
rather than one request per player. Measured against a real, completed
15-game slate (270 starting batters): **23 HTTP requests, ~3.6 seconds**,
versus roughly 850 requests the naive one-request-per-batter version would
have made.

## Usage

```bash
pip install -e .

# Live (requires network access to statsapi.mlb.com):
beat-the-streak 2026-08-19 --source mlbapi

# Offline, against the bundled sample fixture:
beat-the-streak 2026-08-19 --source fixture \
    --fixture-path data/sample_slate_2026-08-19.json --picks 2

# Refresh the park-factor table (run once or twice a season):
python scripts/refresh_park_factors.py
```

## Known gaps

- No real vs-pitcher-hand split endpoint is wired up yet for batters
  (falls back to season average).
- Park factors aren't split by batter handedness (some parks favor lefties
  or righties very differently — Yankee Stadium's short right field is
  the classic example).

## Tests

```bash
pip install -e . -r requirements-dev.txt
pytest
```
