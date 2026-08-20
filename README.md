# Beat the Streak

Ranks MLB batters by estimated probability of getting a hit on a given day,
for [MLB.com's Beat the Streak](https://www.mlb.com/apps/beat-the-streak)
game (pick 1-2 batters/day who you think will record a hit).

## How it works

1. **Data layer** (`beat_the_streak/data.py`) — pulls one `Matchup` per
   probable batter for the day: the batter, the opposing probable pitcher,
   home/away, park factor, and the batter's recent game logs.
   - `MlbStatsApiSource` hits the live public MLB Stats API
     (`statsapi.mlb.com`). Real endpoints, no API key needed — but it does
     require outbound network access to that host.
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

## A note on data access

This was built in a sandboxed environment whose network egress policy
blocks `statsapi.mlb.com`, Baseball Savant, FanGraphs, and
Baseball-Reference — so `MlbStatsApiSource` is implemented against the
real API but untested against live traffic here. Run it from an
environment with normal network access, and double check the API
response shapes against a live call before relying on it (the Stats API
is public but only loosely documented).

Known gaps in the live source, called out in the code:
- No real vs-pitcher-hand split endpoint is wired up yet (falls back to
  season average).
- No confirmed-lineup source (uses full active roster, not today's
  starting 9 — swap in a boxscore poll closer to game time).
- No park-factor table (defaults to neutral, 1.0).

## Usage

```bash
pip install -e .

# Live (requires network access to statsapi.mlb.com):
beat-the-streak 2026-08-19 --source mlbapi

# Offline, against the bundled sample fixture:
beat-the-streak 2026-08-19 --source fixture \
    --fixture-path data/sample_slate_2026-08-19.json --picks 2
```

## Tests

```bash
pip install -e . -r requirements-dev.txt
pytest
```
