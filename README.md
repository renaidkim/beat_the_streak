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
   split (average specifically vs. today's opposing pitcher hand), the
   opposing pitcher's average-against, park factor, home/away.
3. **Ranking** (`beat_the_streak/rank.py`) — a logistic regression, fit on
   a real season of outcomes, predicts P(hit) directly from those signals.
   See [Model backtest](#model-backtest) below for why it looks like this
   and not like a hand-tuned formula. Picking the top N avoids selecting
   two batters who face the *same* pitcher, since their outcomes are
   correlated — one shutdown pitching performance sinks both picks at once.
4. **CLI** (`beat_the_streak/cli.py`) — prints the ranked slate and the
   recommended picks.

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

## Model backtest

The original `rank.py` was a hand-tuned weighted blend (season avg 25%,
recent form 35%, platoon 20%, pitcher quality 20%) converted to a
per-game probability via `1 - (1-p)^4`. It was never checked against real
outcomes. `scripts/fit_hit_probability_model.py` does that: it pulls a
full season (2025) of game logs for the ~100 highest-plate-appearance
batters and their opposing starters, reconstructs — for every game — only
what would have been known *before* that game (season average to date,
last-10-games form, platoon split to date, opposing starter's own
season-to-date average-against, park factor, home/away; no lookahead),
and scores every row two ways: with the shipped heuristic, and with a
logistic regression fit on a date-based holdout (train on the first 70%
of the season, evaluate on the rest, so nothing from "the future" leaks
into training).

**The heuristic failed the check.** On the held-out games:

| Model | Log loss | Brier | AUC |
| :--- | :--- | :--- | :--- |
| Heuristic (as originally shipped) | 0.6580 | 0.2319 | 0.539 |
| Baseline: always predict the training set's average hit rate | 0.6520 | 0.2297 | 0.500 |
| Fitted logistic regression | 0.6502 | 0.2289 | 0.541 |

The heuristic scored *worse* than a baseline that ignores every feature
and just predicts the league-average hit rate every time. Its calibration
table showed why: predicted probabilities ranged from 59% to 80% across
deciles, but the *actual* hit rate in those same deciles barely moved
(58% to 72%) and wasn't even monotonic. The heuristic was overconfident —
it turned small, mostly-noisy differences between batters into large
swings in predicted probability. That's the real, humbling finding here:
whether a specific batter gets a hit in a specific game is close to a
coin flip (raw correlation between any single feature and the outcome
tops out around r=0.04); a season of team-average stats only weakly
narrows that down, and the model has to say so honestly rather than
projecting false confidence.

A bootstrap check (30 resamples) on the fitted coefficients found only
two features with a *consistent* sign across every resample: **park
factor** and **platoon split** (as a delta from season average, not an
absolute number — see the script's docstring for why the absolute version
collapses onto season average and creates artificial collinearity).
Season average, recent form, and pitcher quality all point the expected
direction but are individually noisy at this sample size (~14k
batter-games); they still add value combined, just don't over-trust any
one of them alone.

Two concrete changes came out of this and are now shipped:

- **`rank.py` now runs the fitted logistic regression directly**
  (coefficients in `data/hit_probability_model.json`, loaded at import
  time — no runtime ML dependency, it's just `sigmoid(intercept +
  Σ coef·feature)`), predicting the per-game probability directly instead
  of a per-at-bat rate converted via the `1-(1-p)^4` formula. Live output
  is visibly better-calibrated: predicted probabilities on a real slate
  now span roughly 60-76% instead of 53-84%.
- **Real platoon splits are now wired into `MlbStatsApiSource`** (batched
  into the existing per-batter stats call via `sitCodes=[vl,vr]`, no
  extra requests) instead of falling back to season average for both
  hands. This was a documented known gap before; the backtest is what
  justified prioritizing it — it's one of only two features the bootstrap
  called reliable.

Rerun `python scripts/fit_hit_probability_model.py` (needs
`requirements-analysis.txt`: pandas/numpy/scikit-learn, offline-analysis
only, not a runtime dependency) once or twice a season to refresh the
coefficients against a more recent year.

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

# Refresh the hit-probability model against a more recent season
# (needs requirements-analysis.txt):
pip install -r requirements-analysis.txt
python scripts/fit_hit_probability_model.py
```

## Known gaps

- Park factors aren't split by batter handedness (some parks favor lefties
  or righties very differently — Yankee Stadium's short right field is
  the classic example).
- The model was backtested on one season (2025) and ~14k batter-games
  from the ~100 highest-plate-appearance hitters; it hasn't been
  validated on part-time players, rookies with short track records, or a
  second season. Season average and recent form individually had wide
  bootstrap uncertainty (see [Model backtest](#model-backtest)) — the
  fitted weights are a reasonable current estimate, not a settled result.
- No accounting for a batter's own recent injury/rest status, or same-day
  lineup changes after prediction time.

## Tests

```bash
pip install -e . -r requirements-dev.txt
pytest
```
