# Beat the Streak

Ranks MLB batters by estimated probability of getting a hit on a given day,
for [MLB.com's Beat the Streak](https://www.mlb.com/apps/beat-the-streak)
game (pick 1-2 batters/day who you think will record a hit).

## How it works

1. **Data layer** (`beat_the_streak/data.py`) — pulls one `Matchup` per
   starting batter for the day: the batter, the opposing probable pitcher,
   home/away, park factor, and the batter's lineup slot (1-9) when known.
   - `MlbStatsApiSource` hits the live public MLB Stats API
     (`statsapi.mlb.com`), validated against live traffic. It uses each
     game's confirmed batting order once the boxscore posts it, and falls
     back to the full active roster for games where the lineup isn't out
     yet — so a slate with only a few confirmed lineups still produces
     predictions for everyone else instead of skipping them. See
     [Performance](#performance) for how it keeps request volume down.
   - `FixtureSource` reads a static JSON file instead, for offline dev,
     tests, and demos (`data/sample_slate_2026-08-19.json`).
   - Works up to a few days ahead, not just today. Confirmed lineups only
     ever exist for today's games, so future days always use the roster
     fallback; further out, a growing share of games don't have an
     announced starting pitcher yet either (same-day: ~100% announced;
     +1 day: ~25-30% typically; +2/+3 days: even less — MLB just doesn't
     post them that far ahead most of the time). Rather than dropping
     those batters, unannounced pitchers get a neutral, clearly-labeled
     placeholder (league-average `oba_against`, `confirmed=False`, shown
     as a "TBD" badge and called out in that pick's reasons) so the page
     still has something useful to look at before starters are set —
     just revisit closer to game time as real lineups and starters post.
2. **Features** (`beat_the_streak/features.py`) — turns a `Matchup` into six
   signals: career batting average entering the season, platoon delta
   (average vs. today's opposing pitcher hand, minus in-season overall
   average), the opposing pitcher's average-against, park factor,
   home/away, and batting order slot.
3. **Ranking** (`beat_the_streak/rank.py`) — a gradient-boosted model, fit
   on a real season of outcomes, predicts P(hit) directly from those
   signals. See [Model backtest](#model-backtest) below for the two
   rounds of testing that produced this specific feature set and model
   type — it did not start out looking like this. Picking the top N
   avoids selecting two batters who face the *same* pitcher, since their
   outcomes are correlated — one shutdown pitching performance sinks both
   picks at once.
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

Two concrete changes came out of that first round: `rank.py` switched to
running the fitted logistic regression directly (predicting the per-game
probability, no more per-at-bat-to-per-game conversion), and real platoon
splits got wired into `MlbStatsApiSource` (it had been falling back to
season average for both hands) — the backtest is what justified
prioritizing that fix, since platoon was one of only two features the
bootstrap called reliable.

### Round two: career average, batting order, and gradient boosting

Round one still only barely beat the naive baseline (AUC 0.541, log loss
0.6502 vs. 0.6520). Asked directly "is this actually working, and can it
be improved," the honest first step was testing more of the data the
Stats API actually has available: a batter's **career average** entering
the season (multi-year, not just this season — via `yearByYear` hitting
stats), and each batter's **batting order slot** that game (1-9, from the
same boxscore call already used for confirmed lineups, just previously
discarded). Both required pulling boxscores for essentially the full 2025
season (~2,400 games) to reconstruct point-in-time.

**Both turned out to matter a lot — enough to change what the model
should even be:**

- `career_avg` came back with the single largest, most bootstrap-stable
  coefficient of any feature (bigger than park factor). Makes sense in
  hindsight: a multi-year sample is a far more reliable estimate of a
  hitter's true talent than anything from one partial season.
- `batting_order` was the *most precisely estimated* coefficient in the
  whole model (std/mean ratio far tighter than any other feature) and
  correctly signed — batting higher in the order predicts more hits, both
  because better hitters bat higher and because they get more plate
  appearances per game.
- With `career_avg` in the model, **season average and last-10-games form
  added zero independent predictive power** — tested explicitly (4
  feature-set variants, holding everything else fixed): including or
  excluding them changed holdout log loss by less than 0.0002. They were
  dropped rather than kept for appearances. Once you know a hitter's true
  long-run talent level, this season's hot streak or cold spell isn't
  telling you anything more about tonight.

That called for retesting *how* the features get combined, not just which
ones: a gradient-boosted tree (`HistGradientBoostingClassifier`) was
tested against logistic regression on the same trimmed 6-feature set, and
won cleanly across every metric — reproducibly (identical result across 4
random seeds, so not a lucky fit):

| Model | Log loss | Brier | AUC |
| :--- | :--- | :--- | :--- |
| Naive baseline | 0.6520 | 0.2297 | 0.500 |
| Round-one logistic regression (6 features) | 0.6502 | 0.2289 | 0.541 |
| Round-two logistic regression (career avg + order) | 0.6473 | 0.2276 | 0.563 |
| **Round-two gradient boosting (shipped)** | **0.6461** | **0.2270** | **0.566** |

**The gradient-boosted model initially had a real problem**, though, that
plain unit tests caught before it shipped: point-checking individual
matchups (not just aggregate holdout metrics) showed it predicting a
*tougher* pitcher (.200 average-against) as *more* hittable than a weak
one (.310), and a career .210 hitter above a career .320 hitter — flatly
backwards, and contradicting the same bootstrap that had just confirmed
those features' correct direction. A flexible tree model can do this in
regions of feature space with sparse training data, even while scoring
well in aggregate. Fixed with `monotonic_cst` constraints (`career_avg`,
`pitcher_oba_against`, `park_factor` forced non-decreasing;
`batting_order` non-increasing — every one of these is both basic
baseball logic and what the linear model already confirmed) — cost about
0.001 of holdout log loss, in exchange for ruling out backwards
predictions entirely. `is_home` and `platoon_delta` are left
unconstrained since neither showed a fully reliable direction; platoon
delta in particular turned out to be conditionally important (matters a
lot in some matchup contexts, not at all in others) rather than a
uniform effect, which the tree model can represent and a linear one can't.

Net result: predicted probabilities on a real slate that used to swing
53-84% (round one) now sit in a tighter, better-calibrated ~60-76% band,
and the actual #1 picks now look like what a baseball person would expect
— a leadoff-hitting star at Coors Field outranks a bench bat, not the
other way around.

`rank.py` loads whichever model type `scripts/fit_hit_probability_model.py`
last produced (`data/hit_probability_model.json` for metadata/feature
order, plus `data/hit_probability_model.pkl` for a gradient-boosting
model specifically) — logistic regression needs no runtime ML dependency,
but shipping gradient boosting does: scikit-learn and joblib are now
*runtime* dependencies (`requirements.txt`), not just for the offline
fitting script. That's a real added weight (~100MB+) for what was a
near-zero-dependency app; it was judged worth it because the performance
gain was consistent and reproducible, not marginal or mixed.

Rerun `python scripts/fit_hit_probability_model.py` (needs
`requirements-analysis.txt` for pandas/numpy on top of the runtime deps)
once or twice a season to refresh against a more recent year.

## Hosted site

`scripts/generate_site.py` renders today/tomorrow/day-after-tomorrow's
picks as one static HTML page (`_site/index.html`) — no server, no
client-side JS, just an f-string template. `.github/workflows/publish.yml`
runs it and publishes the result to GitHub Pages:

- **On demand**: open the repo's **Actions** tab → **Publish picks** →
  **Run workflow** (also doable from the GitHub mobile app). Takes under
  a minute. This is the main way to refresh after new lineups post.
- **On a schedule**, as a baseline so the page isn't stale if nobody
  triggers it by hand: every 2 hours, all day. Edit the `cron` line in
  the workflow file to change the cadence.

**One-time setup** (I can't do this part myself — it's a repo setting,
not something available through the tools in this session): in the
repo's **Settings → Pages**, set **Source** to **GitHub Actions**. After
that, the first workflow run publishes the page and prints its URL in
the deploy job's summary.

Cost: $0. GitHub Pages and Actions are free for a repo like this one
(unlimited Actions minutes on a public repo; a generous free monthly
allowance on a private one, and each run here takes well under a minute).

## Performance

`MlbStatsApiSource` batches instead of doing one request per batter:
roster/boxscore calls are made once per game, and player stats (season
average, platoon splits, career year-by-year, or season pitching line)
are fetched via chunked `/people?personIds=...` calls covering up to 50
players each, rather than one request per player. Measured against a
real, completed 15-game slate (270 starting batters): **23 HTTP
requests, ~2 seconds**, versus roughly 850 requests the naive
one-request-per-batter version would have made.

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

# Render the static site locally (see Hosted site below for publishing it):
python scripts/generate_site.py --days 3 --out _site
```

## Known gaps

- `data/hit_probability_model.pkl` is a joblib-pickled scikit-learn model,
  which means it's only guaranteed loadable by the same major/minor
  scikit-learn version it was fit with. `requirements.txt` pins
  `scikit-learn>=1.4` without an upper bound; if a future install picks up
  a materially newer scikit-learn and `joblib.load` starts failing, the
  fix is to re-run `scripts/fit_hit_probability_model.py` with that
  environment's scikit-learn to regenerate the pickle, not to chase
  pickle-compatibility shims.
- Park factors aren't split by batter handedness (some parks favor lefties
  or righties very differently — Yankee Stadium's short right field is
  the classic example).
- The model was backtested on one season (2025) and ~14k batter-games
  from the ~100 highest-plate-appearance hitters; it hasn't been
  validated on part-time players, rookies with short track records, or a
  second season.
- `platoon_delta` turned out to be conditionally important rather than
  reliably important (see [Model backtest](#model-backtest)) — the
  gradient-boosting model can represent "matters here, doesn't matter
  there," but that also means it's the one feature whose effect isn't
  well summarized by a single number, and the least-tested one by the
  bootstrap.
- `is_home` is carried as a feature but never showed a reliable direction
  or magnitude across any round of testing; it's along for the ride more
  than it's doing real work.
- No accounting for a batter's own recent injury/rest status, or same-day
  lineup changes after prediction time.
- The monotonic constraints (career average, pitcher quality, and park
  factor forced non-decreasing; batting order non-increasing) are
  asserted from baseball domain knowledge and the linear model's
  bootstrap-confirmed directions, not independently re-derived by the
  gradient-boosting fit itself — a deliberate choice (see
  [Model backtest](#model-backtest)) to keep the model from being able to
  contradict signal it already validated, but worth knowing that's what
  they are.

## Tests

```bash
pip install -e . -r requirements-dev.txt
pytest
```
