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
   - Each `Matchup` also carries `game_started` (from the schedule's
     `abstractGameState`, live or final both count). Beat the Streak
     only allows picking a player before their game begins, so
     `rank.pick_top` never recommends one whose game has already
     started — but the full ranked table (and the CLI's `--show` list)
     still show them, tagged with a "GAME STARTED" badge, since seeing
     the model's full output is useful for checking its work even for
     matchups you can no longer act on.
2. **Features** (`beat_the_streak/features.py`) — turns a `Matchup` into 13
   signals: batting order slot, the opposing pitcher's career
   strikeouts-per-9 and career ERA (both entering this season) and
   season-to-date average against, park factor, the batter's career
   average/OBP/strikeout rate/walk rate (career, entering this season),
   age, both players' handedness, and this batter's own history against
   this specific opposing pitcher.
3. **Ranking** (`beat_the_streak/rank.py`) — a gradient-boosted model, fit
   on multiple past seasons of outcomes and validated on a season it
   never saw, predicts P(hit) directly from those signals. See
   [Model backtest](#model-backtest) below for the four rounds of
   testing that produced this specific feature set and model type — it
   did not start out looking like this. Picking the top N picks purely by
   probability, *including* two batters who share a pitcher — see
   [Why same-pitcher picks aren't avoided](#why-same-pitcher-picks-arent-avoided)
   for why that's actually the right call for how this game scores a
   two-pick day, not an oversight.
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
once or twice a season to refresh against a more recent year. (Superseded
by round three below — kept as the historical record and because its
6-feature model is still the "prior_production_model" baseline that gets
compared against on every retrain.)

### Round three: multi-season training, a broad feature set, real explainability

Round two was still evaluated with a date split *within* one season
(2025) — a real test, but one where the model could still implicitly pick
up on that season's specific park/weather/league conditions. Asked to
build something "entirely machine-learning driven" with real
explainability rather than hand-written threshold text,
`scripts/fit_ml_model.py` + `scripts/train_ml_model.py` do three things
differently:

1. **Train on 2023-2025, test purely on 2026** — a season the model never
   saw a single row of during fitting or feature selection. Meaningfully
   stronger than a within-season split.
2. **A broad, systematically-built feature set** (~24 features) instead of
   6 hand-picked ones: full career slash line, career walk/strikeout
   rates, a pitcher's career ERA/WHIP/K9/BB9/HR9 (not just season-to-date
   average-against), age, handedness for both sides, rest days since last
   appearance, and month of season, in addition to what round two already
   found. Not pre-filtered by hand — permutation importance decided what
   mattered.
3. **Four real model families compared** (L2 logistic regression, random
   forest, gradient boosting, a small MLP), and **permutation importance**
   on the true 2026 holdout as the explainability method — the actual
   drop in held-out performance when a feature is shuffled, rather than
   impurity-based importance (which inflates high-cardinality/continuous
   features) or hand-picked reason thresholds.

**Only 12 of the 24 features had real (positive) permutation importance**
on the 2026 holdout — the rest (rest days since last appearance for
either player, month, `is_home`, `platoon_delta`, career ISO/BABIP/SLG,
pitcher WHIP/BB9/HR9, switch-hitter indicator) were ~zero or negative,
indistinguishable from noise. That includes `platoon_delta` and
`is_home`, both of which round two's own writeup already flagged as
shaky (conditionally important and never-reliably-signed, respectively)
— round three's broader test confirms neither earns a place once
better-behaved features are available. The model was refit on just the
12 real ones: fewer live API calls, and every remaining feature earned
its place on held-out data rather than intuition.

**The best-scoring model on raw log loss (random forest) had to be
rejected anyway.** Direct point-checks — the same technique that caught
round two's backwards-prediction bug — found it scoring a .230 career
hitter *below* a .200 one at otherwise-default feature values. Identical
pathology, different model family: `RandomForestClassifier` has no
`monotonic_cst` equivalent in scikit-learn, so nothing stops a tree
ensemble from learning a locally-backwards relationship in sparse
regions of feature space even while scoring well in aggregate. The
shipped model is instead the runner-up, gradient boosting with the same
kind of monotonic constraints round two used (`batting_order` and
`pitcher_k9_career`/`career_k_rate` non-increasing; `park_factor`,
`career_avg`, `pitcher_oba_against`, `career_obp`, `pitcher_era_career`
non-decreasing; handedness and age left unconstrained) — verified
monotonic by point-checks across every constrained feature before
shipping, at a cost of about 0.0005 holdout log loss versus the
rejected random forest.

| Model, scored on the true 2026 holdout | Log loss | Brier | AUC |
| :--- | :--- | :--- | :--- |
| Naive baseline (constant training-set rate) | 0.6493 | 0.2284 | 0.500 |
| Round-two model (6 features, within-2025-season validated) | 0.6477 | 0.2277 | 0.533 |
| Logistic regression (12 features) | 0.6480 | 0.2278 | 0.536 |
| Random forest (12 features) — **rejected, see above** | 0.6463 | 0.2270 | 0.546 |
| **Gradient boosting, monotonic (12 features, shipped)** | **0.6468** | **0.2273** | **0.544** |
| Neural net / MLP (12 features) | 0.6507 | 0.2290 | 0.532 |

The gain over round two is real across all three metrics but modest —
this is the honest ceiling for a single-game outcome that's close to a
coin flip no matter how many features get thrown at it (see round one's
finding on raw feature-outcome correlation). What round three actually
delivers is less about squeezing out more log loss and more about the
other two things that were asked for: a feature set decided by
out-of-sample evidence instead of hand-picking, and reason strings in
`rank._explain()` that only describe features the model actually uses
(no more platoon-split callouts for a feature the model no longer sees).

**What this means for whether the app's picks are actually worth
following** — the metric above (log loss/Brier/AUC) measures calibration
across *every* batter-game in the holdout, not "how often would
following the app's top pick have won." That's a different, more
direct number, computed by `scripts/pick_accuracy.py` on the same true
2026 holdout (139 test dates, restricted to the ~100 highest-PA batters
per season the way the whole backtest is — see
[Known gaps](#known-gaps)):

| | Hit rate |
| :--- | :--- |
| Naive "always guess hit" baseline (this dataset's base rate) | 64.7% |
| Naive "just pick the highest career average" — top pick | 73.4% |
| Naive "just pick the highest career average" — top 2 | 71.2% |
| **Model's top daily pick** | **77.7%** |
| **Model's top 2 daily picks** (each pick's own hit rate) | **74.1%** |
| Double-down joint success (*both* of the top 2 hit, per MLB's actual scoring rule) | 54.7% |

Two things worth being explicit about when comparing this to another
tool's advertised "accuracy": first, the ~65% baseline above means any
tool that mostly recommends everyday, good-hitting players is starting
from a high floor before it's contributed anything — "70% accuracy" on
its own doesn't say much without knowing what it's being measured
against. Second, the double-down number (54.7%) is the realistic
expectation if you play two picks a day the way this game actually
scores it (both must hit) — it's necessarily lower than either pick's
own marginal rate, since it's the *product* of two sub-100% events (with
positive correlation from same-pitcher pairs helping some, per
[Why same-pitcher picks aren't avoided](#why-same-pitcher-picks-arent-avoided)).
If you're comparing against a single-pick tool, the top-daily-pick row
(77.7%) is the fairer comparison.

**Recent form, re-tested a third time:** round one's finding that
last-10-games form adds nothing was from a single 2025-only logistic
regression run. `scripts/test_recent_form.py` re-tests it properly —
same multi-season/true-2026-holdout methodology as the shipped model,
four window definitions (5 games, 10 games, 5 calendar days, 10 calendar
days, each as a delta vs. season-to-date average, each added with a
positive monotonic constraint so a real effect couldn't be suppressed by
the wrong sign) — and the answer holds: all four came back at ~zero
permutation importance (-0.00002 to -0.00011) and every one made holdout
log loss very slightly worse, not better, than the shipped model without
any recent-form feature. A handful of recent games is a small, high-
variance sample; career average already captures the player's true
talent level far more reliably, so "hot streaks" mostly look like noise
once that's controlled for.

**Statcast/Baseball Savant, tested and not shipped:** exit velocity,
barrel rate, expected batting average (xBA/xwOBA/xSLG), sprint speed,
whiff rate, and pitcher-side equivalents (`scripts/statcast.py` fetches
season-level leaderboards from Savant's public CSV export,
`scripts/test_statcast.py` tests them) -- same true-2026-holdout
methodology, 17 candidates, each lagged by one season (Statcast
leaderboards are season aggregates, so each row uses the *prior*
completed season's profile; see statcast.py's docstring for why a
single lag rather than a multi-year blend). Batter-side quality-of-
contact metrics (xBA, xwOBA, xSLG, exit velocity, barrel rate) mostly
made holdout log loss *worse* individually -- they largely re-measure
what `career_avg`/`career_obp` already capture, so they added redundant
noise rather than new signal. A few pitcher-side metrics (xBA, whiff%,
K%, BB%, barrel rate allowed) nudged log loss down a little individually
(0.6468 -> ~0.6466), but **all 17 together scored worse than the shipped
model** (0.6470 vs. 0.6468 log loss, 0.5425 vs. 0.5441 AUC) -- the same
dilution pattern seen elsewhere in this project when weakly-signaled
features get added in bulk. None of the individual gains clear a bar
worth trusting over noise (they're the same tiny magnitude that
separated the shipped model from the random-forest candidate rejected
for being non-monotonic), and Baseball Savant's leaderboard endpoint is
an unofficial, undocumented CSV export -- a real added fragility for a
benefit that isn't clearly established. Not implemented.

**Pruning the shipped model's weakest features, tested and not worth
it:** `scripts/test_pruning.py` refits the 12-feature model, ranks
features by fresh permutation importance, then backward-eliminates the
weakest 1-5 (by that ranking) one at a time. Best candidate (drop
`career_obp`, `pitcher_throws_L`, `bats_L` -- the only three with
importance at or below +0.00003) scored log loss 0.6467 vs. the full
model's 0.6468. A paired bootstrap (2000 resamples of the holdout) on
that gap gives a 95% CI of [-0.00024, +0.00007] -- spanning zero, i.e.
not distinguishable from noise at this holdout's size (10,222 rows).
Not shipped: pruning is a legitimate parsimony argument (9 features
instead of 12, one fewer live API field), but there's no measured
accuracy cost *or* benefit either way, so it wasn't judged worth the
code churn absent an actual improvement.

**Opposing bullpen quality, tested two ways -- the second one changed
the answer:** a batter's 2nd/3rd/4th plate appearances in a game are
often against relievers, not the probable starter the model already
sees, so this looked like a real gap. `scripts/test_bullpen.py` first
tried the *prior completed season's* team bullpen ERA (MLB Stats API's
`byDateRange` stat type silently ignores `sitCodes`, verified directly
by comparing a relief-only query against the whole-staff number for the
same date range and getting identical results back -- so a true
mid-season split isn't available that way). That version looked
promising: log loss 0.6468 -> 0.6464, and 91% of a 2000-draw paired
bootstrap favored including it. Promising enough, and hobbled by
staleness badly enough (bullpens churn within a season -- trades,
injuries, callups -- far more than a hitter's underlying skill does),
that it was worth reconstructing properly: `scripts/test_bullpen_pit.py`
rebuilds a true point-in-time bullpen ERA per team per date from
boxscores already fetched for batting order (a boxscore pitching line
with `gamesStarted == 0` is a relief appearance -- no new network calls,
just more parsing of data already cached), summing only appearances
strictly before each row's date. That version came out **worse** than
the baseline (log loss 0.6468 -> 0.6470) with only 11% of the bootstrap
favoring it -- the signal reversed. Likely explanation: the prior-season
number's apparent lift wasn't really about bullpen quality -- it was
probably proxying for overall team quality (good teams tend to have
both good bullpens and good everyday lineups the model already sees via
`career_avg`/`career_obp`) -- and the true point-in-time number is very
noisy early in a season (a bullpen's ERA after a handful of innings can
be 0.00 or 15.00 by small-sample luck), adding noise rather than signal.
Not implemented. Worth remembering as a general lesson: a feature that
looks good on an admittedly-flawed proxy needs to survive the properly-
built version before it ships, not just the first version that compiles.

**Post-hoc probability calibration, tested and not worth it:**
`scripts/test_calibration.py` wraps the shipped model in
`CalibratedClassifierCV` (isotonic and Platt/sigmoid, both fit via
5-fold cross-validation on training data only, never touching the 2026
test set). Calibration is a monotonic transform, so it can't change
rankings (AUC), only how well the predicted numbers themselves match
reality. Isotonic came back essentially a wash (log loss 0.6467 vs.
0.6468 uncalibrated, bootstrap 95% CI [-0.00083, +0.00068] -- a coin
flip). Sigmoid/Platt was clearly worse (0.6487, CI entirely on the
"worse" side, only 1% of bootstrap draws favoring it) -- it assumes a
specific S-shaped miscalibration that doesn't match this model's actual
output. Makes sense in hindsight: `HistGradientBoostingClassifier`
already directly optimizes log loss during training, so there isn't
much miscalibration left to fix. Not implemented.

**Opposing team defense, tested and not worth it:**
`scripts/test_defense.py` tests team-level Outs Above Average (OAA,
Baseball Savant's modern range-based defensive metric) for the
opposing team, prior-completed-season lag (OAA has no point-in-time
reconstruction available the way bullpen ERA did -- it needs play-by-
play difficulty-adjusted tracking data, not just box-score putouts/
errors). Given what the bullpen test's reversal just showed about
prior-season proxies looking better than they are, this was held to a
higher bar going in. It didn't clear even the lower bar: log loss
0.6468 -> 0.6468 (no change), bootstrap 95% CI [-0.00042, +0.00036]
comfortably spanning zero, 57% of draws favoring it -- a coin flip, not
the lopsided 91% the bullpen proxy showed before it reversed. Not
implemented.

### Round four: batter-vs-pitcher history

Six ideas tested in a row above with no real gain (recent form, three
seasons of Statcast data, feature pruning, bullpen quality twice,
opposing defense, calibration) raised a fair question: was the model
near its practical ceiling given what's cheaply available? One more
idea changed the answer. The MLB Stats API endpoint for a batter's
history against one specific pitcher (`vsPlayer`) needs one request per
(batter, pitcher) pair -- no batching -- and the training set has
~28,500 distinct pairs across 2023-2026, so backtesting it directly
would mean tens of thousands of individual requests against an
undocumented public API. Instead, the backtest computed it from data
already fetched for the rest of the pipeline: every batter's own game
logs already identify the opposing pitcher per game, so a batter's
at-bats/hits against that exact pitcher, strictly before each row's
date, cost nothing extra to reconstruct -- with the honest limitation
that it can only see meetings from 2023 onward, not a batter's full
career against a given pitcher.

Even that undercounted version came back clearly positive: log loss
0.6468 -> 0.6466, and **94.8% of a 2000-draw paired bootstrap favored
including it** -- the most one-sided result of everything tested in
rounds three and four (the bullpen-quality proxy's initial 91% is the
next closest, and that one reversed under a properly-built version).
Two things made this more trustworthy than the bullpen result before it
reversed: there's no obvious confound (bullpen ERA plausibly proxied for
"good team" generally; a specific batter-pitcher relationship doesn't
have an equivalent shortcut explanation), and the *live* version is
actually stronger than what got backtested -- production can call
`vsPlayer` for just that day's actual matchups (a few hundred pairs, not
28,500), getting true full-career history rather than a 2023-2026 slice.
Verified monotonic (a batter who's clearly struggled against this
specific pitcher scores lower; the effect flattens out above roughly
neutral history) by point-check before shipping, same discipline as
every other monotonic constraint in this model.

Promoted to production as `bvp_delta` (delta vs. the batter's own
season-to-date average, same reparametrization as `platoon_delta` and
every other delta feature here) -- the model is now 13 features. Live
integration (`MlbStatsApiSource._get_bvp_delta`) is the one place in
this whole data layer that isn't batched: it's one HTTP request per
matchup, which took a real slate from ~2 seconds to ~35-40 seconds (see
[Performance](#performance)) -- a deliberate tradeoff given the
strength of the result, not an oversight.

**Sample-size correction, added right after shipping:** live traffic
surfaced a real problem -- a batter who'd gone 2-for-2 against a pitcher
showed up with "has hit this pitcher well historically," a real number
from a sample far too small to support that claim. `scripts/
test_bvp_sample_size.py` tested whether `bvp_delta` should be
discounted below some minimum at-bat count, two ways: a hard threshold
(zero it out below N at-bats) and empirical-Bayes-style shrinkage
(`delta * ab/(ab+C)`, pulling small samples toward 0 continuously
instead of an all-or-nothing cutoff). Six candidates tested against the
raw always-on version on the true 2026 holdout: the lightest hard
threshold (`>= 3` at-bats) won outright -- log loss 0.6470 -> 0.6469,
82% of a 2000-draw bootstrap favoring it, the best of everything tried.
Heavier thresholds (`>= 5`, `>= 10`) scored *worse* than the raw
version, not better -- there's real signal in the 3-9 at-bat range that
an aggressive cutoff throws away, even though the smallest samples (1-2
at-bats, ~41% of rows have at least one prior at-bat but most of those
are tiny) really are just noise. Promoted `MIN_BVP_AB = 3` into both
the training pipeline and live serving (`data.MIN_BVP_AB`, kept in sync
by comment) -- `bvp_delta` is now 0.0 (no information) below 3 prior
at-bats against the exact pitcher, at both training and prediction
time. The reason text also now spells out the actual at-bat count
("has hit this pitcher well in 11 career at-bats...") instead of saying
"historically," which even 3-9 at-bats doesn't really support.

Rerun `python scripts/fit_ml_model.py` (fetches and caches 2023-2026
data) then `python scripts/train_ml_model.py` (builds features, trains,
picks a winner among the monotonic-safe candidates, writes
`data/hit_probability_model_v2.json`/`.pkl`) once a year, once the
season being used as the test set completes so it can roll into the
next training window; both need `requirements-analysis.txt`. Review the
printed comparison and permutation importances before promoting —
`fit_hit_probability_model.py`'s output can then retire in favor of
this one, but is kept for now as the historical baseline the retrain
compares against.

### Season-to-date average, re-verified (not the same question as recent form)

Reasonable pushback after round four: recent form (5-10 games) being
noise is one thing, but a full season's cumulative average -- hundreds
of at-bats by midseason -- is a categorically larger, lower-variance
sample. Doesn't a .300 career hitter batting .230 this season deserve a
lower prediction than his career average alone suggests? The original
"season average adds nothing" finding (round one/two) was from the old
single-season logistic regression pipeline on 2025 data only, and had
never been re-checked against the current multi-season/GBM pipeline --
worth actually re-verifying rather than assuming it still holds.

`scripts/test_season_avg.py` tests `season_avg_delta` (season-to-date
average minus career average -- no new data needed, `season_avg_to_date`
is already computed by `build_dataset`). Aggregate result confirms the
old finding: log loss 0.6466 -> 0.6466 (no change), permutation
importance +0.00006, bootstrap 57% one-sided (a coin flip). More
tellingly, direct point-checks (a .300 career hitter swept from .350
down to .150 this season, across four different game contexts) show the
model is **essentially flat for any degree of underperformance** --
there's a small real bump for outperforming career average, but a
season-long slump barely moves the prediction anywhere tested. To rule
out that this was just the constrained model's coarse binning in a
sparse region (~8.5% of training rows have a slump this severe), the
same test was rerun with an unconstrained random forest -- log loss got
worse with the feature added, and the point-check came back flat and
non-monotonic (consistent with noise, not a real relationship the
constrained model was missing).

Not implemented -- genuinely no detectable value here, not a modeling
artifact. Best explanation is regression to the mean, one of
sabermetrics' most established findings: batting average has enormous
BABIP-driven variance, and a career-length sample is a more reliable
estimate of true talent than a partial season, even a badly slumping
one -- the same reasoning real projection systems (Marcel, ZiPS,
Steamer) use to weight multiple years over the current season. This
doesn't rule out that a *specific cause* behind a slump (injury, a real
swing change) could matter -- the model has no way to see that either
way, only the raw average.

## Why same-pitcher picks aren't avoided

`pick_top` used to skip a pick if it shared an opposing pitcher with one
already selected, on the theory that two batters facing the same starter
are a correlated bet and correlated bets are riskier. That's backwards
for how this game actually scores a two-pick day.

MLB's "double down" rule: pick two batters, and **both** need to get a
hit for the day to count — it's not "at least one." Successfully done,
it advances your streak by 2 instead of 1, which is the whole incentive
to pick two at all despite the higher bar.

For a bet that requires *both* legs to hit, positive correlation between
the two outcomes raises the joint success probability, it doesn't lower
it. Two batters facing the same pitcher rise and fall together with that
pitcher's performance that day — a good start suppresses both of them
together, a bad one lifts both of them together — the same logic
daily-fantasy players use on purpose when they "stack" hitters against a
bad starter. Concretely, for two batters the model rates at 70% each:
independent (different games), P(both hit) ≈ 70% × 70% = 49%. Sharing a
pitcher, with even a modest positive correlation (~0.3, a plausible
magnitude for a shared-game effect like this), P(both hit) climbs to
roughly 55% — several points higher, not lower. Avoiding that correlation
only makes sense for an at-least-one-of-two bet, which isn't the rule
here.

`pick_top`'s `avoid_same_game` parameter (default `False`) and the CLI's
`--diversify-pitchers` flag still exist for anyone who wants to
diversify anyway, but the ranked top-N is no longer nudged away from a
same-pitcher pair by default.

## Hosted site

`scripts/generate_site.py` renders today/tomorrow/day-after-tomorrow's
picks as one static HTML page (`_site/index.html`) — no server, no
client-side JS, just an f-string template. `.github/workflows/publish.yml`
runs it and publishes the result to GitHub Pages:

- **On demand**: the page itself has a **↻ Refresh predictions** button
  in the top right, which links straight to the workflow's Actions page
  (also reachable via the repo's **Actions** tab → **Publish picks**).
  Click **Run workflow** there (needs you signed into GitHub; also
  doable from the mobile app) and the page updates in under a minute.
  This is the main way to refresh after new lineups post. A real
  in-page button that triggers the run without leaving the page would
  need a token embedded in the public page — not done, since anyone
  viewing the page could then use it to burn your Actions minutes.
- **On a schedule**, as a baseline so the page is current before it's
  actually looked at. Twice a day, timed around actual usage: ~6:00am
  and ~11:30am Pacific (before lineups are typically checked, and again
  before the lunch-hour check-in). GitHub Actions cron has no timezone
  support, so the two `cron` lines in the workflow file are pinned to
  Pacific Daylight Time (UTC-7) and drift an hour during Pacific
  Standard Time (roughly November-March) until manually shifted for the
  season -- see the comment above them in `.github/workflows/publish.yml`.
  Edit those lines to change the cadence.
- The "Generated at" timestamp on the page is Pacific time. The
  Today/Tomorrow/day-after-tomorrow *date boundaries* stay on US
  Eastern internally regardless of the viewer's timezone, though —
  that's the actual convention MLB's own schedule uses (a 7pm Pacific
  game is already "tomorrow" in Eastern terms some nights), so changing
  it to Pacific would occasionally put a game under the wrong day.

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

One exception: batter-vs-pitcher history (`bvp_delta`, added in the
"Model backtest" section below) uses MLB's `vsPlayer` stat type, which
has no batching -- one request per (batter, pitcher) pair, unlike
everything else above. That's roughly one extra request per matchup
(~150-270/day depending on how many games and confirmed lineups), which
brought a full slate from ~2 seconds to **~35-40 seconds**. Still well
under any GitHub Actions timeout and fine for a page that refreshes
twice a day plus on demand, but a real, deliberate tradeoff -- not free
like the rest of the batching in this file.

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

# Refresh the hit-probability model against more recent seasons
# (needs requirements-analysis.txt):
pip install -r requirements-analysis.txt
python scripts/fit_ml_model.py      # fetch + cache 2023-2026 data
python scripts/train_ml_model.py    # build features, train, compare, write data/hit_probability_model_v2.*

# See how often the model's own daily picks would actually have won,
# on the true 2026 holdout (needs train_ml_model.py's cached dataset):
python scripts/pick_accuracy.py

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
- The model is backtested on the ~100 highest-plate-appearance hitters
  per season; it hasn't been validated on part-time players or true
  rookies specifically — `build_dataset` in `train_ml_model.py` actually
  drops rookies with zero prior-season data entirely from training, and
  the live app falls back to season-level/league-average figures for the
  same case (see `data._career_hitting_rates_entering_season` and
  `_career_pitching_rates_entering_season`), so predictions for a
  first-year player are the least-tested case in the whole pipeline.
- `platoon_delta` and `is_home` were both dropped from the model in round
  three — both came back with ~zero permutation importance on the true
  2026 holdout, confirming what round two's bootstrap had already flagged
  as shaky for each of them (see [Model backtest](#model-backtest)).
- No accounting for a batter's own recent injury/rest status, or same-day
  lineup changes after prediction time. Round three did test rest days
  since each player's last appearance explicitly — it came back with
  ~zero permutation importance and was dropped, so this isn't an
  oversight, just a feature that didn't turn out to matter at the
  single-game level.
- The monotonic constraints (see round three in
  [Model backtest](#model-backtest) for the full list) are asserted from
  baseball domain knowledge, not independently re-derived by the
  gradient-boosting fit itself — a deliberate choice to keep the model
  from being able to contradict signal already confirmed by simpler
  models and direct point-checks, but worth knowing that's what they are.
  `RandomForestClassifier` scored marginally better on raw holdout log
  loss but was rejected for exactly this reason (no monotonic-constraint
  mechanism in scikit-learn) — every future retrain needs the same
  point-check discipline before shipping a new winner, not just a log
  loss comparison.
- The pick-accuracy numbers in round three (77.7% top-pick hit rate, etc.)
  come from the same ~100-highest-PA-batters-per-season proxy dataset as
  the rest of the backtest, not a literal replay of what the live app
  would have shown on those 139 days (which sees every confirmed starter
  in real games that day, not just a fixed top-100 list). It's a
  reasonable proxy — Beat the Streak strategy already favors picking
  everyday, high-PA hitters — but not an exact one.
- `bvp_delta`'s backtest (round four) only sees batter-vs-pitcher
  meetings from 2023 onward, not a batter's true full career against a
  given pitcher — the live app doesn't have this limitation (it calls
  `vsPlayer` fresh per matchup, full career), so live behavior for a
  long-tenured pair is based on more information than what was
  validated. The `vsPlayer` stat type is also the one MLB Stats API
  endpoint this app depends on that isn't batchable, and its response
  shape (a `season: null` aggregate row alongside per-season splits) was
  discovered by direct inspection rather than documented anywhere — if
  MLB changes that shape, `_get_bvp_delta` would need updating, not just
  a version-pin.
- The doubleheader edge case in `_compute_bvp_history` (two games can
  share a `(batter, pitcher, date)` key, since `schedule_map` only
  tracks one pitcher per date+team) is resolved by dropping the
  duplicate rather than fixing the underlying ambiguity — affects
  roughly 1% of computed history rows in the backtest, not the live app
  (which fetches real full-career history per matchup instead).

## Tests

```bash
pip install -e . -r requirements-dev.txt
pytest
```
