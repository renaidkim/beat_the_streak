from __future__ import annotations

from beat_the_streak.rank import pick_top, rank_matchups, score_matchup

from .conftest import make_batter, make_matchup, make_pitcher


def test_hit_probability_is_between_0_and_1():
    pick = score_matchup(make_matchup())
    assert 0.0 < pick.hit_probability < 1.0


def test_higher_career_avg_scores_higher():
    strong = make_matchup(batter=make_batter(career_avg=0.320))
    weak = make_matchup(batter=make_batter(career_avg=0.210))
    assert score_matchup(strong).hit_probability > score_matchup(weak).hit_probability


def test_tougher_pitcher_lowers_probability():
    vs_tough = make_matchup(pitcher=make_pitcher(oba_against=0.200))
    vs_weak = make_matchup(pitcher=make_pitcher(oba_against=0.310))
    assert (
        score_matchup(vs_tough).hit_probability
        < score_matchup(vs_weak).hit_probability
    )


def test_hitter_friendly_park_raises_probability():
    neutral = make_matchup(park_factor=1.0)
    hitter_park = make_matchup(park_factor=1.15)
    assert (
        score_matchup(hitter_park).hit_probability
        > score_matchup(neutral).hit_probability
    )


def test_favorable_platoon_split_raises_probability():
    # Batter hits well above his season average against this pitcher's
    # throwing hand specifically. platoon_delta is only weakly/
    # conditionally stable in the fitted model (see the bootstrap note in
    # scripts/fit_hit_probability_model.py -- it's the one feature that
    # isn't sign-stable across resamples), so unlike the other feature
    # tests here this needs a context away from the model's "everything
    # at a bland default" flat region to actually show movement -- the
    # away/leadoff-spot context below is empirically confirmed responsive.
    pitcher = make_pitcher(throws="L", oba_against=0.24)
    context = dict(is_home=False, batting_order=1, park_factor=1.0)
    favorable = make_matchup(
        batter=make_batter(career_avg=0.28, season_avg=0.250, season_avg_vs_lhp=0.350, season_avg_vs_rhp=0.250),
        pitcher=pitcher,
        **context,
    )
    neutral = make_matchup(
        batter=make_batter(career_avg=0.28, season_avg=0.250, season_avg_vs_lhp=0.250, season_avg_vs_rhp=0.250),
        pitcher=pitcher,
        **context,
    )
    assert (
        score_matchup(favorable).hit_probability
        > score_matchup(neutral).hit_probability
    )


def test_batting_near_top_of_order_scores_higher():
    leadoff = make_matchup(batting_order=1)
    ninth = make_matchup(batting_order=9)
    assert score_matchup(leadoff).hit_probability > score_matchup(ninth).hit_probability


def test_unknown_batting_order_falls_back_to_neutral_default():
    unknown = score_matchup(make_matchup(batting_order=None))
    middle = score_matchup(make_matchup(batting_order=5))
    assert unknown.hit_probability == middle.hit_probability


def test_unconfirmed_pitcher_is_flagged_in_reasons():
    matchup = make_matchup(pitcher=make_pitcher(confirmed=False))
    pick = score_matchup(matchup)
    assert any("not yet announced" in reason for reason in pick.reasons)


def test_confirmed_pitcher_has_no_tbd_reason():
    matchup = make_matchup(pitcher=make_pitcher(confirmed=True))
    pick = score_matchup(matchup)
    assert not any("not yet announced" in reason for reason in pick.reasons)


def test_rank_matchups_sorts_descending():
    strong = make_matchup(batter=make_batter(career_avg=0.320))
    weak = make_matchup(batter=make_batter(career_avg=0.210))
    ranked = rank_matchups([weak, strong])
    assert ranked[0].hit_probability >= ranked[1].hit_probability


def test_pick_top_allows_same_pitcher_by_default():
    # MLB's double-down rule needs BOTH picks to hit; positive correlation
    # (two batters sharing a pitcher) helps a both-must-succeed bet rather
    # than hurting it, so the top 2 by probability should be picked as-is
    # even when they share a pitcher. See rank.pick_top's docstring.
    shared_pitcher = make_pitcher(id="shared")
    m1 = make_matchup(batter=make_batter(id="b1"), pitcher=shared_pitcher)
    m2 = make_matchup(batter=make_batter(id="b2"), pitcher=shared_pitcher)
    ranked = rank_matchups([m1, m2])
    result = pick_top(ranked, n=2)
    assert {p.matchup.batter.id for p in result.picks} == {"b1", "b2"}
    assert result.skipped_for_correlation == []


def test_pick_top_diversify_pitchers_opt_in_skips_shared_pitcher():
    shared_pitcher = make_pitcher(id="shared")
    best = make_matchup(
        batter=make_batter(id="b1", career_avg=0.320), pitcher=shared_pitcher
    )
    second_best_same_pitcher = make_matchup(
        batter=make_batter(id="b2", career_avg=0.300), pitcher=shared_pitcher
    )
    third_best_different_pitcher = make_matchup(
        batter=make_batter(id="b3", career_avg=0.280),
        pitcher=make_pitcher(id="other"),
    )
    ranked = rank_matchups(
        [best, second_best_same_pitcher, third_best_different_pitcher]
    )
    result = pick_top(ranked, n=2, avoid_same_game=True)

    picked_batter_ids = {p.matchup.batter.id for p in result.picks}
    assert "b1" in picked_batter_ids
    assert "b2" not in picked_batter_ids  # shares b1's pitcher, should be skipped
    assert len(result.skipped_for_correlation) == 1


def test_pick_top_backfills_when_avoidance_leaves_slate_short():
    shared_pitcher = make_pitcher(id="shared")
    m1 = make_matchup(batter=make_batter(id="b1"), pitcher=shared_pitcher)
    m2 = make_matchup(batter=make_batter(id="b2"), pitcher=shared_pitcher)
    ranked = rank_matchups([m1, m2])
    result = pick_top(ranked, n=2, avoid_same_game=True)
    # Only one distinct pitcher exists, so avoidance alone can't fill 2 picks;
    # backfill should still return 2.
    assert len(result.picks) == 2
