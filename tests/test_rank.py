from __future__ import annotations

from beat_the_streak.rank import pick_top, rank_matchups, score_matchup

from .conftest import make_batter, make_matchup, make_pitcher, make_recent_form


def test_hit_probability_is_between_0_and_1():
    pick = score_matchup(make_matchup())
    assert 0.0 < pick.hit_probability < 1.0


def test_hot_recent_form_scores_higher_than_cold_form():
    hot = make_matchup(recent_form=make_recent_form([1] * 10))
    cold = make_matchup(recent_form=make_recent_form([0] * 10))
    assert score_matchup(hot).hit_probability > score_matchup(cold).hit_probability


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


def test_thin_recent_sample_is_shrunk_toward_season_avg():
    # A 4-game hot streak shouldn't swing the estimate as much as the same
    # rate sustained over a full 10-game window would.
    thin_hot = make_matchup(
        batter=make_batter(season_avg=0.230, season_avg_vs_lhp=0.230, season_avg_vs_rhp=0.230),
        recent_form=make_recent_form([1] * 4),  # 4 games at 1.000
    )
    full_hot = make_matchup(
        batter=make_batter(season_avg=0.230, season_avg_vs_lhp=0.230, season_avg_vs_rhp=0.230),
        recent_form=make_recent_form([1] * 10),  # 10 games at 1.000
    )
    assert (
        score_matchup(thin_hot).hit_probability
        < score_matchup(full_hot).hit_probability
    )


def test_rank_matchups_sorts_descending():
    hot = make_matchup(recent_form=make_recent_form([1] * 10))
    cold = make_matchup(recent_form=make_recent_form([0] * 10))
    ranked = rank_matchups([cold, hot])
    assert ranked[0].hit_probability >= ranked[1].hit_probability


def test_pick_top_avoids_same_pitcher_by_default():
    shared_pitcher = make_pitcher(id="shared")
    best = make_matchup(
        batter=make_batter(id="b1"), pitcher=shared_pitcher, recent_form=make_recent_form([1] * 10)
    )
    second_best_same_pitcher = make_matchup(
        batter=make_batter(id="b2"), pitcher=shared_pitcher, recent_form=make_recent_form([1, 0] * 5)
    )
    third_best_different_pitcher = make_matchup(
        batter=make_batter(id="b3"),
        pitcher=make_pitcher(id="other"),
        recent_form=make_recent_form([1, 0, 0, 1] * 2),
    )
    ranked = rank_matchups(
        [best, second_best_same_pitcher, third_best_different_pitcher]
    )
    result = pick_top(ranked, n=2, avoid_same_game=True)

    picked_batter_ids = {p.matchup.batter.id for p in result.picks}
    assert "b1" in picked_batter_ids
    assert "b2" not in picked_batter_ids  # shares b1's pitcher, should be skipped
    assert len(result.skipped_for_correlation) == 1


def test_pick_top_allows_same_pitcher_when_disabled():
    shared_pitcher = make_pitcher(id="shared")
    m1 = make_matchup(batter=make_batter(id="b1"), pitcher=shared_pitcher)
    m2 = make_matchup(batter=make_batter(id="b2"), pitcher=shared_pitcher)
    ranked = rank_matchups([m1, m2])
    result = pick_top(ranked, n=2, avoid_same_game=False)
    assert len(result.picks) == 2
    assert result.skipped_for_correlation == []


def test_pick_top_backfills_when_avoidance_leaves_slate_short():
    shared_pitcher = make_pitcher(id="shared")
    m1 = make_matchup(batter=make_batter(id="b1"), pitcher=shared_pitcher)
    m2 = make_matchup(batter=make_batter(id="b2"), pitcher=shared_pitcher)
    ranked = rank_matchups([m1, m2])
    result = pick_top(ranked, n=2, avoid_same_game=True)
    # Only one distinct pitcher exists, so avoidance alone can't fill 2 picks;
    # backfill should still return 2.
    assert len(result.picks) == 2
