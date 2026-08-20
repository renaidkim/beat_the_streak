from __future__ import annotations

from pathlib import Path

import pytest

from beat_the_streak.data import FixtureSource

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_slate_2026-08-19.json"


def test_fixture_source_loads_matchups():
    source = FixtureSource(FIXTURE)
    matchups = source.get_matchups("2026-08-19")
    assert len(matchups) == 8
    names = {m.batter.name for m in matchups}
    assert "Marcus Delgado" in names


def test_fixture_source_rejects_wrong_date():
    source = FixtureSource(FIXTURE)
    with pytest.raises(ValueError):
        source.get_matchups("2026-08-20")
