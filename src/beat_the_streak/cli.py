"""Print today's (or any date's) top Beat the Streak picks."""

from __future__ import annotations

import argparse
import sys

from .data import DataSource, FixtureSource, MlbStatsApiSource
from .rank import pick_top, rank_matchups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="Slate date, YYYY-MM-DD")
    parser.add_argument(
        "--source",
        choices=["mlbapi", "fixture"],
        default="mlbapi",
        help="Where to pull the slate from (default: mlbapi, the live MLB "
        "Stats API; use 'fixture' with --fixture-path for offline runs)",
    )
    parser.add_argument(
        "--fixture-path",
        default=None,
        help="Path to a JSON fixture file (required when --source=fixture)",
    )
    parser.add_argument(
        "--picks", type=int, default=2, help="Number of picks to select (default: 2)"
    )
    parser.add_argument(
        "--show", type=int, default=10, help="How many top-ranked matchups to print (default: 10)"
    )
    parser.add_argument(
        "--allow-same-game",
        action="store_true",
        help="Allow two picks facing the same pitcher (correlated risk, off by default)",
    )
    return parser


def build_source(args: argparse.Namespace) -> DataSource:
    if args.source == "fixture":
        if not args.fixture_path:
            raise SystemExit("--fixture-path is required when --source=fixture")
        return FixtureSource(args.fixture_path)
    return MlbStatsApiSource()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source = build_source(args)

    matchups = source.get_matchups(args.date)
    if not matchups:
        print(f"No matchups found for {args.date}.")
        return 1

    ranked = rank_matchups(matchups)
    result = pick_top(ranked, n=args.picks, avoid_same_game=not args.allow_same_game)

    print(f"Top {min(args.show, len(ranked))} matchups for {args.date}:\n")
    for i, pick in enumerate(ranked[: args.show], start=1):
        _print_pick(i, pick)

    print(f"\n{'=' * 60}")
    print(f"RECOMMENDED PICKS ({len(result.picks)}):")
    for i, pick in enumerate(result.picks, start=1):
        _print_pick(i, pick)

    if result.skipped_for_correlation:
        print("\n(skipped for sharing a pitcher with an earlier pick:)")
        for pick in result.skipped_for_correlation:
            print(f"  - {pick.matchup.batter.name} ({pick.hit_probability:.1%})")

    return 0


def _print_pick(rank: int, pick) -> None:
    m = pick.matchup
    loc = "vs" if m.is_home else "@"
    reasons = "; ".join(pick.reasons) if pick.reasons else "no standout factors"
    print(
        f"{rank:>2}. {m.batter.name:<22} {pick.hit_probability:>5.1%}  "
        f"{loc} {m.pitcher.name} ({m.pitcher.throws}HP)  -- {reasons}"
    )


if __name__ == "__main__":
    sys.exit(main())
