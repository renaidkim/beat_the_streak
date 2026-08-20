"""Render today/tomorrow/day-after-tomorrow's picks as one static HTML
page. No client-side JS, no build step -- just an f-string template, so
it can run inside a plain GitHub Actions job and get uploaded straight to
GitHub Pages.

Usage:
    python scripts/generate_site.py [--days 3] [--out _site]
"""

from __future__ import annotations

import argparse
import datetime
import html
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from beat_the_streak.data import MlbStatsApiSource  # noqa: E402
from beat_the_streak.rank import Pick, pick_top, rank_matchups  # noqa: E402

# Used to decide which calendar date is "today" -- MLB's own schedule is
# organized by this same US Eastern "baseball day" boundary (see
# beat_the_streak/data.py), not wherever this script happens to run or
# wherever the reader is. Deliberately NOT swapped for the reader's local
# time: a 7pm Pacific game is already "tomorrow" in Eastern terms some
# nights, so using Pacific here would occasionally show the wrong day's
# slate under "Today". Only the *displayed* generated-at timestamp below
# uses Pacific, for readability -- that doesn't affect which games appear.
EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")

GITHUB_ACTIONS_URL = "https://github.com/renaidkim/beat_the_streak/actions/workflows/publish.yml"


def render_page(days: int) -> str:
    source = MlbStatsApiSource()
    today = datetime.datetime.now(EASTERN).date()
    generated_at = datetime.datetime.now(PACIFIC).strftime("%Y-%m-%d %I:%M %p %Z")

    day_sections = []
    for offset in range(days):
        date = today + datetime.timedelta(days=offset)
        day_sections.append(render_day(source, date, label=_day_label(offset)))

    return _PAGE_TEMPLATE.format(
        generated_at=generated_at,
        github_actions_url=GITHUB_ACTIONS_URL,
        day_sections="\n".join(day_sections),
    )


def _day_label(offset: int) -> str:
    return {0: "Today", 1: "Tomorrow", 2: "Day after tomorrow"}.get(offset, f"+{offset} days")


def render_day(source: MlbStatsApiSource, date: datetime.date, label: str) -> str:
    date_str = date.isoformat()
    try:
        matchups = source.get_matchups(date_str)
    except Exception as exc:  # noqa: BLE001 -- surface the failure on the page, don't kill the whole build
        return f"""
        <section class="day">
          <h2>{html.escape(label)} <span class="date">{date_str}</span></h2>
          <p class="error">Couldn't load this day's slate: {html.escape(str(exc))}</p>
        </section>"""

    if not matchups:
        return f"""
        <section class="day">
          <h2>{html.escape(label)} <span class="date">{date_str}</span></h2>
          <p class="muted">No games scheduled.</p>
        </section>"""

    ranked = rank_matchups(matchups)
    result = pick_top(ranked, n=2)
    n_tbd = sum(1 for p in ranked if not p.matchup.pitcher.confirmed)

    picks_html = "\n".join(_pick_card(p, featured=True) for p in result.picks)
    rows_html = "\n".join(_table_row(rank, p) for rank, p in enumerate(ranked[:25], start=1))

    tbd_note = (
        f'<p class="muted">{n_tbd} of {len(ranked)} matchups have no announced starter yet '
        "and are scored with a league-average assumption -- check back closer to game time.</p>"
        if n_tbd
        else ""
    )

    return f"""
    <section class="day">
      <h2>{html.escape(label)} <span class="date">{date_str}</span></h2>
      <div class="picks">
        {picks_html}
      </div>
      {tbd_note}
      <details>
        <summary>Full ranked slate ({len(ranked)} batters)</summary>
        <table>
          <thead>
            <tr><th>#</th><th>Batter</th><th>P(hit)</th><th>Matchup</th><th>Why</th></tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </details>
    </section>"""


def _pick_card(pick: Pick, featured: bool) -> str:
    m = pick.matchup
    loc = "vs" if m.is_home else "@"
    reasons = "; ".join(pick.reasons) if pick.reasons else "no standout factors"
    cls = "pick-card featured" if featured else "pick-card"
    return f"""
        <div class="{cls}">
          <div class="pick-prob">{pick.hit_probability:.0%}</div>
          <div class="pick-body">
            <div class="pick-name">{html.escape(m.batter.name)}</div>
            <div class="pick-matchup">{loc} {html.escape(m.pitcher.name)} ({html.escape(m.pitcher.throws)}HP)</div>
            <div class="pick-reasons">{html.escape(reasons)}</div>
          </div>
        </div>"""


def _table_row(rank: int, pick: Pick) -> str:
    m = pick.matchup
    loc = "vs" if m.is_home else "@"
    reasons = "; ".join(pick.reasons) if pick.reasons else "—"
    tbd = "" if m.pitcher.confirmed else ' <span class="tbd-badge">TBD</span>'
    return (
        f"<tr><td>{rank}</td>"
        f"<td>{html.escape(m.batter.name)}</td>"
        f"<td>{pick.hit_probability:.1%}</td>"
        f"<td>{loc} {html.escape(m.pitcher.name)}{tbd}</td>"
        f"<td>{html.escape(reasons)}</td></tr>"
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Beat the Streak Picks</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0f1115; --card: #171a21; --text: #e8eaed; --muted: #9aa0a6;
    --accent: #4fc3f7; --border: #2a2e37;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg: #f7f8fa; --card: #ffffff; --text: #1a1a1a; --muted: #666; --accent: #0369a1; --border: #e1e4e8; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1.5rem 1rem 3rem; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 720px; margin-left: auto; margin-right: auto;
  }}
  .header {{
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  .generated {{ color: var(--muted); font-size: 0.85rem; }}
  .refresh-btn {{
    display: inline-flex; align-items: center; gap: 0.4rem; flex-shrink: 0;
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.5rem 0.9rem; color: var(--text); text-decoration: none; font-size: 0.85rem;
    font-weight: 600;
  }}
  .refresh-btn:hover {{ border-color: var(--accent); }}
  .refresh-hint {{ color: var(--muted); font-size: 0.78rem; margin-top: 0.3rem; max-width: 14rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  h2 .date {{ color: var(--muted); font-weight: normal; font-size: 0.85rem; }}
  .picks {{ display: flex; flex-direction: column; gap: 0.6rem; }}
  .pick-card {{
    display: flex; align-items: center; gap: 1rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem;
  }}
  .pick-card.featured {{ border-color: var(--accent); }}
  .pick-prob {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); min-width: 3.2rem; text-align: right; }}
  .pick-name {{ font-weight: 600; }}
  .pick-matchup, .pick-reasons {{ color: var(--muted); font-size: 0.85rem; }}
  .muted {{ color: var(--muted); font-size: 0.9rem; }}
  .error {{ color: #e57373; font-size: 0.9rem; }}
  details {{ margin-top: 1rem; }}
  summary {{ cursor: pointer; color: var(--muted); font-size: 0.9rem; padding: 0.3rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0.6rem; font-size: 0.82rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; }}
  .tbd-badge {{
    display: inline-block; font-size: 0.7rem; padding: 0.05rem 0.35rem; border-radius: 4px;
    background: var(--border); color: var(--muted); margin-left: 0.3rem;
  }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Beat the Streak Picks</h1>
    <p class="generated">Generated {generated_at}</p>
  </div>
  <div>
    <a class="refresh-btn" href="{github_actions_url}" target="_blank" rel="noopener">&#8635; Refresh predictions</a>
    <p class="refresh-hint">Opens GitHub Actions -- click "Run workflow" to pull fresh lineups (~30s).</p>
  </div>
</div>
{day_sections}
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3, help="How many days ahead to render (default: 3)")
    parser.add_argument("--out", default="_site", help="Output directory (default: _site)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(days=args.days)
    (out_dir / "index.html").write_text(page)
    print(f"Wrote {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
