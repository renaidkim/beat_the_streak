"""How often would following the app's actual daily picks have won?

log_loss/brier/AUC (reported by train_ml_model.py) measure calibration
and ranking quality across every batter-game in the holdout, but they
don't directly answer the question a user actually cares about: if I
follow the app's top N picks each day, what fraction of those picks get
a hit? That's "accuracy" in the sense a competing product's marketing
number probably means -- and it needs a well-defined baseline, since
~65% of plate appearances in this dataset already end in a hit, so a
model that always guesses "hit" claims ~65% "accuracy" without knowing
anything. This script reports the honest number: precision of the
actual top-1 and top-2 daily picks, on the same true 2026 out-of-time
holdout, plus what a naive "always pick the highest career average"
baseline would have gotten, for context.

Run after train_ml_model.py (reads its cached test_dataset.csv and the
model it just wrote).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / ".cache" / "ml_model"


def main() -> int:
    test_df = pd.read_csv(CACHE_DIR / "test_dataset.csv", parse_dates=["date"])

    meta = json.loads((REPO_ROOT / "data" / "hit_probability_model.json").read_text())
    model = joblib.load(REPO_ROOT / "data" / "hit_probability_model.pkl")
    features = meta["features"]

    test_df["pred"] = model.predict_proba(test_df[features])[:, 1]

    print(f"Test rows: {len(test_df)}, distinct dates: {test_df['date'].nunique()}")
    print(f"Overall hit rate (naive 'always guess hit' baseline): {test_df['got_hit'].mean():.3f}")

    def top_n_hit_rate(df: pd.DataFrame, n: int, rank_col: str) -> tuple[float, int]:
        picks = (
            df.sort_values(rank_col, ascending=False)
            .groupby("date", group_keys=False)
            .head(n)
        )
        return picks["got_hit"].mean(), len(picks)

    for n in (1, 2):
        rate, count = top_n_hit_rate(test_df, n, "pred")
        print(f"Model's top-{n} daily pick(s): hit rate {rate:.3f} over {count} picks")

    for n in (1, 2):
        rate, count = top_n_hit_rate(test_df, n, "career_avg")
        print(f"Naive top-{n} by career_avg alone: hit rate {rate:.3f} over {count} picks")

    # Double-down (n=2): the rule needs BOTH to hit. Compute the actual
    # joint rate, not the marginal per-pick rate above.
    both_hit = (
        test_df.sort_values("pred", ascending=False)
        .groupby("date", group_keys=False)
        .head(2)
        .groupby("date")["got_hit"]
        .agg(lambda s: int(len(s) == 2 and s.sum() == 2))
    )
    days_with_2 = (
        test_df.groupby("date").size().loc[lambda s: s >= 2].index
    )
    both_hit_valid = both_hit.loc[both_hit.index.isin(days_with_2)]
    print(
        f"\nDouble-down (both of top-2 picks hit): "
        f"{both_hit_valid.mean():.3f} over {len(both_hit_valid)} days with >=2 candidates"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
