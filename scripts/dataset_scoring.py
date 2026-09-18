"""Helpers for `docs/dataset_rubric.md`.

Two pieces:
    - quick_profile(): summary stats (rows/users/items/sparsity/...) for a
      cleaned interaction table (Step 0 in the rubric), used for the C2-C4/C8
      criteria.
    - score_dataset(): applies the Step 2 weights and Step 3 formula/decision
      thresholds to a dict of 0-3 criterion scores. Scoring the criteria
      themselves is a judgment call made by reading the EDA notebook + data
      card — this only does the arithmetic, so the weighted sum in
      `docs/dataset_scores.md` stays reproducible instead of hand-computed.

Usage:
    python scripts/dataset_scoring.py --scores C1=3 C2=3 C3=0 C4=3 C5=1 C6=2 C7=3 C8=2 C9=1 C10=3
"""

import argparse
from typing import Optional

import pandas as pd

# Step 2 weights, in the order they're summed for the Step 3 formula.
CRITERIA_WEIGHTS = {
    "C1": 3,
    "C2": 2,
    "C3": 2,
    "C4": 2,
    "C5": 2,
    "C6": 1,
    "C7": 1,
    "C8": 1,
    "C9": 1,
    "C10": 1,
}
MAX_POINTS = sum(3 * w for w in CRITERIA_WEIGHTS.values())  # 48


def quick_profile(
    df: pd.DataFrame,
    user: str = "user_id",
    item: str = "item_id",
    ts: Optional[str] = "timestamp",
) -> dict:
    """Summary stats for a cleaned interaction table (Step 0 mapping already applied).

    Inputs: df with at least `user`/`item` columns (and `ts` unless None).
    Output: dict with rows, users, items, sparsity, median_inter_per_user,
    pct_users_ge5, time_span_days, repeat_rate, mem_mb.
    """
    per_user = df.groupby(user).size()
    return {
        "rows": len(df),
        "users": df[user].nunique(),
        "items": df[item].nunique(),
        "sparsity": 1 - len(df) / (df[user].nunique() * df[item].nunique()),
        "median_inter_per_user": per_user.median(),
        "pct_users_ge5": (per_user >= 5).mean(),
        "time_span_days": (
            pd.to_datetime(df[ts]).max() - pd.to_datetime(df[ts]).min()
        ).days
        if ts
        else None,
        "repeat_rate": 1 - df[[user, item]].drop_duplicates().shape[0] / len(df),
        "mem_mb": df.memory_usage(deep=True).sum() / 1e6,
    }


def score_dataset(criteria: dict[str, int], exclude: tuple[str, ...] = ()) -> dict:
    """Apply Step 2 weights and the Step 3 formula/decision thresholds.

    Input: criteria, a dict mapping "C1".."C10" to a 0-3 score (all 10 required).
           exclude, criteria to drop from both numerator and denominator (e.g.
           ("C9",) once cross-sell is out of scope) -- the max is rescaled so
           the 0-100 scale and the tier thresholds still apply.
    Output: dict with weighted_points, max_points, score (0-100), decision.
    """
    weights = {k: w for k, w in CRITERIA_WEIGHTS.items() if k not in exclude}
    max_points = sum(3 * w for w in weights.values())
    missing = set(CRITERIA_WEIGHTS) - set(criteria)
    if missing:
        raise ValueError(f"missing criteria scores: {sorted(missing)}")
    for name, value in criteria.items():
        if name not in CRITERIA_WEIGHTS:
            raise ValueError(f"unknown criterion: {name}")
        if not 0 <= value <= 3:
            raise ValueError(f"{name}={value} out of range 0-3")

    weighted_points = sum(criteria[name] * weight for name, weight in weights.items())
    score = round(weighted_points / max_points * 100)

    if score >= 70:
        decision = "Core"
    elif score >= 50:
        decision = "Secondary"
    else:
        decision = "Drop"

    return {
        "weighted_points": weighted_points,
        "max_points": max_points,
        "score": score,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="criterion=value pairs, e.g. C1=3 C2=2 ...",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=(), help="criteria to drop, e.g. C9"
    )
    args = parser.parse_args()

    criteria = {}
    for pair in args.scores:
        name, value = pair.split("=")
        criteria[name] = int(value)

    result = score_dataset(criteria, exclude=tuple(args.exclude))
    print(
        f"{result['weighted_points']}/{result['max_points']} -> score={result['score']} decision={result['decision']}"
    )


if __name__ == "__main__":
    main()
