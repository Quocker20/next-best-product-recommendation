"""Audit the cleaned Expedia bookings for repeat / duplicate structure and its weight in the splits.

Read-only. Input: data/interim/expedia_bookings_flagged.parquet. Output:
results/week4_rebuild/data_audit.json (every number printed and saved by this script).

Questions answered
- How close in time are burst repeats to the booking they repeat (is float32 timestamp order at risk)?
- Same user, same timestamp ties.
- Near-duplicates beyond the burst rule (same trip, other cluster; same destination and cluster, other dates).
- In the valid / test target sets (approximate RecBole split: non-first bookings, ordered by time,
  80/10/10 by position): how many targets are burst repeats, and how much of the sameDest ceiling
  (target cluster already booked at the query destination) is carried by them?

Usage: .venv/Scripts/python.exe scripts/expedia_data_audit.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nbp.paths import INTERIM, ROOT

CLEAN = INTERIM / "expedia_bookings_flagged.parquet"
OUT = ROOT / "results" / "week4_rebuild" / "data_audit.json"
F32_MARGIN_S = 128  # float32 resolution of unix seconds near 1.4e9


def share(mask: np.ndarray) -> dict:
    return {"n": int(mask.sum()), "share": float(mask.mean())}


AGE_BINS = [-1, 128, 3600, 86400, 7 * 86400, 30 * 86400, 1e12]
AGE_LABELS = ["<=128s", "<=1h", "<=1d", "<=7d", "<=30d", ">30d"]


def profile(df: pd.DataFrame) -> dict:
    """User, time and drift facts behind the cleaning strategy (df sorted by user, time)."""
    out: dict = {}
    per_user = df.groupby("user_id", sort=False).size()
    out["bookings_per_user"] = {
        "max": int(per_user.max()),
        "users_at_max": int((per_user == per_user.max()).sum()),
        "users_ge_95": int((per_user >= 95).sum()),
        "users_ge_50": int((per_user >= 50).sum()),
        "rows_share_users_ge_50": float(per_user[per_user >= 50].sum() / len(df)),
    }
    year = df["timestamp"].dt.year
    family = (df["ctx_children"] > 0).to_numpy()
    out["family_share_by_year"] = {
        str(y): float(family[(year == y).to_numpy()].mean()) for y in sorted(year.unique())
    }
    out["rows_by_month_2013_vs_2014"] = {
        str(y): int((year == y).sum()) for y in sorted(year.unique())
    }
    ids = set(
        pd.read_csv(
            INTERIM.parents[1] / "data" / "raw" / "hospitality" / "expedia" / "destinations.csv",
            usecols=["srch_destination_id"],
        )["srch_destination_id"]
    )
    out["destinations_csv_row_coverage"] = float(df["ctx_dest_id"].isin(ids).mean())
    # age of the latest earlier booking of the same cluster at the same destination (sameDest hits)
    t = df["timestamp"].astype("int64") // 10**9
    key = (
        df["user_id"].astype("int64") * 10**9
        + df["ctx_dest_id"].astype("int64") * 128
        + df["item_id"].astype("int64")
    )
    age = (t - t.groupby(key, sort=False).shift()).to_numpy(dtype="float64")
    hit = ~np.isnan(age)
    bins = pd.cut(pd.Series(age[hit]), AGE_BINS, labels=AGE_LABELS).value_counts(sort=False)
    out["sameDest_cluster_hit_age"] = {
        "rows_with_earlier_same_dest_and_cluster": int(hit.sum()),
        **{k: int(v) for k, v in bins.items()},
    }
    for w in (3600, 6 * 3600, 86400):
        f = (age <= w) & ~df["flag_burst_repeat"].to_numpy(dtype=bool)
        out[f"session_repeat_non_burst_le_{w}s"] = int(f.sum())
    return out


def main() -> None:
    df = pd.read_parquet(
        CLEAN,
        columns=[
            "user_id",
            "item_id",
            "timestamp",
            "src_row",
            "ctx_dest_id",
            "ctx_ci",
            "ctx_co",
            "flag_burst_repeat",
            "ctx_children",
            "tgt_orig_distance",
            "ctx_user_country",
        ],
    )
    df = df.sort_values(["user_id", "timestamp", "src_row"], kind="stable").reset_index(drop=True)
    g = df.groupby("user_id", sort=False)
    ts = df["timestamp"].astype("int64") // 10**9
    prev_ts = ts.groupby(df["user_id"], sort=False).shift()
    gap = (ts - prev_ts).to_numpy(dtype="float64")  # seconds to the user's previous booking
    pos = g.cumcount().to_numpy()
    burst = df["flag_burst_repeat"].to_numpy(dtype=bool)

    res: dict = {"rows": len(df)}

    # 1. timing of burst repeats
    bg = gap[burst]
    res["burst_gap_seconds"] = {
        "n": int(burst.sum()),
        "le_0s": int((bg <= 0).sum()),
        "le_128s": int((bg <= F32_MARGIN_S).sum()),
        "le_1h": int((bg <= 3600).sum()),
        "le_1d": int((bg <= 86400).sum()),
        "gt_1d": int((bg > 86400).sum()),
        "median_s": float(np.median(bg)),
    }
    non_first = pos > 0
    res["gap_le_128s_any_pair"] = {
        "n": int(((gap <= F32_MARGIN_S) & non_first).sum()),
        "of_which_burst": int(((gap <= F32_MARGIN_S) & non_first & burst).sum()),
    }
    res["same_timestamp_pairs_same_user"] = int(((gap == 0) & non_first).sum())

    # 2. near-duplicates not caught by the burst rule (vs the user's previous booking)
    def same(c: str) -> np.ndarray:
        return df[c].eq(df.groupby("user_id", sort=False)[c].shift()).to_numpy()

    same_trip = same("ctx_dest_id") & same("ctx_ci") & same("ctx_co")
    same_cluster = same("item_id")
    res["vs_previous_booking"] = {
        "same_trip_same_cluster (= burst rule)": int((same_trip & same_cluster & non_first).sum()),
        "same_trip_other_cluster": int((same_trip & ~same_cluster & non_first).sum()),
        "same_dest_same_cluster_other_dates": int(
            (same("ctx_dest_id") & same_cluster & ~same_trip & non_first).sum()
        ),
    }

    # 3. weight in the targets (approximate RecBole split over non-first bookings)
    tg = df.loc[non_first].sort_values(["timestamp", "user_id", "src_row"], kind="stable")
    n = len(tg)
    cut1, cut2 = int(n * 0.8), int(n * 0.9)
    split = np.array(["train"] * cut1 + ["valid"] * (cut2 - cut1) + ["test"] * (n - cut2))
    tg = tg.assign(split=split)
    # sameDest ceiling: target cluster booked earlier by the same user at the same destination
    key = df["user_id"].astype("int64") * 10**6 + df["ctx_dest_id"].astype("int64")
    kc = pd.Series(key.to_numpy() * 128 + df["item_id"].to_numpy().astype("int64"), index=df.index)
    seen_before = kc.groupby(kc, sort=False).cumcount().to_numpy() > 0
    same_dest_hit = pd.Series(seen_before, index=df.index).loc[tg.index].to_numpy()
    burst_t = tg["flag_burst_repeat"].to_numpy(dtype=bool)
    # a target whose only same-destination hit is the burst-previous booking
    # (approximation: burst target and no other earlier booking of this cluster at this destination)
    prior_cnt = kc.groupby(kc, sort=False).cumcount().to_numpy()
    only_burst = pd.Series(burst & (prior_cnt == 1), index=df.index).loc[tg.index].to_numpy()
    out_split = {}
    for sp in ("train", "valid", "test"):
        m = (tg["split"] == sp).to_numpy()
        out_split[sp] = {
            "targets": int(m.sum()),
            "burst_target": share(burst_t[m]),
            "sameDest_ceiling (target cluster booked earlier at query destination)": share(
                same_dest_hit[m]
            ),
            "ceiling_rows_that_are_burst": share(burst_t[m] & same_dest_hit[m]),
            "ceiling_rows_where_burst_previous_is_the_only_hit": share(only_burst[m]),
            "ceiling_without_burst_only_rows": float((same_dest_hit[m] & ~only_burst[m]).mean()),
        }
    res["targets_approx_split"] = out_split

    # 4. other context defects
    res["missing"] = {
        "orig_destination_distance_null": share(df["tgt_orig_distance"].isna().to_numpy()),
        "user_country_null": share(df["ctx_user_country"].isna().to_numpy()),
    }

    res["profile"] = profile(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
