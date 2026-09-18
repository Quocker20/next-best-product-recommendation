"""Booking-level profile of Expedia Hotel Recommendations for the week-1 report.

Streams `data/raw/hospitality/expedia/train.csv` in chunks, keeps `is_booking == 1`
rows only, and computes the statistics quoted on the report slides (history depth,
repeat share, trip-context mix, check-in seasonality, temporal-split feasibility,
cluster concentration). Every number in `reports/slides/` / the Expedia data card
that is not already in the data card comes from this script's output, per
CLAUDE.md §7.

Input:  data/raw/hospitality/expedia/train.csv (read-only, 37.7M rows)
Output: reports/summary/expedia_profile.json (+ the same values printed to stdout)

Usage:
    python scripts/profile_expedia.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "reports" / "summary" / "expedia_profile.json"

USECOLS = [
    "date_time",
    "user_id",
    "is_booking",
    "hotel_cluster",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_ci",
    "srch_co",
    "orig_destination_distance",
    "is_package",
    "srch_destination_id",
]
CHUNKSIZE = 3_000_000
TEST_FRACTION = 0.2


def party_type(adults: pd.Series, children: pd.Series) -> pd.Series:
    """Map (adults, children) to family / solo / couple / group / unknown.

    unknown = zero adults and zero children (invalid party, kept so shares sum to 1).
    """
    return pd.Series(
        np.where(
            children > 0,
            "family",
            np.where(
                adults == 1,
                "solo",
                np.where(
                    adults == 2, "couple", np.where(adults == 0, "unknown", "group")
                ),
            ),
        ),
        index=adults.index,
    )


def main() -> None:
    total_rows = 0
    parts = []
    for chunk in pd.read_csv(RAW, usecols=USECOLS, chunksize=CHUNKSIZE):
        total_rows += len(chunk)
        parts.append(chunk[chunk["is_booking"] == 1])
    b = pd.concat(parts, ignore_index=True)
    del parts

    b["ts"] = pd.to_datetime(b["date_time"])
    b = b.sort_values("ts").reset_index(drop=True)

    per_user = b.groupby("user_id").size()
    b["repeat_cluster"] = b.groupby("user_id")["hotel_cluster"].transform(
        lambda s: s.duplicated()
    )
    party = party_type(b["srch_adults_cnt"], b["srch_children_cnt"])
    checkin = pd.to_datetime(b["srch_ci"], errors="coerce")
    checkout = pd.to_datetime(b["srch_co"], errors="coerce")
    cluster_share = b["hotel_cluster"].value_counts(normalize=True)

    cut = b["ts"].quantile(1 - TEST_FRACTION)
    train, test = b[b["ts"] < cut], b[b["ts"] >= cut]

    out = {
        "source": str(RAW.relative_to(ROOT)),
        "total_rows": int(total_rows),
        "bookings": len(b),
        "click_rows": int(total_rows - len(b)),
        "booking_users": int(per_user.size),
        "users_single_booking": int((per_user == 1).sum()),
        "users_single_booking_share": round(float((per_user == 1).mean()), 6),
        "users_ge2_bookings": int((per_user >= 2).sum()),
        "bookings_per_user": {
            "median": float(per_user.median()),
            "mean": round(float(per_user.mean()), 3),
            "p75": float(per_user.quantile(0.75)),
            "p90": float(per_user.quantile(0.90)),
            "share_ge3": round(float((per_user >= 3).mean()), 6),
            "share_ge5": round(float((per_user >= 5).mean()), 6),
        },
        "repeat_cluster_share": round(float(b["repeat_cluster"].mean()), 6),
        "party_mix": {
            k: round(float(v), 6) for k, v in party.value_counts(normalize=True).items()
        },
        "is_package_share": round(float(b["is_package"].mean()), 6),
        "hotel_clusters": int(b["hotel_cluster"].nunique()),
        "cluster_share_top1": round(float(cluster_share.iloc[0]), 6),
        "cluster_share_top10": round(float(cluster_share.iloc[:10].sum()), 6),
        "search_destinations": int(b["srch_destination_id"].nunique()),
        "quality_on_bookings": {
            "duplicate_keys_user_datetime_dest_cluster": int(
                b.duplicated(
                    ["user_id", "date_time", "srch_destination_id", "hotel_cluster"]
                ).sum()
            ),
            "checkout_before_checkin": int((checkout < checkin).sum()),
            "zero_adults": int((b["srch_adults_cnt"] == 0).sum()),
            "srch_ci_missing": int(b["srch_ci"].isna().sum()),
            "srch_co_missing": int(b["srch_co"].isna().sum()),
            "orig_destination_distance_missing_share": round(
                float(b["orig_destination_distance"].isna().mean()), 6
            ),
        },
        "booking_date_range": [str(b["ts"].min()), str(b["ts"].max())],
        "bookings_per_month_min": int(b["ts"].dt.to_period("M").value_counts().min()),
        "bookings_per_month_max": int(b["ts"].dt.to_period("M").value_counts().max()),
        "checkin_month_share": {
            int(k): round(float(v), 6)
            for k, v in checkin.dt.month.value_counts(normalize=True)
            .sort_index()
            .items()
        },
        "temporal_split_80_20": {
            "cut": str(cut),
            "train_bookings": len(train),
            "test_bookings": len(test),
            "test_users_seen_in_train_share": round(
                float(test["user_id"].isin(train["user_id"]).mean()), 4
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
