"""Export Expedia bookings to a RecBole atomic .inter file (format change only).

One chunked pass over train.csv keeps is_booking == 1 rows and writes
user_id / item_id (= hotel_cluster) / timestamp (unix seconds of date_time).
No filtering, no features, no split: RecBole builds sequences and the global
temporal split from this file (see configs/smlprec_expedia.yaml).

Usage: python scripts/expedia_to_recbole.py
Output: data/interim/recbole/expedia/expedia.inter

With --with-destination the same rows (same order) are written with one extra column,
srch_destination_id (the destination searched in that booking), to a separate dataset:
data/interim/recbole/expedia_dest/expedia_dest.inter. Row order and item/user ids are
identical, so a checkpoint trained on the plain file still applies; the extra column lets
RecBole carry each target's query destination (used by scripts/late_fusion_destination_prior.py).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
WITH_DEST = "--with-destination" in sys.argv
OUT = (
    ROOT / "data" / "interim" / "recbole" / "expedia_dest" / "expedia_dest.inter"
    if WITH_DEST
    else ROOT / "data" / "interim" / "recbole" / "expedia" / "expedia.inter"
)
COLS = ["date_time", "user_id", "is_booking", "hotel_cluster"] + (
    ["srch_destination_id"] if WITH_DEST else []
)


def main() -> None:
    parts = []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000):
        ch = ch[ch.is_booking == 1]
        cols = {
            "user_id:token": ch["user_id"].values,
            "item_id:token": ch["hotel_cluster"].values,
            # unit-safe: pandas >= 3 parses to microseconds, not nanoseconds
            "timestamp:float": (
                (pd.to_datetime(ch["date_time"]) - pd.Timestamp("1970-01-01"))
                // pd.Timedelta(seconds=1)
            ).values,
        }
        if WITH_DEST:
            cols["srch_destination_id:float"] = ch["srch_destination_id"].values
        parts.append(pd.DataFrame(cols))
    df = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    n_users = df["user_id:token"].nunique()
    print(f"bookings {len(df)} users {n_users} items {df['item_id:token'].nunique()}")
    print(f"sequence targets (bookings minus first per user) {len(df) - n_users}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
