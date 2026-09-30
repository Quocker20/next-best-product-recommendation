"""Export Expedia bookings to a RecBole atomic .inter file (format change only).

One chunked pass over train.csv keeps is_booking == 1 rows and writes
user_id / item_id (= hotel_cluster) / timestamp (unix seconds of date_time).
No filtering, no features, no split: RecBole builds sequences and the global
temporal split from this file (see configs/smlprec_expedia.yaml).

Usage: python scripts/expedia_to_recbole.py
Output: data/interim/recbole/expedia/expedia.inter
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "data" / "interim" / "recbole" / "expedia" / "expedia.inter"
COLS = ["date_time", "user_id", "is_booking", "hotel_cluster"]


def main() -> None:
    parts = []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000):
        ch = ch[ch.is_booking == 1]
        parts.append(
            pd.DataFrame(
                {
                    "user_id:token": ch["user_id"].values,
                    "item_id:token": ch["hotel_cluster"].values,
                    # unit-safe: pandas >= 3 parses to microseconds, not nanoseconds
                    "timestamp:float": (
                        (pd.to_datetime(ch["date_time"]) - pd.Timestamp("1970-01-01"))
                        // pd.Timedelta(seconds=1)
                    ).values,
                }
            )
        )
    df = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    n_users = df["user_id:token"].nunique()
    print(f"bookings {len(df)} users {n_users} items {df['item_id:token'].nunique()}")
    print(f"sequence targets (bookings minus first per user) {len(df) - n_users}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
