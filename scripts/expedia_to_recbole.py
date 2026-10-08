"""Export the cleaned Expedia bookings to a RecBole atomic .inter file (format change only).

Reads data/interim/expedia_bookings.parquet (burst repeats collapsed, see
docs/data_cleaning_strategy.md) and writes user_id / item_id (= hotel_cluster) / timestamp.

`timestamp` is the **event index**: the row position in the parquet, which is sorted by
(time, user_id, source row). It is strictly increasing, exact in float32 (n < 2^24) and gives
RecBole's temporal order and split without ties. Unix seconds stay in `ts_unix` of the parquet;
float32 unix times are only exact to about 128 s and RecBole's sort is not stable.

Usage: python scripts/expedia_to_recbole.py [--with-destination]
Output: data/interim/recbole/expedia_clean/expedia_clean.inter and
        data/interim/expedia_event_cols_clean.npz (destination, hotel market, cluster per event)

With --with-destination the same rows (same order, ids and event index) are written to a
separate dataset, data/interim/recbole/expedia_clean_dest/expedia_clean_dest.inter, with one
extra column `srch_destination_id:float` (the destination searched in that booking). RecBole then
carries each target's query destination and each history position's destination (notebooks 03-06).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "interim" / "expedia_bookings.parquet"
EVENT_COLS = ROOT / "data" / "interim" / "expedia_event_cols_clean.npz"
WITH_DEST = "--with-destination" in sys.argv
NAME = "expedia_clean_dest" if WITH_DEST else "expedia_clean"
OUT = ROOT / "data" / "interim" / "recbole" / NAME / f"{NAME}.inter"


def main() -> None:
    d = pd.read_parquet(SRC, columns=["user_id", "item_id", "ts_unix", "ctx_dest_id", "tgt_market"])
    assert d["ts_unix"].is_monotonic_increasing, "parquet must be sorted by time"
    assert len(d) < 2**24  # event index must be exact as float32
    df = pd.DataFrame(
        {
            "user_id:token": d["user_id"].values,
            "item_id:token": d["item_id"].values,
            "timestamp:float": range(len(d)),
        }
    )
    if WITH_DEST:
        df["srch_destination_id:float"] = d["ctx_dest_id"].values
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    # The RecBole venv has no parquet engine: the prior tables of notebooks 01c and 03-06 read these
    # arrays instead (row i = event i).
    np.savez_compressed(
        EVENT_COLS,
        dest=d["ctx_dest_id"].to_numpy(),
        market=d["tgt_market"].to_numpy(),
        item=d["item_id"].to_numpy(),
    )
    n_users = df["user_id:token"].nunique()
    print(f"bookings {len(df)} users {n_users} items {df['item_id:token'].nunique()}")
    print(f"sequence targets (bookings minus first per user) {len(df) - n_users}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
