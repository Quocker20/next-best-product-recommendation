"""Chunked extraction of Expedia booking rows from the 4 GB `train.csv`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nbp.paths import EXPEDIA_RAW, INTERIM

TRAIN_CSV = EXPEDIA_RAW / "train.csv"
RAW_BOOKINGS = INTERIM / "expedia_bookings_raw.parquet"

# Raw columns kept for every booking row (everything except `is_booking`).
# `hotel_*` columns describe the booked hotel: they are targets/labels, never model inputs.
INT_COLS = [
    "site_name",
    "posa_continent",
    "user_location_country",
    "user_location_region",
    "user_location_city",
    "user_id",
    "is_mobile",
    "is_package",
    "channel",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "srch_destination_id",
    "srch_destination_type_id",
    "cnt",
    "hotel_continent",
    "hotel_country",
    "hotel_market",
    "hotel_cluster",
]
FLOAT_COLS = ["orig_destination_distance"]
DATE_COLS = ["date_time", "srch_ci", "srch_co"]
USECOLS = [*DATE_COLS, *FLOAT_COLS, *INT_COLS, "is_booking"]

TS_DTYPE = "datetime64[ns]"


def to_ns(s: pd.Series) -> pd.Series:
    """Parse a string column to `datetime64[ns]` (pandas 3 may default to another unit).

    Unparseable values become NaT. Forcing one unit keeps int casts and comparisons safe.
    """
    return pd.to_datetime(s, errors="coerce").astype(TS_DTYPE)


def filter_bookings(chunk: pd.DataFrame) -> pd.DataFrame:
    """Keep `is_booking == 1` rows of one raw chunk, parse dates, compact the dtypes.

    Input: raw chunk with `USECOLS`. Output: same columns minus `is_booking`;
    dates are `datetime64[ns]`, id/count columns int32 (`user_id`, `hotel_*` fit in int32).
    """
    out = chunk.loc[chunk["is_booking"] == 1].drop(columns="is_booking").copy()
    for c in DATE_COLS:
        out[c] = to_ns(out[c])
    for c in INT_COLS:
        out[c] = out[c].astype("int32")
    out["orig_destination_distance"] = out["orig_destination_distance"].astype("float32")
    return out.reset_index(drop=True)


def load_bookings(
    path: Path = TRAIN_CSV, chunk_size: int = 1_000_000, verbose: bool = True
) -> pd.DataFrame:
    """One chunked pass over `train.csv`; returns all booking rows in file order.

    Output columns: `DATE_COLS + FLOAT_COLS + INT_COLS`. Row order = file order (stable).
    """
    parts: list[pd.DataFrame] = []
    n_rows = 0
    for i, chunk in enumerate(pd.read_csv(path, usecols=USECOLS, chunksize=chunk_size)):
        n_rows += len(chunk)
        parts.append(filter_bookings(chunk))
        if verbose:
            print(f"chunk {i}: raw rows={n_rows:,} bookings={sum(map(len, parts)):,}", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.attrs["raw_rows"] = n_rows
    return df


def write_raw_bookings(chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Extract bookings and write `data/interim/expedia_bookings_raw.parquet`."""
    df = load_bookings(chunk_size=chunk_size)
    RAW_BOOKINGS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_BOOKINGS, index=False)
    return df
