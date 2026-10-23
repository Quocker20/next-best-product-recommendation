"""Build the per-booking query codes of the cleaned Expedia bookings.

Reads data/interim/expedia_bookings.parquet (burst repeats collapsed) and the raw bookings table
data/interim/expedia_bookings_raw.parquet (rows picked by `src_row`, so the query codes come from
the raw search fields exactly as `nbp.data.query_features` expects) and writes
data/interim/expedia_query_codes_clean.npz: `codes` (n, F) int32 and `fields`. Row i of `codes`
is booking i of the cleaned table.

Usage: python scripts/build_query_codes.py
"""

import time

import numpy as np
import pandas as pd

from nbp.data.query_features import QUERY_FIELDS, RAW_COLS, build_query_codes
from nbp.paths import INTERIM

CLEAN = INTERIM / "expedia_bookings.parquet"
RAW = INTERIM / "expedia_bookings_raw.parquet"
OUT_CODES = INTERIM / "expedia_query_codes_clean.npz"


def main() -> None:
    t0 = time.time()
    clean = pd.read_parquet(CLEAN, columns=["ts_unix", "src_row"])
    assert clean["ts_unix"].is_monotonic_increasing, "parquet must be sorted by time"
    raw = pd.read_parquet(RAW, columns=RAW_COLS)
    codes = build_query_codes(raw.iloc[clean["src_row"].to_numpy()].reset_index(drop=True))
    np.savez_compressed(OUT_CODES, codes=codes, fields=np.array(list(QUERY_FIELDS)))
    print(f"bookings {len(clean)} fields {len(QUERY_FIELDS)}")
    print(f"wrote {OUT_CODES}\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
