"""Export the cleaned Expedia bookings with a row id, plus the per-booking query codes (notebook 1c).

Reads data/interim/expedia_bookings.parquet (burst repeats collapsed) and the raw bookings table
data/interim/expedia_bookings_raw.parquet (rows picked by `src_row`, so the query codes come from
the raw search fields exactly as `nbp.data.query_features` expects) and writes:

- data/interim/recbole/expedia_clean_rowid/expedia_clean_rowid.inter: user_id, item_id,
  timestamp (= event index, see scripts/expedia_to_recbole.py) and `row_id:float` (= event index).
  RecBole copies `row_id` to every target row, so the notebook can look up the target's query
  features without RecBole building history lists for every feature (memory).
- data/interim/expedia_query_codes_clean.npz: `codes` (n, F) int32 and `fields`. Row i of `codes`
  is the booking with row_id i.

Finally checks that the .inter columns equal expedia_clean.inter.

Usage: python scripts/expedia_query_to_recbole.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbp.data.query_features import QUERY_FIELDS, RAW_COLS, build_query_codes
from nbp.paths import INTERIM

CLEAN = INTERIM / "expedia_bookings.parquet"
RAW = INTERIM / "expedia_bookings_raw.parquet"
OUT_INTER = INTERIM / "recbole" / "expedia_clean_rowid" / "expedia_clean_rowid.inter"
OUT_CODES = INTERIM / "expedia_query_codes_clean.npz"
REFERENCE = INTERIM / "recbole" / "expedia_clean" / "expedia_clean.inter"


def main() -> None:
    t0 = time.time()
    clean = pd.read_parquet(CLEAN, columns=["user_id", "item_id", "ts_unix", "src_row"])
    assert clean["ts_unix"].is_monotonic_increasing, "parquet must be sorted by time"
    assert len(clean) < 2**24  # event index must be exact as float32
    raw = pd.read_parquet(RAW, columns=RAW_COLS)
    codes = build_query_codes(raw.iloc[clean["src_row"].to_numpy()].reset_index(drop=True))
    df = pd.DataFrame(
        {
            "user_id:token": clean["user_id"].values,
            "item_id:token": clean["item_id"].values,
            "timestamp:float": range(len(clean)),
            "row_id:float": range(len(clean)),
        }
    )
    OUT_INTER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_INTER, sep="\t", index=False)
    np.savez_compressed(OUT_CODES, codes=codes, fields=np.array(list(QUERY_FIELDS)))
    print(f"bookings {len(df)} users {df['user_id:token'].nunique()} fields {len(QUERY_FIELDS)}")

    ref = pd.read_csv(REFERENCE, sep="\t")
    same = all(
        np.array_equal(ref[c].values, df[c].values)
        for c in ("user_id:token", "item_id:token", "timestamp:float")
    )
    print("rows and columns identical to expedia_clean.inter:", same)
    if not same:
        raise SystemExit("row order or values differ from expedia_clean.inter")
    print(f"wrote {OUT_INTER}\nwrote {OUT_CODES}\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
