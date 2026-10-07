"""Export Expedia bookings with a row id, and the per-booking query codes (notebook 1c).

One chunked pass over train.csv keeps is_booking == 1 rows in file order (the same rows, same
order, same user/item/timestamp values as scripts/expedia_to_recbole.py) and writes:

- data/interim/recbole/expedia_rowid/expedia_rowid.inter: user_id, item_id, timestamp and
  `row_id:float` (0-based position of the booking in the file). RecBole copies `row_id` to every
  target row, so the notebook can look up the target's query features without RecBole building
  history lists for every feature (memory).
- data/interim/expedia_query_codes.npz: `codes` (n, F) int32 from nbp.data.query_features and
  `fields`. Row i of `codes` is the booking with row_id i.

Finally checks that the .inter columns equal data/interim/recbole/expedia/expedia.inter.

Usage: python scripts/expedia_query_to_recbole.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbp.data.query_features import QUERY_FIELDS, RAW_COLS, build_query_codes  # noqa: E402
from nbp.paths import EXPEDIA_RAW, INTERIM  # noqa: E402

RAW = EXPEDIA_RAW / "train.csv"
OUT_INTER = INTERIM / "recbole" / "expedia_rowid" / "expedia_rowid.inter"
OUT_CODES = INTERIM / "expedia_query_codes.npz"
REFERENCE = INTERIM / "recbole" / "expedia" / "expedia.inter"
COLS = ["user_id", "is_booking", "hotel_cluster", *RAW_COLS]


def main() -> None:
    t0 = time.time()
    inter_parts, code_parts = [], []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000):
        ch = ch[ch.is_booking == 1]
        inter_parts.append(
            pd.DataFrame(
                {
                    "user_id:token": ch["user_id"].values,
                    "item_id:token": ch["hotel_cluster"].values,
                    "timestamp:float": (
                        (pd.to_datetime(ch["date_time"]) - pd.Timestamp("1970-01-01"))
                        // pd.Timedelta(seconds=1)
                    ).values,
                }
            )
        )
        code_parts.append(build_query_codes(ch))
    df = pd.concat(inter_parts, ignore_index=True)
    codes = np.concatenate(code_parts)
    assert len(df) == len(codes) < 2**24  # row_id must be exact as float32
    df["row_id:float"] = np.arange(len(df))

    OUT_INTER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_INTER, sep="\t", index=False)
    np.savez_compressed(OUT_CODES, codes=codes, fields=np.array(list(QUERY_FIELDS)))
    print(f"bookings {len(df)} users {df['user_id:token'].nunique()} fields {len(QUERY_FIELDS)}")

    ref = pd.read_csv(REFERENCE, sep="\t")
    same = all(
        np.array_equal(ref[c].values, df[c].values)
        for c in ("user_id:token", "item_id:token", "timestamp:float")
    )
    print("rows and columns identical to expedia.inter:", same)
    if not same:
        raise SystemExit("row order or values differ from expedia.inter")
    print(f"wrote {OUT_INTER}\nwrote {OUT_CODES}\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
