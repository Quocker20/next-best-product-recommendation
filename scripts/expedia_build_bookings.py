"""Build the cleaned Expedia bookings table and print the data-card counts.

Input:  data/raw/hospitality/expedia/train.csv (read-only)
Output: data/interim/expedia_bookings_raw.parquet      (bookings, raw columns)
        data/interim/expedia_bookings_flagged.parquet  (cleaned + all flags, burst repeats kept;
                                                         source of the "with burst" sensitivity run)
        data/interim/expedia_bookings.parquet          (cleaned, burst repeats collapsed; the
                                                         source for every export and notebook)
        results/week4_rebuild/bookings_counts.json  (every number for the data card)

Usage:
    .venv/Scripts/python.exe scripts/expedia_build_bookings.py
"""

from __future__ import annotations

import json

import pandas as pd

from nbp.config import DataConfig, load_config
from nbp.data.clean import clean_bookings, collapse_bursts, summarize
from nbp.data.load import RAW_BOOKINGS, load_bookings
from nbp.paths import CONFIGS, INTERIM, ROOT

FLAGGED = INTERIM / "expedia_bookings_flagged.parquet"
CLEAN = INTERIM / "expedia_bookings.parquet"
OUT = ROOT / "results" / "week4_rebuild" / "bookings_counts.json"

# Figures in docs/data_cards/expedia.md from the Week-1 pass. A mismatch is reported, not hidden.
DATA_CARD = {
    "raw_booking_rows": 3_000_693,
    "users": 813_985,
    "unique_user_item_pairs": 2_360_713,
    "single_booking_users": 315_679,
    "items": 100,
}


def main() -> None:
    cfg = load_config(DataConfig, CONFIGS / "data.yaml")
    raw = load_bookings(chunk_size=cfg.chunk_size)
    raw_rows = raw.attrs["raw_rows"]
    RAW_BOOKINGS.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_BOOKINGS, index=False)

    flagged, steps = clean_bookings(raw)
    flagged.to_parquet(FLAGGED, index=False)
    clean, collapse_steps = collapse_bursts(flagged)
    clean.to_parquet(CLEAN, index=False)

    # Round-trip check: what is on disk is what we counted.
    disk = pd.read_parquet(CLEAN)
    assert len(disk) == len(clean) and str(disk["timestamp"].dtype) == "datetime64[ns]"
    assert collapse_bursts(disk)[1]["burst_dropped"] == 0  # collapsing again changes nothing

    stats = summarize(disk)
    flagged_stats = summarize(pd.read_parquet(FLAGGED))
    # Users/pairs of the raw (pre-dedup) table are comparable to the data card; dedup only drops a few rows.
    raw_stats = {
        "raw_booking_rows": len(raw),
        "users": int(raw["user_id"].nunique()),
        "unique_user_item_pairs": int(raw[["user_id", "hotel_cluster"]].drop_duplicates().shape[0]),
        "single_booking_users": int((raw.groupby("user_id").size() == 1).sum()),
        "items": int(raw["hotel_cluster"].nunique()),
    }
    check = {
        k: {"data_card": v, "computed": raw_stats[k], "match": v == raw_stats[k]}
        for k, v in DATA_CARD.items()
    }
    out = {
        "generated_by": "scripts/expedia_build_bookings.py",
        "raw_csv_rows": raw_rows,
        "cleaning_steps": steps,
        "collapse_steps": collapse_steps,
        "flagged_stats_before_collapse": flagged_stats,
        "raw_bookings_stats": raw_stats,
        "clean_stats": stats,
        "data_card_check": check,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
