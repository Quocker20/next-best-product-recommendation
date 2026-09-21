"""Build the slide-ready Expedia schema table (mentor review point 1).

Merges measured statistics with the field descriptions from the Kaggle data
dictionary and groups the 24 columns of `train.csv` into logical feature
groups. Coverage, cardinality, value range and sample values are measured
here; only the prose description and the group label are authored.

A second light pass over `train.csv` adds min/max (numeric + date columns) and
the distinct-value counts the first EDA pass skipped.

Input:  data/raw/hospitality/expedia/train.csv (read-only)
        reports/summary/expedia_field_table.csv (from expedia_eda_detail.py)
Output: reports/summary/expedia_schema_table.csv
        reports/summary/expedia_schema_table.json
        reports/summary/expedia_schema_table.md (paste-ready table)

Usage:
    .venv/Scripts/python.exe scripts/expedia_schema_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT_DIR = ROOT / "reports" / "summary"
CHUNKSIZE = 2_000_000

NUMERIC_COLS = [
    "site_name",
    "posa_continent",
    "user_location_country",
    "user_location_region",
    "user_location_city",
    "orig_destination_distance",
    "user_id",
    "is_mobile",
    "is_package",
    "channel",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "srch_destination_id",
    "srch_destination_type_id",
    "is_booking",
    "cnt",
    "hotel_continent",
    "hotel_country",
    "hotel_market",
    "hotel_cluster",
]
DATE_COLS = ["date_time", "srch_ci", "srch_co"]

# group label + description: the only authored content in this file
# (descriptions follow the Kaggle "Expedia Hotel Recommendations" data dictionary)
META: dict[str, tuple[str, str]] = {
    "date_time": (
        "Interaction / time",
        "Timestamp of the search event (second resolution); the temporal-split key",
    ),
    "user_id": (
        "User identity",
        "Anonymised traveller id; the only personalisation key in the file",
    ),
    "is_booking": ("Interaction / time", "Event label: 1 = booking, 0 = click"),
    "cnt": (
        "Interaction / time",
        "Number of similar events in the same user session context",
    ),
    "site_name": (
        "Channel / platform",
        "Expedia point-of-sale website (e.g. Expedia.com vs a locale site)",
    ),
    "posa_continent": ("Channel / platform", "Continent of the point of sale"),
    "channel": ("Channel / platform", "Marketing channel the user arrived through"),
    "is_mobile": ("Channel / platform", "1 = event came from a mobile device"),
    "user_location_country": ("User geography", "Country the user is browsing from"),
    "user_location_region": ("User geography", "Region within the user country"),
    "user_location_city": ("User geography", "City the user is browsing from"),
    "orig_destination_distance": (
        "User geography",
        "Physical distance between user city and hotel, in miles; blank when it could not be computed",
    ),
    "srch_ci": (
        "Trip context",
        "Check-in date requested; drives the seasonality slice",
    ),
    "srch_co": (
        "Trip context",
        "Check-out date requested; with srch_ci gives stay length",
    ),
    "srch_adults_cnt": ("Trip context", "Adults in the room request"),
    "srch_children_cnt": ("Trip context", "Children in the room request"),
    "srch_rm_cnt": ("Trip context", "Rooms requested"),
    "is_package": (
        "Trip context",
        "1 = booked as part of a package (flight/car bundled)",
    ),
    "srch_destination_id": (
        "Search / destination",
        "Id of the destination the user searched for",
    ),
    "srch_destination_type_id": (
        "Search / destination",
        "Type of the searched destination (city, region, landmark, ...)",
    ),
    "hotel_continent": ("Hotel / target", "Continent the hotel is in"),
    "hotel_country": ("Hotel / target", "Country the hotel is in"),
    "hotel_market": (
        "Hotel / target",
        "Local market (sub-country travel market) of the hotel",
    ),
    "hotel_cluster": (
        "Hotel / target",
        "TARGET: one of 100 anonymised hotel clusters (proxy for room category / package)",
    ),
}
GROUP_ORDER = [
    "Interaction / time",
    "User identity",
    "Trip context",
    "Search / destination",
    "Hotel / target",
    "User geography",
    "Channel / platform",
]


def write_markdown(df: pd.DataFrame, rows: int) -> str:
    """Render the schema table as a paste-ready markdown table."""
    lines = [
        f"# Expedia `train.csv` schema table ({rows:,} rows, {len(df)} columns)",
        "",
        "Coverage = share of non-null values. `all` = whole file; `bookings` = the",
        "`is_booking == 1` subset the benchmark trains on.",
        "",
        "| No. | Group | Field | Description | Coverage (all) | Coverage (bookings) | Distinct | Range | Sample values |",
        "|----:|-------|-------|-------------|---------------:|--------------------:|---------:|-------|---------------|",
    ]
    for _, r in df.iterrows():
        u = r["n_unique"]
        distinct = (
            "n/a"
            if u is None or (isinstance(u, float) and pd.isna(u))
            else f"{int(u):,}"
        )
        lines.append(
            f"| {r['no']} | {r['group']} | `{r['field']}` | {r['description']} | "
            f"{r['coverage_all']:.2%} | {r['coverage_bookings']:.2%} | {distinct} | "
            f"{r['value_range']} | {r['samples']} |"
        )
    return "\n".join(lines) + "\n"


def main(from_csv: bool = False) -> None:
    if from_csv:
        df = pd.read_csv(OUT_DIR / "expedia_schema_table.csv")
        n_rows = json.loads(
            (OUT_DIR / "expedia_eda_detail.json").read_text(encoding="utf-8")
        )["total_rows"]
        md = write_markdown(df, n_rows)
        (OUT_DIR / "expedia_schema_table.md").write_text(md, encoding="utf-8")
        print(md)
        return

    base = pd.read_csv(OUT_DIR / "expedia_field_table.csv")

    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    date_min: dict[str, pd.Timestamp] = {}
    date_max: dict[str, pd.Timestamp] = {}
    uniq_extra: dict[str, set] = {
        c: set() for c in ["orig_destination_distance", "srch_ci", "srch_co"]
    }
    rows = 0

    for i, chunk in enumerate(
        pd.read_csv(
            RAW, usecols=NUMERIC_COLS + DATE_COLS, chunksize=CHUNKSIZE, low_memory=False
        )
    ):
        rows += len(chunk)
        for c in NUMERIC_COLS:
            col = pd.to_numeric(chunk[c], errors="coerce")
            lo, hi = float(col.min()), float(col.max())
            mins[c] = lo if c not in mins else min(mins[c], lo)
            maxs[c] = hi if c not in maxs else max(maxs[c], hi)
        for c in DATE_COLS:
            d = pd.to_datetime(chunk[c], errors="coerce")
            lo, hi = d.min(), d.max()
            date_min[c] = lo if c not in date_min else min(date_min[c], lo)
            date_max[c] = hi if c not in date_max else max(date_max[c], hi)
        for c in uniq_extra:
            uniq_extra[c].update(chunk[c].dropna().unique().tolist())
        print(f"chunk {i}: rows={rows:,}", flush=True)

    recs = []
    for _, r in base.iterrows():
        field = r["field"]
        group, desc = META[field]
        if field in DATE_COLS:
            vrange = f"{date_min[field].date()} .. {date_max[field].date()}"
            n_unique = len(uniq_extra[field]) if field in uniq_extra else None
        else:
            lo, hi = mins[field], maxs[field]
            fmt = (
                (lambda v: f"{v:,.2f}")
                if field == "orig_destination_distance"
                else (lambda v: f"{v:,.0f}")
            )
            vrange = f"{fmt(lo)} .. {fmt(hi)}"
            n_unique = (
                len(uniq_extra[field])
                if field in uniq_extra
                else (None if pd.isna(r["n_unique_all"]) else int(r["n_unique_all"]))
            )
        samples = ", ".join(eval(r["samples"])) if isinstance(r["samples"], str) else ""
        recs.append(
            {
                "group": group,
                "field": field,
                "description": desc,
                "coverage_all": round(float(r["coverage_all"]), 6),
                "coverage_bookings": round(float(r["coverage_bookings"]), 6),
                "n_unique": n_unique,
                "value_range": vrange,
                "samples": samples,
            }
        )

    df = pd.DataFrame(recs)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
    df = df.sort_values(["group", "field"]).reset_index(drop=True)
    df.insert(0, "no", np.arange(1, len(df) + 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "expedia_schema_table.csv", index=False)
    (OUT_DIR / "expedia_schema_table.json").write_text(
        json.dumps(df.to_dict("records"), indent=2, default=str), encoding="utf-8"
    )

    md = write_markdown(df, rows)
    (OUT_DIR / "expedia_schema_table.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    import sys

    main(from_csv="--from-csv" in sys.argv)
