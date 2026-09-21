"""Session/user/item depth profile of the Trivago 2019 backup dataset.

The week-1 deck quotes "56% of users appear only once" for Trivago; the
benchmark pass counts rows, not sessions, so this script measures the three
depth definitions separately (rows per user, sessions per user, clickouts per
user, clickouts per item) to state the right one on the slide.

Input:  data/raw/hospitality/trivago_2019/train.csv (read-only)
        data/raw/hospitality/trivago_2019/item_metadata.csv
Output: reports/summary/trivago_profile.json

Usage:
    .venv/Scripts/python.exe scripts/trivago_profile.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "trivago_2019"
OUT = ROOT / "reports" / "summary" / "trivago_profile.json"
CHUNKSIZE = 2_000_000


def main() -> None:
    rows = 0
    rows_per_user = pd.Series(dtype="int64")
    user_session_pairs: list[pd.DataFrame] = []
    clickouts_per_user = pd.Series(dtype="int64")
    clickouts_per_item = pd.Series(dtype="int64")
    sessions_seen: set[str] = set()
    impressions_lengths: list[int] = []
    price_sample: list[float] = []
    cities: set[str] = set()
    platforms: set[str] = set()
    devices: pd.Series = pd.Series(dtype="int64")

    for i, chunk in enumerate(
        pd.read_csv(RAW / "train.csv", chunksize=CHUNKSIZE, dtype=str, low_memory=False)
    ):
        rows += len(chunk)
        rows_per_user = rows_per_user.add(chunk["user_id"].value_counts(), fill_value=0)
        sessions_seen.update(chunk["session_id"].dropna().unique().tolist())
        user_session_pairs.append(chunk[["user_id", "session_id"]].drop_duplicates())
        co = chunk[chunk["action_type"] == "clickout item"]
        clickouts_per_user = clickouts_per_user.add(
            co["user_id"].value_counts(), fill_value=0
        )
        clickouts_per_item = clickouts_per_item.add(
            co["reference"].value_counts(), fill_value=0
        )
        imp = co["impressions"].dropna()
        impressions_lengths.extend(imp.head(200_000).str.count(r"\|").add(1).tolist())
        if len(price_sample) < 200_000:
            pr = co["prices"].dropna().head(50_000)
            for row in pr:
                price_sample.extend(float(x) for x in row.split("|")[:25])
        cities.update(chunk["city"].dropna().unique().tolist())
        platforms.update(chunk["platform"].dropna().unique().tolist())
        devices = devices.add(chunk["device"].value_counts(), fill_value=0)
        print(f"chunk {i}: rows={rows:,}", flush=True)

    pairs = pd.concat(user_session_pairs, ignore_index=True).drop_duplicates()
    sess_counts = pairs.groupby("user_id").size()
    prices = pd.Series(price_sample, dtype="float64")
    meta = pd.read_csv(RAW / "item_metadata.csv")
    prop_counts = meta["properties"].str.count(r"\|").add(1)

    out = {
        "source": "data/raw/hospitality/trivago_2019/",
        "generated_by": "scripts/trivago_profile.py",
        "rows": rows,
        "n_users": int(rows_per_user.size),
        "n_sessions": len(sessions_seen),
        "clickouts": int(clickouts_per_user.sum()),
        "n_clicked_items": int(clickouts_per_item.size),
        "rows_per_user": {
            "median": float(rows_per_user.median()),
            "mean": round(float(rows_per_user.mean()), 3),
            "share_eq1": round(float((rows_per_user == 1).mean()), 6),
        },
        "sessions_per_user": {
            "median": float(sess_counts.median()),
            "mean": round(float(sess_counts.mean()), 3),
            "share_eq1": round(float((sess_counts == 1).mean()), 6),
            "share_ge2": round(float((sess_counts >= 2).mean()), 6),
        },
        "clickouts_per_user": {
            "users_with_clickout": int((clickouts_per_user > 0).sum()),
            "share_users_with_no_clickout": round(
                1 - float((clickouts_per_user > 0).sum()) / rows_per_user.size, 6
            ),
            "median_among_clickers": float(
                clickouts_per_user[clickouts_per_user > 0].median()
            ),
            "share_eq1_among_clickers": round(
                float((clickouts_per_user[clickouts_per_user > 0] == 1).mean()), 6
            ),
        },
        "clickouts_per_item": {
            "share_items_eq1": round(float((clickouts_per_item == 1).mean()), 6),
            "median": float(clickouts_per_item.median()),
            "max": int(clickouts_per_item.max()),
        },
        "impressions_per_clickout": {
            "sampled_rows": len(impressions_lengths),
            "median": float(pd.Series(impressions_lengths).median()),
            "mean": round(float(pd.Series(impressions_lengths).mean()), 2),
            "max": int(max(impressions_lengths)),
        },
        "displayed_price": {
            "sampled_values": int(prices.size),
            "median": float(prices.median()),
            "p10": float(prices.quantile(0.10)),
            "p90": float(prices.quantile(0.90)),
            "max": float(prices.max()),
        },
        "catalog": {
            "n_cities": len(cities),
            "n_platforms": len(platforms),
            "device_share": {
                k: round(float(v / rows), 6)
                for k, v in devices.sort_values(ascending=False).items()
            },
            "item_metadata_rows": len(meta),
            "item_metadata_properties_median": float(prop_counts.median()),
            "item_metadata_properties_max": int(prop_counts.max()),
            "n_distinct_properties": int(
                meta["properties"].str.split("|").explode().nunique()
            ),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
