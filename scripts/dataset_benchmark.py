"""Like-for-like quantitative benchmark of the candidate datasets.

Answers the mentor's "justify the pick with numbers, not adjectives" review
point: one comparable row per dataset (rows, columns, users, items, history
depth, time span, missing rate, sparsity, signal type), all measured from the
raw files rather than copied from the data cards.

Expedia numbers are read back from reports/summary/expedia_eda_detail.json
(produced by scripts/expedia_eda_detail.py) so the two reports cannot drift.

Input:  data/raw/** (read-only) + reports/summary/expedia_eda_detail.json
Output: reports/summary/dataset_benchmark.json
        reports/summary/dataset_benchmark.csv

Usage:
    .venv/Scripts/python.exe scripts/dataset_benchmark.py [name ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "reports" / "summary"


def blank_row(name: str, domain: str, role: str) -> dict[str, Any]:
    return {
        "dataset": name,
        "domain": domain,
        "role": role,
        "file_scanned": None,
        "file_size_mb": None,
        "rows": None,
        "columns": None,
        "n_users": None,
        "user_field": None,
        "n_items": None,
        "item_field": None,
        "interactions": None,
        "interactions_per_user_median": None,
        "interactions_per_user_mean": None,
        "share_users_ge2": None,
        "time_field": None,
        "time_min": None,
        "time_max": None,
        "span_days": None,
        "missing_rate_overall": None,
        "n_cols_zero_missing": None,
        "worst_missing_col": None,
        "worst_missing_rate": None,
        "sparsity": None,
        "signal_type": None,
        "notes": None,
    }


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1)


def missing_summary(nonnull: pd.Series, rows: int) -> tuple[float, int, str, float]:
    """(overall cell missing rate, cols with 0 missing, worst col, worst rate)."""
    miss = 1 - (nonnull / rows)
    overall = float(1 - nonnull.sum() / (rows * len(nonnull)))
    worst = miss.idxmax()
    return (
        round(overall, 6),
        int((miss == 0).sum()),
        str(worst),
        round(float(miss.max()), 6),
    )


# --------------------------------------------------------------------------
# Expedia — read back from the detailed EDA output (booking-level benchmark)
# --------------------------------------------------------------------------
def expedia() -> dict[str, Any]:
    src = OUT_DIR / "expedia_eda_detail.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    ft = pd.DataFrame(d["field_table"])
    rows, bookings = d["total_rows"], d["bookings"]
    n_users = int(ft.loc[ft["field"] == "user_id", "n_unique_all"].iloc[0])
    prof = json.loads((OUT_DIR / "expedia_profile.json").read_text(encoding="utf-8"))

    overall_missing = float(1 - ft["nonnull_all"].sum() / (rows * len(ft)))
    miss = 1 - ft["coverage_all"]
    worst_i = int(miss.idxmax())

    r = blank_row("Expedia Hotel Recommendations", "travel", "core (chosen)")
    r.update(
        file_scanned="train.csv",
        file_size_mb=size_mb(RAW / "hospitality" / "expedia" / "train.csv"),
        rows=rows,
        columns=d["n_columns"],
        n_users=n_users,
        user_field="user_id",
        n_items=d["cluster_popularity"]["n_clusters"],
        item_field="hotel_cluster",
        interactions=bookings,
        interactions_per_user_median=prof["bookings_per_user"]["median"],
        interactions_per_user_mean=prof["bookings_per_user"]["mean"],
        share_users_ge2=round(1 - prof["users_single_booking_share"], 6),
        time_field="date_time",
        time_min=prof["booking_date_range"][0][:10],
        time_max=prof["booking_date_range"][1][:10],
        missing_rate_overall=round(overall_missing, 6),
        n_cols_zero_missing=int((miss == 0).sum()),
        worst_missing_col=str(ft["field"].iloc[worst_i]),
        worst_missing_rate=round(float(miss.iloc[worst_i]), 6),
        signal_type="booking (is_booking=1), clicks separable",
        notes=(
            f"{prof['booking_users']:,} booking users; "
            f"clicks {d['clicks']:,} rows ({d['clicks'] / rows:.2%})"
        ),
    )
    r["sparsity"] = round(
        1 - bookings / (prof["booking_users"] * d["cluster_popularity"]["n_clusters"]),
        6,
    )
    return r


# --------------------------------------------------------------------------
# Trivago RecSys 2019 — chunked pass over train.csv
# --------------------------------------------------------------------------
def trivago() -> dict[str, Any]:
    path = RAW / "hospitality" / "trivago_2019" / "train.csv"
    nonnull: pd.Series | None = None
    rows = 0
    user_counts = pd.Series(dtype="int64")
    session_counts = pd.Series(dtype="int64")
    action_counts = pd.Series(dtype="int64")
    click_items = pd.Series(dtype="int64")
    ts_min, ts_max = np.inf, -np.inf

    for chunk in pd.read_csv(path, chunksize=2_000_000, dtype=str, low_memory=False):
        rows += len(chunk)
        nn = chunk.notna().sum()
        nonnull = nn if nonnull is None else nonnull.add(nn, fill_value=0)
        user_counts = user_counts.add(chunk["user_id"].value_counts(), fill_value=0)
        session_counts = session_counts.add(
            chunk["session_id"].value_counts(), fill_value=0
        )
        action_counts = action_counts.add(
            chunk["action_type"].value_counts(), fill_value=0
        )
        t = pd.to_numeric(chunk["timestamp"], errors="coerce")
        ts_min, ts_max = min(ts_min, float(t.min())), max(ts_max, float(t.max()))
        co = chunk[chunk["action_type"] == "clickout item"]
        click_items = click_items.add(co["reference"].value_counts(), fill_value=0)
        print(f"  trivago rows={rows:,}", flush=True)

    n_users = int(user_counts.size)
    n_sessions = int(session_counts.size)
    n_items = int(click_items.size)
    clickouts = int(action_counts.get("clickout item", 0))
    overall, zero_cols, worst_col, worst_rate = missing_summary(nonnull, rows)

    r = blank_row("Trivago RecSys Challenge 2019", "travel", "backup (contingency)")
    r.update(
        file_scanned="train.csv",
        file_size_mb=size_mb(path),
        rows=rows,
        columns=len(nonnull),
        n_users=n_users,
        user_field="user_id",
        n_items=n_items,
        item_field="reference (clickout item)",
        interactions=clickouts,
        interactions_per_user_median=float(user_counts.median()),
        interactions_per_user_mean=round(float(user_counts.mean()), 3),
        share_users_ge2=round(float((user_counts >= 2).mean()), 6),
        time_field="timestamp (unix)",
        time_min=str(pd.to_datetime(ts_min, unit="s").date()),
        time_max=str(pd.to_datetime(ts_max, unit="s").date()),
        missing_rate_overall=overall,
        n_cols_zero_missing=zero_cols,
        worst_missing_col=worst_col,
        worst_missing_rate=worst_rate,
        sparsity=round(1 - clickouts / (n_users * n_items), 8),
        signal_type="clickout from impression list (no purchase)",
        notes=(
            f"{n_sessions:,} sessions; actions: "
            + ", ".join(
                f"{k}={int(v):,}"
                for k, v in action_counts.sort_values(ascending=False).head(4).items()
            )
        ),
    )
    return r


# --------------------------------------------------------------------------
# Airbnb New User Bookings
# --------------------------------------------------------------------------
def airbnb() -> dict[str, Any]:
    path = RAW / "hospitality" / "airbnb_new_user" / "train_users_2.csv"
    df = pd.read_csv(path, low_memory=False)
    rows = len(df)
    nonnull = df.notna().sum()
    overall, zero_cols, worst_col, worst_rate = missing_summary(nonnull, rows)
    booked = df[df["country_destination"] != "NDF"]
    created = pd.to_datetime(df["date_account_created"], errors="coerce")

    r = blank_row("Airbnb New User Bookings", "travel", "dropped")
    r.update(
        file_scanned="train_users_2.csv",
        file_size_mb=size_mb(path),
        rows=rows,
        columns=int(df.shape[1]),
        n_users=int(df["id"].nunique()),
        user_field="id",
        n_items=int(df["country_destination"].nunique()),
        item_field="country_destination",
        interactions=len(booked),
        interactions_per_user_median=1.0,
        interactions_per_user_mean=round(len(booked) / df["id"].nunique(), 3),
        share_users_ge2=0.0,
        time_field="date_account_created",
        time_min=str(created.min().date()),
        time_max=str(created.max().date()),
        missing_rate_overall=overall,
        n_cols_zero_missing=zero_cols,
        worst_missing_col=worst_col,
        worst_missing_rate=worst_rate,
        sparsity=round(1 - len(booked) / (df["id"].nunique() * 12), 6),
        signal_type="first booking country (one per user, 12 classes)",
        notes=(
            f"NDF (no booking) = {(df['country_destination'] == 'NDF').mean():.2%} of users; "
            f"US = {(df['country_destination'] == 'US').mean():.2%}"
        ),
    )
    return r


# --------------------------------------------------------------------------
# Hotel Booking Demand
# --------------------------------------------------------------------------
def hotel_booking_demand() -> dict[str, Any]:
    path = RAW / "hospitality" / "hotel_booking_demand" / "hotel_bookings.csv"
    df = pd.read_csv(path, low_memory=False)
    rows = len(df)
    nonnull = df.notna().sum()
    overall, zero_cols, worst_col, worst_rate = missing_summary(nonnull, rows)
    res_date = pd.to_datetime(df["reservation_status_date"], errors="coerce")

    r = blank_row("Hotel Booking Demand", "travel", "dropped")
    r.update(
        file_scanned="hotel_bookings.csv",
        file_size_mb=size_mb(path),
        rows=rows,
        columns=int(df.shape[1]),
        n_users=None,
        user_field="none (no guest id)",
        n_items=int(df["reserved_room_type"].nunique()),
        item_field="reserved_room_type",
        interactions=rows,
        time_field="reservation_status_date",
        time_min=str(res_date.min().date()),
        time_max=str(res_date.max().date()),
        missing_rate_overall=overall,
        n_cols_zero_missing=zero_cols,
        worst_missing_col=worst_col,
        worst_missing_rate=worst_rate,
        signal_type="hotel reservation record (no personalization key)",
        notes=(
            f"exact duplicate rows = {df.duplicated().mean():.2%}; "
            f"cancelled = {df['is_canceled'].mean():.2%}"
        ),
    )
    return r


# --------------------------------------------------------------------------
# Akeed (food, domain contrast)
# --------------------------------------------------------------------------
def akeed() -> dict[str, Any]:
    path = RAW / "food" / "akeed" / "orders.csv"
    df = pd.read_csv(path, low_memory=False)
    rows = len(df)
    nonnull = df.notna().sum()
    overall, zero_cols, worst_col, worst_rate = missing_summary(nonnull, rows)
    created = pd.to_datetime(df["created_at"], errors="coerce")
    per_user = df.groupby("customer_id").size()
    n_vendors = int(df["vendor_id"].nunique())

    r = blank_row("Akeed Restaurant Recommendation", "food", "deferred (out of scope)")
    r.update(
        file_scanned="orders.csv",
        file_size_mb=size_mb(path),
        rows=rows,
        columns=int(df.shape[1]),
        n_users=int(per_user.size),
        user_field="customer_id",
        n_items=n_vendors,
        item_field="vendor_id",
        interactions=rows,
        interactions_per_user_median=float(per_user.median()),
        interactions_per_user_mean=round(float(per_user.mean()), 3),
        share_users_ge2=round(float((per_user >= 2).mean()), 6),
        time_field="created_at",
        time_min=str(created.min().date()),
        time_max=str(created.max().date()),
        missing_rate_overall=overall,
        n_cols_zero_missing=zero_cols,
        worst_missing_col=worst_col,
        worst_missing_rate=worst_rate,
        sparsity=round(1 - rows / (per_user.size * n_vendors), 6),
        signal_type="delivered food order",
        notes=(
            f"deliverydistance <= 0 on {(df['deliverydistance'] <= 0).mean():.2%} of orders; "
            f"duplicate akeed_order_id = {int(df['akeed_order_id'].duplicated().sum()):,}"
        ),
    )
    return r


# --------------------------------------------------------------------------
# Porto Taxi (ride, domain contrast)
# --------------------------------------------------------------------------
def porto() -> dict[str, Any]:
    path = RAW / "ride" / "porto_taxi" / "train.csv"
    cols = [
        "TRIP_ID",
        "CALL_TYPE",
        "ORIGIN_CALL",
        "ORIGIN_STAND",
        "TAXI_ID",
        "TIMESTAMP",
        "DAY_TYPE",
        "MISSING_DATA",
    ]
    rows = 0
    nonnull: pd.Series | None = None
    taxis: set[str] = set()
    origin_calls: set[str] = set()
    ts_min, ts_max = np.inf, -np.inf
    missing_flag = 0
    for chunk in pd.read_csv(
        path, usecols=cols, chunksize=500_000, dtype=str, low_memory=False
    ):
        rows += len(chunk)
        nn = chunk.notna().sum()
        nonnull = nn if nonnull is None else nonnull.add(nn, fill_value=0)
        taxis.update(chunk["TAXI_ID"].dropna().unique().tolist())
        origin_calls.update(chunk["ORIGIN_CALL"].dropna().unique().tolist())
        t = pd.to_numeric(chunk["TIMESTAMP"], errors="coerce")
        ts_min, ts_max = min(ts_min, float(t.min())), max(ts_max, float(t.max()))
        missing_flag += int((chunk["MISSING_DATA"] == "True").sum())
        print(f"  porto rows={rows:,}", flush=True)

    overall, zero_cols, worst_col, worst_rate = missing_summary(nonnull, rows)
    r = blank_row("Porto Taxi Trajectory", "ride", "dropped (no rider id)")
    r.update(
        file_scanned="train.csv (9 cols, POLYLINE skipped)",
        file_size_mb=size_mb(path),
        rows=rows,
        columns=9,
        n_users=None,
        user_field="none (TAXI_ID = driver, not rider)",
        n_items=None,
        item_field="destination = last GPS point (must be discretized)",
        interactions=rows,
        time_field="TIMESTAMP (unix)",
        time_min=str(pd.to_datetime(ts_min, unit="s").date()),
        time_max=str(pd.to_datetime(ts_max, unit="s").date()),
        missing_rate_overall=overall,
        n_cols_zero_missing=zero_cols,
        worst_missing_col=worst_col,
        worst_missing_rate=worst_rate,
        signal_type="taxi trip trajectory (no rider identity)",
        notes=(
            f"{len(taxis):,} distinct drivers; ORIGIN_CALL (phone-dispatch id) present on "
            f"{float(nonnull['ORIGIN_CALL']) / rows:.2%} of trips, {len(origin_calls):,} distinct; "
            f"MISSING_DATA=True on {missing_flag:,} trips"
        ),
    )
    return r


BUILDERS = {
    "expedia": expedia,
    "trivago": trivago,
    "airbnb": airbnb,
    "hotel_booking_demand": hotel_booking_demand,
    "akeed": akeed,
    "porto": porto,
}


def main(names: list[str] | None = None) -> None:
    todo = names or list(BUILDERS)
    out_json = OUT_DIR / "dataset_benchmark.json"
    existing: dict[str, Any] = {}
    if out_json.exists():
        existing = {r["dataset"]: r for r in json.loads(out_json.read_text("utf-8"))}

    for name in todo:
        print(f"[{name}] scanning ...", flush=True)
        row = BUILDERS[name]()
        existing[row["dataset"]] = row
        print(f"[{name}] done: rows={row['rows']:,}", flush=True)

    for row in existing.values():
        if row["time_min"] and row["time_max"]:
            row["span_days"] = int(
                (pd.Timestamp(row["time_max"]) - pd.Timestamp(row["time_min"])).days
            )

    order = [
        "Expedia Hotel Recommendations",
        "Trivago RecSys Challenge 2019",
        "Airbnb New User Bookings",
        "Hotel Booking Demand",
        "Akeed Restaurant Recommendation",
        "Porto Taxi Trajectory",
    ]
    rows_out = [existing[k] for k in order if k in existing]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "dataset_benchmark.csv", index=False)
    print(f"written: {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or None)
