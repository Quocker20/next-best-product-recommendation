"""Detailed EDA pass over Expedia `train.csv` for the week-1 report revision.

Adds the statistics the week-1 slide deck is missing after mentor review:
per-field coverage + sample values (schema table), daily/hourly/weekday
transaction time series with anomaly flags, and distribution/segmentation
tables (lead time, stay length, distance, cluster popularity, party mix).

One chunked pass over the 37.7M-row file; booking rows (`is_booking == 1`)
are retained in memory downcast, click rows only contribute to aggregates.

Input:  data/raw/hospitality/expedia/train.csv (read-only)
Output: reports/summary/expedia_eda_detail.json
        reports/summary/expedia_daily_series.csv   (date, clicks, bookings)
        reports/summary/expedia_field_table.csv    (schema table for slides)

Usage:
    .venv/Scripts/python.exe scripts/expedia_eda_detail.py
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
DATE_COLS = ["date_time", "srch_ci", "srch_co"]
DTYPES: dict[str, str] = {
    "site_name": "int16",
    "posa_continent": "int8",
    "user_location_country": "int16",
    "user_location_region": "int16",
    "user_location_city": "int32",
    "orig_destination_distance": "float32",
    "user_id": "int32",
    "is_mobile": "int8",
    "is_package": "int8",
    "channel": "int8",
    "srch_adults_cnt": "int8",
    "srch_children_cnt": "int8",
    "srch_rm_cnt": "int8",
    "srch_destination_id": "int32",
    "srch_destination_type_id": "int8",
    "is_booking": "int8",
    "cnt": "int16",
    "hotel_continent": "int8",
    "hotel_country": "int16",
    "hotel_market": "int16",
    "hotel_cluster": "int8",
}
# columns whose exact distinct count we track over the whole file
UNIQUE_TRACK = [
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
    "is_booking",
    "cnt",
    "hotel_continent",
    "hotel_country",
    "hotel_market",
    "hotel_cluster",
]
KEEP_COLS = list(DTYPES) + DATE_COLS


def party_type(adults: pd.Series, children: pd.Series) -> pd.Series:
    """Map (adults, children) to family / solo / couple / group / unknown.

    Same rule as scripts/profile_expedia.py so shares stay comparable.
    """
    return pd.Series(
        np.where(
            children > 0,
            "family",
            np.where(
                adults == 1,
                "solo",
                np.where(
                    adults == 2, "couple", np.where(adults == 0, "unknown", "group")
                ),
            ),
        ),
        index=adults.index,
    )


def add_counts(store: dict, key, values: pd.Series) -> None:
    """Accumulate a value_counts Series into a plain dict keyed by `key`."""
    vc = values.value_counts()
    target = store.setdefault(key, {})
    for k, v in vc.items():
        target[k] = target.get(k, 0) + int(v)


def flag_anomalies(series: pd.Series, window: int = 28, z: float = 4.0) -> pd.DataFrame:
    """Robust outlier flags on a daily series (rolling median + MAD).

    Returns a frame with the value, rolling median, robust z-score and a
    boolean flag for |z| > `z`. Columns: value, median, mad_z, is_anomaly.
    """
    med = series.rolling(window, center=True, min_periods=7).median()
    abs_dev = (series - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=7).median()
    scale = (1.4826 * mad).replace(0, np.nan)
    mad_z = (series - med) / scale
    return pd.DataFrame(
        {
            "value": series,
            "median": med,
            "mad_z": mad_z,
            "is_anomaly": mad_z.abs() > z,
        }
    )


def main(nrows: int | None = None, out_prefix: str = "expedia") -> None:
    nonnull_all: dict[str, int] = {c: 0 for c in KEEP_COLS}
    nonnull_bk: dict[str, int] = {c: 0 for c in KEEP_COLS}
    uniques: dict[str, set] = {c: set() for c in UNIQUE_TRACK}
    samples: dict[str, list] = {c: [] for c in KEEP_COLS}
    daily: dict[str, dict] = {}
    hourly: dict[str, dict] = {}
    weekday: dict[str, dict] = {}
    booking_parts: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(
        RAW,
        usecols=KEEP_COLS,
        dtype=DTYPES,
        chunksize=CHUNKSIZE,
        low_memory=False,
        nrows=nrows,
    )
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        is_bk = chunk["is_booking"] == 1
        bk = chunk[is_bk]

        for col in KEEP_COLS:
            nonnull_all[col] += int(chunk[col].notna().sum())
            nonnull_bk[col] += int(bk[col].notna().sum())
            if len(samples[col]) < 3:
                vals = chunk[col].dropna()
                for v in vals.head(200).tolist():
                    if v not in samples[col]:
                        samples[col].append(v)
                    if len(samples[col]) >= 3:
                        break
        for col in UNIQUE_TRACK:
            uniques[col].update(chunk[col].dropna().unique().tolist())

        ts = pd.to_datetime(chunk["date_time"], errors="coerce")
        add_counts(daily, "click", ts[~is_bk].dt.date)
        add_counts(daily, "booking", ts[is_bk].dt.date)
        add_counts(hourly, "click", ts[~is_bk].dt.hour)
        add_counts(hourly, "booking", ts[is_bk].dt.hour)
        add_counts(weekday, "click", ts[~is_bk].dt.dayofweek)
        add_counts(weekday, "booking", ts[is_bk].dt.dayofweek)

        bk = bk.copy()
        bk["ts"] = ts[is_bk]
        bk["ci"] = pd.to_datetime(bk["srch_ci"], errors="coerce")
        bk["co"] = pd.to_datetime(bk["srch_co"], errors="coerce")
        booking_parts.append(bk.drop(columns=DATE_COLS))
        print(f"chunk {i}: rows={total_rows:,} bookings_kept={len(bk):,}", flush=True)

    b = pd.concat(booking_parts, ignore_index=True)
    del booking_parts
    b = b.sort_values("ts").reset_index(drop=True)
    n_bk = len(b)
    print(f"bookings in memory: {n_bk:,}", flush=True)

    # ---------- daily series + anomalies ----------
    daily_df = (
        pd.DataFrame(
            {
                "clicks": pd.Series(daily["click"]),
                "bookings": pd.Series(daily["booking"]),
            }
        )
        .fillna(0)
        .astype("int64")
        .sort_index()
    )
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df.index.name = "date"
    daily_df["total"] = daily_df["clicks"] + daily_df["bookings"]
    daily_df["booking_rate"] = daily_df["bookings"] / daily_df["total"]

    anom_bk = flag_anomalies(daily_df["bookings"])
    anom_tot = flag_anomalies(daily_df["total"])
    daily_df["bookings_mad_z"] = anom_bk["mad_z"]
    daily_df["bookings_is_anomaly"] = anom_bk["is_anomaly"]
    daily_df["total_mad_z"] = anom_tot["mad_z"]
    daily_df["total_is_anomaly"] = anom_tot["is_anomaly"]

    top_spikes = (
        anom_bk[anom_bk["is_anomaly"] & (anom_bk["mad_z"] > 0)]
        .sort_values("mad_z", ascending=False)
        .head(10)
    )
    top_drops = (
        anom_bk[anom_bk["is_anomaly"] & (anom_bk["mad_z"] < 0)]
        .sort_values("mad_z")
        .head(10)
    )

    # weekly periodicity: autocorrelation of daily bookings
    s = daily_df["bookings"].astype(float)
    acf = {lag: round(float(s.autocorr(lag)), 4) for lag in (1, 7, 14, 28, 30, 365)}

    # ---------- derived booking features ----------
    lead = (b["ci"] - b["ts"].dt.normalize()).dt.days
    stay = (b["co"] - b["ci"]).dt.days
    party = party_type(b["srch_adults_cnt"], b["srch_children_cnt"])
    b["party"] = party
    per_user = b.groupby("user_id").size()
    dist = b["orig_destination_distance"]

    cluster_counts = b["hotel_cluster"].value_counts().sort_values(ascending=False)
    cluster_share_sorted = (cluster_counts / n_bk).round(6).tolist()
    dest_counts = b["srch_destination_id"].value_counts()

    def q(series: pd.Series, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict:
        d = series.dropna()
        return {f"p{int(x * 100)}": round(float(d.quantile(x)), 3) for x in qs}

    def hist(series: pd.Series, bins: list[float], labels: list[str]) -> dict:
        d = series.dropna()
        cut = pd.cut(d, bins=bins, labels=labels, right=False)
        vc = cut.value_counts().reindex(labels).fillna(0)
        return {
            lbl: {"count": int(vc[lbl]), "share": round(float(vc[lbl] / len(d)), 6)}
            for lbl in labels
        }

    lead_bins = [-10_000, 0, 1, 4, 8, 15, 31, 61, 91, 181, 10_000]
    lead_labels = [
        "negative",
        "same day",
        "1-3d",
        "4-7d",
        "8-14d",
        "15-30d",
        "31-60d",
        "61-90d",
        "91-180d",
        "180d+",
    ]
    stay_bins = [-10_000, 1, 2, 3, 4, 5, 8, 15, 31, 10_000]
    stay_labels = [
        "<=0 nights",
        "1 night",
        "2 nights",
        "3 nights",
        "4 nights",
        "5-7 nights",
        "8-14 nights",
        "15-30 nights",
        "30+ nights",
    ]
    dist_bins = [0, 50, 200, 500, 1000, 2000, 5000, 1_000_000]
    dist_labels = [
        "<50mi",
        "50-200mi",
        "200-500mi",
        "500-1000mi",
        "1000-2000mi",
        "2000-5000mi",
        "5000mi+",
    ]
    depth_bins = [1, 2, 3, 5, 11, 21, 1_000_000]
    depth_labels = ["1", "2", "3-4", "5-10", "11-20", "21+"]

    # party × context cross-tabs
    party_pkg = b.groupby("party", observed=True)["is_package"].mean().round(6)
    party_lead = lead.groupby(party).median()
    party_stay = stay.groupby(party).median()
    party_mobile = b.groupby("party", observed=True)["is_mobile"].mean().round(6)

    ci_month = b["ci"].dt.month.value_counts(normalize=True).sort_index()
    ci_by_party = (
        pd.crosstab(b["ci"].dt.month, party, normalize="columns").round(6).to_dict()
    )

    monthly = b.groupby(b["ts"].dt.to_period("M")).size()
    monthly_pkg = b.groupby(b["ts"].dt.to_period("M"))["is_package"].mean().round(6)

    out = {
        "source": str(RAW.relative_to(ROOT)),
        "generated_by": "scripts/expedia_eda_detail.py",
        "total_rows": int(total_rows),
        "bookings": int(n_bk),
        "clicks": int(total_rows - n_bk),
        "n_columns": len(KEEP_COLS),
        "field_table": [
            {
                "no": i + 1,
                "field": col,
                "nonnull_all": nonnull_all[col],
                "coverage_all": round(nonnull_all[col] / total_rows, 6),
                "nonnull_bookings": nonnull_bk[col],
                "coverage_bookings": round(nonnull_bk[col] / n_bk, 6),
                "n_unique_all": (len(uniques[col]) if col in uniques else None),
                "samples": [str(v) for v in samples[col]],
            }
            for i, col in enumerate(KEEP_COLS)
        ],
        "daily": {
            "days_covered": len(daily_df),
            "date_min": str(daily_df.index.min().date()),
            "date_max": str(daily_df.index.max().date()),
            "bookings_mean": round(float(daily_df["bookings"].mean()), 1),
            "bookings_median": float(daily_df["bookings"].median()),
            "bookings_min": int(daily_df["bookings"].min()),
            "bookings_max": int(daily_df["bookings"].max()),
            "bookings_min_date": str(daily_df["bookings"].idxmin().date()),
            "bookings_max_date": str(daily_df["bookings"].idxmax().date()),
            "booking_rate_mean": round(float(daily_df["booking_rate"].mean()), 6),
            "autocorrelation": acf,
            "n_anomalies_bookings": int(daily_df["bookings_is_anomaly"].sum()),
            "n_anomalies_total": int(daily_df["total_is_anomaly"].sum()),
            "top_spikes": [
                {
                    "date": str(idx.date()),
                    "bookings": int(row["value"]),
                    "rolling_median": round(float(row["median"]), 1),
                    "mad_z": round(float(row["mad_z"]), 2),
                }
                for idx, row in top_spikes.iterrows()
            ],
            "top_drops": [
                {
                    "date": str(idx.date()),
                    "bookings": int(row["value"]),
                    "rolling_median": round(float(row["median"]), 1),
                    "mad_z": round(float(row["mad_z"]), 2),
                }
                for idx, row in top_drops.iterrows()
            ],
        },
        "hour_of_day": {
            "bookings": {int(k): int(v) for k, v in sorted(hourly["booking"].items())},
            "clicks": {int(k): int(v) for k, v in sorted(hourly["click"].items())},
        },
        "weekday": {
            "bookings": {int(k): int(v) for k, v in sorted(weekday["booking"].items())},
            "clicks": {int(k): int(v) for k, v in sorted(weekday["click"].items())},
        },
        "monthly_bookings": {str(k): int(v) for k, v in monthly.items()},
        "monthly_package_share": {str(k): float(v) for k, v in monthly_pkg.items()},
        "checkin_month_share": {
            int(k): round(float(v), 6) for k, v in ci_month.items()
        },
        "checkin_month_share_by_party": {
            str(k): {int(m): round(float(x), 6) for m, x in v.items()}
            for k, v in ci_by_party.items()
        },
        "lead_time_days": {
            "quantiles": q(lead),
            "mean": round(float(lead.dropna().mean()), 3),
            "negative_count": int((lead < 0).sum()),
            "histogram": hist(lead, lead_bins, lead_labels),
        },
        "stay_length_nights": {
            "quantiles": q(stay),
            "mean": round(float(stay.dropna().mean()), 3),
            "nonpositive_count": int((stay <= 0).sum()),
            "histogram": hist(stay, stay_bins, stay_labels),
        },
        "orig_destination_distance": {
            "missing_share_bookings": round(float(dist.isna().mean()), 6),
            "quantiles": q(dist),
            "mean": round(float(dist.dropna().mean()), 3),
            "max": round(float(dist.dropna().max()), 3),
            "histogram": hist(dist, dist_bins, dist_labels),
        },
        "user_depth": {
            "histogram": hist(per_user.astype(float), depth_bins, depth_labels),
            "quantiles": q(per_user.astype(float)),
            "gini_note": "share of bookings held by top user deciles",
            "top10pct_users_booking_share": round(
                float(
                    per_user.sort_values(ascending=False)
                    .head(int(len(per_user) * 0.1))
                    .sum()
                    / n_bk
                ),
                6,
            ),
        },
        "cluster_popularity": {
            "n_clusters": int(cluster_counts.size),
            "share_sorted_desc": cluster_share_sorted,
            "top5": {int(k): int(v) for k, v in cluster_counts.head(5).items()},
            "bottom5": {int(k): int(v) for k, v in cluster_counts.tail(5).items()},
            "share_top10": round(float(cluster_counts.head(10).sum() / n_bk), 6),
            "share_top20": round(float(cluster_counts.head(20).sum() / n_bk), 6),
            "share_bottom50": round(float(cluster_counts.tail(50).sum() / n_bk), 6),
        },
        "destinations": {
            "n_destinations_bookings": int(dest_counts.size),
            "share_top10": round(float(dest_counts.head(10).sum() / n_bk), 6),
            "share_top100": round(float(dest_counts.head(100).sum() / n_bk), 6),
            "share_top1000": round(float(dest_counts.head(1000).sum() / n_bk), 6),
            "n_dest_single_booking": int((dest_counts == 1).sum()),
        },
        "segments": {
            "party_share": {
                k: round(float(v), 6)
                for k, v in party.value_counts(normalize=True).items()
            },
            "party_package_share": {k: float(v) for k, v in party_pkg.items()},
            "party_lead_median": {k: float(v) for k, v in party_lead.items()},
            "party_stay_median": {k: float(v) for k, v in party_stay.items()},
            "party_mobile_share": {k: float(v) for k, v in party_mobile.items()},
            "is_mobile_share": round(float(b["is_mobile"].mean()), 6),
            "is_package_share": round(float(b["is_package"].mean()), 6),
            "channel_share": {
                int(k): round(float(v), 6)
                for k, v in b["channel"]
                .value_counts(normalize=True)
                .sort_index()
                .items()
            },
            "site_name_top5": {
                int(k): round(float(v), 6)
                for k, v in b["site_name"].value_counts(normalize=True).head(5).items()
            },
            "hotel_continent_share": {
                int(k): round(float(v), 6)
                for k, v in b["hotel_continent"]
                .value_counts(normalize=True)
                .sort_index()
                .items()
            },
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = (
        f"{out_prefix}_eda_detail.json",
        f"{out_prefix}_daily_series.csv",
        f"{out_prefix}_field_table.csv",
    )
    (OUT_DIR / names[0]).write_text(json.dumps(out, indent=2), encoding="utf-8")
    daily_df.to_csv(OUT_DIR / names[1])
    pd.DataFrame(out["field_table"]).to_csv(OUT_DIR / names[2], index=False)
    print("written:")
    for f in names:
        print(" ", (OUT_DIR / f).relative_to(ROOT))


if __name__ == "__main__":
    import sys

    smoke = "--smoke" in sys.argv
    main(nrows=500_000 if smoke else None, out_prefix="smoke" if smoke else "expedia")
