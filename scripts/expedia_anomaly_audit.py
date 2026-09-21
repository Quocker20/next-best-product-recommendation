"""Anomaly / outlier audit of Expedia `train.csv`, split by clicks vs bookings.

Quantifies the defects the schema scan exposed (check-in dates in 2558, stays
in the past, zero-adult parties, extreme lead times) so the report can state
each one with a count and a share instead of a qualitative "data is clean".

Input:  data/raw/hospitality/expedia/train.csv (read-only)
Output: reports/summary/expedia_anomaly_audit.json

Usage:
    .venv/Scripts/python.exe scripts/expedia_anomaly_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "reports" / "summary" / "expedia_anomaly_audit.json"

USECOLS = [
    "date_time",
    "srch_ci",
    "srch_co",
    "is_booking",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_rm_cnt",
    "cnt",
    "orig_destination_distance",
]
CHUNKSIZE = 2_000_000
# last event in the file is 2014-12-31; anything checking in after 2016 is
# treated as a typo rather than a very early booking
PLAUSIBLE_CI_MAX = pd.Timestamp("2016-12-31")


def main() -> None:
    checks = [
        "srch_ci_missing",
        "srch_co_missing",
        "srch_ci_before_search_date",
        "srch_ci_after_2016",
        "srch_ci_before_2013",
        "lead_time_gt_365d",
        "srch_co_before_srch_ci",
        "srch_co_equals_srch_ci",
        "stay_gt_30_nights",
        "zero_adults",
        "zero_rooms",
        "adults_gt_4",
        "children_gt_4",
        "guests_gt_rooms_x4",
        "cnt_gt_50",
        "distance_missing",
        "distance_lt_1mi",
    ]
    counts = {"all": dict.fromkeys(checks, 0), "booking": dict.fromkeys(checks, 0)}
    rows = {"all": 0, "booking": 0}
    ci_year: dict[str, dict[int, int]] = {"all": {}, "booking": {}}
    lead_max = {"all": -(10**9), "booking": -(10**9)}
    stay_max = {"all": -(10**9), "booking": -(10**9)}

    for i, chunk in enumerate(pd.read_csv(RAW, usecols=USECOLS, chunksize=CHUNKSIZE)):
        ts = pd.to_datetime(chunk["date_time"], errors="coerce")
        ci = pd.to_datetime(chunk["srch_ci"], errors="coerce")
        co = pd.to_datetime(chunk["srch_co"], errors="coerce")
        lead = (ci - ts.dt.normalize()).dt.days
        stay = (co - ci).dt.days
        is_bk = chunk["is_booking"] == 1

        flags = {
            "srch_ci_missing": chunk["srch_ci"].isna(),
            "srch_co_missing": chunk["srch_co"].isna(),
            "srch_ci_before_search_date": lead < 0,
            "srch_ci_after_2016": ci > PLAUSIBLE_CI_MAX,
            "srch_ci_before_2013": ci < pd.Timestamp("2013-01-01"),
            "lead_time_gt_365d": lead > 365,
            "srch_co_before_srch_ci": stay < 0,
            "srch_co_equals_srch_ci": stay == 0,
            "stay_gt_30_nights": stay > 30,
            "zero_adults": chunk["srch_adults_cnt"] == 0,
            "zero_rooms": chunk["srch_rm_cnt"] == 0,
            "adults_gt_4": chunk["srch_adults_cnt"] > 4,
            "children_gt_4": chunk["srch_children_cnt"] > 4,
            "guests_gt_rooms_x4": (
                chunk["srch_adults_cnt"] + chunk["srch_children_cnt"]
            )
            > (chunk["srch_rm_cnt"] * 4),
            "cnt_gt_50": chunk["cnt"] > 50,
            "distance_missing": chunk["orig_destination_distance"].isna(),
            "distance_lt_1mi": chunk["orig_destination_distance"] < 1,
        }
        rows["all"] += len(chunk)
        rows["booking"] += int(is_bk.sum())
        for name, mask in flags.items():
            m = mask.fillna(False)
            counts["all"][name] += int(m.sum())
            counts["booking"][name] += int((m & is_bk).sum())

        for scope, sel in (("all", slice(None)), ("booking", is_bk)):
            yrs = ci[sel].dt.year.value_counts()
            for y, c in yrs.items():
                if not np.isnan(y):
                    ci_year[scope][int(y)] = ci_year[scope].get(int(y), 0) + int(c)
            lv, sv = lead[sel].max(), stay[sel].max()
            if pd.notna(lv):
                lead_max[scope] = max(lead_max[scope], int(lv))
            if pd.notna(sv):
                stay_max[scope] = max(stay_max[scope], int(sv))
        print(f"chunk {i}: rows={rows['all']:,}", flush=True)

    out = {
        "source": str(RAW.relative_to(ROOT)),
        "generated_by": "scripts/expedia_anomaly_audit.py",
        "rows_all": rows["all"],
        "rows_booking": rows["booking"],
        "plausible_checkin_max": str(PLAUSIBLE_CI_MAX.date()),
        "checks": {
            name: {
                "all_count": counts["all"][name],
                "all_share": round(counts["all"][name] / rows["all"], 8),
                "booking_count": counts["booking"][name],
                "booking_share": round(counts["booking"][name] / rows["booking"], 8),
            }
            for name in checks
        },
        "checkin_year_histogram": {
            scope: dict(sorted(ci_year[scope].items())) for scope in ci_year
        },
        "max_lead_time_days": lead_max,
        "max_stay_nights": stay_max,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
