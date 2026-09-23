"""Recompute hotel_cluster booking counts restricted to the locked temporal
train split, to check a claim that the full-dataset per-cluster minimum
(2,465 bookings, from scripts/expedia_eda_detail.py) is being misquoted as a
"training" figure in reports/methodology_selection*.docx.

Reuses the exact split definition from scripts/expedia_context_signal.py:
temporal cut at the 80th percentile of booking date_time, train = ts < cut.

Input:  data/raw/hospitality/expedia/train.csv (read-only)
Output: reports/summary/expedia_train_split_cluster_check.json

Usage:
    python scripts/expedia_train_split_cluster_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "reports" / "summary" / "expedia_train_split_cluster_check.json"

USECOLS = ["date_time", "is_booking", "hotel_cluster"]
CHUNKSIZE = 2_000_000
TEST_FRACTION = 0.2


def main() -> None:
    parts = []
    rows = 0
    for i, chunk in enumerate(pd.read_csv(RAW, usecols=USECOLS, chunksize=CHUNKSIZE)):
        rows += len(chunk)
        parts.append(chunk[chunk["is_booking"] == 1])
        print(f"chunk {i}: rows={rows:,}", flush=True)
    b = pd.concat(parts, ignore_index=True)
    del parts
    b["ts"] = pd.to_datetime(b["date_time"])
    b = b.sort_values("ts").reset_index(drop=True)
    n_all = len(b)

    cut = b["ts"].quantile(1 - TEST_FRACTION)
    train = b[b["ts"] < cut]
    test = b[b["ts"] >= cut]

    counts_all = b["hotel_cluster"].value_counts().sort_values(ascending=False)
    counts_train = train["hotel_cluster"].value_counts().sort_values(ascending=False)

    result = {
        "source": str(RAW.relative_to(ROOT)),
        "generated_by": "scripts/expedia_train_split_cluster_check.py",
        "split": {
            "cut": str(cut),
            "n_all_bookings": n_all,
            "n_train_bookings": int(len(train)),
            "n_test_bookings": int(len(test)),
        },
        "cluster_counts_full_dataset": {
            "n_clusters": int(counts_all.size),
            "min_cluster": int(counts_all.idxmin()),
            "min_count": int(counts_all.min()),
            "bottom5": {int(k): int(v) for k, v in counts_all.tail(5).items()},
        },
        "cluster_counts_train_split_only": {
            "n_clusters": int(counts_train.size),
            "min_cluster": int(counts_train.idxmin()),
            "min_count": int(counts_train.min()),
            "bottom5": {int(k): int(v) for k, v in counts_train.tail(5).items()},
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
