"""Extra EDA numbers for the week-6 deck on the date split (companion of expedia_eda_split.py).

Inputs : data/interim/expedia_bookings.parquet, data/processed/expedia_date_split/rows.npz
Output : results/week6_retrain/eda_split_extra.json
  repeat_by_window : warm rows (L >= 1) per window split into new cluster / same as the previous
                     booking / older cluster already in the history (full history of the user)
  destination_spectrum : destinations and bookings by number of bookings per destination (all bookings)

Usage: .venv/Scripts/python.exe scripts/expedia_eda_split_extra.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nbp.paths import INTERIM, PROCESSED, ROOT

OUT = ROOT / "results" / "week6_retrain" / "eda_split_extra.json"
WN = ["train", "valid", "test"]
BUCKETS = [("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19), ("20-99", 20, 99), ("100-999", 100, 999), ("1000+", 1000, 10**12)]  # fmt: skip


def main() -> None:
    d = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=["item_id", "ctx_dest_id"])
    z = np.load(PROCESSED / "expedia_date_split" / "rows.npz")
    n = len(d)
    win, L = z["window"].astype(int), z["L"].astype("int64")
    uid = z["user"].astype("int64")
    y = d["item_id"].to_numpy().astype("int64")
    dest = d["ctx_dest_id"].to_numpy().astype("int64")
    assert len(win) == n

    first = pd.Series(np.arange(n)).groupby(uid * 128 + y, sort=False).transform("min").to_numpy()
    in_hist = first < np.arange(n)  # cluster booked earlier by the same user (full history)
    seq, seq_len = z["seq"], z["seq_len"].astype(int)
    last = np.where(seq_len > 0, seq[np.arange(n), np.maximum(seq_len - 1, 0)], -1) - 1
    is_last = (L >= 1) & (y == last)
    res: dict = {"repeat_by_window": {}}
    for i, k in enumerate(WN):
        m = (win == i) & (L >= 1)
        c_last = int((is_last & m).sum())
        c_old = int((in_hist & ~is_last & m).sum())
        c_new = int((~in_hist & m).sum())
        tot = int(m.sum())
        assert c_last + c_old + c_new == tot
        res["repeat_by_window"][k] = {
            "warm_rows": tot,
            "new_cluster": c_new,
            "same_as_previous": c_last,
            "older_cluster": c_old,
            "share_new": round(c_new / tot, 4),
            "share_same_as_previous": round(c_last / tot, 4),
            "share_older": round(c_old / tot, 4),
            "share_repeat_total": round((c_last + c_old) / tot, 4),
        }
    per_dest = np.bincount(dest)
    per_dest = per_dest[per_dest > 0]
    res["destination_spectrum"] = {
        "destinations": int(len(per_dest)),
        "bookings": int(per_dest.sum()),
        "buckets": {
            b: {
                "destinations": int(((per_dest >= lo) & (per_dest <= hi)).sum()),
                "bookings": int(per_dest[(per_dest >= lo) & (per_dest <= hi)].sum()),
                "share_destinations": round(float(((per_dest >= lo) & (per_dest <= hi)).mean()), 4),
                "share_bookings": round(
                    float(per_dest[(per_dest >= lo) & (per_dest <= hi)].sum() / per_dest.sum()), 4
                ),
            }
            for b, lo, hi in BUCKETS
        },
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1)[:2500])


if __name__ == "__main__":
    main()
