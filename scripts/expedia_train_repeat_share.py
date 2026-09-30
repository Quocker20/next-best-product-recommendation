"""Repeat-vs-new share of bookings per Expedia temporal split (train / valid / test).

Splits = earliest 80 % / next 10 % / last 10 % of bookings by timestamp (RecBole RS 0.8/0.1/0.1, order TO).
Each booking is compared with ALL earlier bookings of the same user (history may come from
earlier splits): same as last booking / in older history (not last) / brand-new cluster.
First-ever bookings have no history and are counted separately.

Input:  data/interim/recbole/expedia/expedia.inter
Output: reports/summary/expedia_train_repeat_share.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTER = ROOT / "data" / "interim" / "recbole" / "expedia" / "expedia.inter"
OUT = ROOT / "reports" / "summary" / "expedia_train_repeat_share.json"

df = pd.read_csv(INTER, sep="	", names=["u", "i", "t"], header=0)
df = df.sort_values("t", kind="stable").reset_index(drop=True)
n_all = len(df)
a, b = int(n_all * 0.8), int(n_all * 0.9)
df["split"] = np.where(df.index < a, "train", np.where(df.index < b, "valid", "test"))
df = df.sort_values(["u", "t"], kind="stable").reset_index(drop=True)

df["prev"] = df.groupby("u")["i"].shift(1)
df["has_hist"] = df["prev"].notna()
df["is_last"] = df["i"] == df["prev"]
df["seen_before"] = df.groupby(["u", "i"]).cumcount() > 0
df["in_old"] = df["seen_before"] & ~df["is_last"]
df["is_new"] = df["has_hist"] & ~df["seen_before"]

out = {}
for name in ["train", "valid", "test"]:
    tr = df[df.split == name]
    n = len(tr)
    h = tr[tr.has_hist]
    nh = len(h)
    res = {
        "bookings": n,
        "users": int(tr.u.nunique()),
        "no_history_first_booking": int((~tr.has_hist).sum()),
        "bookings_with_history": nh,
        "same_as_last": int(h.is_last.sum()),
        "in_old_history_not_last": int(h.in_old.sum()),
        "in_any_history_incl_last": int(h.seen_before.sum()),
        "new_cluster": int(h.is_new.sum()),
    }
    for k in ["same_as_last", "in_old_history_not_last", "in_any_history_incl_last", "new_cluster"]:
        res[f"pct_{k}_of_with_history"] = round(100 * res[k] / nh, 2)
        res[f"pct_{k}_of_all"] = round(100 * res[k] / n, 2)
    res["pct_no_history_of_all"] = round(100 * res["no_history_first_booking"] / n, 2)
    out[name] = res
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

# --- breakdown by history length (# earlier bookings of the user) ---
df["hist_len"] = df.groupby("u").cumcount()
bins = [-1, 0, 1, 2, 3, 4, 9, np.inf]
labels = ["0 (new)", "1", "2", "3", "4", "5-9", "10+"]
df["bucket"] = pd.cut(df["hist_len"], bins=bins, labels=labels)
by = {}
for name in ["train", "valid", "test"]:
    s = df[df.split == name]
    rows = {}
    for lab, g in s.groupby("bucket", observed=True):
        m = len(g)
        rows[str(lab)] = {
            "bookings": m,
            "pct_of_split": round(100 * m / len(s), 2),
            "pct_same_as_last": round(100 * g.is_last.mean(), 2),
            "pct_old_not_last": round(100 * g.in_old.mean(), 2),
            "pct_any_history": round(100 * g.seen_before.mean(), 2),
            "pct_new_cluster": round(100 * g.is_new.mean(), 2),
        }
    by[name] = rows
    print(f"\n== {name} ==")
    print(pd.DataFrame(rows).T.to_string())
OUT2 = ROOT / "reports" / "summary" / "expedia_repeat_share_by_history_len.json"
OUT2.write_text(json.dumps(by, indent=2))
