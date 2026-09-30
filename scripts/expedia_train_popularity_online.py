"""Leak-free (expanding-window) popularity ranks on Expedia train bookings, by destination support.

For every train booking (same split as scripts/expedia_history_slices.py: global cut at the 80th
percentile of booking date_time, train = date_time <= cut) the popularity prior is built ONLY from
bookings with a strictly earlier date_time. Bookings sharing a timestamp never see each other.
The booking is assigned to exactly one bucket by n_dest = earlier bookings of its searched
destination:

- n_dest == 0     -> global popularity (earlier bookings, all destinations)
- 1 <= n_dest < 20 -> destination counts smoothed to market, m = 5
- n_dest >= 20     -> same smoothed formula, m = 5

  p(k|dest) = (n_dest,k + m * p_market,k) / (n_dest + m)

Market = modal hotel_market of earlier bookings at that destination; p_market = cluster
distribution of earlier bookings in that market. Also reported per bucket for comparison: global
popularity and raw destination popularity (no smoothing). The booking's own hotel_market is never used.

Rank of the true cluster among 100 = 1 + #clusters with a strictly higher score + 0.5 * (#tied - 1)
(average rank over ties, so an empty prior gives 50.5, not 1).

Usage: python scripts/expedia_train_popularity_online.py
Output: reports/summary/week2_methodology/expedia_train_popularity_online.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = (
    ROOT
    / "reports"
    / "summary"
    / "week2_methodology"
    / "expedia_train_popularity_online.json"
)
COLS = [
    "date_time",
    "user_id",
    "is_booking",
    "srch_destination_id",
    "hotel_market",
    "hotel_cluster",
]
N_CLUSTERS = 100
M_SMOOTH = 5
DEST_THRESHOLD = 20
BUCKETS = ["0", "1-19", "20+"]
METHODS = ["primary", "global", "dest_raw"]
KS = (1, 5, 10, 20)


def load_bookings() -> pd.DataFrame:
    """Chunked pass over train.csv keeping is_booking == 1 rows."""
    dtypes = {c: "int32" for c in COLS if c != "date_time"}
    parts = []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000, dtype=dtypes):
        ch = ch[ch.is_booking == 1].drop(columns="is_booking")
        ch["date_time"] = pd.to_datetime(ch["date_time"])
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def avg_rank(score: np.ndarray, y: int) -> float:
    """Average rank (1 = best) of cluster y under score, ties share the mean rank."""
    s = score[y]
    return 1.0 + float((score > s).sum()) + 0.5 * (float((score == s).sum()) - 1.0)


def main() -> None:
    bk = load_bookings().sort_values("date_time", kind="stable").reset_index(drop=True)
    cut = bk["date_time"].quantile(0.8)
    tr = bk[bk["date_time"] <= cut].reset_index(drop=True)
    print(f"bookings total={len(bk):,}  train={len(tr):,}  cut={cut}")

    # user history = the user's bookings with strictly earlier date_time (any destination)
    hist = (tr.groupby("user_id")["date_time"].rank(method="min") - 1).astype(int).values
    ts = tr["date_time"].values.astype("int64")
    dest = tr["srch_destination_id"].values
    mkt = tr["hotel_market"].values
    cl = tr["hotel_cluster"].values
    n = len(tr)

    glob = np.zeros(N_CLUSTERS, np.float64)
    dest_cnt: dict[int, np.ndarray] = {}
    mkt_cnt: dict[int, np.ndarray] = {}
    dest_mkt: dict[int, dict[int, int]] = {}
    dest_modal: dict[int, int] = {}

    ranks = {m: np.zeros(n, np.float32) for m in METHODS}
    bucket = np.zeros(n, np.int8)
    n_dest_arr = np.zeros(n, np.int32)

    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        for i in range(a, b):
            d, y = dest[i], cl[i]
            dc = dest_cnt.get(d)
            nd = 0 if dc is None else dc.sum()
            n_dest_arr[i] = nd
            r_glob = avg_rank(glob, y)
            ranks["global"][i] = r_glob
            if nd == 0:
                bucket[i] = 0
                ranks["primary"][i] = r_glob
                ranks["dest_raw"][i] = r_glob
                continue
            bucket[i] = 1 if nd < DEST_THRESHOLD else 2
            mc = mkt_cnt[dest_modal[d]]
            p_mkt = mc / mc.sum()
            ranks["primary"][i] = avg_rank((dc + M_SMOOTH * p_mkt) / (nd + M_SMOOTH), y)
            ranks["dest_raw"][i] = avg_rank(dc, y)
        for i in range(a, b):  # update only after the whole timestamp group is scored
            d, m, y = dest[i], mkt[i], cl[i]
            glob[y] += 1
            dest_cnt.setdefault(d, np.zeros(N_CLUSTERS))[y] += 1
            mkt_cnt.setdefault(m, np.zeros(N_CLUSTERS))[y] += 1
            dm = dest_mkt.setdefault(d, {})
            dm[m] = dm.get(m, 0) + 1
            cur = dest_modal.get(d)
            if cur is None or dm[m] > dm[cur]:
                dest_modal[d] = m

    def summarize(mask: np.ndarray) -> dict:
        """Per-bucket rank metrics restricted to bookings where mask is True."""
        out = {"n_bookings": int(mask.sum()), "buckets": {}}
        for bi, name in enumerate(BUCKETS):
            sel = (bucket == bi) & mask
            row = {"n": int(sel.sum()), "share": round(float(sel.sum() / mask.sum()), 6)}
            for m in METHODS:
                r = ranks[m][sel]
                row[m] = {
                    "mean_rank": round(float(r.mean()), 3),
                    "median_rank": float(np.median(r)),
                    "mrr": round(float((1.0 / r).mean()), 4),
                    **{f"recall@{k}": round(float((r <= k).mean()), 4) for k in KS},
                }
            out["buckets"][name] = row
        out["overall_primary_mean_rank"] = round(float(ranks["primary"][mask].mean()), 3)
        return out

    res = {
        "cut": str(cut),
        "n_train": int(n),
        "m_smooth": M_SMOOTH,
        "dest_threshold": DEST_THRESHOLD,
        "all_users": summarize(np.ones(n, bool)),
        "user_history_lt5": summarize(hist < 5),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
