"""History-length and destination-support slices on Expedia bookings.

Evidence for the cold-start hybrid thresholds (history buckets, destination
support buckets, prior smoothing strength). Same split as
scripts/expedia_context_signal.py: global temporal cut at the 80th percentile of
booking date_time. Every count/prior comes from train bookings only; history for
a test event is every earlier booking of that user ("online"), and train-only
history is reported separately.

Heuristic rankers compared (Recall@5, full ranking over 100 clusters):
- global / dest_type / market_via_dest: popularity per key
- dest_raw_backoff_market: destination popularity, unseen destination -> market
- dest_smooth_m{5,20,50}: (n_dest,k + m * p_market,k) / (n_dest + m)
- history_then_dest: user's past clusters (recency weight 0.7^age) ranked first
- same_dest_history_then_dest[_smooth20]: only past clusters booked at the same
  srch_destination_id as the query ranked first

Market is the train-modal hotel_market of the searched destination, never the
event's own hotel_market (that field describes the booked hotel -> label leak).

Usage: python scripts/expedia_history_slices.py
Output: results/week2_methodology/expedia_history_slices.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "results" / "week2_methodology" / "expedia_history_slices.json"
COLS = [
    "date_time",
    "user_id",
    "is_booking",
    "srch_destination_id",
    "srch_destination_type_id",
    "hotel_market",
    "hotel_cluster",
]
K = 5
N_CLUSTERS = 100
RECENCY_DECAY = 0.7
CHUNK = 50_000
HISTORY_BUCKETS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("10+", 10, 10**9),
]
SUPPORT_BUCKETS = [
    ("0", 0, 0),
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20-99", 20, 99),
    ("100-999", 100, 999),
    ("1000+", 1000, 10**12),
]
METHODS = [
    "global",
    "dest_type",
    "market_via_dest",
    "dest_raw_backoff_market",
    "dest_smooth_m5",
    "dest_smooth_m20",
    "dest_smooth_m50",
    "history_then_dest",
    "same_dest_history_then_dest",
    "same_dest_history_then_dest_smooth20",
]


def load_bookings() -> pd.DataFrame:
    """Chunked pass over train.csv keeping is_booking == 1 rows (COLS minus is_booking)."""
    dtypes = {c: "int32" for c in COLS if c != "date_time"}
    parts = []
    for ch in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000, dtype=dtypes):
        ch = ch[ch.is_booking == 1].drop(columns="is_booking")
        ch["date_time"] = pd.to_datetime(ch["date_time"])
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def count_table(tr: pd.DataFrame, key: str) -> pd.DataFrame:
    """key x 100 matrix of train booking counts per cluster."""
    g = tr.groupby([key, "hotel_cluster"]).size().unstack(fill_value=0)
    return g.reindex(columns=range(N_CLUSTERS), fill_value=0)


def take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Rows of values at idx; idx == -1 (unseen key) gives a zero row."""
    out = np.zeros((len(idx), N_CLUSTERS), np.float32)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def norm(x: np.ndarray) -> np.ndarray:
    s = x.sum(1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x), where=s > 0)


def main() -> None:
    bk = load_bookings().sort_values(["user_id", "date_time"], kind="stable")
    bk = bk.reset_index(drop=True)
    cut = bk["date_time"].quantile(0.8)
    bk["is_test"] = bk["date_time"] > cut
    bk["hist_online"] = bk.groupby("user_id").cumcount()
    g = bk.groupby("user_id")["is_test"]
    n_train_user = g.transform("size") - g.transform("sum")
    bk["hist_train"] = np.where(bk["is_test"], n_train_user, bk["hist_online"])

    tr = bk[~bk.is_test]
    te_idx = np.flatnonzero(bk.is_test.values)
    te = bk.iloc[te_idx]
    y = te["hotel_cluster"].values
    n_te = len(te)

    glob_p = np.bincount(tr.hotel_cluster, minlength=N_CLUSTERS).astype(np.float32)
    glob_p /= glob_p.sum()
    dest_tab = count_table(tr, "srch_destination_id")
    dtype_tab = count_table(tr, "srch_destination_type_id")
    mkt_tab = count_table(tr, "hotel_market")

    dm = (
        tr.groupby(["srch_destination_id", "hotel_market"]).size().reset_index(name="n")
    )
    modal = dm.sort_values("n", ascending=False).drop_duplicates("srch_destination_id")
    dest2mkt = modal.set_index("srch_destination_id")["hotel_market"]
    purity = float(modal["n"].sum() / dm["n"].sum())

    d_idx = dest_tab.index.get_indexer(te.srch_destination_id.values)
    mk = dest2mkt.reindex(te.srch_destination_id.values).values
    m_idx = mkt_tab.index.get_indexer(np.where(np.isnan(mk), -1, mk).astype(np.int64))
    t_idx = dtype_tab.index.get_indexer(te.srch_destination_type_id.values)
    dv = dest_tab.values.astype(np.float32)
    mv = mkt_tab.values.astype(np.float32)
    tv = dtype_tab.values.astype(np.float32)
    n_dest = np.where(d_idx >= 0, dv.sum(1)[np.maximum(d_idx, 0)], 0)

    cl = bk.hotel_cluster.values
    dst = bk.srch_destination_id.values
    hist_len = bk["hist_online"].values
    hits = {k: np.zeros(n_te, bool) for k in METHODS}
    in_hist = np.zeros(n_te, bool)
    in_hist_same = np.zeros(n_te, bool)

    for a in range(0, n_te, CHUNK):
        b = min(a + CHUNK, n_te)
        n, yy = b - a, y[a:b]
        d, m, t = take(dv, d_idx[a:b]), take(mv, m_idx[a:b]), take(tv, t_idx[a:b])
        nd, nm = d.sum(1), m.sum(1)
        p_glob = np.broadcast_to(glob_p, (n, N_CLUSTERS))
        p_mkt = np.where(nm[:, None] > 0, norm(m), p_glob)
        p_dtype = np.where(t.sum(1)[:, None] > 0, norm(t), p_glob)
        p_dest = np.where(nd[:, None] > 0, norm(d), p_mkt)

        def smooth(s: float, d=d, p_mkt=p_mkt, nd=nd) -> np.ndarray:
            return (d + s * p_mkt) / (nd[:, None] + s)

        h_any = np.zeros((n, N_CLUSTERS), np.float32)
        h_same = np.zeros((n, N_CLUSTERS), np.float32)
        for j in range(n):
            i = te_idx[a + j]
            h = hist_len[i]
            if h == 0:
                continue
            past = cl[i - h : i]
            w = RECENCY_DECAY ** np.arange(h - 1, -1, -1)
            np.add.at(h_any[j], past, w)
            same = dst[i - h : i] == dst[i]
            if same.any():
                np.add.at(h_same[j], past[same], w[same])
        r = np.arange(n)
        in_hist[a:b] = h_any[r, yy] > 0
        in_hist_same[a:b] = h_same[r, yy] > 0

        scores = {
            "global": p_glob,
            "dest_type": p_dtype,
            "market_via_dest": p_mkt,
            "dest_raw_backoff_market": p_dest,
            "dest_smooth_m5": smooth(5),
            "dest_smooth_m20": smooth(20),
            "dest_smooth_m50": smooth(50),
            "history_then_dest": h_any * 10 + p_dest,
            "same_dest_history_then_dest": h_same * 10 + p_dest,
            "same_dest_history_then_dest_smooth20": h_same * 10 + smooth(20),
        }
        for k, v in scores.items():
            top = np.argpartition(-np.asarray(v), K, axis=1)[:, :K]
            hits[k][a:b] = (top == yy[:, None]).any(1)

    def summarize(mask: np.ndarray) -> dict:
        out = {"n": int(mask.sum()), "share": round(float(mask.mean()), 6)}
        for k, h in hits.items():
            out[f"R@5_{k}"] = round(float(h[mask].mean()), 6) if mask.any() else None
        out["target_in_history"] = round(float(in_hist[mask].mean()), 6)
        out["target_in_same_dest_history"] = round(float(in_hist_same[mask].mean()), 6)
        return out

    ho, ht = te.hist_online.values, te.hist_train.values
    res = {
        "source": str(RAW.relative_to(ROOT)),
        "generated_by": "scripts/expedia_history_slices.py",
        "cut": str(cut),
        "n_train": len(tr),
        "n_test": int(n_te),
        "candidate_rows_train_x100": len(tr) * N_CLUSTERS,
        "candidate_rows_test_x100": int(n_te) * N_CLUSTERS,
        "dest_to_modal_market_purity": round(purity, 6),
        "test_dest_seen_in_train_share": round(float((d_idx >= 0).mean()), 6),
        "overall": summarize(np.ones(n_te, bool)),
        "by_history_online": {
            b: summarize((ho >= lo) & (ho <= hi)) for b, lo, hi in HISTORY_BUCKETS
        },
        "by_history_train_only": {
            b: summarize((ht >= lo) & (ht <= hi)) for b, lo, hi in HISTORY_BUCKETS
        },
        "by_dest_support": {
            b: summarize((n_dest >= lo) & (n_dest <= hi))
            for b, lo, hi in SUPPORT_BUCKETS
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
