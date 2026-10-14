"""EDA of the whole booking table (all 2,952,675 Expedia bookings, no split) for the week-6 v2 deck.

Output: results/week6_retrain/eda_all_bookings.json
Usage: .venv/Scripts/python.exe scripts/eda_all_bookings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/workspace/PROJECTS/next-best-product-recommendation/data/interim/expedia_bookings.parquet")
OUT = ROOT / "results" / "week6_retrain" / "eda_all_bookings.json"


def main() -> None:
    d = pd.read_parquet(SRC, columns=["user_id", "item_id", "timestamp", "ctx_dest_id", "ctx_checkin_month",
                                      "ctx_party_type", "ctx_is_package", "ctx_is_mobile", "ctx_lead_days",
                                      "ctx_stay_nights"])
    d = d.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    n = len(d)
    out: dict = {"bookings": n, "users": int(d.user_id.nunique())}
    out["first_date"], out["last_date"] = str(d.timestamp.min().date()), str(d.timestamp.max().date())

    m = d.timestamp.dt.to_period("M").astype(str).value_counts().sort_index()
    out["monthly_bookings"] = {k: int(v) for k, v in m.items()}
    ci = d.ctx_checkin_month.value_counts().sort_index()
    out["checkin_month_share"] = {str(int(k)): float(v / n) for k, v in ci.items()}
    out["party_share"] = {k: float(v / n) for k, v in d.ctx_party_type.value_counts().items()}
    out["package_share"] = float(d.ctx_is_package.mean())
    out["mobile_share"] = float(d.ctx_is_mobile.mean())
    out["lead_days_median"] = float(d.ctx_lead_days.median())
    out["stay_nights_median"] = float(d.ctx_stay_nights.median())

    cnt = d.groupby("user_id").size()
    out["bookings_per_user"] = {
        "median": float(cnt.median()), "mean": float(cnt.mean()),
        "share_1": float((cnt == 1).mean()), "share_2": float((cnt == 2).mean()),
        "share_3_4": float(cnt.between(3, 4).mean()), "share_5_9": float(cnt.between(5, 9).mean()),
        "share_10_plus": float((cnt >= 10).mean()),
    }
    g = d.groupby("user_id").timestamp.diff().dt.total_seconds().dropna() / 86400
    out["gap_days"] = {"median": float(g.median()), "share_within_1h": float((g <= 1 / 24).mean()),
                       "share_within_1d": float((g <= 1).mean()), "n_pairs": int(len(g))}

    edges = [0, 1 / 24, 1, 7, 30, 90, np.inf]
    labels = ["le_1h", "1h_1d", "1d_7d", "7d_30d", "30d_90d", "gt_90d"]
    b = pd.cut(g, bins=edges, labels=labels, include_lowest=True).value_counts(normalize=True)
    out["gap_buckets"] = {k: float(b[k]) for k in labels}
    # repeat shares: booking whose cluster / destination appears earlier in the same user's history
    has_hist = d.groupby("user_id").cumcount() > 0
    seen_c = d.groupby(["user_id", "item_id"]).cumcount() > 0
    seen_d = d.groupby(["user_id", "ctx_dest_id"]).cumcount() > 0
    out["repeat_given_history"] = {"rows_with_history": int(has_hist.sum()),
                                   "cluster_seen_before": float(seen_c[has_hist].mean()),
                                   "destination_seen_before": float(seen_d[has_hist].mean())}

    dest = d.ctx_dest_id.value_counts()
    out["destinations"] = {"n": int(len(dest)), "top10": float(dest.iloc[:10].sum() / n),
                           "top100": float(dest.iloc[:100].sum() / n),
                           "top1000": float(dest.iloc[:1000].sum() / n)}
    ranks = np.unique(np.round(np.logspace(0, np.log10(len(dest)), 60)).astype(int))
    cum = dest.cumsum().to_numpy() / n
    out["destination_curve"] = {"ranks": ranks.tolist(), "cum": [float(cum[r - 1]) for r in ranks]}
    cl = d.item_id.value_counts()
    out["clusters"] = {"n": int(len(cl)), "top1": float(cl.iloc[0] / n), "top10": float(cl.iloc[:10].sum() / n),
                       "share_by_rank": [float(v / n) for v in cl.to_numpy()]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("destination_curve", "monthly_bookings")}, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
