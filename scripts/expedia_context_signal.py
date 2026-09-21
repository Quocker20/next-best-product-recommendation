"""How much does trip context move the hotel-cluster distribution? (EDA probe)

Two measurements, both on the booking subset of Expedia `train.csv`:

1. Distribution shift: Jensen-Shannon divergence between the global
   hotel_cluster distribution and the distribution conditioned on check-in
   month, party type, package flag, destination and market.
2. A cheap popularity probe under the locked protocol split (temporal 80/20,
   full ranking over 100 clusters): global popularity vs popularity
   conditioned on one context field vs repeat-last-cluster.

The probe is EDA evidence that context carries signal, NOT the Phase-2
benchmark: no tuning, no model family, single seed, one metric family.

Input:  data/raw/hospitality/expedia/train.csv (read-only)
Output: reports/summary/expedia_context_signal.json

Usage:
    .venv/Scripts/python.exe scripts/expedia_context_signal.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
OUT = ROOT / "reports" / "summary" / "expedia_context_signal.json"

USECOLS = [
    "date_time",
    "user_id",
    "is_booking",
    "hotel_cluster",
    "srch_ci",
    "srch_adults_cnt",
    "srch_children_cnt",
    "srch_destination_id",
    "hotel_market",
    "is_package",
]
CHUNKSIZE = 2_000_000
N_CLUSTERS = 100
TEST_FRACTION = 0.2
K = 5


def party_type(adults: pd.Series, children: pd.Series) -> pd.Series:
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


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two distributions."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(series: pd.Series) -> np.ndarray:
    return (
        series.value_counts().reindex(range(N_CLUSTERS)).fillna(0).to_numpy(dtype=float)
    )


def topk_from_counts(df: pd.DataFrame, by: str, k: int) -> dict:
    """{context value: [top-k clusters]} from training counts."""
    counts = df.groupby([by, "hotel_cluster"]).size().rename("n").reset_index()
    counts = counts.sort_values([by, "n"], ascending=[True, False])
    return (
        counts.groupby(by)["hotel_cluster"]
        .apply(lambda s: s.head(k).tolist())
        .to_dict()
    )


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
    b["ci_month"] = pd.to_datetime(b["srch_ci"], errors="coerce").dt.month
    b["party"] = party_type(b["srch_adults_cnt"], b["srch_children_cnt"])
    b = b.sort_values("ts").reset_index(drop=True)
    n = len(b)
    print(f"bookings: {n:,}", flush=True)

    # ---------- 1. distribution shift ----------
    p_global = dist(b["hotel_cluster"])
    global_top = list(b["hotel_cluster"].value_counts().head(K).index.astype(int))

    def shift_table(key: str, min_rows: int = 1000, limit: int | None = None) -> dict:
        g = b.groupby(key)
        recs = []
        for val, sub in g:
            if len(sub) < min_rows:
                continue
            d = dist(sub["hotel_cluster"])
            recs.append(
                {
                    "value": (
                        int(val) if isinstance(val, (int, np.integer)) else str(val)
                    ),
                    "bookings": len(sub),
                    "share_of_all": round(len(sub) / n, 6),
                    "js_divergence_bits": round(js_divergence(d, p_global), 5),
                    "top_cluster": int(np.argmax(d)),
                    "top_cluster_share": round(float(d.max() / d.sum()), 6),
                }
            )
        recs.sort(key=lambda r: r["js_divergence_bits"], reverse=True)
        return {
            "n_groups_measured": len(recs),
            "js_mean": round(
                float(np.mean([r["js_divergence_bits"] for r in recs])), 5
            ),
            "js_max": round(float(max(r["js_divergence_bits"] for r in recs)), 5),
            "n_groups_with_different_top_cluster": sum(
                1 for r in recs if r["top_cluster"] != int(np.argmax(p_global))
            ),
            "groups": recs[:limit] if limit else recs,
        }

    shifts = {
        "checkin_month": shift_table("ci_month"),
        "party": shift_table("party", min_rows=1),
        "is_package": shift_table("is_package", min_rows=1),
        "hotel_market_top": shift_table("hotel_market", min_rows=5000, limit=15),
        "srch_destination_top": shift_table(
            "srch_destination_id", min_rows=5000, limit=15
        ),
    }

    # ---------- 2. popularity probe on the locked temporal split ----------
    cut = b["ts"].quantile(1 - TEST_FRACTION)
    train, test = b[b["ts"] < cut].copy(), b[b["ts"] >= cut].copy()
    truth = test["hotel_cluster"].to_numpy()

    def recall_at_k(pred_lists: list[list[int]], k: int) -> float:
        hit = sum(1 for p, t in zip(pred_lists, truth) if t in p[:k])
        return round(hit / len(truth), 6)

    def context_pred(key: str) -> list[list[int]]:
        table = topk_from_counts(train, key, K)
        return [table.get(v, global_top) for v in test[key].tolist()]

    last_cluster = train.groupby("user_id")["hotel_cluster"].last().to_dict()
    repeat_pred = [
        ([last_cluster[u]] + [c for c in global_top if c != last_cluster[u]])
        if u in last_cluster
        else global_top
        for u in test["user_id"].tolist()
    ]
    covered = {
        key: context_pred(key)
        for key in ("srch_destination_id", "hotel_market", "ci_month", "party")
    }
    global_pred = [global_top] * len(test)

    # does a context field add signal ON TOP of destination? back off
    # (destination, context) -> destination -> global
    dest_table = topk_from_counts(train, "srch_destination_id", K)

    def pair_pred(key: str) -> list[list[int]]:
        train_pair = train.assign(
            _pair=list(zip(train["srch_destination_id"], train[key]))
        )
        pair_table = topk_from_counts(train_pair, "_pair", K)
        out = []
        for dest, ctx in zip(test["srch_destination_id"].tolist(), test[key].tolist()):
            out.append(
                pair_table.get((dest, ctx)) or dest_table.get(dest) or global_top
            )
        return out

    pair_preds = {
        f"popularity_by_destination_x_{k}": pair_pred(k)
        for k in ("ci_month", "party", "is_package")
    }

    # how much does the destination mix itself move with the season?
    dest_month_js = []
    dest_global = b["srch_destination_id"].value_counts().rename("n").to_frame()
    dest_index = dest_global.index
    p_dest_global = (dest_global["n"] / dest_global["n"].sum()).to_numpy()
    for m, sub in b.groupby("ci_month"):
        q = (
            sub["srch_destination_id"]
            .value_counts()
            .reindex(dest_index)
            .fillna(0)
            .to_numpy(dtype=float)
        )
        dest_month_js.append(
            {
                "checkin_month": int(m),
                "js_divergence_bits": round(js_divergence(q, p_dest_global), 5),
            }
        )

    probe = {
        "protocol": (
            "temporal split at the 80th percentile of booking date_time; "
            "full ranking over 100 clusters; single run, no tuning"
        ),
        "cut": str(cut),
        "train_bookings": len(train),
        "test_bookings": len(test),
        "test_users_unseen_in_train_share": round(
            float((~test["user_id"].isin(train["user_id"])).mean()), 6
        ),
        "recall_at_1": {
            "global_popularity": recall_at_k(global_pred, 1),
            "popularity_by_srch_destination_id": recall_at_k(
                covered["srch_destination_id"], 1
            ),
            "popularity_by_hotel_market": recall_at_k(covered["hotel_market"], 1),
            "popularity_by_checkin_month": recall_at_k(covered["ci_month"], 1),
            "popularity_by_party": recall_at_k(covered["party"], 1),
            "repeat_last_cluster": recall_at_k(repeat_pred, 1),
            **{k: recall_at_k(v, 1) for k, v in pair_preds.items()},
        },
        "recall_at_5": {
            "global_popularity": recall_at_k(global_pred, K),
            "popularity_by_srch_destination_id": recall_at_k(
                covered["srch_destination_id"], K
            ),
            "popularity_by_hotel_market": recall_at_k(covered["hotel_market"], K),
            "popularity_by_checkin_month": recall_at_k(covered["ci_month"], K),
            "popularity_by_party": recall_at_k(covered["party"], K),
            "repeat_last_cluster": recall_at_k(repeat_pred, K),
            **{k: recall_at_k(v, K) for k, v in pair_preds.items()},
        },
        "coverage_of_context_tables": {
            "srch_destination_id": round(
                float(
                    test["srch_destination_id"]
                    .isin(set(train["srch_destination_id"]))
                    .mean()
                ),
                6,
            ),
            "hotel_market": round(
                float(test["hotel_market"].isin(set(train["hotel_market"])).mean()), 6
            ),
        },
        "repeat_last_available_share": round(
            float(test["user_id"].isin(last_cluster.keys()).mean()), 6
        ),
    }

    out = {
        "source": str(RAW.relative_to(ROOT)),
        "generated_by": "scripts/expedia_context_signal.py",
        "bookings": int(n),
        "global_top5_clusters": [int(c) for c in global_top],
        "global_top1_share": round(float(p_global.max() / p_global.sum()), 6),
        "distribution_shift": shifts,
        "destination_mix_shift_by_checkin_month": dest_month_js,
        "popularity_probe": probe,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"popularity_probe": probe}, indent=2))
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
