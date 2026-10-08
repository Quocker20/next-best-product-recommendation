"""Leakage audit of the query token fields and the sameDest feature on the collapsed bookings.

Read-only. Inputs: data/interim/expedia_bookings.parquet (event index = row position) and
data/interim/expedia_query_codes_clean.npz (query codes, row i = event i). Output:
results/week4_rebuild/leakage_audit.json.

Checks
1. No forbidden raw column (hotel_*, orig_destination_distance, cnt, is_booking, user region/city)
   feeds the query fields.
2. Event order: inside every user the event index and the int64 unix seconds are in the same order;
   count of same-second pairs (their order is decided by the source row).
3. History vs target: for valid / test targets the history is the user's earlier events only
   (max history event index < target event index, target never in its own history).
4. Single-field strength: for each query field, a majority-cluster-per-value rule fitted on train
   targets and scored on valid targets (Recall@5, full ranking over 100 clusters). Destination is
   expected to be strong; any other field far above the popularity baseline is a leak suspect.
5. Trip overlap: share of valid / test targets whose trip (destination, check-in, check-out) or
   trip plus cluster already appears in the user's history. After the burst collapse the exact
   trip plus cluster rows are the remaining near-duplicates.

Usage: .venv/Scripts/python.exe scripts/expedia_leakage_audit.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nbp.data.query_features import (
    FORBIDDEN_EXACT,
    FORBIDDEN_PREFIX,
    QUERY_FIELDS,
    RAW_COLS,
)
from nbp.paths import INTERIM, ROOT

CLEAN = INTERIM / "expedia_bookings.parquet"
CODES = INTERIM / "expedia_query_codes_clean.npz"
OUT = ROOT / "results" / "week4_rebuild" / "leakage_audit.json"
SPLIT = (0.8, 0.1, 0.1)
MAX_HIST = 20
N_CLUSTERS = 100
MIN_COUNT = 5  # same minimum train count as the destination and country vocabularies


def topk_recall(
    table: np.ndarray, value_idx: np.ndarray, y: np.ndarray, glob: np.ndarray, k: int = 5
):
    """Recall@k of 'rank clusters by count among train targets with the same value' (fallback: global)."""
    n = len(y)
    hit = 0
    for start in range(0, n, 200_000):
        sl = slice(start, min(start + 200_000, n))
        rows = table[value_idx[sl]].astype(np.float64)
        unseen = rows.sum(1) == 0
        rows[unseen] = glob
        top = np.argpartition(-rows, k, axis=1)[:, :k]
        hit += int((top == y[sl][:, None]).any(1).sum())
    return hit / n


def main() -> None:
    d = pd.read_parquet(
        CLEAN,
        columns=["user_id", "item_id", "ts_unix", "ctx_dest_id", "ctx_ci", "ctx_co"],
    )
    n = len(d)
    z = np.load(CODES)
    codes, fields = z["codes"], list(z["fields"])
    assert fields == list(QUERY_FIELDS) and codes.shape[0] == n
    res: dict = {"events": n}

    # 1. forbidden raw columns
    bad = [c for c in RAW_COLS if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIX)]
    res["forbidden_raw_columns_in_query_fields"] = bad

    # 2. event order inside users
    pos = d.groupby("user_id", sort=False).cumcount().to_numpy()
    ts = d["ts_unix"].to_numpy()
    prev = d.groupby("user_id", sort=False)["ts_unix"].shift().to_numpy()
    nonfirst = pos > 0
    res["order"] = {
        "ts_non_decreasing_within_user_violations": int((ts[nonfirst] < prev[nonfirst]).sum()),
        "same_second_pairs_within_user": int((ts[nonfirst] == prev[nonfirst]).sum()),
    }

    # split over next-booking targets, in event order
    tgt = np.flatnonzero(nonfirst)
    n_tr, n_va = int(SPLIT[0] * len(tgt)), int(SPLIT[1] * len(tgt))
    parts = {"train": tgt[:n_tr], "valid": tgt[n_tr : n_tr + n_va], "test": tgt[n_tr + n_va :]}
    res["targets"] = {k: len(v) for k, v in parts.items()}

    # 3. history vs target: previous event of the same user has a smaller index
    prev_event = (
        pd.Series(np.arange(n)).groupby(d["user_id"].to_numpy(), sort=False).shift().to_numpy()
    )
    res["history_before_target"] = {
        sp: {
            "targets_checked": len(idx),
            "violations": int((prev_event[idx] >= idx).sum()),
            "histories_capped_at_20_share": float((pos[idx] > MAX_HIST).mean()),
        }
        for sp, idx in parts.items()
        if sp != "train"
    }

    # 4. single-field strength (majority-cluster-per-value rule, fit on train targets)
    y_all = d["item_id"].to_numpy().astype(np.int64)
    glob = np.bincount(y_all[parts["train"]], minlength=N_CLUSTERS).astype(np.float64)
    valid = parts["valid"]
    base = float((np.argsort(-glob)[:5][None, :] == y_all[valid][:, None]).any(1).mean())
    fld = {"global_popularity_recall@5": base}
    for j, name in enumerate(fields):
        vals, inv = np.unique(codes[:, j], return_inverse=True)
        table = np.zeros((len(vals), N_CLUSTERS), np.int64)
        np.add.at(table, (inv[parts["train"]], y_all[parts["train"]]), 1)
        fld[name] = {
            "distinct_values": len(vals),
            "recall@5_valid": topk_recall(table, inv[valid], y_all[valid], glob),
        }
    res["single_field_strength"] = fld

    # 5. trip overlap with the user's history (earlier events of the same user)
    ci = d["ctx_ci"].astype("int64").to_numpy()
    co = d["ctx_co"].astype("int64").to_numpy()
    dest = d["ctx_dest_id"].to_numpy().astype("int64")
    uid = d["user_id"].to_numpy().astype("int64")
    k_trip = pd.Series(list(zip(uid, dest, ci, co))).astype("object")
    trip_seen = k_trip.groupby(k_trip, sort=False).cumcount().to_numpy() > 0
    k_trip_cl = pd.Series(list(zip(uid, dest, ci, co, y_all))).astype("object")
    trip_cl_seen = k_trip_cl.groupby(k_trip_cl, sort=False).cumcount().to_numpy() > 0
    res["trip_overlap_with_history"] = {
        sp: {
            "same_trip_any_cluster": float(trip_seen[idx].mean()),
            "same_trip_same_cluster": float(trip_cl_seen[idx].mean()),
            "same_trip_same_cluster_n": int(trip_cl_seen[idx].sum()),
        }
        for sp, idx in parts.items()
        if sp != "train"
    }

    # 6. candidate query fields not used so far (plan stage B): sparsity and single-field strength
    extra = pd.read_parquet(CLEAN, columns=["ctx_ci", "ctx_user_region", "ctx_user_city"])
    cand_codes = {
        "checkin_weekday": extra["ctx_ci"].dt.dayofweek.to_numpy().astype("int64"),
        "user_region": extra["ctx_user_region"].to_numpy().astype("int64"),
        "user_city": extra["ctx_user_city"].to_numpy().astype("int64"),
    }
    cand = {}
    for name, v in cand_codes.items():
        vals, inv = np.unique(v, return_inverse=True)
        table = np.zeros((len(vals), N_CLUSTERS), np.int64)
        np.add.at(table, (inv[parts["train"]], y_all[parts["train"]]), 1)
        in_vocab = table.sum(1)[inv[valid]] >= MIN_COUNT
        cand[name] = {
            "distinct_values": len(vals),
            f"valid_targets_with_value_seen_ge_{MIN_COUNT}_in_train": float(in_vocab.mean()),
            "recall@5_valid": topk_recall(table, inv[valid], y_all[valid], glob),
        }
    res["candidate_fields"] = cand

    # 7. where the sameDest signal comes from: age of the latest earlier booking of the same cluster
    # at the same destination, for the valid / test targets whose answer is such a repeat
    sec = d["ts_unix"].to_numpy().astype("int64")
    k_dc = pd.Series(uid * 10**9 + dest * 128 + y_all)
    age = (pd.Series(sec) - pd.Series(sec).groupby(k_dc, sort=False).shift()).to_numpy(
        dtype="float64"
    )
    bins = [-1, 3600, 86400, 7 * 86400, 30 * 86400, 1e12]
    labels = ["<=1h", "<=1d", "<=7d", "<=30d", ">30d"]
    res["sameDest_signal_age"] = {}
    for sp, idx in parts.items():
        if sp == "train":
            continue
        a = age[idx]
        hit = ~np.isnan(a)
        counts = pd.cut(pd.Series(a[hit]), bins, labels=labels).value_counts(sort=False)
        res["sameDest_signal_age"][sp] = {
            "targets": len(idx),
            "targets_with_same_dest_cluster_repeat": int(hit.sum()),
            "share_of_targets": float(hit.mean()),
            **{f"age_{k}": int(v) for k, v in counts.items()},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
