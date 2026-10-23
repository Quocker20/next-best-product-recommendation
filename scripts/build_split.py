"""Build the date split and the per-row history arrays shared by every model.

Reads data/interim/expedia_bookings.parquet (cleaned, burst repeats collapsed, time-sorted) and the
query codes data/interim/expedia_query_codes_clean.npz (row i = booking i), applies the dates of
configs/split.yaml and writes

- data/processed/expedia_date_split/rows.npz: the arrays of `nbp.data.split.build_rows` plus
  `event`, `user`, `window`, `dest`, `codes`, `fields`, `dates`, `ts_unix`, `checkin_month`;
- results/split_counts.json: bookings, users, warm / cold rows and the h distribution
  per window (every number in the docs is copied from this file).

Checks (fail loudly): time order, no booking after test_end, codes aligned with the bookings, and
for a random sample of rows the history equals an independent per-user recomputation.

Run with the project .venv: python scripts/build_split.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml

from nbp.data.query_features import QUERY_FIELDS
from nbp.data.split import WINDOW_NAMES, assign_window, build_rows
from nbp.paths import CONFIGS, INTERIM, PROCESSED
from nbp.paths import ROOT as P_ROOT

BOOKINGS = INTERIM / "expedia_bookings.parquet"
CODES = INTERIM / "expedia_query_codes_clean.npz"
OUT = PROCESSED / "expedia_date_split" / "rows.npz"
COUNTS = P_ROOT / "results" / "split_counts.json"
N_CHECK = 10_000
H_BUCKETS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, None, "6+")]


def h_bucket_counts(h: np.ndarray) -> dict:
    out = {}
    for lo, hi, name in H_BUCKETS:
        m = h >= lo if hi is None else (h >= lo) & (h <= hi)
        out[name] = int(m.sum())
    return out


def main() -> None:
    t0 = time.time()
    cfg = yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))
    dates = (cfg["valid_start"], cfg["test_start"], cfg["test_end"])
    max_len = int(cfg["max_history"])

    df = pd.read_parquet(
        BOOKINGS,
        columns=["user_id", "item_id", "timestamp", "ts_unix", "ctx_dest_id", "ctx_checkin_month"],
    )
    assert df["ts_unix"].is_monotonic_increasing, "bookings must be sorted by time"
    z = np.load(CODES)
    codes, fields = z["codes"], [str(f) for f in z["fields"]]
    assert fields == list(QUERY_FIELDS), fields
    assert len(codes) == len(df), (len(codes), len(df))
    dest = codes[:, fields.index("dest")]
    n_dest_diff = int((dest != df["ctx_dest_id"].to_numpy()).sum())
    print("dest code vs ctx_dest_id mismatches:", n_dest_diff)

    window = assign_window(df["timestamp"].to_numpy(), *dates)
    if (window < 0).any():
        raise SystemExit(f"{int((window < 0).sum())} bookings at or after test_end {dates[2]}")
    user = pd.factorize(df["user_id"])[0].astype(np.int32)
    rows = build_rows(user, df["item_id"].to_numpy(), dest, window, max_len)

    # independent check on a random sample: history = the user's earlier bookings, last max_len
    rng = np.random.default_rng(0)
    sample = rng.choice(len(df), N_CHECK, replace=False)
    by_user = pd.Series(np.arange(len(df))).groupby(user).apply(np.asarray)
    for i in sample:
        ev = by_user[user[i]]
        prev = ev[ev < i][-max_len:]
        k = len(prev)
        assert rows["seq_len"][i] == k
        assert (rows["h_event"][i, :k] == prev).all() and (rows["h_event"][i, k:] == -1).all()
        assert (rows["seq"][i, :k] == df["item_id"].to_numpy()[prev] + 1).all()
        assert rows["h"][i] == int((window[ev[ev < i]] == window[i]).sum())
    print(f"history check passed on {N_CHECK} random rows")

    month = df["ctx_checkin_month"].astype("Int64").fillna(0).to_numpy().astype(np.int8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        event=np.arange(len(df), dtype=np.int32),
        user=user,
        window=window,
        dest=dest.astype(np.int32),
        ts_unix=df["ts_unix"].to_numpy(),
        checkin_month=month,
        codes=codes,
        fields=np.array(fields),
        dates=np.array(dates),
        **rows,
    )

    counts: dict = {
        "source": str(BOOKINGS.relative_to(P_ROOT)),
        "dates": dict(zip(("valid_start", "test_start", "test_end"), dates)),
        "max_history": max_len,
        "bookings": len(df),
        "users": int(user.max() + 1),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "dest_code_vs_ctx_dest_id_mismatches": n_dest_diff,
        "windows": {},
    }
    for w, name in WINDOW_NAMES.items():
        m = window == w
        h = rows["h"][m]
        counts["windows"][name] = {
            "bookings": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "users": len(np.unique(user[m])),
            "warm_L_ge_1": int((rows["L"][m] >= 1).sum()),
            "cold_L_0": int((rows["L"][m] == 0).sum()),
            "h_ge_1": int((h >= 1).sum()),
            "h_ge_1_share": round(float((h >= 1).mean()), 4),
            "h_buckets": h_bucket_counts(h),
            "max_h": int(h.max()),
            "mean_seq_len": round(float(rows["seq_len"][m].mean()), 3),
        }
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    COUNTS.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(
        f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB) and {COUNTS} in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
