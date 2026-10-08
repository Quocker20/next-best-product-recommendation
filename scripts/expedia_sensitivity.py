"""Sensitivity of the sameDest gain and the query-token advantage, from the saved per-row test ranks.

Read-only. Inputs: the per-row warm test ranks written by notebook 01c
(`experiments/<run>/ranks_test_warm.npz`), the logistic-regression / ItemKNN ranks of notebook 07
(`data/interim/basic_baselines_ranks_test.parquet`) and the collapsed bookings. Output:
`results/week4_rebuild/sensitivity.json`.

Analyses (test, warm rows)
1. sameDest gain (query token + sameDest minus query token) on all rows, and after removing the
   targets that repeat a cluster the user booked at the same destination within 1 hour / 1 day
   (re-booking inside a session, not a long-term preference). Paired bootstrap 95 % CI.
2. Query token vs logistic regression and ItemKNN (paired bootstrap), same rows.

Usage: .venv/Scripts/python.exe scripts/expedia_sensitivity.py [run_dir]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import summarize
from nbp.paths import INTERIM, ROOT

RUNS = sorted((ROOT / "experiments").glob("*_expedia_smlp4rec_query-1c-clean"))
RUN = RUNS[-1] if len(sys.argv) < 2 else ROOT / sys.argv[1]
BASE_RANKS = INTERIM / "basic_baselines_ranks_test.parquet"
CLEAN = INTERIM / "expedia_bookings.parquet"
OUT = ROOT / "results" / "week4_rebuild" / ("sensitivity_cold_train.json" if "1d" in RUN.name else "sensitivity.json")
LR = "Logistic regression (C=10.0, destination + context, no history)"
KNN = "ItemKNN (cosine, K=100, history only)"
WINDOWS = {"within_1h": 3600, "within_1d": 86400}


def main() -> None:
    z = np.load(RUN / "ranks_test_warm.npz")
    row = z["row"]  # event index of each warm test target
    d = pd.read_parquet(CLEAN, columns=["user_id", "item_id", "ctx_dest_id", "ts_unix", "src_row"])
    uid = d["user_id"].to_numpy().astype("int64")
    dest = d["ctx_dest_id"].to_numpy().astype("int64")
    item = d["item_id"].to_numpy().astype("int64")
    sec = d["ts_unix"].to_numpy().astype("int64")
    key = pd.Series(uid * 10**9 + dest * 128 + item)
    age = (pd.Series(sec) - pd.Series(sec).groupby(key, sort=False).shift()).to_numpy("float64")
    age_t = age[row]  # NaN = the user never booked this cluster at this destination before
    ranks = {
        k: z[k]
        for k in ("plain", "plain_prior", "old_hybrid", "query_token", "query_token_sameDest")
    }

    res: dict = {"run": RUN.name, "rows": len(row)}
    masks = {"all_warm": np.ones(len(row), bool)}
    for name, w in WINDOWS.items():
        masks[f"excluding_rebook_{name}"] = ~(age_t <= w)
    res["sameDest_gain"] = {}
    for name, m in masks.items():
        a, b = ranks["query_token_sameDest"][m], ranks["query_token"][m]
        res["sameDest_gain"][name] = {
            "rows": int(m.sum()),
            "query_token": summarize(b, ks=(5,)),
            "query_token_sameDest": summarize(a, ks=(5,)),
            "gain_bootstrap": paired_bootstrap(a, b, k=5),
        }
    res["rows_removed"] = {
        name: int((~masks[f"excluding_rebook_{name}"]).sum()) for name in WINDOWS
    }

    # baselines by source row
    base = pd.read_parquet(BASE_RANKS)
    base = base[base["slice"] == "warm"].set_index("src_row")
    src = d["src_row"].to_numpy()[row]
    assert base.index.is_unique and np.isin(src, base.index).all(), (
        "warm rows differ from notebook 07"
    )
    lr, knn = base.loc[src, LR].to_numpy(), base.loc[src, KNN].to_numpy()
    res["vs_baselines_warm"] = {
        "query_token minus logistic_regression": paired_bootstrap(ranks["query_token"], lr, k=5),
        "query_token_sameDest minus logistic_regression": paired_bootstrap(
            ranks["query_token_sameDest"], lr, k=5
        ),
        "query_token minus ItemKNN": paired_bootstrap(ranks["query_token"], knn, k=5),
        "query_token_sameDest minus ItemKNN": paired_bootstrap(
            ranks["query_token_sameDest"], knn, k=5
        ),
        "logistic_regression": summarize(lr, ks=(5,)),
        "ItemKNN": summarize(knn, ks=(5,)),
    }
    # cold (L = 0) rows: ranks written by scripts/query_token_cold_users.py
    cold_file = RUN / "ranks_test_cold.npz"
    if cold_file.exists():
        zc = np.load(cold_file)
        crow = zc["row"]
        bc = pd.read_parquet(BASE_RANKS)
        bc = bc[bc["slice"] == "cold"].set_index("src_row")
        csrc = d["src_row"].to_numpy()[crow]
        assert bc.index.is_unique and np.isin(csrc, bc.index).all(), (
            "cold rows differ from notebook 07"
        )
        lr_c = bc.loc[csrc, LR].to_numpy()
        qt_c, prior_c, qtp_c = (
            zc["query_token_alone"],
            zc["destination_prior_only"],
            zc["query_token_+_prior"],
        )
        res["cold_rows"] = {
            "rows": len(crow),
            "query_token minus logistic_regression": paired_bootstrap(qt_c, lr_c, k=5),
            "query_token minus destination_prior": paired_bootstrap(qt_c, prior_c, k=5),
            "logistic_regression minus destination_prior": paired_bootstrap(lr_c, prior_c, k=5),
            "query_token_plus_prior minus logistic_regression": paired_bootstrap(qtp_c, lr_c, k=5),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
