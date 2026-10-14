"""Test-window slices of the five models (reported seed, autoregressive protocol) for the report and the slides.

Inputs : data/interim/week6/ranks_*.npz (ranks of the true cluster per test row), data/processed/expedia_date_split/rows.npz,
         data/interim/expedia_bookings.parquet, configs/split.yaml (report_seed)
Output : results/week6_retrain/results_slices.json
Slices : history length L (earlier bookings of the user, any window), check-in season, party type, package,
         searched destination known / new for the user (warm rows only), row sets (all / warm / cold).
Each cell: n, NDCG@10, Recall@10 of every model. Paired bootstrap (users resampled) of main minus query embedding
on the known-destination rows.

Usage: .venv/Scripts/python.exe scripts/expedia_results_slices.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import ndcg_at_k, recall_at_k
from nbp.paths import CONFIGS, INTERIM, PROCESSED, ROOT

OUT = ROOT / "results" / "week6_retrain" / "results_slices.json"
SEED = int(yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))["report_seed"])
R = INTERIM / "week6"
FILES = {
    "itemknn": "ranks_itemknn.npz",
    "popularity": "ranks_popularity.npz",
    "plain": f"ranks_plain_s{SEED}.npz",
    "query": f"ranks_query_s{SEED}.npz",
    "main": f"ranks_main_s{SEED}.npz",
}
L_BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10**9)]
SEASONS = {"Th.11-12": (11, 12), "Th.1-3": (1, 2, 3), "Th.4-10": (4, 5, 6, 7, 8, 9, 10)}
PARTIES = ["solo", "couple", "family", "group"]


def cell(ranks: dict, mask: np.ndarray) -> dict:
    return {
        m: {"ndcg@10": float(ndcg_at_k(r[mask], 10)), "recall@10": float(recall_at_k(r[mask], 10))} for m, r in ranks.items()
    } | {"n": int(mask.sum())}


def main() -> None:
    z = np.load(PROCESSED / "expedia_date_split" / "rows.npz")
    bk = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=["ctx_party_type", "ctx_is_package", "ctx_dest_id", "user_id"])
    ranks, ev = {}, None
    for m, f in FILES.items():
        zz = np.load(R / f)
        ranks[m] = zz["test_ar"].astype(np.int64)
        ev = zz["event_test"] if ev is None else ev
        assert np.array_equal(ev, zz["event_test"]), m
    L = z["L"][ev].astype(np.int64)
    n_all = len(bk)
    idx = np.arange(n_all)
    uid = z["user"].astype(np.int64)
    dest = bk["ctx_dest_id"].to_numpy().astype(np.int64)
    first = pd.Series(idx).groupby(uid * 10**6 + dest, sort=False).transform("min").to_numpy()
    dest_known = (first < idx)[ev]  # an earlier booking of the same user at the searched destination
    month = z["checkin_month"][ev].astype(int)
    party = bk["ctx_party_type"].astype(str).to_numpy()[ev]
    pkg = bk["ctx_is_package"].to_numpy().astype(int)[ev]
    users = z["user"][ev]
    res: dict = {"reported_seed": SEED, "protocol": "autoregressive", "window": "test", "models": list(FILES)}
    res["row_set"] = {"all": cell(ranks, np.ones(len(L), bool)), "warm": cell(ranks, L >= 1), "cold": cell(ranks, L == 0)}
    res["history_length"] = {b: cell(ranks, (L >= lo) & (L <= hi)) for b, lo, hi in L_BUCKETS}
    res["season_checkin"] = {k: cell(ranks, np.isin(month, v)) for k, v in SEASONS.items()}
    res["party"] = {p: cell(ranks, party == p) for p in PARTIES}
    res["package"] = {"package": cell(ranks, pkg == 1), "no_package": cell(ranks, pkg == 0)}
    warm = L >= 1
    res["destination_warm"] = {"known": cell(ranks, warm & dest_known), "new": cell(ranks, warm & ~dest_known)}
    mk = warm & dest_known
    b = paired_bootstrap(ranks["main"][mk], ranks["query"][mk], k=10, n_boot=1000, seed=0, groups=users[mk])
    res["samedest_gain_known_destination"] = {k: {"diff": v["diff"], "ci95": list(v["ci95"])} for k, v in b.items() if k in ("ndcg@10", "recall@10")}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k in ("row_set", "history_length", "season_checkin", "party", "package", "destination_warm"):
        print(k)
        for name, c in res[k].items():
            print(f"  {name:10s} n={c['n']:>7,} " + " ".join(f"{m}={c[m]['ndcg@10']:.4f}" for m in FILES))
    print("samedest gain, known destination:", res["samedest_gain_known_destination"])


if __name__ == "__main__":
    main()
