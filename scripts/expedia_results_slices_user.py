"""Slices of the user-level test (fixed top-20 per user, reranker m2_merge, reported seed) for the results slides.

Inputs : data/interim/week6/user_top20.npz (written by scripts/eval_user_top20.py), rows.npz, expedia_bookings.parquet,
         configs/split.yaml (report_seed)
Output : results/week6_retrain/user_top20_slices.json
Unit   : the test user; every attribute is that of the user's first test booking (the one the first prediction step reads).
Slices : user set (all / warm / cold), history length L (earlier bookings, any window), check-in season, party type, package,
         searched destination known / new for the user (warm users only).
Each cell: n users, NDCG@10 and Recall@10 of the five models. Paired bootstrap (users resampled, 1000 draws, seed 0) of the
NDCG@10 differences between neighbouring models on all users, and of the sameDest gain on the known-destination users.

Usage: .venv/Scripts/python.exe scripts/expedia_results_slices_user.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from nbp.eval import runner as R
from nbp.eval.two_step import user_plan
from nbp.eval.user_level import fixed_top_user_metrics
from nbp.paths import CONFIGS, INTERIM, PROCESSED, ROOT

OUT = ROOT / "results" / "week6_retrain" / "user_top20_slices.json"
SEED = str(yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))["report_seed"])
CHOSEN = "m2_merge"
MODELS = ["popularity", "itemknn", "plain", "query", "main"]
L_BUCKETS = [("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10**9)]
SEASONS = {"Th.11-12": (11, 12), "Th.1-3": (1, 2, 3), "Th.4-10": (4, 5, 6, 7, 8, 9, 10)}
PARTIES = ["solo", "couple", "family", "group"]
N_BOOT = 1000

rows = R.load_rows(PROCESSED / "expedia_date_split" / "rows.npz")
v = R.window_views(rows)["test"]
it = R.window_index(rows)["test"]
plan = user_plan(v, rows["user"][it], rows["y"][it].astype(np.int64))
z = np.load(INTERIM / "week6" / "user_top20.npz")
assert np.array_equal(z["user"], plan.user)
first = plan.first
ev1 = v.event[first]  # global event index of each user's first test booking

d = pd.read_parquet(INTERIM / "expedia_bookings.parquet", columns=["ctx_ci", "ctx_adults", "ctx_children", "ctx_is_package"])
d1 = d.iloc[ev1].reset_index(drop=True)
L1 = rows["L"][it][first].astype(int)
ci_month = d1["ctx_ci"].dt.month.to_numpy()
adults = d1["ctx_adults"].astype("float").to_numpy()
children = d1["ctx_children"].fillna(0).astype("float").to_numpy()
party = np.where(children > 0, "family", np.where(adults == 1, "solo", np.where(adults == 2, "couple",
        np.where(np.isnan(adults) | (adults == 0), "unknown", "group"))))
package = d1["ctx_is_package"].to_numpy() == 1
warm = L1 >= 1
slot = np.arange(v.seq.shape[1])[None, :] < v.seq_len[first][:, None]
known = (slot & (v.h_dest[first] == v.dest[first][:, None])).any(1)

per = {}
for m in MODELS:
    key = f"{m}_{'-' if m in ('popularity', 'itemknn') else SEED}_{CHOSEN}"
    top = z[key].astype(np.int64) + 1  # hotel_cluster -> model id
    per[m] = fixed_top_user_metrics(top, plan.truths, (10,))["per_user"]
nd = {m: per[m]["ndcg@10"] for m in MODELS}
rc = {m: per[m]["recall@10"] for m in MODELS}


def cell(mask: np.ndarray) -> dict:
    return {"n": int(mask.sum()), **{m: {"ndcg@10": float(nd[m][mask].mean()), "recall@10": float(rc[m][mask].mean())} for m in MODELS}}


def boot(a: str, b: str, mask: np.ndarray) -> dict:
    diff = (nd[a] - nd[b])[mask]
    rng = np.random.default_rng(0)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return {"diff": float(diff.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


allm = np.ones(len(first), bool)
out = {
    "seed": SEED, "reranker": CHOSEN, "unit": "test user, attributes of the first test booking",
    "row_set": {"all": cell(allm), "warm": cell(warm), "cold": cell(~warm)},
    "history_length": {k: cell((L1 >= lo) & (L1 <= hi)) for k, lo, hi in L_BUCKETS},
    "season": {k: cell(np.isin(ci_month, ms)) for k, ms in SEASONS.items()},
    "party": {k: cell(party == k) for k in PARTIES},
    "package": {"package": cell(package), "no_package": cell(~package)},
    "destination_warm": {"known": cell(warm & known), "new": cell(warm & ~known)},
    "known_share_of_warm": float((warm & known).sum() / warm.sum()),
    "bootstrap_ndcg@10": {
        "query_minus_plain": boot("query", "plain", allm), "main_minus_query": boot("main", "query", allm),
        "plain_minus_itemknn": boot("plain", "itemknn", allm), "main_minus_popularity": boot("main", "popularity", allm),
        "main_minus_itemknn": boot("main", "itemknn", allm),
        "main_minus_query_known_destination": boot("main", "query", warm & known),
    },
}
OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", OUT)
for k in ("row_set", "history_length", "season", "party", "package", "destination_warm"):
    for g, c in out[k].items():
        print(k, g, c["n"], " ".join(f"{m} {c[m]['ndcg@10']:.4f}" for m in MODELS))
print(json.dumps(out["bootstrap_ndcg@10"], indent=1))
