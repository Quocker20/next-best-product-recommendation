"""`test_with_result.csv`: the split `test.csv` plus each model's fixed per-user top-10 (user-level protocol).

Reads `data/interim/week6/user_top20.npz` (written by `scripts/eval_user_top20.py`). Every booking of
a user carries the same list: the first 10 of that user's reranked top-20, one column per model
(`top10_<model>`, hotel_cluster ids, space-separated, best first), reranker m2_merge. SMLP4Rec columns
use the reported seed (`results/week6_retrain/reported_seed.json`). The row-level columns of the superseded protocol
(`top10_main_ar`, `truth_rank_main_ar`) are dropped.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from nbp.eval import runner as R  # noqa: E402
from nbp.paths import INTERIM, PROCESSED  # noqa: E402

TOPN = 10
CSV = PROCESSED / "expedia_date_split" / "csv"
SEED = json.loads((ROOT / "results" / "week6_retrain" / "reported_seed.json").read_text(encoding="utf-8"))["reported_seed"]
CHOSEN = "m2_merge"
COLS = [(k, "-" if k in ("popularity", "itemknn") else str(SEED), CHOSEN, f"top10_{k}")
        for k in ("popularity", "itemknn", "plain", "query", "main")]

z = np.load(INTERIM / "week6" / "user_top20.npz")
src = pd.read_csv(CSV / "test.csv")
rows = R.load_rows(PROCESSED / "expedia_date_split" / "rows.npz")
it = R.window_index(rows)["test"]
assert np.array_equal(src["event_id"].to_numpy(), rows["event"][it]), "test.csv row order differs from rows.npz"
u = rows["user"][it]  # internal user id (the csv user_id is the raw Expedia id)
pos = np.searchsorted(z["user"], u)
assert (z["user"][np.minimum(pos, len(z["user"]) - 1)] == u).all(), "a test user has no prediction"

out = src.drop(columns=["top10_main_ar", "truth_rank_main_ar"], errors="ignore")
for key, seed, meth, col in COLS:
    top = z[f"{key}_{seed}_{meth}"][:, :TOPN]
    lists = np.array([" ".join(map(str, t)) for t in top.tolist()], dtype=object)
    out[col] = lists[pos]
out.to_csv(CSV / "test_with_result.csv", index=False)

# check: recall@10 per booking row recomputed from the written text (true cluster among the 10 ids)
for *_, col in COLS:
    hit = np.mean([str(h) in s.split() for h, s in zip(out["hotel_cluster"], out[col])])
    print(f"{col}: booking-row hit@10 = {hit:.4f}")
print(out.shape, "->", CSV / "test_with_result.csv", f"(SMLP4Rec seed {SEED})")
