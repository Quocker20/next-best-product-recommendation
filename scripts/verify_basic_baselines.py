"""Independent check of notebook 07 (basic baselines: ItemKNN, logistic regression).

Shares no code with the notebook: it rebuilds the split, the histories, the ItemKNN similarity and
the logistic-regression features with its own code, re-scores the rows, and compares with what the
notebook saved. Each model is checked on its own.

  1. split        warm / cold eval rows rebuilt from data/interim/expedia_bookings.parquet equal the
                  rows in the notebook's per-row rank files; training data ends before the first eval row.
  2. metrics      every Recall@K / NDCG@K in results/week4_rebuild/basic_baselines.json recomputed from
                  the saved per-row ranks.
  3. itemknn      cosine similarity recomputed with numpy from the training bookings equals the saved
                  (implicit) similarity on its kept entries; test warm rows re-scored from histories
                  rebuilt with pandas; ranks equal the saved ranks.
  4. logreg       saved model re-applied with numpy (softmax of X W^T + b) on features rebuilt here;
                  ranks of all test and valid rows equal the saved ranks; no forbidden feature.
  5. hardcode     no result-like decimal literal in the notebook's code cells.

Output: results/week4_rebuild/basic_baselines_verify.json. Usage: python scripts/verify_basic_baselines.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOKINGS = ROOT / "data" / "interim" / "expedia_bookings.parquet"
NB = ROOT / "notebooks" / "hospitality" / "smlp4rec" / "07_basic_baselines.ipynb"
RES = ROOT / "results" / "week4_rebuild" / "basic_baselines.json"
OUT = ROOT / "results" / "week4_rebuild" / "basic_baselines_verify.json"
INTERIM = ROOT / "data" / "interim"


def rank_of_target(score: np.ndarray, y: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """1-based rank; order = score desc, then popularity desc, then cluster id asc."""
    s_t = score[np.arange(len(y)), y][:, None]
    p_t = pop[y][:, None]
    c = np.arange(score.shape[1])[None, :]
    better = (score > s_t) | ((score == s_t) & ((pop[None, :] > p_t) | ((pop[None, :] == p_t) & (c < y[:, None]))))
    return 1 + better.sum(1)


def metric_table(r: np.ndarray) -> dict:
    out = {"n": int(len(r))}
    for k in (5, 10, 20):
        hit = r <= k
        out[f"recall@{k}"] = round(float(hit.mean()), 4)
        out[f"ndcg@{k}"] = round(float((hit / np.log2(r + 1)).mean()), 4)
    return out


def main() -> None:
    res = json.loads(RES.read_text(encoding="utf-8"))
    mod = joblib.load(INTERIM / "basic_baselines_models.joblib")
    pop, cut = mod["pop"], int(mod["cut"])
    rep: dict = {}

    # 1. split, rebuilt independently
    d = pd.read_parquet(BOOKINGS)
    d["ts"] = d["timestamp"].astype("int64") // 10**9
    d = d.sort_values(["ts", "src_row"], kind="mergesort").reset_index(drop=True)
    d["pos"] = d.groupby("user_id").cumcount()
    targets = d.index[d["pos"] > 0]
    n = len(targets)
    tr, va = targets[: int(0.8 * n)], targets[int(0.8 * n) : int(0.9 * n)]
    te = targets[int(0.9 * n) :]
    t_tr, t_va = d.loc[tr, "ts"].max(), d.loc[va, "ts"].max()
    first = d["pos"] == 0
    sets = {
        ("valid", "warm"): set(d.loc[va, "src_row"]),
        ("test", "warm"): set(d.loc[te, "src_row"]),
        ("valid", "cold"): set(d.loc[first & (d["ts"] > t_tr) & (d["ts"] <= t_va), "src_row"]),
        ("test", "cold"): set(d.loc[first & (d["ts"] > t_va), "src_row"]),
    }
    saved = {s: pd.read_parquet(INTERIM / f"basic_baselines_ranks_{s}.parquet") for s in ("valid", "test")}
    rep["split"] = {
        f"{s}/{k}": {"rebuilt": len(v), "saved": int((saved[s]["slice"] == k).sum()),
                     "same_rows": v == set(saved[s].loc[saved[s]["slice"] == k, "src_row"])}
        for (s, k), v in sets.items()
    }
    rep["split"]["cut_equals_last_train_target_minus_128"] = bool(cut == t_tr - 128)
    rep["split"]["training_ends_before_first_valid_target"] = bool(d.loc[d["ts"] < cut, "ts"].max() < d.loc[va, "ts"].min())
    assert all(v["same_rows"] for k, v in rep["split"].items() if isinstance(v, dict))
    assert rep["split"]["cut_equals_last_train_target_minus_128"] and rep["split"]["training_ends_before_first_valid_target"]

    # 2. metrics from saved ranks
    seen = set(d.loc[tr, "user_id"])
    users = d.set_index("src_row")["user_id"]
    mism = []
    models = [c for c in saved["test"].columns if c not in ("src_row", "slice")]
    for s in ("valid", "test"):
        f = saved[s]
        warm = f[f["slice"] == "warm"]
        cold = f[f["slice"] == "cold"]
        is_seen = users.loc[warm["src_row"]].isin(seen).to_numpy()
        for m in models:
            rw, rc = warm[m].to_numpy().astype(np.int64), cold[m].to_numpy().astype(np.int64)
            mine = {
                "all events": metric_table(np.concatenate([rw, rc])),
                "warm (all L>=1 targets)": metric_table(rw),
                "seen users": metric_table(rw[is_seen]),
                "unseen, L>=1": metric_table(rw[~is_seen]),
                "unseen, L=0 (first booking)": metric_table(rc),
            }
            for sl, v in mine.items():
                if v != res["results"][s][m][sl]:
                    mism.append([s, m, sl, v, res["results"][s][m][sl]])
    rep["metrics_recomputed_from_saved_ranks"] = {"mismatches": mism}
    assert not mism

    # 3. ItemKNN alone
    knn_name = next(m for m in models if m.startswith("ItemKNN"))
    k_pick = res["models"][knn_name]["K_pick"]
    trn = d[d["ts"] < cut]
    ui = pd.crosstab(trn["user_id"], trn["item_id"]).reindex(columns=range(100), fill_value=0).to_numpy(np.float64)
    gram = ui.T @ ui
    norm = np.sqrt(np.diag(gram))
    cos = gram / np.outer(norm, norm)
    sim = mod["knn_similarity"].toarray()
    kept = sim != 0
    rep["itemknn"] = {
        "K_pick": k_pick,
        "max_kept_entries_per_item": int(kept.sum(1).max()),
        "max_abs_diff_on_kept_entries_vs_numpy_cosine": float(np.abs(sim[kept] - cos[kept]).max()),
        "kept_entries_are_the_top_K_of_numpy_cosine": bool(all(
            np.isin(np.flatnonzero(kept[i]), np.argsort(-cos[i], kind="stable")[: kept[i].sum() + 1]).all()
            for i in range(100)
        )),
    }
    assert rep["itemknn"]["max_abs_diff_on_kept_entries_vs_numpy_cosine"] < 1e-5
    assert rep["itemknn"]["max_kept_entries_per_item"] <= k_pick
    te_rows = d.loc[te].sample(5000, random_state=7)
    by_user = d.groupby("user_id")
    hist = np.zeros((len(te_rows), 100))
    for i, (u, p) in enumerate(zip(te_rows["user_id"], te_rows["pos"])):
        g = by_user.get_group(u).sort_values("pos")
        h = g[(g["pos"] < p) & (g["pos"] >= p - 20)]
        assert (h["ts"] <= te_rows["ts"].iloc[i]).all()
        np.add.at(hist[i], h["item_id"].to_numpy(), 1.0)
    r_knn = rank_of_target(hist @ sim, te_rows["item_id"].to_numpy().astype(np.int64), pop)
    sv = saved["test"].set_index("src_row").loc[te_rows["src_row"], knn_name].to_numpy()
    rep["itemknn"]["test_sample_rows"] = len(te_rows)
    rep["itemknn"]["test_sample_rank_mismatches"] = int((r_knn != sv).sum())
    rep["itemknn"]["test_sample_metrics_rescored"] = metric_table(r_knn)
    assert rep["itemknn"]["test_sample_rank_mismatches"] == 0

    # 4. logistic regression alone
    lr_name = next(m for m in models if m.startswith("Logistic"))
    lr, enc = mod["lr"], mod["lr_encoder"]
    feats = list(enc.feature_names_in_)
    bad = [c for c in feats if c.startswith("tgt_") or c in ("ctx_user_city", "ctx_user_region", "user_id", "item_id", "src_row")]
    lead_bins = res["models"][lr_name]["lead_bins"]
    stay_clip = res["models"][lr_name]["stay_clip"]

    def features(rows: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=rows.index)
        for c in feats:
            if c == "stay_bucket":
                v = rows["ctx_stay_nights"].astype("Float64").clip(0, stay_clip)
            elif c == "lead_bucket":
                v = pd.Series(np.searchsorted(lead_bins, rows["ctx_lead_days"].astype("Float64").fillna(-10**9).to_numpy(), side="left") - 1,
                              index=rows.index).astype("Float64")
                v[rows["ctx_lead_days"].isna().to_numpy()] = pd.NA
            else:
                v = rows[c].astype("Float64")
            f[c] = v.fillna(-1).astype("int64").astype(str)
        return f

    lr_rep = {"features": feats, "forbidden_features": bad, "epochs_run": int(np.max(lr.n_iter_))}
    for s, idx in (("valid", va), ("test", te)):
        for k, rows in (("warm", d.loc[idx]), ("cold", d[d["src_row"].isin(sets[(s, "cold")])])):
            x = enc.transform(features(rows))
            z = np.asarray(x @ lr.coef_.T) + lr.intercept_
            prob = np.exp(z - z.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            r = rank_of_target(prob, rows["item_id"].to_numpy().astype(np.int64), pop)
            sv = saved[s].set_index("src_row").loc[rows["src_row"], lr_name].to_numpy()
            lr_rep[f"{s}/{k}"] = {"rows": len(rows), "rank_mismatches": int((r != sv).sum()),
                                  "ndcg@5_rescored": metric_table(r)["ndcg@5"]}
    rep["logreg"] = lr_rep
    assert not bad
    assert all(v["rank_mismatches"] == 0 for k, v in lr_rep.items() if isinstance(v, dict))

    # 5. hardcoded result-like numbers in the notebook's code
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    rep["hardcode_scan"] = {"decimal_literals_with_3plus_digits": sorted(set(re.findall(r"(?<![\w.])0\.\d{3,}", code)))}
    assert not rep["hardcode_scan"]["decimal_literals_with_3plus_digits"]

    OUT.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, default=str))
    print("ALL INDEPENDENT CHECKS PASSED; saved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
