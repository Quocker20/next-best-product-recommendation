"""Consolidated results package for the final report (Expedia, test split).

Covers only the models of the final report:
- original code: SMLP4Rec plain (history only), trained on warm rows only (nb01, 3 epochs) and
  trained on all events (warm + first bookings, 01e, early stopping);
- improved model: SMLP4Rec + query token, and + sameDest (the locked main model), 01d trained on
  all events with early stopping;
- baselines: ItemKNN and logistic regression trained on all events with early stopping (07b).

Nothing is retrained. Inputs (read-only):
- run JSONs in `results/week4_rebuild/` (headline numbers, `source` column names file and key);
- per-row test ranks: `experiments/2026-10-10_*/ranks_test*.npz`, `data/interim/baselines_es_ranks_test.parquet`;
- top-20 lists of the SMLP4Rec models: `data/interim/final_topk_test.npz`
  (`scripts/final_export_topk_smlp4rec.py`, smlp4rec venv, loads the checkpoints only);
- top-10 lists of the baselines: recomputed here from `data/interim/baselines_es_models.joblib`
  (the fitted 07b models), checked against the saved ranks;
- `data/interim/expedia_bookings.parquet` (context columns for the slices; row index = event index).

Every headline number read from a JSON is recomputed from the per-row ranks and asserted equal.

Outputs (`results/week5_final/`): `final_results.json`, `final_benchmark.csv`, `slices_*.csv`,
`beyond_accuracy_*.csv`, `final_tables.md` (the printed markdown tables). Everything is also
printed to stdout.

Limitations: `hotel_cluster` is a proxy for room category / package (anonymous ids); one dataset;
offline only; the early-stopping runs are seed 2022 only.

Usage: .venv/Scripts/python.exe scripts/final_results_summary.py
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import summarize
from nbp.paths import EXPERIMENTS, INTERIM, ROOT

RES = ROOT / "results" / "week4_rebuild"
OUT = ROOT / "results" / "week5_final"
BOOKINGS = INTERIM / "expedia_bookings.parquet"
TOPK_NPZ = INTERIM / "final_topk_test.npz"
BASE_RANKS = INTERIM / "baselines_es_ranks_test.parquet"
BASE_MODELS = INTERIM / "baselines_es_models.joblib"
FINAL_DIR = EXPERIMENTS / "2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es"
PLAIN_ALL_DIR = EXPERIMENTS / "2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es"

F_PLAIN_WARM = "smlprec_plain_run.json"
F_PLAIN_ALL = "smlprec_plain_cold_train_es_run.json"
F_FINAL = "smlprec_query_cold_train_es_run.json"
F_FINAL_COLD = "smlprec_query_cold_L0_cold_train_es.json"
F_BASE = "baselines_es_run.json"
F_BASE_CMP = "baselines_vs_final_es.json"
F_SEEDS = "seeds_summary.json"
F_EDA = "eda_clean.json"
F_SPLIT_EXPORT = "split_export_counts.json"
F_3EP_COLD = "smlprec_query_cold_L0_cold_train.json"
F_LR07 = "basic_baselines.json"
LR07_RANKS = INTERIM / "basic_baselines_ranks_test.parquet"
LR07_KEY = "Logistic regression (C=10.0, destination + context, no history)"

KNN_KEY = "ItemKNN (cosine, K=50, history only; cold: first-booking popularity)"
LR_KEY = "Logistic regression (C=1.0, pass 8, destination + context, no history)"
KS = (5, 10, 20)
METRICS = [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
TOP_BA = 10  # cutoff of the beyond-accuracy metrics
N_CLUSTERS = 100
MAX_HIST = 20
LEAD_BINS = [-1, 0, 2, 6, 13, 29, 59, 89, 179, 364, 10_000]  # as notebook 07b
STAY_CLIP = 14
LR_FEATURES = [
    "ctx_dest_id", "ctx_dest_type", "ctx_checkin_month", "stay_bucket", "lead_bucket",
    "ctx_adults", "ctx_children", "ctx_rooms", "ctx_is_package", "ctx_channel", "ctx_is_mobile",
    "ctx_site_name", "ctx_posa_continent", "ctx_user_country",
]  # fmt: skip
MIN_DEST_ROWS = 20  # destinations with fewer test rows are left out of coverage per destination
SMALL_BUCKET = 1000  # slice buckets below this row count are flagged
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}  # fmt: skip

# model key -> (display name, training set label)
MODELS = {
    "plain_warm": ("SMLP4Rec plain (original code)", "warm rows only (L>=1), 3 epochs fixed, nb01"),
    "plain_all": ("SMLP4Rec plain (original code)", "all events (warm + L=0), early stopping, 01e"),
    "query_token": ("SMLP4Rec + query token", "all events (warm + L=0), early stopping, 01d"),
    "main": ("SMLP4Rec + query token + sameDest (main)", "all events (warm + L=0), early stopping, 01d"),
    "itemknn": ("ItemKNN (K=50)", "all events (warm + L=0), K on valid, 07b"),
    "logreg": ("Logistic regression (C=1.0)", "all events (warm + L=0), early stopping, 07b"),
}  # fmt: skip
SLICE_MODELS = list(MODELS)
MD: list[str] = []  # markdown tables, also written to final_tables.md


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def emit(title: str, df: pd.DataFrame, note: str = "") -> None:
    """Print a table as markdown and keep it for final_tables.md."""
    cols = list(df.columns)
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    text = "\n".join(lines)
    print(text)
    MD.append(text)


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative exposure vector (0 = equal exposure)."""
    x = np.sort(np.asarray(counts, np.float64))
    n = len(x)
    if x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


# ----------------------------------------------------------------------------------------------
# 1. per-row ranks and top lists on the 263,331 test-window events
# ----------------------------------------------------------------------------------------------
def load_rows() -> tuple[pd.DataFrame, dict, dict]:
    """Per-row frame (event order) with ranks of every model and the context columns.

    Returns (rows, top lists {model: (n, TOP_BA) int or None}, extra arrays).
    """
    z = np.load(TOPK_NPZ)
    ev = z["event"]
    rows = pd.DataFrame({"event": ev, "n_prior": z["n_prior"], "truth": z["truth"].astype(int)})
    rows["row_set"] = np.where(rows["n_prior"] > 0, "warm", "cold")
    rows["known_dest"] = z["known_dest_last20"]
    for k in ("plain_warm", "plain_all", "query_token", "main"):
        rows[f"rank_{k}"] = z[f"rank_{k}"].astype(np.int64)
    tops = {
        k: z[f"top_{k}"][:, :TOP_BA].astype(np.int64)
        for k in ("plain_warm", "plain_all", "query_token", "main")
    }
    agreement = json.loads(str(z["agreement"]))

    # bookings: row index = event index (sorted by time, user, src_row)
    b = pd.read_parquet(BOOKINGS, columns=[
        "user_id", "item_id", "src_row", "ts_unix", "ctx_dest_id", "ctx_dest_type", "ctx_checkin_month",
        "ctx_stay_nights", "ctx_lead_days", "ctx_adults", "ctx_children", "ctx_rooms", "ctx_party_type",
        "ctx_is_package", "ctx_channel", "ctx_is_mobile", "ctx_site_name", "ctx_posa_continent",
        "ctx_user_country"])  # fmt: skip
    assert (b["item_id"].to_numpy()[ev].astype(int) == rows["truth"].to_numpy()).all(), (
        "event index mismatch"
    )

    # baselines: saved ranks by src_row
    base = pd.read_parquet(BASE_RANKS)
    ev_of_src = pd.Series(np.arange(len(b)), index=b["src_row"].to_numpy())
    base["event"] = ev_of_src.loc[base["src_row"].to_numpy()].to_numpy()
    base = base.set_index("event").loc[ev]
    assert (base["slice"].to_numpy() == rows["row_set"].to_numpy()).all(), "warm / cold rows differ"
    rows["rank_itemknn"] = base[KNN_KEY].to_numpy().astype(np.int64)
    rows["rank_logreg"] = base[LR_KEY].to_numpy().astype(np.int64)
    # sensitivity only: logistic regression of notebook 07 (C=10, 3 SAGA passes, trained on the
    # bookings before the training cut), same test rows
    b07 = pd.read_parquet(LR07_RANKS)
    b07["event"] = ev_of_src.loc[b07["src_row"].to_numpy()].to_numpy()
    b07 = b07.set_index("event").loc[ev]
    assert (b07["slice"].to_numpy() == rows["row_set"].to_numpy()).all()
    rows["rank_logreg_nb07"] = b07[LR07_KEY].to_numpy().astype(np.int64)

    ctx = b.iloc[ev].reset_index(drop=True)
    rows["dest"] = ctx["ctx_dest_id"].to_numpy()
    rows["checkin_month"] = (
        ctx["ctx_checkin_month"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    )
    rows["party_type"] = ctx["ctx_party_type"].astype(str).to_numpy()
    rows["is_package"] = (
        ctx["ctx_is_package"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    )

    tops_b, pop_share, sim = baseline_tops(b, ev, rows)
    tops.update(tops_b)
    extra = {
        "agreement": agreement,
        "pop_share": pop_share,
        "sim": sim,
        "bookings": b,
        "alpha": float(z["alpha"]),
        "ws_new": float(z["ws_new"]),
        "ws_known": float(z["ws_known"]),
    }
    return rows, tops, extra


def baseline_tops(
    b: pd.DataFrame, ev: np.ndarray, rows: pd.DataFrame
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Top-10 lists of ItemKNN and logistic regression from the fitted 07b models.

    Same training set, features, history and tie rule (ties -> more popular training cluster) as
    notebook 07b; the recomputed ranks are asserted equal to the saved ones.
    Returns ({model: (n, 10)}, training popularity share (100,), cluster cosine similarity (100, 100)).
    """
    fin = load(F_FINAL)
    last_train = fin["splits"]["train"]["last_target_event"]
    item = b["item_id"].to_numpy().astype(np.int64)
    user = b["user_id"].to_numpy()
    pos = b.groupby("user_id", sort=False).cumcount().to_numpy()
    train_ev = np.arange(last_train + 1)  # every booking up to the last train target (07b, 01d)
    assert len(train_ev) == fin["splits"]["train"]["targets"] + fin["cold_training_rows"]
    pop = np.bincount(item[train_ev], minlength=N_CLUSTERS).astype(np.float64)
    pop_share = pop / pop.sum()
    first_tr = train_ev[pos[train_ev] == 0]
    first_tr = first_tr[
        first_tr < last_train
    ]  # 07b: first bookings strictly before the last train target
    assert len(first_tr) == fin["cold_training_rows"]
    pop_first = np.bincount(item[first_tr], minlength=N_CLUSTERS).astype(np.float64)
    order = np.argsort(-pop, kind="stable")
    inv = np.empty(N_CLUSTERS, np.int64)
    inv[order] = np.arange(N_CLUSTERS)

    def rank_top(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = scores[:, order]
        st = s[np.arange(len(y)), inv[y]][:, None]
        rank = (
            1
            + (s > st).sum(1)
            + ((s == st) & (np.arange(N_CLUSTERS)[None] < inv[y][:, None])).sum(1)
        )
        top = order[np.argsort(-s, axis=1, kind="stable")[:, :TOP_BA]]
        return rank, top

    m = joblib.load(BASE_MODELS)
    warm = rows["row_set"].to_numpy() == "warm"
    y = rows["truth"].to_numpy()
    n = len(ev)
    top_knn = np.empty((n, TOP_BA), np.int64)
    top_lr = np.empty((n, TOP_BA), np.int64)
    rk_knn = np.empty(n, np.int64)
    rk_lr = np.empty(n, np.int64)

    # ItemKNN: counts of the last 20 bookings before the target @ cosine similarity; cold = first-booking popularity
    by_user = np.lexsort((np.arange(len(b)), user))
    u_sorted = user[by_user]
    starts = np.r_[0, np.flatnonzero(u_sorted[1:] != u_sorted[:-1]) + 1]
    start_of = np.empty(len(b), np.int64)
    start_of[by_user] = np.repeat(starts, np.diff(np.r_[starts, len(b)]))
    ew = ev[warm]
    p = pos[ew]
    s0 = start_of[ew] + np.maximum(0, p - MAX_HIST)
    ln = np.minimum(p, MAX_HIST)
    idx = np.repeat(s0, ln) + (np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln))
    r_i = np.repeat(np.arange(len(ew)), ln)
    hist = sp.csr_matrix(
        (np.ones(len(idx), np.float32), (r_i, item[by_user[idx]])), shape=(len(ew), N_CLUSTERS)
    )
    rk, tp = rank_top((hist @ m["knn_similarity"]).toarray(), y[warm])
    rk_knn[warm], top_knn[warm] = rk, tp
    rk, tp = rank_top(np.tile(pop_first / pop_first.sum(), (int((~warm).sum()), 1)), y[~warm])
    rk_knn[~warm], top_knn[~warm] = rk, tp

    # logistic regression: one-hot query features (07b lr_frame)
    d = b.iloc[ev].reset_index(drop=True)
    f = pd.DataFrame(index=d.index)
    f["stay_bucket"] = d["ctx_stay_nights"].clip(0, STAY_CLIP)
    f["lead_bucket"] = pd.cut(d["ctx_lead_days"], LEAD_BINS, labels=False)
    for c in LR_FEATURES:
        if c not in f:
            f[c] = d[c]
    x = m["lr_encoder"].transform(
        f[LR_FEATURES].astype("Float64").fillna(-1).astype("int64").astype(str)
    )
    assert list(m["lr"].classes_) == list(range(N_CLUSTERS))
    rk_lr, top_lr = rank_top(m["lr"].predict_proba(x), y)

    for name, rk_new, col in (
        ("ItemKNN", rk_knn, "rank_itemknn"),
        ("logistic regression", rk_lr, "rank_logreg"),
    ):
        agree = float((rk_new == rows[col].to_numpy()).mean())
        print(f"recomputed {name} ranks equal the saved 07b ranks on {agree:.6f} of rows")
        assert agree > 0.9999, name

    # cluster similarity for intra-list diversity: cosine of cluster columns of the training
    # user x cluster booking-count matrix (the matrix ItemKNN is fitted on, all neighbours kept)
    uc = pd.factorize(user[train_ev])[0]
    ui = sp.csr_matrix(
        (np.ones(len(train_ev)), (uc, item[train_ev])), shape=(uc.max() + 1, N_CLUSTERS)
    )
    g = (ui.T @ ui).toarray()
    nrm = np.sqrt(np.diag(g))
    sim = g / np.outer(nrm, nrm)
    return {"itemknn": top_knn, "logreg": top_lr}, pop_share, sim


# ----------------------------------------------------------------------------------------------
# 2. headline table (JSON numbers, each re-derived from the per-row ranks)
# ----------------------------------------------------------------------------------------------
def headline(rows: pd.DataFrame) -> pd.DataFrame:
    pw, pa, fin, fc, bs = (
        load(F_PLAIN_WARM),
        load(F_PLAIN_ALL),
        load(F_FINAL),
        load(F_FINAL_COLD),
        load(F_BASE),
    )
    bt = fin["samedest_hybrid"]["benchmark_test"]
    cold_pop = np.load(FINAL_DIR / "ranks_test_cold.npz")
    w = rows["row_set"].to_numpy() == "warm"
    c = ~w
    # cold-row popularity fallback used by 01d for the warm-only plain model, aligned by event
    pos_ev = pd.Series(np.arange(len(rows)), index=rows["event"].to_numpy())
    pop_c = np.empty(len(rows), np.int64)
    pop_c[pos_ev.loc[cold_pop["row"]].to_numpy()] = cold_pop["global_popularity"]
    plain_warm_fb = np.where(w, rows["rank_plain_warm"].to_numpy(), pop_c)

    R = {k: rows[f"rank_{k}"].to_numpy() for k in SLICE_MODELS}
    recs = []

    def add(model, row_set, js, src, ranks, scored_by):
        calc = summarize(ranks, ks=KS)
        if js is not None:
            for mtr in ["n"] + METRICS:
                assert abs(js[mtr] - calc[mtr]) < 1e-9, (model, row_set, mtr, js[mtr], calc[mtr])
        val = js if js is not None else calc
        name, train = MODELS[model]
        recs.append(
            {
                "model": name,
                "model_key": model,
                "training_set": train,
                "seed": "2022",
                "row_set": row_set,
                "scored_by": scored_by,
                **{k: val[k] for k in ["n"] + METRICS},
                "source": src,
            }
        )

    B = "results.test"
    add(
        "plain_warm",
        "all",
        bt["all_events"]["SMLP4Rec plain"],
        f"{F_FINAL}: samedest_hybrid.benchmark_test.all_events['SMLP4Rec plain']",
        plain_warm_fb,
        "model on warm rows + global-popularity fallback on cold rows",
    )
    add(
        "plain_warm",
        "cold",
        fc["results"]["test"]["global popularity"],
        f"{F_FINAL_COLD}: results.test['global popularity']",
        pop_c[c],
        "NOT SCORED by the model (never trained on an empty history); global-popularity fallback",
    )
    add("plain_warm", "warm", pw["test"], f"{F_PLAIN_WARM}: test", R["plain_warm"][w], "model")
    for rs, key, m_ in (
        ("all", "all events", None),
        ("cold", "cold (L=0)", c),
        ("warm", "warm", w),
    ):
        mask = np.ones(len(rows), bool) if m_ is None else m_
        add(
            "plain_all",
            rs,
            pa["test"][key],
            f"{F_PLAIN_ALL}: test['{key}']",
            R["plain_all"][mask],
            "model (cold rows: one constant ranking, empty history)" if rs == "cold" else "model",
        )
    add(
        "query_token",
        "all",
        bt["all_events"]["Query token"],
        f"{F_FINAL}: samedest_hybrid.benchmark_test.all_events['Query token']",
        R["query_token"],
        "model",
    )
    add(
        "query_token",
        "cold",
        fc["results"]["test"]["query token alone (empty history)"],
        f"{F_FINAL_COLD}: results.test['query token alone (empty history)']",
        R["query_token"][c],
        "model",
    )
    add(
        "query_token",
        "warm",
        bt["warm"]["Query token"],
        f"{F_FINAL}: samedest_hybrid.benchmark_test.warm['Query token']",
        R["query_token"][w],
        "model",
    )
    add(
        "main",
        "all",
        bt["all_events"]["Query token + sameDest"],
        f"{F_FINAL}: samedest_hybrid.benchmark_test.all_events['Query token + sameDest']",
        R["main"],
        "model",
    )
    add(
        "main",
        "cold",
        fc["results"]["test"]["query token alone (empty history)"],
        f"{F_FINAL_COLD}: results.test['query token alone (empty history)']",
        R["main"][c],
        "identical to query token: no history, sameDest has no input (weight 0 for a new destination)",
    )
    add(
        "main",
        "warm",
        bt["warm"]["Query token + sameDest"],
        f"{F_FINAL}: samedest_hybrid.benchmark_test.warm['Query token + sameDest']",
        R["main"][w],
        "model",
    )
    for model, key in (("itemknn", KNN_KEY), ("logreg", LR_KEY)):
        for rs, jkey, m_ in (
            ("all", "all events", None),
            ("cold", "cold (L=0)", c),
            ("warm", "warm", w),
        ):
            mask = np.ones(len(rows), bool) if m_ is None else m_
            note = "model"
            if model == "itemknn" and rs == "cold":
                note = "NOT SCORED by the model (no history); first-booking popularity fallback"
            elif model == "itemknn" and rs == "all":
                note = "model on warm rows + first-booking popularity on cold rows"
            add(
                model,
                rs,
                bs["results"]["test"][key][jkey],
                f"{F_BASE}: {B}['{key}']['{jkey}']",
                R[model][mask],
                note,
            )
    return pd.DataFrame(recs)


def lr_sensitivity(rows: pd.DataFrame) -> pd.DataFrame:
    """Notebook 07 logistic regression (C=10, 3 passes) on the same rows, and main minus it."""
    js = load(F_LR07)["results"]["test"][LR07_KEY]
    w = rows["row_set"].to_numpy() == "warm"
    recs = []
    for rs, key, m in (
        ("all", "all events", np.ones(len(rows), bool)),
        ("warm", "warm (all L>=1 targets)", w),
        ("cold", None, ~w),
    ):
        r = rows["rank_logreg_nb07"].to_numpy()[m]
        calc = summarize(r, ks=KS)
        src = "recomputed from basic_baselines_ranks_test.parquet"
        if key is not None and key in js:
            for mtr in ["n"] + METRICS:
                assert abs(js[key][mtr] - calc[mtr]) < 1e-9, (rs, mtr)
            src = f"{F_LR07}: results.test['{LR07_KEY}']['{key}']"
        bo = paired_bootstrap(rows["rank_main"].to_numpy()[m], r, k=5)
        recs.append(
            {
                "row_set": rs,
                **calc,
                "main_minus_lr07_recall@5": bo["recall@5"]["diff"],
                "ci95_recall@5": str(bo["recall@5"]["ci95"]),
                "main_minus_lr07_ndcg@5": bo["ndcg@5"]["diff"],
                "ci95_ndcg@5": str(bo["ndcg@5"]["ci95"]),
                "source": src,
            }
        )
    return pd.DataFrame(recs)


def seeds_rows() -> pd.DataFrame:
    """Mean +- std over seeds 2022/2023/2024 of the 3-epoch 01d runs (no seed runs exist for ES)."""
    s = load(F_SEEDS)["per_variant"]["01d"]
    recs = []
    for key, name, model in (
        ("qt", "SMLP4Rec + query token", "query_token"),
        ("qt_sameDest", "SMLP4Rec + query token + sameDest (main)", "main"),
    ):
        for rs, rs_key in (("all", "all_events"), ("cold", "cold"), ("warm", "warm")):
            if rs == "cold":
                key_ = "qt"  # cold: sameDest has no input, the main model equals the query token
            else:
                key_ = key
            r = {
                "model": name,
                "model_key": model,
                "training_set": "all events (warm + L=0), 3 epochs fixed, 01d (3-epoch seed runs)",
                "seed": "mean+-std (2022, 2023, 2024)",
                "row_set": rs,
                "scored_by": "model"
                if not (rs == "cold" and model == "main")
                else "identical to query token",
            }
            for mtr in ("recall@5", "ndcg@5"):
                e = s.get(f"{rs_key}/{key_}/{mtr}")
                r[mtr] = e["mean"] if e else np.nan
                r[f"{mtr}_std"] = e["std"] if e else np.nan
                r[f"{mtr}_values"] = " ".join(map(str, e["values"])) if e else ""
            r["source"] = f"{F_SEEDS}: per_variant['01d']['{rs_key}/{key_}/<metric>']"
            recs.append(r)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------------------------
# 3. paired bootstrap
# ----------------------------------------------------------------------------------------------
def bootstraps(rows: pd.DataFrame, extra: dict) -> pd.DataFrame:
    w = rows["row_set"].to_numpy() == "warm"
    sets = {"all": np.ones(len(rows), bool), "cold": ~w, "warm": w}
    comps = {
        "all": [("main", "logreg"), ("main", "itemknn"), ("main", "plain_all"), ("main", "query_token"),
                ("query_token", "plain_all")],
        "cold": [("main", "logreg"), ("main", "itemknn"), ("main", "plain_all")],
        "warm": [("main", "logreg"), ("main", "itemknn"), ("main", "plain_all"), ("main", "plain_warm"),
                 ("main", "query_token"), ("query_token", "plain_all"), ("plain_all", "plain_warm")],
    }  # fmt: skip
    recs = []
    for rs, pairs in comps.items():
        m = sets[rs]
        for a, b_ in pairs:
            for k in (5, 10):
                t = time.time()
                r = paired_bootstrap(
                    rows[f"rank_{a}"].to_numpy()[m], rows[f"rank_{b_}"].to_numpy()[m], k=k
                )
                for mtr, v in r.items():
                    recs.append(
                        {
                            "row_set": rs,
                            "n": int(m.sum()),
                            "a": a,
                            "b": b_,
                            "comparison": f"{MODELS[a][0]} [{a}] minus {MODELS[b_][0]} [{b_}]",
                            "metric": mtr,
                            "diff": v["diff"],
                            "ci95_lo": v["ci95"][0],
                            "ci95_hi": v["ci95"][1],
                            "note": (
                                "cold: main = query token; ItemKNN = first-booking popularity"
                                if rs == "cold"
                                else ""
                            ),
                        }
                    )
                print(f"  bootstrap {rs} {a}-{b_} k={k} ({time.time() - t:.0f}s)")

    # sameDest gain without re-books of the same cluster at the same destination within 1 h / 1 d
    # (definition of scripts/expedia_sensitivity.py; conditions on the label, so a lower bound)
    b = extra["bookings"]
    uid = b["user_id"].to_numpy().astype("int64")
    key = pd.Series(
        uid * 10**9
        + b["ctx_dest_id"].to_numpy().astype("int64") * 128
        + b["item_id"].to_numpy().astype("int64")
    )
    sec = pd.Series(b["ts_unix"].to_numpy().astype("int64"))
    age = (sec - sec.groupby(key, sort=False).shift()).to_numpy("float64")[rows["event"].to_numpy()]
    for name, win in (("within_1h", 3600), ("within_1d", 86400)):
        m = w & ~(age <= win)
        r = paired_bootstrap(
            rows["rank_main"].to_numpy()[m], rows["rank_query_token"].to_numpy()[m], k=5
        )
        for mtr, v in r.items():
            recs.append(
                {
                    "row_set": f"warm excluding re-books {name}",
                    "n": int(m.sum()),
                    "a": "main",
                    "b": "query_token",
                    "comparison": "sameDest gain: main minus query token",
                    "metric": mtr,
                    "diff": v["diff"],
                    "ci95_lo": v["ci95"][0],
                    "ci95_hi": v["ci95"][1],
                    "note": f"rows removed: {int((w & (age <= win)).sum())}",
                }
            )
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------------------------
# 4. slices
# ----------------------------------------------------------------------------------------------
def slice_table(
    rows: pd.DataFrame, slice_name: str, bucket: pd.Series, row_sets=("all", "warm")
) -> pd.DataFrame:
    recs = []
    w = rows["row_set"].to_numpy() == "warm"
    for rs in row_sets:
        base = {"all": np.ones(len(rows), bool), "warm": w, "cold": ~w}[rs]
        for bk in sorted(bucket[base].unique(), key=str):
            m = base & (bucket.to_numpy() == bk)
            n = int(m.sum())
            for model in SLICE_MODELS:
                if model == "plain_warm" and (m & ~w).any():
                    continue  # not scored on cold rows: only in buckets without cold rows
                s = summarize(rows[f"rank_{model}"].to_numpy()[m], ks=KS)
                p = s["recall@5"]
                recs.append(
                    {
                        "slice": slice_name,
                        "bucket": bk,
                        "row_set": rs,
                        "model_key": model,
                        "model": MODELS[model][0],
                        "training_set": MODELS[model][1],
                        **s,
                        "ci95_halfwidth_recall@5": round(1.96 * np.sqrt(p * (1 - p) / n), 4),
                        "small_bucket": n < SMALL_BUCKET,
                    }
                )
    return pd.DataFrame(recs)


def slices(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    L = rows["n_prior"].to_numpy()
    lb = pd.Series(
        np.select(
            [L == 0, L == 1, L <= 4, L <= 9], ["L=0 (cold)", "L=1", "L=2-4", "L=5-9"], "L>=10"
        )
    )
    known = pd.Series(
        np.where(
            rows["row_set"] == "cold",
            "cold (no history)",
            np.where(rows["known_dest"], "known destination", "new destination"),
        )
    )
    mon = rows["checkin_month"]
    month = pd.Series(
        np.where(mon.isna(), "unknown", mon.fillna(0).astype(int).astype(str).str.zfill(2))
    )
    season = pd.Series(
        [SEASONS.get(int(x), "unknown") if not np.isnan(x) else "unknown" for x in mon]
    )
    pkg = pd.Series(
        np.where(
            rows["is_package"].isna(),
            "unknown",
            np.where(rows["is_package"] == 1, "package", "no package"),
        )
    )
    return {
        "history_length": slice_table(rows, "history length L", lb, ("all",)),
        "destination_known": slice_table(
            rows, "query destination in last-20 history", known, ("all",)
        ),
        "checkin_month": slice_table(rows, "check-in month", month),
        "season": slice_table(rows, "check-in season", season),
        "party_type": slice_table(rows, "party type", pd.Series(rows["party_type"])),
        "is_package": slice_table(rows, "is_package", pkg),
    }


# ----------------------------------------------------------------------------------------------
# 5. beyond accuracy (top-10 lists)
# ----------------------------------------------------------------------------------------------
def beyond_accuracy(rows: pd.DataFrame, tops: dict, extra: dict) -> dict[str, pd.DataFrame]:
    pop, S = extra["pop_share"], extra["sim"]
    head = np.argsort(-pop, kind="stable")[:10]
    w = rows["row_set"].to_numpy() == "warm"
    truth = rows["truth"].to_numpy()
    agg, dest_rows, season_rows = [], [], []
    for rs in ("all", "warm", "cold"):
        base = {"all": np.ones(len(rows), bool), "warm": w, "cold": ~w}[rs]
        for model in SLICE_MODELS:
            if model == "plain_warm" and rs != "warm":
                continue
            T = tops[model][base]
            n = len(T)
            dist = 1.0 - S[T[:, :, None], T[:, None, :]]  # (n, 10, 10)
            ild = dist.sum((1, 2)) / (TOP_BA * (TOP_BA - 1))
            arp = pop[T].mean(1)
            expo = np.bincount(T.ravel(), minlength=N_CLUSTERS)
            agg.append(
                {
                    "row_set": rs,
                    "model_key": model,
                    "model": MODELS[model][0],
                    "n": n,
                    f"ILD@{TOP_BA}": round(float(ild.mean()), 4),
                    f"ARP@{TOP_BA}": round(float(arp.mean()), 5),
                    "ARP_target": round(float(pop[truth[base]].mean()), 5),
                    "ARP_ratio": round(float(arp.mean() / pop[truth[base]].mean()), 4),
                    "head10_slot_share": round(float(np.isin(T, head).mean()), 4),
                    "head10_target_share": round(float(np.isin(truth[base], head).mean()), 4),
                    f"catalog_coverage@{TOP_BA}": int((expo > 0).sum()),
                    "exposure_gini": round(gini(expo), 4),
                }
            )
            # coverage per destination (destinations with >= MIN_DEST_ROWS rows in this row set)
            dsub = pd.DataFrame({"dest": rows["dest"].to_numpy()[base], "truth": truth[base]})
            dsub["row"] = np.arange(n)
            cnt = dsub["dest"].value_counts()
            keep = cnt[cnt >= MIN_DEST_ROWS].index
            rec_d, bk_d, cov_d = [], [], []
            for _, g in dsub[dsub["dest"].isin(keep)].groupby("dest"):
                rec = set(np.unique(T[g["row"].to_numpy()]))
                booked = set(g["truth"].unique())
                rec_d.append(len(rec))
                bk_d.append(len(booked))
                cov_d.append(len(booked & rec) / len(booked))
            dest_rows.append(
                {
                    "row_set": rs,
                    "model_key": model,
                    "model": MODELS[model][0],
                    "destinations": len(keep),
                    "rows_in_these_destinations": int(cnt[keep].sum()),
                    f"mean_distinct_recommended@{TOP_BA}": round(float(np.mean(rec_d)), 2),
                    "mean_distinct_booked": round(float(np.mean(bk_d)), 2),
                    f"mean_booked_cluster_coverage@{TOP_BA}": round(float(np.mean(cov_d)), 4),
                }
            )
            # coverage per check-in month
            mon = rows["checkin_month"].to_numpy()[base]
            for mo in sorted(set(mon[~np.isnan(mon)].astype(int))) + ["unknown"]:
                mm = np.isnan(mon) if mo == "unknown" else (mon == mo)
                if mm.sum() == 0:
                    continue
                e = np.bincount(T[mm].ravel(), minlength=N_CLUSTERS)
                season_rows.append(
                    {
                        "row_set": rs,
                        "model_key": model,
                        "model": MODELS[model][0],
                        "checkin_month": str(mo).zfill(2) if mo != "unknown" else mo,
                        "n": int(mm.sum()),
                        f"distinct_recommended@{TOP_BA}": int((e > 0).sum()),
                        "distinct_booked": len(np.unique(truth[base][mm])),
                        "exposure_gini": round(gini(e), 4),
                        "small_bucket": bool(mm.sum() < SMALL_BUCKET),
                    }
                )
    return {
        "aggregate": pd.DataFrame(agg),
        "per_destination": pd.DataFrame(dest_rows),
        "per_checkin_month": pd.DataFrame(season_rows),
    }


# ----------------------------------------------------------------------------------------------
# 6. existing results and consistency checks
# ----------------------------------------------------------------------------------------------
def existing_and_checks(hl: pd.DataFrame) -> dict:
    fin, fc, eda = load(F_FINAL), load(F_FINAL_COLD), load(F_EDA)
    out = {
        "field_permutation_test_warm_main_query_token": {
            "source": f"{F_FINAL}: field_permutation_test",
            "values": fin["field_permutation_test"],
        },
        "diagnostics_test_warm_query_token": {
            "source": f"{F_FINAL}: diagnostics_test",
            "values": fin["diagnostics_test"],
        },
        "cold_destination_in_vocab_query_token": {
            "source": f"{F_FINAL_COLD}: test_slices_ndcg5_recall5[*]['query token alone (empty history)']",
            "values": {
                k: v["query token alone (empty history)"]
                for k, v in fc["test_slices_ndcg5_recall5"].items()
            },
        },
        "eda_checkin_month_share_by_split": {
            "source": f"{F_EDA}: E7.composition[*].checkin_month",
            "values": {k: v.get("checkin_month") for k, v in eda["E7"]["composition"].items()},
        },
        "eda_party_share_by_split": {
            "source": f"{F_EDA}: E7.composition[*].party",
            "values": {k: v.get("party") for k, v in eda["E7"]["composition"].items()},
        },
    }
    checks = {}
    # benchmark_results.csv (stage A table) against the final numbers
    bench = pd.read_csv(ROOT / "reports" / "benchmark_results.csv")
    checks["benchmark_results_csv_has_early_stopping_runs"] = bool(
        bench["source"].str.contains("_es").any()
    )
    pairs = [
        ("Logistic regression", "logreg"),
        ("ItemKNN", "itemknn"),
        ("SMLP4Rec plain (history only)", "plain_warm"),
    ]
    diffs = []
    for bname, key in pairs:
        for rs_b, rs in (("warm", "warm"), ("cold", "cold"), ("all_events", "all")):
            br = bench[(bench["model"] == bname) & (bench["row_set"] == rs_b)]
            fr = hl[(hl["model_key"] == key) & (hl["row_set"] == rs)]
            if len(br) and len(fr):
                diffs.append(
                    {
                        "benchmark_results.csv model": bname,
                        "row_set": rs,
                        "csv recall@5": float(br["recall@5"].iloc[0]),
                        "final recall@5": float(fr["recall@5"].iloc[0]),
                        "csv source": br["source"].iloc[0],
                        "final source": fr["source"].iloc[0],
                    }
                )
    checks["benchmark_results_csv_vs_final"] = diffs
    # split_export_counts.json (notebook 08 top-10 export) against the ES and the 3-epoch final model
    se, c3 = load(F_SPLIT_EXPORT), load(F_3EP_COLD)
    checks["split_export_counts_test_recall@10"] = {
        "split_export_counts warm": round(se["test_recall_at_10_warm"], 4),
        "split_export_counts cold": round(se["test_recall_at_10_cold"], 4),
        "ES main model warm": float(
            hl[(hl.model_key == "main") & (hl.row_set == "warm")]["recall@10"].iloc[0]
        ),
        "ES main model cold": float(
            hl[(hl.model_key == "main") & (hl.row_set == "cold")]["recall@10"].iloc[0]
        ),
        "3-epoch 01d cold (smlprec_query_cold_L0_cold_train.json)": c3["results"]["test"][
            "query token alone (empty history)"
        ]["recall@10"],
    }
    # baselines_vs_final_es.json must equal the headline
    cmp = load(F_BASE_CMP)["models"]
    names = {
        "main": "SMLP4Rec + query token + sameDest (final, best epoch 8)",
        "query_token": "SMLP4Rec + query token (final, best epoch 8)",
        "plain_all": "SMLP4Rec plain, no query token (01e, best epoch 2)",
        "itemknn": KNN_KEY,
        "logreg": LR_KEY,
    }
    mism = []
    for key, nm in names.items():
        for rs, jk in (("all", "all events"), ("cold", "cold (L=0)"), ("warm", "warm")):
            fr = hl[(hl.model_key == key) & (hl.row_set == rs)].iloc[0]
            for mtr in METRICS:
                if abs(cmp[nm][jk][mtr] - fr[mtr]) > 1e-9:
                    mism.append((key, rs, mtr))
    checks["baselines_vs_final_es_json_mismatches"] = mism
    return {"existing": out, "checks": checks}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, tops, extra = load_rows()
    print(
        "\nrank agreement of final_topk_test.npz with the notebooks' saved ranks:",
        json.dumps(extra["agreement"], indent=1),
    )
    print(
        f"rows: all {len(rows):,} | warm {(rows.row_set == 'warm').sum():,} | cold {(rows.row_set == 'cold').sum():,} "
        f"| known destination (warm) {int(rows.known_dest.sum()):,}"
    )

    hl = headline(rows)
    sd = seeds_rows()
    bench = pd.concat([hl, sd], ignore_index=True)
    bench.insert(0, "dataset", "expedia (collapsed bookings, test split, 100 clusters)")
    bench["n"] = bench["n"].astype("Int64")  # seed rows carry no row count
    bench.to_csv(OUT / "final_benchmark.csv", index=False)

    show = hl.copy()
    show["model (training set)"] = show["model"] + " [" + show["training_set"] + "]"
    for rs in ("all", "cold", "warm"):
        t = show[show.row_set == rs][["model (training set)", "n"] + METRICS + ["scored_by"]]
        emit(f"Headline, test, row set = {rs} (Expedia, seed 2022)", t.reset_index(drop=True))
    t = sd[
        [
            "model",
            "row_set",
            "recall@5",
            "recall@5_std",
            "ndcg@5",
            "ndcg@5_std",
            "recall@5_values",
            "ndcg@5_values",
        ]
    ]
    emit(
        "Seed variation: 01d 3-epoch runs, seeds 2022/2023/2024 (mean, sample std)",
        t,
        "The early-stopping runs in the headline are seed 2022 only; no ES seed runs exist.",
    )

    print("\nbootstrap (1000 paired resamples over test rows, nbp.eval.bootstrap) ...")
    bo = bootstraps(rows, extra)
    bo.to_csv(OUT / "bootstrap_ci.csv", index=False)
    bt = bo.copy()
    bt["ci95"] = (
        "[" + bt["ci95_lo"].map("{:+.4f}".format) + ", " + bt["ci95_hi"].map("{:+.4f}".format) + "]"
    )
    bt["diff"] = bt["diff"].map("{:+.4f}".format)
    emit(
        "Paired bootstrap, test (difference a minus b, 95% CI)",
        bt[["row_set", "n", "comparison", "metric", "diff", "ci95", "note"]],
    )

    lrs = lr_sensitivity(rows)
    lrs.to_csv(OUT / "sensitivity_logreg_nb07.csv", index=False)
    emit(
        "Sensitivity: logistic regression of notebook 07 (C=10, 3 SAGA passes) on the same test rows",
        lrs,
        "Not the like-for-like baseline (different C / pass selection); shown because it is stronger than the 07b run.",
    )

    sl = slices(rows)
    for name, df in sl.items():
        df.to_csv(OUT / f"slices_{name}.csv", index=False)
        piv = df.pivot_table(
            index=["row_set", "bucket"], columns="model_key", values="ndcg@5", aggfunc="first"
        )
        piv = piv[[m for m in SLICE_MODELS if m in piv.columns]]
        nn = df.groupby(["row_set", "bucket"])[["n", "small_bucket"]].first()
        rc = df.pivot_table(
            index=["row_set", "bucket"], columns="model_key", values="recall@5", aggfunc="first"
        )
        tab = pd.concat([nn, piv.add_prefix("NDCG@5 ")], axis=1).reset_index()
        tab2 = pd.concat(
            [nn, rc[[m for m in SLICE_MODELS if m in rc.columns]].add_prefix("R@5 ")], axis=1
        ).reset_index()
        emit(f"Slice: {df['slice'].iloc[0]} - NDCG@5", tab)
        emit(f"Slice: {df['slice'].iloc[0]} - Recall@5", tab2)

    ba = beyond_accuracy(rows, tops, extra)
    for name, df in ba.items():
        df.to_csv(OUT / f"beyond_accuracy_{name}.csv", index=False)
    emit("Beyond accuracy, top-10, aggregate", ba["aggregate"].drop(columns=["model"]))
    emit(
        f"Coverage per destination (destinations with >= {MIN_DEST_ROWS} rows in the row set)",
        ba["per_destination"].drop(columns=["model"]),
    )
    pm = ba["per_checkin_month"]
    piv = pm.pivot_table(
        index=["row_set", "checkin_month"],
        columns="model_key",
        values="exposure_gini",
        aggfunc="first",
    )
    nn = pm.groupby(["row_set", "checkin_month"])[["n", "distinct_booked", "small_bucket"]].first()
    dr = pm.pivot_table(
        index=["row_set", "checkin_month"],
        columns="model_key",
        values=f"distinct_recommended@{TOP_BA}",
        aggfunc="first",
    )
    dr = dr.astype("Int64").astype(str).replace("<NA>", "")
    emit(
        "Coverage per check-in month: distinct clusters recommended in top-10",
        pd.concat([nn, dr[[m for m in SLICE_MODELS if m in dr.columns]]], axis=1).reset_index(),
    )
    emit(
        "Coverage per check-in month: exposure Gini of top-10 slots",
        pd.concat([nn, piv[[m for m in SLICE_MODELS if m in piv.columns]]], axis=1).reset_index(),
    )

    ex = existing_and_checks(hl)
    fp = ex["existing"]["field_permutation_test_warm_main_query_token"]["values"]
    emit(
        "Existing: per-field permutation on the query token, test warm (01d ES)",
        pd.DataFrame([{"field": k, **v} for k, v in fp.items()]),
    )
    dg = ex["existing"]["diagnostics_test_warm_query_token"]["values"]
    emit(
        "Existing: query / history ablation of the query token, test warm (01d ES)",
        pd.DataFrame([{"input": k, **v} for k, v in dg.items()]),
    )
    cv = ex["existing"]["cold_destination_in_vocab_query_token"]["values"]
    emit(
        "Existing: cold rows by destination vocabulary, query token (01d ES)",
        pd.DataFrame([{"slice": k, **v} for k, v in cv.items()]),
    )
    cm = ex["existing"]["eda_checkin_month_share_by_split"]["values"]
    emit(
        "Existing (EDA): check-in month share by split", pd.DataFrame(cm).reset_index(names="month")
    )
    emit(
        "Check: reports/benchmark_results.csv vs final numbers",
        pd.DataFrame(ex["checks"]["benchmark_results_csv_vs_final"]),
    )
    print(
        "checks:",
        json.dumps(
            {k: v for k, v in ex["checks"].items() if k != "benchmark_results_csv_vs_final"},
            indent=1,
        ),
    )

    result = {
        "dataset": "Expedia Hotel Recommendations, bookings, burst repeats collapsed; item = hotel_cluster (100, proxy)",
        "split": "global temporal 80/10/10 over next-booking targets; test = 213,869 warm + 49,462 cold rows",
        "generated_by": "scripts/final_results_summary.py",
        "inputs": {
            "json": [
                F_LR07,
                F_PLAIN_WARM,
                F_PLAIN_ALL,
                F_FINAL,
                F_FINAL_COLD,
                F_BASE,
                F_BASE_CMP,
                F_SEEDS,
                F_EDA,
                F_SPLIT_EXPORT,
                F_3EP_COLD,
            ],
            "per_row": [
                str(p.relative_to(ROOT))
                for p in (
                    TOPK_NPZ,
                    BASE_RANKS,
                    LR07_RANKS,
                    BASE_MODELS,
                    FINAL_DIR / "ranks_test_warm.npz",
                    FINAL_DIR / "ranks_test_cold.npz",
                    PLAIN_ALL_DIR / "ranks_test.npz",
                )
            ],
        },
        "models": {k: {"name": v[0], "training_set": v[1]} for k, v in MODELS.items()},
        "samedest_weights": {
            "alpha": extra["alpha"],
            "w_s_new_destination": extra["ws_new"],
            "w_s_known_destination": extra["ws_known"],
            "source": f"{F_FINAL}: samedest_hybrid.tuned_on_valid.A",
        },
        "row_counts": {
            "all": len(rows),
            "warm": int((rows.row_set == "warm").sum()),
            "cold": int((rows.row_set == "cold").sum()),
            "warm_known_destination": int(rows.known_dest.sum()),
        },
        "rank_agreement_topk_export": extra["agreement"],
        "headline": hl.to_dict("records"),
        "seeds_3epoch_01d": sd.to_dict("records"),
        "bootstrap": bo.to_dict("records"),
        "sensitivity_logreg_nb07": lrs.to_dict("records"),
        "slices": {k: v.to_dict("records") for k, v in sl.items()},
        "beyond_accuracy": {k: v.to_dict("records") for k, v in ba.items()},
        "beyond_accuracy_definitions": {
            f"ILD@{TOP_BA}": "mean over rows of the mean pairwise (1 - cosine) of the top-10 clusters; cosine of cluster columns of the training user x cluster booking-count matrix",
            f"ARP@{TOP_BA}": "mean training popularity share of the recommended clusters; ARP_ratio = ARP / mean training share of the booked clusters (1 = no bias)",
            "head10_slot_share": "share of top-10 slots taken by the 10 most popular training clusters (head10_target_share = share of booked clusters in that head)",
            "exposure_gini": "Gini of how often each cluster appears in the top-10 lists (0 = equal exposure)",
            "per_destination": f"destinations with >= {MIN_DEST_ROWS} rows; booked-cluster coverage = share of the clusters booked there that appear in at least one top-10 list there",
        },
        "existing_results": ex["existing"],
        "consistency_checks": ex["checks"],
    }
    (OUT / "final_results.json").write_text(
        json.dumps(result, indent=1, default=str), encoding="utf-8"
    )
    (OUT / "final_tables.md").write_text(
        "<!-- generated by scripts/final_results_summary.py; do not edit -->\n\n" + "\n".join(MD),
        encoding="utf-8",
    )
    print(
        f"\nwrote {OUT.relative_to(ROOT)}/ (final_results.json, final_benchmark.csv, bootstrap_ci.csv, slices_*.csv, "
        f"beyond_accuracy_*.csv, final_tables.md) in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
