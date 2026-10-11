"""Independent verification of the final results package (`results/week5_final/`).

Written separately from `scripts/final_results_summary.py` (none of its functions are imported).
It recomputes the key numbers from the per-row sources and compares them with the package:

- per-row ranks: the notebooks' own saved ranks (`experiments/2026-10-10_*_query-1d-clean-cold-es/
  ranks_test_{warm,cold}.npz`, `experiments/2026-10-10_*_plain-1e-clean-cold-es/ranks_test.npz`),
  the 07b baseline ranks (`data/interim/baselines_es_ranks_test.parquet`) and the checkpoint
  re-scoring (`data/interim/final_topk_test.npz`);
- row sets: warm / cold recomputed from `data/interim/expedia_bookings.parquet` (prior bookings per
  user, counted here);
- headline Recall / NDCG at 5, 10, 20 for every model x {all, cold, warm} (`nbp.eval.metrics`);
- two paired bootstrap CIs (main vs logistic regression, warm and cold) with a different seed;
- the history-length slice and three beyond-accuracy numbers (coverage, exposure Gini, ARP@10);
- the identity of the early-stopping main run (checkpoint path, best epoch, best.pth = epoch 8);
- labelling rules (training set and cold-row handling) in the CSV and the markdown tables;
- every number of `docs/plans/final_prep/01_results_summary.md` against `final_tables.md`;
- spot checks of numbers quoted in `02_report_toc.md` and `04_demo_options.md`.

Output: `results/week5_final/verification_final.json` (one record per check, PASS / FAIL / NOTE)
and a printed summary. Nothing is retrained; `data/raw/` is not read.

Usage: .venv/Scripts/python.exe scripts/verify_final_results.py
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import ndcg_at_k, recall_at_k
from nbp.paths import EXPERIMENTS, INTERIM, ROOT

W4 = ROOT / "results" / "week4_rebuild"
W5 = ROOT / "results" / "week5_final"
DOCS = ROOT / "docs" / "plans" / "final_prep"
ES_DIR = EXPERIMENTS / "2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es"
PLAIN_ES_DIR = EXPERIMENTS / "2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es"
KNN_COL = "ItemKNN (cosine, K=50, history only; cold: first-booking popularity)"
LR_COL = "Logistic regression (C=1.0, pass 8, destination + context, no history)"
KS = (5, 10, 20)
METRICS = [f"recall@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
EXPECTED_ROWS = {"all": 263_331, "cold": 49_462, "warm": 213_869}
TOL_ROUND = 5.1e-5  # package values are rounded to 4 decimals
BOOT_SEED = 20261011  # different from the package (seed 0)
BOOT_TOL = 0.0015  # allowed CI endpoint gap between two bootstrap seeds (MC error)

CHECKS: list[dict] = []


def check(name: str, ok: bool | None, detail: object = "") -> None:
    """Record one check: ok True -> PASS, False -> FAIL, None -> NOTE (informational)."""
    status = "NOTE" if ok is None else ("PASS" if ok else "FAIL")
    CHECKS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status}] {name}" + (f" | {detail}" if detail not in ("", None) else ""))


def metrics(rank: np.ndarray) -> dict:
    rank = np.asarray(rank, np.float64)
    out = {"n": len(rank)}
    out.update({f"recall@{k}": recall_at_k(rank, k) for k in KS})
    out.update({f"ndcg@{k}": ndcg_at_k(rank, k) for k in KS})
    return out


def gini_lorenz(x: np.ndarray) -> float:
    """Gini as 1 - 2 * area under the Lorenz curve (trapezoids); 0 = equal exposure."""
    x = np.sort(np.asarray(x, np.float64))
    cum = np.r_[0.0, np.cumsum(x) / x.sum()]
    return float(1 - (cum[1:] + cum[:-1]).sum() / len(x))


# ----------------------------------------------------------------------------------------------
def load_rows() -> pd.DataFrame:
    """One row per test-window event with every model's rank, built from the notebooks' files."""
    b = pd.read_parquet(
        INTERIM / "expedia_bookings.parquet", columns=["user_id", "item_id", "src_row"]
    )
    n_prior = b.groupby("user_id", sort=False).cumcount().to_numpy()
    run = json.loads((W4 / "smlprec_query_cold_train_es_run.json").read_text(encoding="utf-8"))
    first_test = run["splits"]["test"]["first_target_event"]
    ev = np.arange(first_test, len(b))
    rows = pd.DataFrame(
        {"event": ev, "n_prior": n_prior[ev], "truth": b["item_id"].to_numpy()[ev].astype(int)}
    )
    rows["warm"] = rows["n_prior"] > 0
    pos = pd.Series(np.arange(len(rows)), index=ev)

    zw, zc = np.load(ES_DIR / "ranks_test_warm.npz"), np.load(ES_DIR / "ranks_test_cold.npz")
    ze = np.load(PLAIN_ES_DIR / "ranks_test.npz")
    iw, ic = pos.loc[zw["row"]].to_numpy(), pos.loc[zc["row"].astype(np.int64)].to_numpy()
    check(
        "01d saved warm / cold row ids = warm / cold events recomputed from bookings",
        bool(rows["warm"].to_numpy()[iw].all() and (~rows["warm"].to_numpy()[ic]).all())
        and len(iw) + len(ic) == len(rows),
        {"warm_rows_file": len(iw), "cold_rows_file": len(ic)},
    )
    r = {k: np.full(len(rows), -1, np.int64) for k in ("main", "query_token", "plain_all", "pw")}
    r["main"][iw], r["main"][ic] = zw["query_token_sameDest"], zc["query_token_alone"]
    r["query_token"][iw], r["query_token"][ic] = zw["query_token"], zc["query_token_alone"]
    r["plain_all"][iw], r["plain_all"][ic] = ze["warm"], ze["cold"]
    r["pw"][iw], r["pw"][ic] = zw["plain"], zc["global_popularity"]
    for k, v in r.items():
        rows[k if k != "pw" else "plain_warm_fb"] = v

    base = pd.read_parquet(INTERIM / "baselines_es_ranks_test.parquet")
    ev_of_src = pd.Series(np.arange(len(b)), index=b["src_row"].to_numpy())
    bev = ev_of_src.loc[base["src_row"].to_numpy()].to_numpy()
    base.index = pos.loc[bev].to_numpy()
    base = base.sort_index()
    check(
        "07b baseline parquet covers exactly the test-window events, same warm / cold split",
        bool((base.index.to_numpy() == np.arange(len(rows))).all())
        and bool(((base["slice"] == "warm").to_numpy() == rows["warm"].to_numpy()).all()),
        {"rows": len(base)},
    )
    rows["itemknn"] = base[KNN_COL].to_numpy().astype(np.int64)
    rows["logreg"] = base[LR_COL].to_numpy().astype(np.int64)

    # checkpoint re-scoring (final_topk_test.npz) agrees with the notebooks' ranks
    z = np.load(INTERIM / "final_topk_test.npz")
    assert (z["event"] == ev).all()
    agree = {
        k: float((z[f"rank_{k}"].astype(np.int64) == rows[k].to_numpy()).mean())
        for k in ("main", "query_token", "plain_all")
    }
    w = rows["warm"].to_numpy()
    agree["plain_warm (warm rows)"] = float(
        (z["rank_plain_warm"].astype(np.int64)[w] == rows["plain_warm_fb"].to_numpy()[w]).mean()
    )
    check(
        "final_topk_test.npz ranks = notebooks' saved ranks (>= 0.9999 of rows)",
        min(agree.values()) >= 0.9999,
        {k: round(v, 6) for k, v in agree.items()},
    )
    check(
        "final_topk_test.npz n_prior = prior bookings recomputed here",
        bool((z["n_prior"] == rows["n_prior"].to_numpy()).all()),
    )
    rows["plain_warm_model"] = np.where(w, z["rank_plain_warm"].astype(np.int64), -1)
    rows.attrs["top_main"] = z["top_main"][:, :10].astype(np.int64)
    rows.attrs["item"] = b["item_id"].to_numpy().astype(np.int64)
    rows.attrs["last_train"] = run["splits"]["train"]["last_target_event"]
    return rows


# ----------------------------------------------------------------------------------------------
def verify_counts(rows: pd.DataFrame) -> None:
    w = rows["warm"].to_numpy()
    got = {"all": len(rows), "cold": int((~w).sum()), "warm": int(w.sum())}
    check("row counts all / cold / warm = 263,331 / 49,462 / 213,869", got == EXPECTED_ROWS, got)
    fr = json.loads((W5 / "final_results.json").read_text(encoding="utf-8"))["row_counts"]
    check(
        "final_results.json row_counts agree",
        all(fr[k] == v for k, v in got.items()),
        {k: fr[k] for k in got},
    )


def verify_headline(rows: pd.DataFrame) -> dict:
    bench = pd.read_csv(W5 / "final_benchmark.csv")
    bench = bench[bench["seed"].astype(str) == "2022"]
    w = rows["warm"].to_numpy()
    sets = {"all": np.ones(len(rows), bool), "cold": ~w, "warm": w}
    rank_of = {
        "main": "main",
        "query_token": "query_token",
        "plain_all": "plain_all",
        "plain_warm": "plain_warm_fb",  # model on warm rows, global popularity on cold rows
        "itemknn": "itemknn",
        "logreg": "logreg",
    }
    mism, n_cmp, mine = [], 0, {}
    for key, col in rank_of.items():
        for rs, m in sets.items():
            calc = metrics(rows[col].to_numpy()[m])
            mine[f"{key}/{rs}"] = {k: round(v, 4) if k != "n" else v for k, v in calc.items()}
            pk = bench[(bench["model_key"] == key) & (bench["row_set"] == rs)]
            if len(pk) != 1:
                mism.append((key, rs, "missing row"))
                continue
            pk = pk.iloc[0]
            if int(pk["n"]) != calc["n"]:
                mism.append((key, rs, "n", int(pk["n"]), calc["n"]))
            for mt in METRICS:
                n_cmp += 1
                if abs(float(pk[mt]) - calc[mt]) > TOL_ROUND:
                    mism.append((key, rs, mt, float(pk[mt]), round(calc[mt], 6)))
    check(
        f"headline: 6 models x 3 row sets x 6 metrics recomputed = final_benchmark.csv ({n_cmp} values)",
        not mism,
        mism or "0 mismatches",
    )
    # the same numbers in final_tables.md
    md = (W5 / "final_tables.md").read_text(encoding="utf-8")
    missing = []
    for key in rank_of:
        for rs in sets:
            v = mine[f"{key}/{rs}"]
            cells = " | ".join(f"{v[m]:.4f}" for m in METRICS)
            if f"| {v['n']} | {cells} |" not in md:
                missing.append(f"{key}/{rs}")
    check("headline rows appear verbatim in final_tables.md", not missing, missing or "18/18 rows")
    # main cold = query token cold (sameDest has no input on an empty history)
    check(
        "main model on cold rows is identical to the query token (rank by rank)",
        bool((rows["main"].to_numpy()[~w] == rows["query_token"].to_numpy()[~w]).all()),
    )
    # ItemKNN on cold rows: one constant ranking (popularity), not a model score
    knn_cold_top = (
        pd.Series(rows["truth"].to_numpy()[~w])
        .groupby(rows["itemknn"].to_numpy()[~w])
        .nunique()
        .max()
    )
    check(
        "ItemKNN cold rows: each rank position maps to one cluster (constant popularity ranking)",
        bool(knn_cold_top == 1),
    )
    return mine


def verify_bootstrap(rows: pd.DataFrame) -> list:
    pkg = pd.read_csv(W5 / "bootstrap_ci.csv")
    w = rows["warm"].to_numpy()
    out = []
    for rs, m in (("warm", w), ("cold", ~w)):
        for k in (5, 10):
            r = paired_bootstrap(
                rows["main"].to_numpy()[m], rows["logreg"].to_numpy()[m], k=k, seed=BOOT_SEED
            )
            for mt, v in r.items():
                p = pkg[
                    (pkg.row_set == rs)
                    & (pkg.a == "main")
                    & (pkg.b == "logreg")
                    & (pkg.metric == mt)
                ].iloc[0]
                gap = max(abs(v["ci95"][0] - p["ci95_lo"]), abs(v["ci95"][1] - p["ci95_hi"]))
                ok = abs(v["diff"] - p["diff"]) < 1e-9 and gap <= BOOT_TOL
                out.append(
                    {
                        "row_set": rs,
                        "metric": mt,
                        "diff_mine": v["diff"],
                        "diff_pkg": p["diff"],
                        "ci_mine": v["ci95"],
                        "ci_pkg": [p["ci95_lo"], p["ci95_hi"]],
                        "max_endpoint_gap": round(gap, 4),
                        "ok": ok,
                    }
                )
    check(
        f"bootstrap main - logistic regression (warm, cold; K=5,10) with seed {BOOT_SEED}: "
        f"same diff, CI endpoints within {BOOT_TOL}",
        all(o["ok"] for o in out),
        [{k: v for k, v in o.items() if k != "ok"} for o in out],
    )
    return out


def verify_slice_L(rows: pd.DataFrame) -> None:
    pkg = pd.read_csv(W5 / "slices_history_length.csv")
    L = rows["n_prior"].to_numpy()
    bucket = np.select(
        [L == 0, L == 1, L <= 4, L <= 9], ["L=0 (cold)", "L=1", "L=2-4", "L=5-9"], "L>=10"
    )
    cols = {
        "main": "main",
        "query_token": "query_token",
        "plain_all": "plain_all",
        "itemknn": "itemknn",
        "logreg": "logreg",
        "plain_warm": "plain_warm_model",
    }
    mism, n = [], 0
    for bk in np.unique(bucket):
        m = bucket == bk
        for key, col in cols.items():
            if key == "plain_warm" and bk == "L=0 (cold)":
                continue
            calc = metrics(rows[col].to_numpy()[m])
            p = pkg[(pkg.bucket == bk) & (pkg.model_key == key)]
            if len(p) != 1:
                mism.append((bk, key, "missing"))
                continue
            p = p.iloc[0]
            if int(p["n"]) != calc["n"]:
                mism.append((bk, key, "n"))
            for mt in METRICS:
                n += 1
                if abs(p[mt] - calc[mt]) > TOL_ROUND:
                    mism.append((bk, key, mt, p[mt], round(calc[mt], 6)))
    check(
        f"slice history length L recomputed = slices_history_length.csv ({n} values)",
        not mism,
        mism or "0 mismatches",
    )


def verify_beyond_accuracy(rows: pd.DataFrame) -> None:
    pkg = pd.read_csv(W5 / "beyond_accuracy_aggregate.csv")
    p = pkg[(pkg.row_set == "all") & (pkg.model_key == "main")].iloc[0]
    top = rows.attrs["top_main"]
    expo = np.bincount(top.ravel(), minlength=100)
    item, last = rows.attrs["item"], rows.attrs["last_train"]
    pop = np.bincount(item[: last + 1], minlength=100).astype(float)
    share = pop / pop.sum()
    arp = float(share[top].mean())
    got = {
        "catalog_coverage@10": int((expo > 0).sum()),
        "exposure_gini (Lorenz)": round(gini_lorenz(expo), 4),
        "ARP@10": round(arp, 5),
    }
    want = {
        "catalog_coverage@10": int(p["catalog_coverage@10"]),
        "exposure_gini (Lorenz)": float(p["exposure_gini"]),
        "ARP@10": float(p["ARP@10"]),
    }
    ok = (
        got["catalog_coverage@10"] == want["catalog_coverage@10"]
        and abs(got["exposure_gini (Lorenz)"] - want["exposure_gini (Lorenz)"]) <= 1e-4
        and abs(got["ARP@10"] - want["ARP@10"]) <= 1e-5
    )
    check(
        "beyond accuracy, main model, all rows: coverage, exposure Gini, ARP@10",
        ok,
        {"mine": got, "pkg": want},
    )


def verify_es_run() -> None:
    run = json.loads((W4 / "smlprec_query_cold_train_es_run.json").read_text(encoding="utf-8"))
    cold = json.loads((W4 / "smlprec_query_cold_L0_cold_train_es.json").read_text(encoding="utf-8"))
    met = json.loads((ES_DIR / "metrics.json").read_text(encoding="utf-8"))
    ck = str(run["checkpoint"]).replace("\\", "/")
    best = torch.load(ES_DIR / "best.pth", weights_only=False, map_location="cpu")
    ep8 = torch.load(ES_DIR / "epoch8.pth", weights_only=False, map_location="cpu")
    same8 = all(torch.equal(best["state_dict"][k], ep8[k]) for k in best["state_dict"])
    detail = {
        "run_json_checkpoint": ck.split("/experiments/")[-1],
        "cold_json_checkpoint": str(cold["checkpoint"])
        .replace("\\", "/")
        .split("/experiments/")[-1],
        "best_epoch_by_valid": run["best_epoch_by_valid"],
        "best.pth epoch field": best.get("epoch"),
        "best.pth weights == epoch8.pth": same8,
        "metrics.json test == run json test": met.get("test") == run.get("test"),
        "stop_reason": run["stop_reason"],
    }
    ok = (
        ES_DIR.name in ck
        and ES_DIR.name in str(cold["checkpoint"])
        and run["best_epoch_by_valid"] == 8
        and best.get("epoch") == 8
        and same8
        and detail["metrics.json test == run json test"]
    )
    check(
        "main model = early-stopping run in experiments/2026-10-10_..._query-1d-clean-cold-es (epoch 8)",
        ok,
        detail,
    )
    z = np.load(INTERIM / "final_topk_test.npz")
    tv = run["samedest_hybrid"]["tuned_on_valid"]["A"]
    check(
        "sameDest weights in final_topk_test.npz = tuned_on_valid in the ES run json",
        float(z["alpha"]) == tv["alpha"]
        and float(z["ws_new"]) == tv["weights"]["1"][0]
        and float(z["ws_known"]) == tv["weights"]["1"][1],
        {"alpha": float(z["alpha"]), "w_new": float(z["ws_new"]), "w_known": float(z["ws_known"])},
    )
    pl = json.loads((W4 / "smlprec_plain_cold_train_es_run.json").read_text(encoding="utf-8"))
    bs = json.loads((W4 / "baselines_es_run.json").read_text(encoding="utf-8"))
    check(
        "01e plain and 07b baselines trained on the same rows as 01d (warm + first bookings)",
        "identical to 01d" in pl["training_set"]
        and "identical to 01d" in bs["training_set"]["rule"]
        and bs["training_set"]["first_bookings"] == run["cold_training_rows"]
        and pl["cold_rows"]["train"] == run["cold_training_rows"],
        {"01e": pl["training_set"], "07b": bs["training_set"]},
    )


def verify_labels() -> None:
    bench = pd.read_csv(W5 / "final_benchmark.csv")
    hl = bench[bench["seed"].astype(str) == "2022"]
    bad = []
    for _, r in hl.iterrows():
        ts, src = r["training_set"], r["source"]
        if r["model_key"] == "plain_warm":
            if "warm rows only" not in ts:
                bad.append((r["model_key"], r["row_set"], "warm-only label missing"))
        elif "all events (warm + L=0)" not in ts:
            bad.append((r["model_key"], r["row_set"], "all-events label missing"))
        if r["model_key"] in ("main", "query_token") and ("01d" not in ts or "_es" not in src):
            bad.append((r["model_key"], r["row_set"], "not 01d ES"))
        if r["model_key"] in ("itemknn", "logreg") and (
            "07b" not in ts or "baselines_es_run" not in src
        ):
            bad.append((r["model_key"], r["row_set"], "not 07b"))
        if r["model_key"] == "plain_all" and ("01e" not in ts or "plain_cold_train_es" not in src):
            bad.append((r["model_key"], r["row_set"], "not 01e ES"))
    check(
        "training-set labels: plain_warm warm-only; all others all events; 01d ES / 01e / 07b sources",
        not bad,
        bad or "18 rows ok",
    )
    cold = hl[hl.row_set == "cold"].set_index("model_key")["scored_by"]
    want = {
        "main": "identical to query token",
        "itemknn": "NOT SCORED by the model",
        "plain_warm": "NOT SCORED by the model",
        "plain_all": "one constant ranking",
    }
    miss = {k: cold[k] for k, v in want.items() if v not in cold[k]}
    check(
        "cold-row handling labelled (main = query token; ItemKNN / plain warm-only = popularity fallback)",
        not miss,
        miss or "4/4 labels",
    )
    doc = (DOCS / "01_results_summary.md").read_text(encoding="utf-8")
    check(
        "01_results_summary.md labels the original-code warm-only run 'warm users only'",
        "**warm users only**" in doc,
    )


# ----------------------------------------------------------------------------------------------
NUM = re.compile(r"(?<![\w.])[+-]?\d[\d,]*\.?\d*(?![\w])")


def numbers(text: str) -> set[str]:
    return {m.group(0).lstrip("+").rstrip(".,") for m in NUM.finditer(text)}


def verify_doc01() -> dict:
    """Table rows of 01 must appear verbatim in final_tables.md; prose numbers must exist in a source."""
    doc = (DOCS / "01_results_summary.md").read_text(encoding="utf-8").splitlines()
    md = (W5 / "final_tables.md").read_text(encoding="utf-8")
    md_lines = set(md.splitlines())
    corpus = md
    for f in sorted(W4.glob("*.json")):
        corpus += f.read_text(encoding="utf-8")
    corpus += (W5 / "final_results.json").read_text(encoding="utf-8")
    for f in sorted(W5.glob("*.csv")):
        corpus += f.read_text(encoding="utf-8")
    corpus_nums = numbers(corpus) | {n.replace(",", "") for n in numbers(corpus)}
    # long floats in the JSON (e.g. 0.9999953...) are quoted rounded: add 3-6 decimal roundings
    for n in [x for x in corpus_nums if "." in x and len(x.split(".")[1]) > 6]:
        try:
            corpus_nums |= {f"{float(n):.{d}f}" for d in (3, 4, 5, 6)}
        except ValueError:
            pass
    in_section1 = False
    table_miss, prose_miss = [], []
    for i, line in enumerate(doc, 1):
        if line.startswith("## "):
            in_section1 = line.startswith(("## 1.", "## 9."))  # hand-written tables
        if line.startswith("|") and not line.startswith("|---"):
            if line in md_lines or in_section1:
                continue
            table_miss.append(i)
            continue
        if line.startswith(("```", "#")):
            continue
        for n in numbers(line):
            plain = n.replace(",", "")
            if not re.search(r"\d", n) or len(plain.replace(".", "")) < 3:
                continue  # 1-2 digit counts, K values, section numbers
            cand = {n, plain, plain.rstrip("0"), plain.lstrip("0")}
            if (
                plain.replace(".", "").isdigit()
                and "." not in plain
                and int(plain) in range(1990, 2100)
            ):
                continue  # years
            if not cand & corpus_nums:
                prose_miss.append((i, n))
    # line-level threshold claim: "months 4-10 below 0.03 each" (test check-in month share)
    eda = json.loads((W4 / "eda_clean.json").read_text(encoding="utf-8"))
    share = eda["E7"]["composition"]["test"]["checkin_month"]
    mx = max(share[str(m)] for m in range(4, 11))
    check(
        "01 prose: test check-in month share of months 4-10 below 0.03 each", mx < 0.03, {"max": mx}
    )
    if mx < 0.03:
        prose_miss = [x for x in prose_miss if x[1] != "0.03"]
    check(
        "01_results_summary.md: every table row outside the hand-written sections 1 and 9 is verbatim in final_tables.md",
        not table_miss,
        table_miss or "all rows found",
    )
    check(
        "01_results_summary.md: every prose number (>= 3 digits) occurs in final_tables.md / week4 JSON / week5 CSV",
        not prose_miss,
        prose_miss or "all found",
    )
    return {"table_rows_not_found": table_miss, "prose_numbers_not_found": prose_miss}


def spot_checks() -> None:
    """Numbers quoted in 02 / 04 against their source files."""
    js = {
        f: json.loads((W4 / f).read_text(encoding="utf-8"))
        for f in (
            "bookings_counts.json",
            "eda_clean.json",
            "baselines_es_run.json",
            "smlprec_query_cold_train_es_run.json",
            "smlprec_plain_cold_train_es_run.json",
            "baselines_vs_final_es.json",
        )
    }
    run, eda, bs = (
        js["smlprec_query_cold_train_es_run.json"],
        js["eda_clean.json"],
        js["baselines_es_run.json"],
    )
    hl = pd.read_csv(W5 / "final_benchmark.csv")
    hl = hl[hl["seed"].astype(str) == "2022"].set_index(["model_key", "row_set"])
    t02 = (DOCS / "02_report_toc.md").read_text(encoding="utf-8")
    t04 = (DOCS / "04_demo_options.md").read_text(encoding="utf-8")
    items = [
        (
            "02",
            "2,952,675 bookings",
            js["bookings_counts.json"]["collapse_steps"]["rows_out"],
            2952675,
        ),
        ("02", "813,985 users", eda["E2"]["users"], 813985),
        (
            "02",
            "39.34 % single-booking users",
            round(100 * eda["E2"]["share_single_booking"], 2),
            39.34,
        ),
        (
            "02",
            "48,010 burst repeats",
            js["bookings_counts.json"]["collapse_steps"]["burst_dropped"],
            48010,
        ),
        ("02", "713,754 first bookings", bs["training_set"]["first_bookings"], 713754),
        ("02", "train 1,710,952 warm targets", bs["split_counts"]["train"], 1710952),
        ("02", "valid cold 50,769", bs["split_counts"]["valid_cold"], 50769),
        (
            "02",
            "33,702 parameters (plain)",
            js["smlprec_plain_cold_train_es_run.json"]["n_parameters"],
            33702,
        ),
        ("02", "11,128 s train + eval", round(run["timing_seconds"]["train_and_eval"]), 11128),
        ("02", "best epoch 8", run["best_epoch_by_valid"], 8),
        (
            "02",
            "JSD check-in month train vs test 0.439 bits",
            round(eda["E7"]["jsd_train_vs_test_bits_per_field"]["ci_month"], 3),
            0.439,
        ),
        ("02", "warm R@5 ItemKNN 0.3036", float(hl.loc[("itemknn", "warm"), "recall@5"]), 0.3036),
        ("02", "warm NDCG@5 logreg 0.3354", float(hl.loc[("logreg", "warm"), "ndcg@5"]), 0.3354),
        (
            "02",
            "warm R@5 / NDCG@5 main 0.6104 / 0.4461",
            float(hl.loc[("main", "warm"), "recall@5"]),
            0.6104,
        ),
        (
            "04",
            "warm Main model Recall@5 (table in section 0)",
            float(hl.loc[("main", "warm"), "recall@5"]),
            None,
        ),
        (
            "04",
            "cold Main model Recall@5 0.5252",
            float(hl.loc[("main", "cold"), "recall@5"]),
            0.5252,
        ),
        (
            "04",
            "warm logreg Recall@5 0.4969",
            float(hl.loc[("logreg", "warm"), "recall@5"]),
            0.4969,
        ),
        (
            "04",
            "cold ItemKNN Recall@5 0.1318",
            float(hl.loc[("itemknn", "cold"), "recall@5"]),
            0.1318,
        ),
        ("04", "first test event 2689344", run["splits"]["test"]["first_target_event"], 2689344),
        ("04", "last train event 2424705", run["splits"]["train"]["last_target_event"], 2424705),
    ]
    res = []
    for doc, label, src, quoted in items:
        if quoted is None:  # 04 table: read what the doc says
            m = re.search(r"\| warm \(n = 213,869\) \| ([0-9.]+) \|", t04)
            quoted = float(m.group(1)) if m else None
        ok = quoted is not None and abs(float(src) - float(quoted)) < 1e-9
        res.append({"doc": doc, "claim": label, "doc_value": quoted, "source_value": src, "ok": ok})
    for r in res:
        check(
            f"spot check {r['doc']}: {r['claim']}",
            r["ok"],
            {"doc": r["doc_value"], "source": r["source_value"]},
        )
    check(
        "02 test window (mid-November to December) = eda_clean.json splits.test dates",
        eda["splits"]["test"]["first_date"] == "2014-11-19"
        and eda["splits"]["test"]["last_date"] == "2014-12-31"
        and "mid-November to December" in t02,
        eda["splits"]["test"],
    )


def main() -> None:
    rows = load_rows()
    verify_counts(rows)
    mine = verify_headline(rows)
    boot = verify_bootstrap(rows)
    verify_slice_L(rows)
    verify_beyond_accuracy(rows)
    verify_es_run()
    verify_labels()
    doc = verify_doc01()
    spot_checks()
    bench = pd.read_csv(ROOT / "reports" / "benchmark_results.csv")
    check(
        "reports/benchmark_results.csv contains the early-stopping runs (stale if not)",
        None,
        {"has_es_rows": bool(bench["source"].astype(str).str.contains("_es").any())},
    )
    summary = {s: sum(c["status"] == s for c in CHECKS) for s in ("PASS", "FAIL", "NOTE")}
    out = {
        "generated_by": "scripts/verify_final_results.py",
        "dataset": "Expedia (collapsed bookings), test split, 100 clusters",
        "bootstrap_seed": BOOT_SEED,
        "summary": summary,
        "checks": CHECKS,
        "recomputed_headline": mine,
        "bootstrap_recomputed": boot,
        "doc01_diff": doc,
    }
    (W5 / "verification_final.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8"
    )
    print("\nsummary:", summary, "-> results/week5_final/verification_final.json")


if __name__ == "__main__":
    main()
