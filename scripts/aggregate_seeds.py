"""Mean +- std over seeds for notebooks 01c and 01d, and the paired 01d - 01c difference per seed.

Read-only. Inputs (results/week4_rebuild/): smlprec_query_run[_seed<S>].json (01c),
smlprec_query_cold_train_run[_seed<S>].json (01d), the matching cold JSONs
(smlprec_query_cold_L0[...].json, smlprec_query_cold_L0_cold_train[...].json) and sensitivity JSONs.
Seed 2022 is the reference run without a suffix. Output: results/week4_rebuild/seeds_summary.json.

Std is the sample std (ddof = 1) over the seeds; with 3 seeds it is a rough guide only.

Usage: .venv/Scripts/python.exe scripts/aggregate_seeds.py
"""

from __future__ import annotations

import json

import numpy as np

from nbp.paths import ROOT

RES = ROOT / "results" / "week4_rebuild"
SEEDS = [2022, 2023, 2024]
MODELS = {
    "01c": ("smlprec_query_run", "smlprec_query_cold_L0", "sensitivity"),
    "01d": (
        "smlprec_query_cold_train_run",
        "smlprec_query_cold_L0_cold_train",
        "sensitivity_cold_train",
    ),
}
METRICS = ("recall@5", "ndcg@5")


def path(stem: str, seed: int):
    return RES / (f"{stem}.json" if seed == 2022 else f"{stem}_seed{seed}.json")


def per_seed(variant: str) -> dict:
    run_stem, cold_stem, sens_stem = MODELS[variant]
    out: dict = {}
    for s in SEEDS:
        run = json.loads(path(run_stem, s).read_text(encoding="utf-8"))
        cold = json.loads(path(cold_stem, s).read_text(encoding="utf-8"))
        sens = json.loads(path(sens_stem, s).read_text(encoding="utf-8"))
        bench = run["samedest_hybrid"]["benchmark_test"]
        row = {"best_epoch": run["best_epoch_by_valid"], "valid_ndcg@10": run["valid"]["ndcg@10"]}
        for sc in ("warm", "all_events"):
            for name, tag in (("Query token", "qt"), ("Query token + sameDest", "qt_sameDest")):
                for m in METRICS:
                    row[f"{sc}/{tag}/{m}"] = bench[sc][name][m]
            for m in METRICS:
                row[f"{sc}/sameDest_gain/{m}"] = row[f"{sc}/qt_sameDest/{m}"] - row[f"{sc}/qt/{m}"]
        for m in METRICS:
            row[f"cold/qt/{m}"] = cold["results"]["test"]["query token alone (empty history)"][m]
            row[f"cold/prior_only/{m}"] = cold["results"]["test"]["destination prior only"][m]
            row[f"warm/sameDest_gain_excl_rebook_1h/{m}"] = sens["sameDest_gain"][
                "excluding_rebook_within_1h"
            ]["gain_bootstrap"][m]["diff"]
            row[f"warm/qt_minus_logreg/{m}"] = sens["vs_baselines_warm"][
                "query_token minus logistic_regression"
            ][m]["diff"]
            row[f"warm/qt_sameDest_minus_logreg/{m}"] = sens["vs_baselines_warm"][
                "query_token_sameDest minus logistic_regression"
            ][m]["diff"]
            row[f"cold/qt_minus_logreg/{m}"] = sens["cold_rows"][
                "query_token minus logistic_regression"
            ][m]["diff"]
        out[s] = row
    return out


def stats(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std(ddof=1)), 4),
        "values": [round(float(x), 4) for x in v],
    }


def main() -> None:
    data = {v: per_seed(v) for v in MODELS}
    keys = [k for k in data["01c"][SEEDS[0]] if "/" in k]
    summary: dict = {"seeds": SEEDS, "per_variant": {}, "paired_01d_minus_01c": {}}
    for v in data:
        summary["per_variant"][v] = {k: stats([data[v][s][k] for s in SEEDS]) for k in keys}
        summary["per_variant"][v]["valid_ndcg@10"] = stats(
            [data[v][s]["valid_ndcg@10"] for s in SEEDS]
        )
        summary["per_variant"][v]["best_epoch"] = [data[v][s]["best_epoch"] for s in SEEDS]
    for k in keys:
        diffs = [data["01d"][s][k] - data["01c"][s][k] for s in SEEDS]
        st = stats(diffs)
        st["seeds_where_01d_higher"] = int(sum(d > 0 for d in diffs))
        summary["paired_01d_minus_01c"][k] = st
    (RES / "seeds_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(s):
        return f"{s['mean']:.4f} +- {s['std']:.4f}"

    for k in (
        "warm/qt/recall@5",
        "warm/qt/ndcg@5",
        "warm/qt_sameDest/recall@5",
        "warm/qt_sameDest/ndcg@5",
        "all_events/qt_sameDest/ndcg@5",
        "cold/qt/recall@5",
        "cold/qt/ndcg@5",
        "warm/sameDest_gain/ndcg@5",
        "warm/sameDest_gain_excl_rebook_1h/ndcg@5",
        "warm/qt_minus_logreg/ndcg@5",
        "cold/qt_minus_logreg/recall@5",
    ):
        p = summary["paired_01d_minus_01c"][k]
        print(
            f"{k:44s} 01c {fmt(summary['per_variant']['01c'][k])} | 01d {fmt(summary['per_variant']['01d'][k])}"
            f" | paired 01d-01c {p['mean']:+.4f} +- {p['std']:.4f} ({p['seeds_where_01d_higher']}/3 seeds)"
        )


if __name__ == "__main__":
    main()
