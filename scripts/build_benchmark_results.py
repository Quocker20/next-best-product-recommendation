"""Collect benchmark rows from the committed result JSON files into one CSV.

One row per dataset x model x seed x row set (test split only). No number is typed here: every
value is read from a JSON in results/week4_rebuild/.

Inputs : results/week4_rebuild/{smlprec_query_run,basic_baselines,smlprec_query_cold_L0}.json
Output : reports/benchmark_results.csv
Columns: dataset, model, role, seed, row_set, n, recall@5/10/20, ndcg@5/10/20, source
Usage  : python scripts/build_benchmark_results.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "week4_rebuild"
OUT = ROOT / "reports" / "benchmark_results.csv"
SEED = 2022  # RecBole seed of notebook 01c and random_state of notebook 07
METRICS = ["recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"]

# model name in the source JSON -> (name in the table, role)
QUERY_RUN_MODELS = {
    "SMLP4Rec plain": ("SMLP4Rec plain (history only)", "main-model ablation"),
    "Query token": ("SMLP4Rec query token", "main-model ablation"),
    "Query token + sameDest": ("SMLP4Rec query token + sameDest", "main model"),
    "Hybrid cũ: plain + prior + sameDest (nb 04)": (
        "Old hybrid (plain + prior + sameDest)",
        "internal reference",
    ),
}
BASELINES = {
    "ItemKNN (cosine, K=100, history only)": ("ItemKNN", "baseline"),
    "Logistic regression (C=10.0, destination + context, no history)": (
        "Logistic regression",
        "baseline",
    ),
}
BASELINE_ROWS = {
    "warm (all L>=1 targets)": "warm",
    "unseen, L=0 (first booking)": "cold",
    "all events": "all_events",
}


def row(model: str, role: str, row_set: str, seed: str, m: dict, source: str) -> dict:
    """One CSV row from a metric dict with keys n and METRICS."""
    return {
        "dataset": "expedia",
        "model": model,
        "role": role,
        "seed": seed,
        "row_set": row_set,
        "n": m["n"],
        **{k: m[k] for k in METRICS},
        "source": source,
    }


def main() -> None:
    rows = []
    q = json.loads((RES / "smlprec_query_run.json").read_text(encoding="utf8"))
    bench = q["samedest_hybrid"]["benchmark_test"]
    for row_set in ("warm", "all_events"):
        for src_name, (name, role) in QUERY_RUN_MODELS.items():
            rows.append(
                row(
                    name,
                    role,
                    row_set,
                    str(SEED),
                    bench[row_set][src_name],
                    "smlprec_query_run.json",
                )
            )

    b = json.loads((RES / "basic_baselines.json").read_text(encoding="utf8"))["results"]["test"]
    for src_name, (name, role) in BASELINES.items():
        seed = "deterministic" if name == "ItemKNN" else str(SEED)
        for src_set, row_set in BASELINE_ROWS.items():
            rows.append(
                row(name, role, row_set, seed, b[src_name][src_set], "basic_baselines.json")
            )

    cold = json.loads((RES / "smlprec_query_cold_L0.json").read_text(encoding="utf8"))
    qt = cold["results"]["test"]["query token alone (empty history)"]
    rows.append(
        row(
            "SMLP4Rec query token (empty history)",
            "main-model ablation",
            "cold",
            str(SEED),
            qt,
            "smlprec_query_cold_L0.json",
        )
    )

    # variant trained with the first bookings (L = 0 rows), notebook 01d
    qd_path = RES / "smlprec_query_cold_train_run.json"
    if qd_path.exists():
        bd = json.loads(qd_path.read_text(encoding="utf8"))["samedest_hybrid"]["benchmark_test"]
        for row_set in ("warm", "all_events"):
            for src_name in ("Query token", "Query token + sameDest"):
                name, _ = QUERY_RUN_MODELS[src_name]
                rows.append(
                    row(
                        f"{name} [trained with L=0 rows]",
                        "main-model variant (01d)",
                        row_set,
                        str(SEED),
                        bd[row_set][src_name],
                        qd_path.name,
                    )
                )
        cd_path = RES / "smlprec_query_cold_L0_cold_train.json"
        if cd_path.exists():
            qd = json.loads(cd_path.read_text(encoding="utf8"))
            rows.append(
                row(
                    "SMLP4Rec query token (empty history) [trained with L=0 rows]",
                    "main-model variant (01d)",
                    "cold",
                    str(SEED),
                    qd["results"]["test"]["query token alone (empty history)"],
                    cd_path.name,
                )
            )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT}")
    print(df.drop(columns=["source"]).to_string(index=False))


if __name__ == "__main__":
    main()
