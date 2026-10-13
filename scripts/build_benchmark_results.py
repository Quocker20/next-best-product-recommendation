"""Rebuild reports/benchmark_results.csv from results/week6_retrain/final_benchmark.csv.

One row per dataset x model x seed x row set x protocol (test window), with the `source` file.
The input is written by notebooks/hospitality/smlp4rec/04_compare.ipynb; never edit either CSV by hand.

Usage: python scripts/build_benchmark_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbp.eval.metrics import KS

SRC = ROOT / "results" / "week6_retrain" / "final_benchmark.csv"
OUT = ROOT / "reports" / "benchmark_results.csv"
ROLE = {
    "ItemKNN": "baseline",
    "Logistic regression": "baseline",
    "SMLP4Rec plain": "main-model ablation",
    "SMLP4Rec + query embedding": "main-model ablation",
    "SMLP4Rec + query embedding + sameDest (main)": "main model",
}


def main() -> None:
    d = pd.read_csv(SRC)
    unknown = set(d["model"]) - set(ROLE)
    if unknown:
        raise SystemExit(f"models without a role: {sorted(unknown)}")
    d["role"] = d["model"].map(ROLE)
    d["source"] = SRC.relative_to(ROOT).as_posix()
    cols = ["dataset", "model", "role", "trained_on", "seed", "window", "row_set", "protocol", "n"]
    cols += [f"ndcg@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["source"]
    d[cols].to_csv(OUT, index=False)
    print(f"{len(d)} rows -> {OUT}")


if __name__ == "__main__":
    main()
