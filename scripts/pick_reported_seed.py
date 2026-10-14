"""Pick the seed whose results are closest to the 3-seed mean; write results/week6_retrain/reported_seed.json.

Criterion: for each SMLP4Rec model (plain, + query embedding, main), each row set (all, warm, cold) and
each protocol (AR, TF) of the test window, the absolute gap of the seed's six metrics (NDCG@5/10/20,
Recall@5/10/20) to the 3-seed mean, divided by the std over seeds; the mean of these z gaps per seed.
The seed with the smallest value is the reported seed used for every table, figure and slide of the report
(configs/split.yaml `report_seed` must hold it). Input: results/week6_retrain/final_benchmark.csv
(written by notebook 04). Hyperparameters and the sameDest weights stay selected on the reference seed 2022.

Usage: python scripts/pick_reported_seed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbp.eval.metrics import KS

SRC = ROOT / "results" / "week6_retrain" / "final_benchmark.csv"
OUT = ROOT / "results" / "week6_retrain" / "reported_seed.json"
COLS = [f"ndcg@{k}" for k in KS] + [f"recall@{k}" for k in KS]


def main() -> None:
    b = pd.read_csv(SRC)
    b = b[b["model"].str.startswith("SMLP4Rec") & (b["window"] == "test")]
    gaps = []
    for (proto, rs, model), g in b.groupby(["protocol", "row_set", "model"]):
        mu, sd = g[COLS].mean(), g[COLS].std(ddof=1)
        for _, r in g.iterrows():
            z = ((r[COLS].astype(float) - mu).abs() / sd).mean()
            gaps.append({"protocol": proto, "row_set": rs, "model": model, "seed": int(r["seed"]), "z_gap": float(z)})
    d = pd.DataFrame(gaps)
    per_seed = d.groupby("seed")["z_gap"].mean()
    primary = d[(d.protocol == "ar") & (d.row_set == "all")].groupby("seed")["z_gap"].mean()
    seed = int(per_seed.idxmin())
    assert seed == int(primary.idxmin()), "the primary table and all tables disagree"
    out = {
        "reported_seed": seed,
        "criterion": "mean over models, row sets and protocols of |metric - 3-seed mean| / std over seeds, six metrics (test window)",
        "mean_z_gap_all_tables": {str(s): round(float(v), 4) for s, v in per_seed.items()},
        "mean_z_gap_primary_table_ar_all_rows": {str(s): round(float(v), 4) for s, v in primary.items()},
        "source": SRC.relative_to(ROOT).as_posix(),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
