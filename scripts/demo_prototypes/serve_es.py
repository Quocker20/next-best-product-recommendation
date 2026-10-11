"""Prototype: run the existing demo UI on the early-stopping checkpoint without editing demo_ui.py.

Overrides `RUNS` so only seed 2022 is served, from
experiments/2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es/best.pth (epoch 8) with the sameDest
weights tuned on valid in results/week4_rebuild/smlprec_query_cold_train_es_run.json.
Baselines are unchanged (data/interim/demo_baselines.npz, exported from notebook 07).

Usage (smlp4rec venv; -u so the "ready" line is not buffered):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe -u scripts/demo_prototypes/serve_es.py --port 8766
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo_ui

demo_ui.RUNS = {
    2022: (
        "2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es",
        "smlprec_query_cold_train_es_run.json",
    ),
}

if __name__ == "__main__":
    demo_ui.main()
