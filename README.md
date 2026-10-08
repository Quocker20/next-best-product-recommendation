# Next-Best-Product Recommendation (Topic C1)

Research benchmark for next-best-product recommendation in hospitality (Vinpearl). Output is a report, not a production system.

**Scope (revised 2026-10-08):** travel domain, one dataset (**Expedia Hotel Recommendations**), one main model (**SMLP4Rec with a query token + sameDest**), two classical baselines (**ItemKNN**, **logistic regression**). Remaining work: data processing, model tuning, RQ1 / RQ4 evaluation, final report. Everything else (other models, food delivery, taxi, cross-sell) is out of scope until reopened. Details: `CLAUDE.md`, active plan `docs/plans/plan_oct_08_replan.md`, decision log `PROGRESS.md` (`CLAUDE.md` and `PROGRESS.md` are local, gitignored).

**Status:** the main model reaches test NDCG@5 0.4475 / Recall@5 0.6060 on warm users (one seed, 3 epochs), against 0.2195 / 0.3161 for ItemKNN and 0.3561 / 0.5219 for logistic regression. Numbers come from `results/week4_rebuild/`.

## Layout

| Folder | Holds |
|---|---|
| `docs/` | Reference docs, data cards, `plans/`, `explainers/` |
| `reports/summary/<week>/` | Only reports and slides (docx, pdf, html, md) |
| `results/<week>/` | Committed script outputs (json, csv, logs, `figures/{en,vi}`) |
| `experiments/` | Training-run outputs (gitignored) |
| `scripts/` | Analysis, run, report and slide builders |
| `notebooks/` | EDA per domain; modelling in `hospitality/smlp4rec/` (`01c` main model, `07` baselines) |
| `src/nbp/` | Tested reusable code: `data`, `eval`, `hybrid`, `models`, `baselines` |
| `configs/`, `tests/`, `sample_data/` | Configs, pytest, small samples |
| `data/` | Gitignored: `raw/` (read-only), `interim/`, `processed/` |

Week folders: `week1_dataset_selection`, `week2_methodology`, `week3_implementation`, `week4_rebuild`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

SMLP4Rec runs through RecBole 1.0.1, which needs numpy 1.23 and pandas 1.5, so it uses a separate Python 3.11 venv at a **short** path (long Windows paths break torch DLL loading):

```bash
py -3.11 -m venv C:/Users/<user>/.venvs/smlp4rec
# install torch (CPU), numpy==1.23.5, pandas==1.5.3, scipy==1.10.1, PyYAML, tqdm, ipykernel, then:
pip install --no-deps -e <path-to-patched-MLP4Rec-clone>
python -m ipykernel install --user --name smlp4rec
```

RecBole patches: `weights_only=False` in `torch.load`; anomaly detection off (`sine.py` enables it on import).

## Reproduce the main model

```bash
python scripts/download.py                     # raw data into data/raw/ (needs Kaggle token in env)
python scripts/expedia_build_bookings.py       # cleaned bookings parquet
python scripts/expedia_to_recbole.py           # RecBole interaction file
python scripts/expedia_query_to_recbole.py     # query-token codes
```

Then run `notebooks/hospitality/smlp4rec/01c_train_test_smlp4rec_query.ipynb` (kernel `smlp4rec`) for the main model and `07_basic_baselines.ipynb` (kernel `nbp`, the `.venv`) for the baselines. Tests: `pytest`.
