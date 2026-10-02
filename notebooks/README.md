# Notebooks

Thư mục lưu trữ các Jupyter Notebook phục vụ quá trình khám phá dữ liệu (EDA), thử nghiệm mô hình và trực quan hóa dữ liệu.

## `hospitality/smlp4rec/` — train / test / evaluate SMLP4Rec on Expedia

Run in order with the `smlp4rec` kernel (Python 3.11 venv with RecBole 1.0.1). Each notebook writes to `experiments/<date>_expedia_smlp4rec_week3-repro/` and ends with a check against the script-era JSONs in `reports/summary/week3/`. Metrics: Recall@K and NDCG@K (`src/nbp/eval/metrics.py`).

1. `01_train_test_smlp4rec.ipynb` — explicit training loop, per-epoch valid/test, best epoch, heuristics.
2. `02_top5_rerank_rules.ipynb` — past/novel rules on the top 5 (fixed quota, L-dependent cap).
3. `03_destination_prior_hybrid.ipynb` — destination prior, late fusion, cold users, sameDest hybrid, bootstrap.
