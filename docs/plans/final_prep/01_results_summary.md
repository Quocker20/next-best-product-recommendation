# 01 Final results summary (Expedia, test split)

Consolidated results package for the final report. Scope: the original code (SMLP4Rec plain, history only), the improved model (SMLP4Rec + query embedding, and + sameDest = the locked main model, notebook 01d with early stopping) and the two baselines (ItemKNN, logistic regression). Explorations (old hybrid, rerank rules, destination prior, behaviour split) are left out.

- Built 2026-10-11 by `scripts/final_results_summary.py` (project `.venv`) from the run JSONs in `results/week4_rebuild/` and the saved per-row ranks; top-10 lists of the SMLP4Rec models from `scripts/final_export_topk_smlp4rec.py` (smlp4rec venv, loads checkpoints only, no training).
- Outputs: `results/week5_final/final_results.json`, `final_benchmark.csv` (one row per model x training set x row set x seed-or-mean, `source` column = input file and key), `bootstrap_ci.csv`, `slices_*.csv`, `beyond_accuracy_*.csv`, `sensitivity_logreg_nb07.csv`, `final_tables.md`.
- **Every table below is pasted verbatim from `results/week5_final/final_tables.md`** (the script's printed output). Every headline number read from a JSON was re-derived from the per-row ranks and asserted equal by the script.
- Dataset and rows: Expedia Hotel Recommendations, bookings only, burst repeats collapsed; item = `hotel_cluster` (100 anonymous clusters, a proxy for room category / package). Global temporal 80/10/10 split over next-booking targets; test = all events n = 263,331, cold (L = 0, first booking) n = 49,462, warm (L >= 1) n = 213,869. Full ranking over 100 clusters.

## 1. Models and training sets

| Model key | Model | Training set | Stopping / selection | Seed |
|---|---|---|---|---|
| `plain_warm` | SMLP4Rec plain (original code, history only), nb01 | **warm users only** (L >= 1 next-booking targets of the train window) | fixed 3 epochs | 2022 |
| `plain_all` | SMLP4Rec plain (original code, history only), 01e | all events: warm targets + first bookings (L = 0, empty history) of the train window, identical to 01d | early stopping on valid NDCG@10 (best epoch 2) | 2022 |
| `query_embedding` | SMLP4Rec + query embedding, 01d | all events (as above) | early stopping (best epoch 8) | 2022 |
| `main` | SMLP4Rec + query embedding + sameDest (locked main model), 01d | all events; sameDest weights tuned on valid (alpha 0.2, w_s 0 for a new destination, 0.75 for a known one) | early stopping (best epoch 8) | 2022 |
| `itemknn` | ItemKNN (implicit cosine, K = 50, history only), 07b | all events | K on valid NDCG@10 | deterministic |
| `logreg` | Multinomial logistic regression (destination + trip context, no history), 07b | all events | C on valid, one SAGA pass at a time with early stopping (C = 1.0, pass 8) | 2022 |

Rows that a model cannot score are filled with a stated fallback, never left blank (column `scored_by`):
- `plain_warm` on cold rows: **not scored by the model** (it never saw an empty history). The cold and all-events numbers in the source JSON use global popularity for cold rows; they are shown with that label.
- `itemknn` on cold rows: no history, so first-booking popularity (07b design).
- `main` on cold rows: identical to `query_embedding` (no history, so the sameDest term has no input; its weight for a new destination is 0).
- `plain_all` on cold rows: the model scores them, but an empty history gives one constant ranking for every cold row (a learned first-booking popularity).

**Which original-code row is the fair comparison.** The like-for-like original-code row is `plain_all` (01e): same training set (warm + L = 0 rows), same early-stopping rule, same rows and scoring code as the main model, and it scores cold rows with the model itself. `plain_warm` is the original training recipe (warm users only, 3 epochs) and is reported, labelled, as the code as first run. On warm rows the two are statistically indistinguishable (bootstrap below: `plain_all` minus `plain_warm` NDCG@5 +0.0003, 95 % CI [-0.0002, +0.0007]), so the choice changes no conclusion; it matters only for the cold and all-events rows, where `plain_warm` has no model score.

## 2. Headline tables

#### Headline, test, row set = all (Expedia, seed 2022)

| model (training set) | n | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | scored_by |
|---|---|---|---|---|---|---|---|---|
| SMLP4Rec plain (original code) [warm rows only (L>=1), 3 epochs fixed, nb01] | 263331 | 0.2856 | 0.4105 | 0.5734 | 0.2000 | 0.2401 | 0.2811 | model on warm rows + global-popularity fallback on cold rows |
| SMLP4Rec plain (original code) [all events (warm + L=0), early stopping, 01e] | 263331 | 0.2856 | 0.4110 | 0.5736 | 0.2001 | 0.2405 | 0.2814 | model |
| SMLP4Rec + query embedding [all events (warm + L=0), early stopping, 01d] | 263331 | 0.5831 | 0.7613 | 0.9050 | 0.4122 | 0.4700 | 0.5066 | model |
| SMLP4Rec + query embedding + sameDest (main) [all events (warm + L=0), early stopping, 01d] | 263331 | 0.5944 | 0.7664 | 0.9066 | 0.4295 | 0.4853 | 0.5210 | model |
| ItemKNN (K=50) [all events (warm + L=0), K on valid, 07b] | 263331 | 0.2713 | 0.3989 | 0.5626 | 0.1850 | 0.2260 | 0.2671 | model on warm rows + first-booking popularity on cold rows |
| Logistic regression (C=1.0) [all events (warm + L=0), early stopping, 07b] | 263331 | 0.4948 | 0.6869 | 0.8560 | 0.3339 | 0.3961 | 0.4391 | model |


#### Headline, test, row set = cold (Expedia, seed 2022)

| model (training set) | n | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | scored_by |
|---|---|---|---|---|---|---|---|---|
| SMLP4Rec plain (original code) [warm rows only (L>=1), 3 epochs fixed, nb01] | 49462 | 0.1318 | 0.2270 | 0.3589 | 0.0814 | 0.1115 | 0.1444 | NOT SCORED by the model (never trained on an empty history); global-popularity fallback |
| SMLP4Rec plain (original code) [all events (warm + L=0), early stopping, 01e] | 49462 | 0.1307 | 0.2288 | 0.3599 | 0.0809 | 0.1124 | 0.1454 | model (cold rows: one constant ranking, empty history) |
| SMLP4Rec + query embedding [all events (warm + L=0), early stopping, 01d] | 49462 | 0.5252 | 0.7137 | 0.8807 | 0.3578 | 0.4189 | 0.4614 | model |
| SMLP4Rec + query embedding + sameDest (main) [all events (warm + L=0), early stopping, 01d] | 49462 | 0.5252 | 0.7137 | 0.8807 | 0.3578 | 0.4189 | 0.4614 | identical to query embedding: no history, sameDest has no input (weight 0 for a new destination) |
| ItemKNN (K=50) [all events (warm + L=0), K on valid, 07b] | 49462 | 0.1318 | 0.2288 | 0.3590 | 0.0814 | 0.1123 | 0.1449 | NOT SCORED by the model (no history); first-booking popularity fallback |
| Logistic regression (C=1.0) [all events (warm + L=0), early stopping, 07b] | 49462 | 0.4857 | 0.6747 | 0.8482 | 0.3275 | 0.3887 | 0.4328 | model |


#### Headline, test, row set = warm (Expedia, seed 2022)

| model (training set) | n | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | scored_by |
|---|---|---|---|---|---|---|---|---|
| SMLP4Rec plain (original code) [warm rows only (L>=1), 3 epochs fixed, nb01] | 213869 | 0.3212 | 0.4530 | 0.6230 | 0.2274 | 0.2698 | 0.3127 | model |
| SMLP4Rec plain (original code) [all events (warm + L=0), early stopping, 01e] | 213869 | 0.3214 | 0.4532 | 0.6230 | 0.2277 | 0.2701 | 0.3129 | model |
| SMLP4Rec + query embedding [all events (warm + L=0), early stopping, 01d] | 213869 | 0.5965 | 0.7723 | 0.9107 | 0.4248 | 0.4818 | 0.5171 | model |
| SMLP4Rec + query embedding + sameDest (main) [all events (warm + L=0), early stopping, 01d] | 213869 | 0.6104 | 0.7786 | 0.9125 | 0.4461 | 0.5007 | 0.5348 | model |
| ItemKNN (K=50) [all events (warm + L=0), K on valid, 07b] | 213869 | 0.3036 | 0.4383 | 0.6097 | 0.2089 | 0.2522 | 0.2954 | model |
| Logistic regression (C=1.0) [all events (warm + L=0), early stopping, 07b] | 213869 | 0.4969 | 0.6897 | 0.8578 | 0.3354 | 0.3978 | 0.4406 | model |


## 3. Seeds

The early-stopping runs above (01d, 01e, 07b) are **seed 2022 only**; seeds 2023 and 2024 were not rerun with early stopping. The only seed spread available for the main model comes from the 3-epoch 01d runs (same training set, fixed 3 epochs). It is shown as an indication of seed noise, not as the error bar of the headline run. Only Recall@5 and NDCG@5 were aggregated across seeds (`seeds_summary.json`). ItemKNN is deterministic; the logistic regression and both plain runs have one seed.

#### Seed variation: 01d 3-epoch runs, seeds 2022/2023/2024 (mean, sample std)

The early-stopping runs in the headline are seed 2022 only; no ES seed runs exist.

| model | row_set | recall@5 | recall@5_std | ndcg@5 | ndcg@5_std | recall@5_values | ndcg@5_values |
|---|---|---|---|---|---|---|---|
| SMLP4Rec + query embedding | all | 0.5737 | 0.0025 | 0.4044 | 0.0016 | 0.5748 0.5708 0.5755 | 0.4046 0.4028 0.4059 |
| SMLP4Rec + query embedding | cold | 0.5109 | 0.0011 | 0.3488 | 0.0017 | 0.5104 0.5102 0.5122 | 0.3471 0.3487 0.3505 |
| SMLP4Rec + query embedding | warm | 0.5882 | 0.0029 | 0.4173 | 0.0018 | 0.5897 0.5849 0.5901 | 0.4179 0.4153 0.4188 |
| SMLP4Rec + query embedding + sameDest (main) | all | 0.5857 | 0.0022 | 0.4231 | 0.0016 | 0.5866 0.5832 0.5873 | 0.4233 0.4214 0.4245 |
| SMLP4Rec + query embedding + sameDest (main) | cold | 0.5109 | 0.0011 | 0.3488 | 0.0017 | 0.5104 0.5102 0.5122 | 0.3471 0.3487 0.3505 |
| SMLP4Rec + query embedding + sameDest (main) | warm | 0.6030 | 0.0025 | 0.4403 | 0.0018 | 0.6042 0.6001 0.6047 | 0.4409 0.4382 0.4417 |


## 4. Paired bootstrap (headline comparisons)

1000 paired resamples of test rows, `nbp.eval.bootstrap.paired_bootstrap` (seed 0), K = 5 and 10. On cold rows the main model is the query embedding and ItemKNN is first-booking popularity. The last four rows repeat the sameDest gain after removing targets that re-book the same cluster at the same destination within 1 h / 1 d (definition of `scripts/expedia_sensitivity.py`; it conditions on the label, so it is a lower bound).

#### Paired bootstrap, test (difference a minus b, 95% CI)

| row_set | n | comparison | metric | diff | ci95 | note |
|---|---|---|---|---|---|---|
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@5 | +0.0996 | [+0.0978, +0.1015] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@5 | +0.0956 | [+0.0942, +0.0969] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@10 | +0.0796 | [+0.0781, +0.0811] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@10 | +0.0892 | [+0.0880, +0.0903] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@5 | +0.3230 | [+0.3210, +0.3251] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@5 | +0.2445 | [+0.2431, +0.2461] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@10 | +0.3675 | [+0.3652, +0.3697] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@10 | +0.2593 | [+0.2581, +0.2607] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@5 | +0.3088 | [+0.3066, +0.3108] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@5 | +0.2294 | [+0.2279, +0.2309] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@10 | +0.3554 | [+0.3533, +0.3575] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@10 | +0.2448 | [+0.2435, +0.2462] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | recall@5 | +0.0113 | [+0.0108, +0.0118] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | ndcg@5 | +0.0173 | [+0.0168, +0.0178] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | recall@10 | +0.0052 | [+0.0048, +0.0055] |  |
| all | 263331 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | ndcg@10 | +0.0153 | [+0.0148, +0.0157] |  |
| all | 263331 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | recall@5 | +0.2975 | [+0.2954, +0.2996] |  |
| all | 263331 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | ndcg@5 | +0.2121 | [+0.2106, +0.2136] |  |
| all | 263331 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | recall@10 | +0.3503 | [+0.3480, +0.3524] |  |
| all | 263331 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | ndcg@10 | +0.2295 | [+0.2282, +0.2310] |  |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@5 | +0.0394 | [+0.0358, +0.0433] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@5 | +0.0303 | [+0.0279, +0.0327] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@10 | +0.0390 | [+0.0357, +0.0422] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@10 | +0.0302 | [+0.0279, +0.0322] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@5 | +0.3934 | [+0.3886, +0.3986] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@5 | +0.2764 | [+0.2727, +0.2803] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@10 | +0.4849 | [+0.4797, +0.4901] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@10 | +0.3066 | [+0.3031, +0.3101] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@5 | +0.3945 | [+0.3895, +0.3994] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@5 | +0.2769 | [+0.2730, +0.2807] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@10 | +0.4849 | [+0.4797, +0.4901] | cold: main = query embedding; ItemKNN = first-booking popularity |
| cold | 49462 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@10 | +0.3065 | [+0.3030, +0.3100] | cold: main = query embedding; ItemKNN = first-booking popularity |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@5 | +0.1135 | [+0.1115, +0.1153] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@5 | +0.1107 | [+0.1091, +0.1122] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | recall@10 | +0.0890 | [+0.0870, +0.0907] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus Logistic regression (C=1.0) [logreg] | ndcg@10 | +0.1028 | [+0.1014, +0.1041] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@5 | +0.3068 | [+0.3046, +0.3091] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@5 | +0.2372 | [+0.2356, +0.2388] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | recall@10 | +0.3404 | [+0.3382, +0.3428] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus ItemKNN (K=50) [itemknn] | ndcg@10 | +0.2484 | [+0.2471, +0.2500] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@5 | +0.2889 | [+0.2868, +0.2912] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@5 | +0.2184 | [+0.2169, +0.2200] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | recall@10 | +0.3255 | [+0.3232, +0.3278] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_all] | ndcg@10 | +0.2306 | [+0.2292, +0.2321] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_warm] | recall@5 | +0.2892 | [+0.2870, +0.2914] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_warm] | ndcg@5 | +0.2187 | [+0.2172, +0.2203] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_warm] | recall@10 | +0.3257 | [+0.3235, +0.3279] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec plain (original code) [plain_warm] | ndcg@10 | +0.2309 | [+0.2295, +0.2323] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | recall@5 | +0.0139 | [+0.0132, +0.0145] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | ndcg@5 | +0.0213 | [+0.0206, +0.0219] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | recall@10 | +0.0064 | [+0.0060, +0.0068] |  |
| warm | 213869 | SMLP4Rec + query embedding + sameDest (main) [main] minus SMLP4Rec + query embedding [query_embedding] | ndcg@10 | +0.0188 | [+0.0183, +0.0194] |  |
| warm | 213869 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | recall@5 | +0.2750 | [+0.2728, +0.2774] |  |
| warm | 213869 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | ndcg@5 | +0.1971 | [+0.1955, +0.1988] |  |
| warm | 213869 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | recall@10 | +0.3191 | [+0.3168, +0.3214] |  |
| warm | 213869 | SMLP4Rec + query embedding [query_embedding] minus SMLP4Rec plain (original code) [plain_all] | ndcg@10 | +0.2118 | [+0.2103, +0.2133] |  |
| warm | 213869 | SMLP4Rec plain (original code) [plain_all] minus SMLP4Rec plain (original code) [plain_warm] | recall@5 | +0.0003 | [-0.0007, +0.0011] |  |
| warm | 213869 | SMLP4Rec plain (original code) [plain_all] minus SMLP4Rec plain (original code) [plain_warm] | ndcg@5 | +0.0003 | [-0.0002, +0.0007] |  |
| warm | 213869 | SMLP4Rec plain (original code) [plain_all] minus SMLP4Rec plain (original code) [plain_warm] | recall@10 | +0.0002 | [-0.0007, +0.0012] |  |
| warm | 213869 | SMLP4Rec plain (original code) [plain_all] minus SMLP4Rec plain (original code) [plain_warm] | ndcg@10 | +0.0003 | [-0.0001, +0.0007] |  |
| warm excluding re-books within_1h | 206975 | sameDest gain: main minus query embedding | recall@5 | +0.0097 | [+0.0091, +0.0102] | rows removed: 6894 |
| warm excluding re-books within_1h | 206975 | sameDest gain: main minus query embedding | ndcg@5 | +0.0139 | [+0.0133, +0.0145] | rows removed: 6894 |
| warm excluding re-books within_1d | 205031 | sameDest gain: main minus query embedding | recall@5 | +0.0085 | [+0.0080, +0.0091] | rows removed: 8838 |
| warm excluding re-books within_1d | 205031 | sameDest gain: main minus query embedding | ndcg@5 | +0.0119 | [+0.0114, +0.0124] | rows removed: 8838 |


**Baseline strength check.** The 07b logistic regression (the like-for-like run, trained on all events with early stopping) is weaker than the logistic regression of notebook 07 (C = 10, 3 SAGA passes, trained on the bookings before a time cut at the end of the train window). The table below scores the notebook-07 model on the same rows. The main model still wins on all three row sets, but the warm NDCG@5 gap is +0.0893 against notebook 07 versus +0.1107 against 07b. The report should either cite the smaller gap as the conservative number or explain the difference (07b restarts SAGA's gradient memory each pass and selected C = 1.0).

#### Sensitivity: logistic regression of notebook 07 (C=10, 3 SAGA passes) on the same test rows

Not the like-for-like baseline (different C / pass selection); shown because it is stronger than the 07b run.

| row_set | n | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 | main_minus_lr07_recall@5 | ci95_recall@5 | main_minus_lr07_ndcg@5 | ci95_ndcg@5 | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 263331 | 0.5198 | 0.7089 | 0.8724 | 0.3554 | 0.4166 | 0.4581 | 0.0745 | [0.0728, 0.0763] | 0.0742 | [0.0729, 0.0754] | basic_baselines.json: results.test['Logistic regression (C=10.0, destination + context, no history)']['all events'] |
| warm | 213869 | 0.5222 | 0.7115 | 0.8739 | 0.3568 | 0.4181 | 0.4594 | 0.0882 | [0.0862, 0.0901] | 0.0893 | [0.0877, 0.0909] | basic_baselines.json: results.test['Logistic regression (C=10.0, destination + context, no history)']['warm (all L>=1 targets)'] |
| cold | 49462 | 0.5096 | 0.6979 | 0.8659 | 0.3490 | 0.4099 | 0.4526 | 0.0155 | [0.0121, 0.0188] | 0.0088 | [0.0068, 0.011] | recomputed from basic_baselines_ranks_test.parquet |


## 5. RQ1 slices (same rows and per-row ranks for every model)

`small_bucket` = fewer than 1,000 rows. The CSV files also give Recall / NDCG at 10 and 20 and a binomial 95 % half-width for Recall@5 per bucket. `plain_warm` appears only in buckets without cold rows.

### 5.1 History length L

#### Slice: history length L - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | L=0 (cold) | 49462 | False |  | 0.0809 | 0.3578 | 0.3578 | 0.0814 | 0.3275 |
| all | L=1 | 39778 | False | 0.1798 | 0.1792 | 0.3934 | 0.4011 | 0.1719 | 0.3285 |
| all | L=2-4 | 67266 | False | 0.2114 | 0.2112 | 0.4135 | 0.4276 | 0.1996 | 0.3326 |
| all | L=5-9 | 46340 | False | 0.2391 | 0.2397 | 0.4340 | 0.4562 | 0.2149 | 0.3356 |
| all | L>=10 | 60485 | False | 0.2675 | 0.2688 | 0.4511 | 0.4885 | 0.2390 | 0.3428 |


#### Slice: history length L - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | L=0 (cold) | 49462 | False |  | 0.1307 | 0.5252 | 0.5252 | 0.1318 | 0.4857 |
| all | L=1 | 39778 | False | 0.2528 | 0.2516 | 0.5625 | 0.5670 | 0.2377 | 0.4919 |
| all | L=2-4 | 67266 | False | 0.3004 | 0.3009 | 0.5854 | 0.5933 | 0.2941 | 0.4927 |
| all | L=5-9 | 46340 | False | 0.3395 | 0.3397 | 0.6062 | 0.6214 | 0.3144 | 0.4973 |
| all | L>=10 | 60485 | False | 0.3753 | 0.3762 | 0.6237 | 0.6494 | 0.3491 | 0.5046 |


### 5.2 Known vs new destination (query destination among the last 20 bookings, the sameDest definition)

#### Slice: query destination in last-20 history - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | cold (no history) | 49462 | False |  | 0.0809 | 0.3578 | 0.3578 | 0.0814 | 0.3275 |
| all | known destination | 60586 | False | 0.4405 | 0.4408 | 0.5166 | 0.5917 | 0.4088 | 0.3345 |
| all | new destination | 153283 | False | 0.1432 | 0.1435 | 0.3885 | 0.3885 | 0.1299 | 0.3357 |


#### Slice: query destination in last-20 history - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | cold (no history) | 49462 | False |  | 0.1307 | 0.5252 | 0.5252 | 0.1318 | 0.4857 |
| all | known destination | 60586 | False | 0.5617 | 0.5614 | 0.6783 | 0.7273 | 0.5501 | 0.4957 |
| all | new destination | 153283 | False | 0.2261 | 0.2266 | 0.5641 | 0.5641 | 0.2061 | 0.4974 |


### 5.3 Season (check-in month and meteorological season)

**The season slice is degenerate on the test window.** Test bookings are made 2014-11-19 to 2014-12-31, so check-in months are concentrated in December and January (EDA table in section 8: test share 0.5156 December, 0.2114 January; months 4-10 below 0.03 each). Months 6-10 have few rows (October has 925 rows overall and is flagged as small). The slice shows that the model ranking is the same in every month; it cannot show a seasonal effect, because every bucket is booked in the same 6 weeks. Answering the season part of RQ1 properly would need a rolling-origin split or pooling valid + test (open item, user decision).

#### Slice: check-in month - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | 01 | 56096 | False |  | 0.1907 | 0.4084 | 0.4271 | 0.1752 | 0.3323 |
| all | 02 | 18078 | False |  | 0.1698 | 0.4023 | 0.4185 | 0.1576 | 0.3261 |
| all | 03 | 10134 | False |  | 0.1679 | 0.4050 | 0.4202 | 0.1534 | 0.3259 |
| all | 04 | 6688 | False |  | 0.1614 | 0.4022 | 0.4191 | 0.1467 | 0.3199 |
| all | 05 | 5566 | False |  | 0.1727 | 0.3891 | 0.4074 | 0.1559 | 0.3091 |
| all | 06 | 3670 | False |  | 0.1619 | 0.3954 | 0.4123 | 0.1443 | 0.3155 |
| all | 07 | 3134 | False |  | 0.1644 | 0.3887 | 0.4074 | 0.1489 | 0.3225 |
| all | 08 | 2021 | False |  | 0.1612 | 0.3956 | 0.4120 | 0.1403 | 0.3092 |
| all | 09 | 1629 | False |  | 0.1640 | 0.3864 | 0.4056 | 0.1458 | 0.2997 |
| all | 10 | 925 | True |  | 0.1939 | 0.3810 | 0.4041 | 0.1729 | 0.3116 |
| all | 11 | 21578 | False |  | 0.2297 | 0.4290 | 0.4479 | 0.2140 | 0.3459 |
| all | 12 | 133812 | False |  | 0.2119 | 0.4163 | 0.4329 | 0.1968 | 0.3378 |
| warm | 01 | 45209 | False | 0.2198 | 0.2206 | 0.4214 | 0.4447 | 0.2013 | 0.3333 |
| warm | 02 | 13988 | False | 0.2039 | 0.2054 | 0.4134 | 0.4344 | 0.1891 | 0.3229 |
| warm | 03 | 7819 | False | 0.2035 | 0.2025 | 0.4151 | 0.4347 | 0.1837 | 0.3202 |
| warm | 04 | 5107 | False | 0.1955 | 0.1966 | 0.4084 | 0.4305 | 0.1767 | 0.3070 |
| warm | 05 | 4366 | False | 0.2043 | 0.2058 | 0.3974 | 0.4207 | 0.1838 | 0.3053 |
| warm | 06 | 2900 | False | 0.1892 | 0.1898 | 0.4077 | 0.4291 | 0.1679 | 0.3159 |
| warm | 07 | 2544 | False | 0.1856 | 0.1887 | 0.3977 | 0.4207 | 0.1683 | 0.3241 |
| warm | 08 | 1655 | False | 0.1863 | 0.1834 | 0.4034 | 0.4233 | 0.1578 | 0.3090 |
| warm | 09 | 1355 | False | 0.1893 | 0.1882 | 0.3890 | 0.4121 | 0.1655 | 0.2997 |
| warm | 10 | 750 | True | 0.2207 | 0.2274 | 0.3933 | 0.4217 | 0.2004 | 0.3102 |
| warm | 11 | 17903 | False | 0.2525 | 0.2532 | 0.4409 | 0.4638 | 0.2337 | 0.3470 |
| warm | 12 | 110273 | False | 0.2366 | 0.2365 | 0.4296 | 0.4498 | 0.2182 | 0.3413 |


#### Slice: check-in month - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | 01 | 56096 | False |  | 0.2714 | 0.5771 | 0.5902 | 0.2569 | 0.4907 |
| all | 02 | 18078 | False |  | 0.2417 | 0.5711 | 0.5815 | 0.2315 | 0.4845 |
| all | 03 | 10134 | False |  | 0.2386 | 0.5727 | 0.5819 | 0.2255 | 0.4833 |
| all | 04 | 6688 | False |  | 0.2315 | 0.5697 | 0.5828 | 0.2186 | 0.4714 |
| all | 05 | 5566 | False |  | 0.2443 | 0.5593 | 0.5699 | 0.2328 | 0.4592 |
| all | 06 | 3670 | False |  | 0.2357 | 0.5591 | 0.5711 | 0.2147 | 0.4722 |
| all | 07 | 3134 | False |  | 0.2390 | 0.5546 | 0.5683 | 0.2265 | 0.4872 |
| all | 08 | 2021 | False |  | 0.2350 | 0.5720 | 0.5804 | 0.2157 | 0.4592 |
| all | 09 | 1629 | False |  | 0.2388 | 0.5500 | 0.5629 | 0.2241 | 0.4543 |
| all | 10 | 925 | True |  | 0.2724 | 0.5438 | 0.5578 | 0.2432 | 0.4605 |
| all | 11 | 21578 | False |  | 0.3255 | 0.6022 | 0.6144 | 0.3113 | 0.5123 |
| all | 12 | 133812 | False |  | 0.3029 | 0.5887 | 0.5992 | 0.2882 | 0.5007 |
| warm | 01 | 45209 | False | 0.3091 | 0.3100 | 0.5912 | 0.6074 | 0.2919 | 0.4921 |
| warm | 02 | 13988 | False | 0.2842 | 0.2874 | 0.5826 | 0.5961 | 0.2731 | 0.4788 |
| warm | 03 | 7819 | False | 0.2848 | 0.2835 | 0.5837 | 0.5956 | 0.2664 | 0.4799 |
| warm | 04 | 5107 | False | 0.2747 | 0.2775 | 0.5741 | 0.5913 | 0.2589 | 0.4553 |
| warm | 05 | 4366 | False | 0.2852 | 0.2865 | 0.5671 | 0.5806 | 0.2703 | 0.4553 |
| warm | 06 | 2900 | False | 0.2717 | 0.2721 | 0.5741 | 0.5893 | 0.2462 | 0.4759 |
| warm | 07 | 2544 | False | 0.2669 | 0.2728 | 0.5617 | 0.5786 | 0.2543 | 0.4866 |
| warm | 08 | 1655 | False | 0.2671 | 0.2647 | 0.5789 | 0.5891 | 0.2411 | 0.4556 |
| warm | 09 | 1355 | False | 0.2679 | 0.2716 | 0.5506 | 0.5661 | 0.2517 | 0.4531 |
| warm | 10 | 750 | True | 0.3067 | 0.3160 | 0.5573 | 0.5747 | 0.2773 | 0.4680 |
| warm | 11 | 17903 | False | 0.3556 | 0.3561 | 0.6131 | 0.6278 | 0.3374 | 0.5148 |
| warm | 12 | 110273 | False | 0.3355 | 0.3348 | 0.6033 | 0.6160 | 0.3172 | 0.5052 |


#### Slice: check-in season - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | DJF | 207986 | False |  | 0.2025 | 0.4129 | 0.4301 | 0.1876 | 0.3353 |
| all | JJA | 8825 | False |  | 0.1626 | 0.3931 | 0.4105 | 0.1450 | 0.3166 |
| all | MAM | 22388 | False |  | 0.1671 | 0.4002 | 0.4167 | 0.1520 | 0.3199 |
| all | SON | 24132 | False |  | 0.2239 | 0.4243 | 0.4434 | 0.2078 | 0.3415 |
| warm | DJF | 169470 | False | 0.2294 | 0.2297 | 0.4261 | 0.4472 | 0.2113 | 0.3376 |
| warm | JJA | 7099 | False | 0.1872 | 0.1879 | 0.4031 | 0.4247 | 0.1657 | 0.3172 |
| warm | MAM | 17292 | False | 0.2013 | 0.2016 | 0.4086 | 0.4299 | 0.1817 | 0.3125 |
| warm | SON | 20008 | False | 0.2471 | 0.2478 | 0.4356 | 0.4587 | 0.2278 | 0.3425 |


#### Slice: check-in season - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | DJF | 207986 | False |  | 0.2891 | 0.5841 | 0.5953 | 0.2748 | 0.4966 |
| all | JJA | 8825 | False |  | 0.2367 | 0.5605 | 0.5722 | 0.2192 | 0.4746 |
| all | MAM | 22388 | False |  | 0.2379 | 0.5685 | 0.5792 | 0.2253 | 0.4738 |
| all | SON | 24132 | False |  | 0.3176 | 0.5964 | 0.6087 | 0.3028 | 0.5064 |
| warm | DJF | 169470 | False | 0.3242 | 0.3243 | 0.5984 | 0.6121 | 0.3068 | 0.4995 |
| warm | JJA | 7099 | False | 0.2689 | 0.2706 | 0.5708 | 0.5854 | 0.2479 | 0.4750 |
| warm | MAM | 17292 | False | 0.2819 | 0.2825 | 0.5767 | 0.5906 | 0.2652 | 0.4664 |
| warm | SON | 20008 | False | 0.3478 | 0.3489 | 0.6068 | 0.6217 | 0.3294 | 0.5089 |


### 5.4 Trip context: party type and package

Party type = family (children > 0), solo (1 adult), couple (2), group (3+), unknown (`clean.party_type`). The `unknown` bucket is small.

#### Slice: party type - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | couple | 131067 | False |  | 0.1863 | 0.4063 | 0.4216 | 0.1719 | 0.3354 |
| all | family | 36389 | False |  | 0.1769 | 0.4073 | 0.4188 | 0.1625 | 0.3400 |
| all | group | 24337 | False |  | 0.1832 | 0.4000 | 0.4157 | 0.1660 | 0.3260 |
| all | solo | 71293 | False |  | 0.2434 | 0.4299 | 0.4544 | 0.2272 | 0.3307 |
| all | unknown | 245 | True |  | 0.1513 | 0.3766 | 0.4027 | 0.1498 | 0.3396 |
| warm | couple | 104026 | False | 0.2144 | 0.2152 | 0.4189 | 0.4382 | 0.1967 | 0.3369 |
| warm | family | 28249 | False | 0.1985 | 0.1984 | 0.4174 | 0.4322 | 0.1796 | 0.3403 |
| warm | group | 19555 | False | 0.2084 | 0.2084 | 0.4073 | 0.4269 | 0.1871 | 0.3240 |
| warm | solo | 61881 | False | 0.2685 | 0.2682 | 0.4437 | 0.4718 | 0.2498 | 0.3341 |
| warm | unknown | 158 | True | 0.2204 | 0.2167 | 0.3922 | 0.4327 | 0.2192 | 0.3359 |


#### Slice: party type - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | couple | 131067 | False |  | 0.2675 | 0.5789 | 0.5888 | 0.2539 | 0.4971 |
| all | family | 36389 | False |  | 0.2600 | 0.5770 | 0.5845 | 0.2449 | 0.5018 |
| all | group | 24337 | False |  | 0.2672 | 0.5671 | 0.5778 | 0.2474 | 0.4821 |
| all | solo | 71293 | False |  | 0.3386 | 0.5995 | 0.6153 | 0.3251 | 0.4913 |
| all | unknown | 245 | True |  | 0.2082 | 0.5347 | 0.5633 | 0.2245 | 0.5020 |
| warm | couple | 104026 | False | 0.3043 | 0.3053 | 0.5923 | 0.6048 | 0.2875 | 0.4990 |
| warm | family | 28249 | False | 0.2888 | 0.2885 | 0.5887 | 0.5983 | 0.2684 | 0.5026 |
| warm | group | 19555 | False | 0.3013 | 0.3015 | 0.5749 | 0.5883 | 0.2770 | 0.4805 |
| warm | solo | 61881 | False | 0.3706 | 0.3700 | 0.6140 | 0.6322 | 0.3550 | 0.4958 |
| warm | unknown | 158 | True | 0.3101 | 0.2848 | 0.5506 | 0.5949 | 0.3228 | 0.5127 |


#### Slice: is_package - NDCG@5

| row_set | bucket | n | small_bucket | NDCG@5 plain_warm | NDCG@5 plain_all | NDCG@5 query_embedding | NDCG@5 main | NDCG@5 itemknn | NDCG@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | no package | 238003 | False |  | 0.2079 | 0.4120 | 0.4302 | 0.1916 | 0.3315 |
| all | package | 25328 | False |  | 0.1268 | 0.4146 | 0.4228 | 0.1227 | 0.3571 |
| warm | no package | 197340 | False | 0.2319 | 0.2321 | 0.4248 | 0.4469 | 0.2123 | 0.3340 |
| warm | package | 16529 | False | 0.1737 | 0.1748 | 0.4244 | 0.4370 | 0.1691 | 0.3525 |


#### Slice: is_package - Recall@5

| row_set | bucket | n | small_bucket | R@5 plain_warm | R@5 plain_all | R@5 query_embedding | R@5 main | R@5 itemknn | R@5 logreg |
|---|---|---|---|---|---|---|---|---|---|
| all | no package | 238003 | False |  | 0.2964 | 0.5826 | 0.5944 | 0.2811 | 0.4914 |
| all | package | 25328 | False |  | 0.1845 | 0.5873 | 0.5936 | 0.1793 | 0.5272 |
| warm | no package | 197340 | False | 0.3275 | 0.3277 | 0.5965 | 0.6108 | 0.3089 | 0.4948 |
| warm | package | 16529 | False | 0.2454 | 0.2468 | 0.5960 | 0.6057 | 0.2400 | 0.5225 |


## 6. RQ4 beyond-accuracy (top-10 lists)

Definitions (computed in `beyond_accuracy()`):
- **ILD@10**: mean over rows of the mean pairwise (1 - cosine) of the 10 recommended clusters; cosine between cluster columns of the training user x cluster booking-count matrix (the matrix ItemKNN is fitted on). Clusters are anonymous, so co-booking is the only similarity available. Values sit in a narrow band (most cluster pairs are rarely co-booked), so differences are small.
- **ARP@10**: mean training popularity share of the recommended clusters. `ARP_ratio` = ARP@10 / mean training share of the booked clusters (1 = recommends as popular as what people book; > 1 = popularity bias). `head10_slot_share` = share of top-10 slots taken by the 10 most popular training clusters; `head10_target_share` = share of booked clusters that are in that head.
- **Catalog coverage@10**: distinct clusters in any top-10 list. **exposure_gini**: Gini of how often each cluster appears in the top-10 lists (0 = equal exposure).
- **Coverage per destination**: destinations with >= 20 rows in the row set; distinct clusters recommended across their rows, distinct clusters booked there, and booked-cluster coverage (share of the clusters booked at the destination that appear in at least one top-10 list there).
- **Coverage per check-in month**: distinct clusters recommended and exposure Gini per check-in month (same degeneracy caveat as 5.3).

Global catalog coverage is 100/100 for every learned model on warm and all rows, as expected with 100 clusters. The informative views are exposure concentration and per-destination coverage.

#### Beyond accuracy, top-10, aggregate

| row_set | model_key | n | ILD@10 | ARP@10 | ARP_target | ARP_ratio | head10_slot_share | head10_target_share | catalog_coverage@10 | exposure_gini |
|---|---|---|---|---|---|---|---|---|---|---|
| all | plain_all | 263331 | 0.8719 | 0.0179 | 0.0142 | 1.2568 | 0.4691 | 0.2447 | 100 | 0.6208 |
| all | query_embedding | 263331 | 0.9008 | 0.0140 | 0.0142 | 0.9828 | 0.2286 | 0.2447 | 100 | 0.3637 |
| all | main | 263331 | 0.9009 | 0.0140 | 0.0142 | 0.9825 | 0.2284 | 0.2447 | 100 | 0.3630 |
| all | itemknn | 263331 | 0.8762 | 0.0171 | 0.0142 | 1.2039 | 0.4365 | 0.2447 | 100 | 0.5805 |
| all | logreg | 263331 | 0.9058 | 0.0144 | 0.0142 | 1.0132 | 0.2584 | 0.2447 | 100 | 0.4196 |
| warm | plain_warm | 213869 | 0.8731 | 0.0168 | 0.0143 | 1.1755 | 0.3632 | 0.2488 | 100 | 0.5641 |
| warm | plain_all | 213869 | 0.8731 | 0.0168 | 0.0143 | 1.1754 | 0.3694 | 0.2488 | 100 | 0.5623 |
| warm | query_embedding | 213869 | 0.8996 | 0.0140 | 0.0143 | 0.9779 | 0.2301 | 0.2488 | 100 | 0.3675 |
| warm | main | 213869 | 0.8998 | 0.0140 | 0.0143 | 0.9775 | 0.2298 | 0.2488 | 100 | 0.3667 |
| warm | itemknn | 213869 | 0.8783 | 0.0159 | 0.0143 | 1.1108 | 0.3293 | 0.2488 | 100 | 0.5143 |
| warm | logreg | 213869 | 0.9053 | 0.0145 | 0.0143 | 1.0101 | 0.2618 | 0.2488 | 100 | 0.4257 |
| cold | plain_all | 49462 | 0.8669 | 0.0223 | 0.0137 | 1.6249 | 0.9000 | 0.2270 | 10 | 0.9000 |
| cold | query_embedding | 49462 | 0.9058 | 0.0138 | 0.0137 | 1.0052 | 0.2220 | 0.2270 | 100 | 0.3540 |
| cold | main | 49462 | 0.9058 | 0.0138 | 0.0137 | 1.0052 | 0.2220 | 0.2270 | 100 | 0.3540 |
| cold | itemknn | 49462 | 0.8669 | 0.0223 | 0.0137 | 1.6249 | 0.9000 | 0.2270 | 10 | 0.9000 |
| cold | logreg | 49462 | 0.9081 | 0.0141 | 0.0137 | 1.0273 | 0.2437 | 0.2270 | 100 | 0.3971 |


#### Coverage per destination (destinations with >= 20 rows in the row set)

| row_set | model_key | destinations | rows_in_these_destinations | mean_distinct_recommended@10 | mean_distinct_booked | mean_booked_cluster_coverage@10 |
|---|---|---|---|---|---|---|
| all | plain_all | 1878 | 216738 | 65.5300 | 15.6600 | 0.9335 |
| all | query_embedding | 1878 | 216738 | 23.0200 | 15.6600 | 0.8985 |
| all | main | 1878 | 216738 | 23.6400 | 15.6600 | 0.9159 |
| all | itemknn | 1878 | 216738 | 70.9400 | 15.6600 | 0.9515 |
| all | logreg | 1878 | 216738 | 18.8700 | 15.6600 | 0.8094 |
| warm | plain_warm | 1606 | 170169 | 67.8500 | 15.9400 | 0.9471 |
| warm | plain_all | 1606 | 170169 | 68.0000 | 15.9400 | 0.9461 |
| warm | query_embedding | 1606 | 170169 | 23.6300 | 15.9400 | 0.9036 |
| warm | main | 1606 | 170169 | 24.3300 | 15.9400 | 0.9227 |
| warm | itemknn | 1606 | 170169 | 73.5600 | 15.9400 | 0.9624 |
| warm | logreg | 1606 | 170169 | 18.6300 | 15.9400 | 0.8052 |
| cold | plain_all | 416 | 31371 | 10.0000 | 18.5800 | 0.2044 |
| cold | query_embedding | 416 | 31371 | 18.2000 | 18.5800 | 0.7384 |
| cold | main | 416 | 31371 | 18.2000 | 18.5800 | 0.7384 |
| cold | itemknn | 416 | 31371 | 10.0000 | 18.5800 | 0.2044 |
| cold | logreg | 416 | 31371 | 20.0600 | 18.5800 | 0.7622 |


#### Coverage per check-in month: distinct clusters recommended in top-10

| row_set | checkin_month | n | distinct_booked | small_bucket | plain_warm | plain_all | query_embedding | main | itemknn | logreg |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 01 | 56096 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 02 | 18078 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 03 | 10134 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 04 | 6688 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 05 | 5566 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 06 | 3670 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 07 | 3134 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 08 | 2021 | 99 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 09 | 1629 | 97 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 10 | 925 | 96 | True |  | 98 | 99 | 99 | 98 | 99 |
| all | 11 | 21578 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| all | 12 | 133812 | 100 | False |  | 100 | 100 | 100 | 100 | 100 |
| cold | 01 | 10887 | 100 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 02 | 4090 | 100 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 03 | 2315 | 99 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 04 | 1581 | 99 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 05 | 1200 | 100 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 06 | 770 | 95 | True |  | 10 | 100 | 100 | 10 | 99 |
| cold | 07 | 590 | 95 | True |  | 10 | 100 | 100 | 10 | 100 |
| cold | 08 | 366 | 87 | True |  | 10 | 100 | 100 | 10 | 100 |
| cold | 09 | 274 | 79 | True |  | 10 | 99 | 99 | 10 | 99 |
| cold | 10 | 175 | 75 | True |  | 10 | 99 | 99 | 10 | 96 |
| cold | 11 | 3675 | 99 | False |  | 10 | 100 | 100 | 10 | 100 |
| cold | 12 | 23539 | 100 | False |  | 10 | 100 | 100 | 10 | 100 |
| warm | 01 | 45209 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 02 | 13988 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 03 | 7819 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 04 | 5107 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 05 | 4366 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 06 | 2900 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 07 | 2544 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 08 | 1655 | 99 | False | 100 | 100 | 99 | 99 | 100 | 99 |
| warm | 09 | 1355 | 96 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 10 | 750 | 93 | True | 98 | 98 | 99 | 99 | 98 | 99 |
| warm | 11 | 17903 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |
| warm | 12 | 110273 | 100 | False | 100 | 100 | 100 | 100 | 100 | 100 |


#### Coverage per check-in month: exposure Gini of top-10 slots

| row_set | checkin_month | n | distinct_booked | small_bucket | plain_warm | plain_all | query_embedding | main | itemknn | logreg |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 01 | 56096 | 100 | False |  | 0.6201 | 0.3443 | 0.3436 | 0.5791 | 0.3948 |
| all | 02 | 18078 | 100 | False |  | 0.6219 | 0.3199 | 0.3190 | 0.5815 | 0.3698 |
| all | 03 | 10134 | 100 | False |  | 0.6231 | 0.3194 | 0.3191 | 0.5806 | 0.3768 |
| all | 04 | 6688 | 100 | False |  | 0.6280 | 0.3299 | 0.3293 | 0.5877 | 0.3833 |
| all | 05 | 5566 | 100 | False |  | 0.6132 | 0.3607 | 0.3600 | 0.5781 | 0.4138 |
| all | 06 | 3670 | 100 | False |  | 0.6219 | 0.3706 | 0.3702 | 0.5785 | 0.4113 |
| all | 07 | 3134 | 100 | False |  | 0.6151 | 0.3795 | 0.3793 | 0.5663 | 0.4186 |
| all | 08 | 2021 | 99 | False |  | 0.6037 | 0.3738 | 0.3741 | 0.5619 | 0.4263 |
| all | 09 | 1629 | 97 | False |  | 0.6095 | 0.4140 | 0.4125 | 0.5628 | 0.4482 |
| all | 10 | 925 | 96 | True |  | 0.6115 | 0.3793 | 0.3786 | 0.5742 | 0.4357 |
| all | 11 | 21578 | 100 | False |  | 0.6247 | 0.4280 | 0.4274 | 0.5851 | 0.4823 |
| all | 12 | 133812 | 100 | False |  | 0.6256 | 0.4050 | 0.4044 | 0.5852 | 0.4663 |
| cold | 01 | 10887 | 100 | False |  | 0.9000 | 0.3358 | 0.3358 | 0.9000 | 0.3790 |
| cold | 02 | 4090 | 100 | False |  | 0.9000 | 0.3208 | 0.3208 | 0.9000 | 0.3620 |
| cold | 03 | 2315 | 99 | False |  | 0.9000 | 0.3163 | 0.3163 | 0.9000 | 0.3611 |
| cold | 04 | 1581 | 99 | False |  | 0.9000 | 0.3204 | 0.3204 | 0.9000 | 0.3701 |
| cold | 05 | 1200 | 100 | False |  | 0.9000 | 0.3415 | 0.3415 | 0.9000 | 0.3903 |
| cold | 06 | 770 | 95 | True |  | 0.9000 | 0.3611 | 0.3611 | 0.9000 | 0.3872 |
| cold | 07 | 590 | 95 | True |  | 0.9000 | 0.3755 | 0.3755 | 0.9000 | 0.3982 |
| cold | 08 | 366 | 87 | True |  | 0.9000 | 0.3736 | 0.3736 | 0.9000 | 0.4170 |
| cold | 09 | 274 | 79 | True |  | 0.9000 | 0.4173 | 0.4173 | 0.9000 | 0.4472 |
| cold | 10 | 175 | 75 | True |  | 0.9000 | 0.3709 | 0.3709 | 0.9000 | 0.4193 |
| cold | 11 | 3675 | 99 | False |  | 0.9000 | 0.4506 | 0.4506 | 0.9000 | 0.4833 |
| cold | 12 | 23539 | 100 | False |  | 0.9000 | 0.4058 | 0.4058 | 0.9000 | 0.4514 |
| warm | 01 | 45209 | 100 | False | 0.5625 | 0.5588 | 0.3483 | 0.3474 | 0.5099 | 0.3999 |
| warm | 02 | 13988 | 100 | False | 0.5518 | 0.5476 | 0.3224 | 0.3212 | 0.4984 | 0.3737 |
| warm | 03 | 7819 | 100 | False | 0.5533 | 0.5478 | 0.3242 | 0.3238 | 0.4950 | 0.3842 |
| warm | 04 | 5107 | 100 | False | 0.5557 | 0.5526 | 0.3365 | 0.3358 | 0.5031 | 0.3898 |
| warm | 05 | 4366 | 100 | False | 0.5469 | 0.5447 | 0.3693 | 0.3685 | 0.5030 | 0.4229 |
| warm | 06 | 2900 | 100 | False | 0.5607 | 0.5562 | 0.3756 | 0.3750 | 0.5036 | 0.4191 |
| warm | 07 | 2544 | 100 | False | 0.5591 | 0.5581 | 0.3835 | 0.3833 | 0.4999 | 0.4251 |
| warm | 08 | 1655 | 99 | False | 0.5499 | 0.5482 | 0.3781 | 0.3785 | 0.4985 | 0.4309 |
| warm | 09 | 1355 | 96 | False | 0.5596 | 0.5626 | 0.4165 | 0.4147 | 0.5083 | 0.4506 |
| warm | 10 | 750 | 93 | True | 0.5512 | 0.5515 | 0.3861 | 0.3851 | 0.5068 | 0.4439 |
| warm | 11 | 17903 | 100 | False | 0.5731 | 0.5743 | 0.4252 | 0.4245 | 0.5282 | 0.4832 |
| warm | 12 | 110273 | 100 | False | 0.5735 | 0.5728 | 0.4062 | 0.4055 | 0.5254 | 0.4702 |


Reading: the context-aware models (query embedding, main) recommend clusters about as popular as the booked ones (ARP_ratio 0.9775-1.0052 across row sets) and spread exposure most evenly (lowest exposure Gini). The history-only models (plain, ItemKNN) over-recommend the popular head. Per destination it is the other way round: the query-embedding models concentrate on about 23-24 distinct clusters per destination, against 65-74 for the history-only models (which ignore the destination). They keep booked-cluster coverage near the history-only level (0.9159 vs 0.9335 for `plain_all`, all rows), above the logistic regression (0.8094).

## 7. Existing results reused (01d early-stopping run)

#### Existing: per-field permutation on the query embedding, test warm (01d ES)

| field | recall@5 | ndcg@5 | ndcg@5_drop |
|---|---|---|---|
| dest | 0.1313 | 0.0857 | 0.3391 |
| dest_type | 0.5908 | 0.4204 | 0.0044 |
| ci_month | 0.5947 | 0.4235 | 0.0013 |
| lead | 0.5915 | 0.4210 | 0.0038 |
| stay | 0.5917 | 0.4207 | 0.0041 |
| adults | 0.5924 | 0.4211 | 0.0037 |
| children | 0.5944 | 0.4232 | 0.0016 |
| rooms | 0.5938 | 0.4226 | 0.0022 |
| package | 0.5953 | 0.4239 | 0.0009 |
| mobile | 0.5967 | 0.4249 | -0.0001 |
| channel | 0.5959 | 0.4242 | 0.0006 |
| site | 0.5955 | 0.4238 | 0.0010 |
| posa | 0.5959 | 0.4242 | 0.0006 |
| country | 0.5942 | 0.4229 | 0.0019 |


#### Existing: query / history ablation of the query embedding, test warm (01d ES)

| input | n | recall@5 | recall@10 | recall@20 | ndcg@5 | ndcg@10 | ndcg@20 |
|---|---|---|---|---|---|---|---|
| real | 213869 | 0.5965 | 0.7723 | 0.9107 | 0.4248 | 0.4818 | 0.5171 |
| query shuffled | 213869 | 0.1233 | 0.2077 | 0.3513 | 0.0807 | 0.1077 | 0.1437 |
| history masked | 213869 | 0.5312 | 0.7233 | 0.8860 | 0.3616 | 0.4239 | 0.4653 |
| both (shuffled + masked) | 213869 | 0.0761 | 0.1451 | 0.2731 | 0.0456 | 0.0677 | 0.0997 |


#### Existing: cold rows by destination vocabulary, query embedding (01d ES)

| slice | n | recall@5 | ndcg@5 |
|---|---|---|---|
| destination in query-embedding vocab | 48578 | 0.5301 | 0.3613 |
| destination out of vocab | 884 | 0.2523 | 0.1655 |


## 8. EDA context for the season slice

#### Existing (EDA): check-in month share by split

| month | train | valid | test |
|---|---|---|---|
| 1 | 0.0418 | 0.0550 | 0.2114 |
| 2 | 0.0479 | 0.0253 | 0.0654 |
| 3 | 0.0685 | 0.0180 | 0.0366 |
| 4 | 0.0775 | 0.0127 | 0.0239 |
| 5 | 0.0916 | 0.0119 | 0.0204 |
| 6 | 0.1008 | 0.0077 | 0.0136 |
| 7 | 0.1164 | 0.0067 | 0.0119 |
| 8 | 0.1337 | 0.0049 | 0.0077 |
| 9 | 0.1199 | 0.0034 | 0.0063 |
| 10 | 0.0989 | 0.2476 | 0.0035 |
| 11 | 0.0540 | 0.4397 | 0.0837 |
| 12 | 0.0491 | 0.1672 | 0.5156 |


## 9. RQ mapping

| RQ | Answered by | Status |
|---|---|---|
| RQ1: does a context-aware session model beat traditional CF on sparse, seasonal, trip-context-dependent travel data? | Headline tables (section 2), bootstrap (section 4), slices by L (5.1), known / new destination (5.2), trip context (5.4), existing ablations (section 7) | Answered for history length, destination novelty, party type, package and cold vs warm. **Seasonality is not answerable on this split** (5.3); the slice is reported with row counts only. |
| RQ1, sparsity / new users | Cold rows (L = 0) and L = 1 bucket | Answered: the query embedding scores first bookings, while history-only models fall back to a constant ranking. |
| RQ4: accuracy vs diversity / coverage | Section 6 (ILD, ARP, head share, exposure Gini, coverage per destination and per check-in month) plus the accuracy tables | Answered with the definitions above. ILD differences are small because clusters are anonymous (co-booking similarity only); per-season coverage inherits the season degeneracy. |
| RQ2 (food delivery), RQ3 (cross-sell) | none | Out of scope (CLAUDE.md section 3); no data in this project. |

Not answerable with this data: real room-category / package semantics (clusters are anonymous IDs, no price column), online / business effect (offline only), in-stay services (no public data), generalisation to other travel datasets (one dataset).

## 10. Inconsistencies found between sources

1. **`reports/benchmark_results.csv` is stale.** It contains no early-stopping run (the script prints `benchmark_results_csv_has_early_stopping_runs: false`). Its main-model rows are the 3-epoch 01c / 01d runs. Its logistic regression is notebook 07 (warm Recall@5 0.5222 vs 0.4969 for the final 07b run), and its ItemKNN is notebook 07 with K = 100 (0.3038 vs 0.3036). It should be rebuilt from `results/week5_final/final_benchmark.csv` or `scripts/build_benchmark_results.py` updated (not done here; the file is protocol-owned).
2. **`results/week4_rebuild/split_export_counts.json` and `data/processed/splits_csv/test.csv` (notebook 08 top-10 export) come from the 3-epoch 01d checkpoint, not the early-stopping final model.** Their test Recall@10 is 0.7722 warm / 0.7021 cold. The final model has 0.7786 / 0.7137, and 0.7021 equals the 3-epoch cold value in `smlprec_query_cold_L0_cold_train.json`. Cause: notebook 08 and `scripts/demo_ui.py` read `demo_ui.RUNS`, which points to `2026-10-09_..._query-1d-clean-cold`. The demo UI therefore also shows the 3-epoch model.
3. `smlprec_query_cold_L0_cold_train_es.json` field `run` says "checkpoint of notebook 01c", but its `checkpoint` path is the 01d early-stopping checkpoint (the numbers are 01d ES). Text label only.
4. The "SMLP4Rec plain" all-events row in `smlprec_query_cold_train_es_run.json` (`samedest_hybrid.benchmark_test`) is the warm-only plain model plus global popularity on cold rows, not a model score on cold rows. It is labelled as such here.
5. Two different popularity fallbacks are used for cold rows: global popularity in `smlprec_query_cold_L0_cold_train_es.json` (cold Recall@10 0.2270) and first-booking popularity in 07b (0.2288). Small, but the "popularity on cold rows" numbers differ between sources.
6. The 07b logistic regression is weaker than the notebook-07 one (section 4), so "trained on all events with early stopping" produced a weaker baseline than the earlier run. See the sensitivity table.
7. `docs/eval_protocol.md` says ItemKNN cold rows use global popularity (07b uses first-booking popularity) and still mentions a "3-epoch budget until stage C"; CLAUDE.md section 6 asks for >= 3 seeds of the main model, which the early-stopping run does not have.
8. Checked and consistent: `baselines_vs_final_es.json` equals the headline for every model, row set and metric (0 mismatches); recomputed ItemKNN and logistic-regression ranks equal the saved 07b ranks on every row; the checkpoint re-scoring reproduces the notebooks' saved ranks (1.0 agreement, and 0.999995 for main / nb01 plain, which is one warm row with a tie).

#### Check: reports/benchmark_results.csv vs final numbers

| benchmark_results.csv model | row_set | csv recall@5 | final recall@5 | csv source | final source |
|---|---|---|---|---|---|
| Logistic regression | warm | 0.5222 | 0.4969 | basic_baselines.json | baselines_es_run.json: results.test['Logistic regression (C=1.0, pass 8, destination + context, no history)']['warm'] |
| Logistic regression | cold | 0.5096 | 0.4857 | basic_baselines.json | baselines_es_run.json: results.test['Logistic regression (C=1.0, pass 8, destination + context, no history)']['cold (L=0)'] |
| Logistic regression | all | 0.5198 | 0.4948 | basic_baselines.json | baselines_es_run.json: results.test['Logistic regression (C=1.0, pass 8, destination + context, no history)']['all events'] |
| ItemKNN | warm | 0.3038 | 0.3036 | basic_baselines.json | baselines_es_run.json: results.test['ItemKNN (cosine, K=50, history only; cold: first-booking popularity)']['warm'] |
| ItemKNN | cold | 0.1318 | 0.1318 | basic_baselines.json | baselines_es_run.json: results.test['ItemKNN (cosine, K=50, history only; cold: first-booking popularity)']['cold (L=0)'] |
| ItemKNN | all | 0.2715 | 0.2713 | basic_baselines.json | baselines_es_run.json: results.test['ItemKNN (cosine, K=50, history only; cold: first-booking popularity)']['all events'] |
| SMLP4Rec plain (history only) | warm | 0.3212 | 0.3212 | smlprec_query_run.json | smlprec_plain_run.json: test |
| SMLP4Rec plain (history only) | all | 0.2856 | 0.2856 | smlprec_query_run.json | smlprec_query_cold_train_es_run.json: samedest_hybrid.benchmark_test.all_events['SMLP4Rec plain'] |


## 11. Caveats and open items

- `hotel_cluster` is a proxy for room category / package (anonymous IDs, no price column); one dataset (Expedia); offline evaluation only; no in-stay service data exists publicly; only two simple baselines.
- **Single seed for the early-stopping runs** (main model, 01e plain, 07b logistic regression). The seed spread shown is from the 3-epoch 01d runs. Open: rerun 01d (and ideally 01e) with early stopping for seeds 2023 / 2024, applying `min_delta` 7e-4 from the start.
- The early-stopping `min_delta` (7e-4) was set after seeing the valid curve of the first 01d pass (valid data only; test not used). See PROGRESS.md "01d with early stopping".
- **Season slice degenerate** on the test window (section 5.3). Options: rolling-origin split, or pooling valid + test for the slice (needs a protocol decision).
- Cold-row "first booking" is the first *observed* booking (data starts 2013-01-07, left-censored).
- The sameDest re-book exclusion conditions on the label (lower bound of the true long-term gain).
- The 07b logistic regression may be under-trained (C = 1.0 selected; SAGA memory restarts each pass); see the notebook-07 sensitivity row.
- Beyond-accuracy metrics use co-booking similarity because clusters are anonymous; ILD differences are small and should not be over-read.
- Per-destination coverage uses only destinations with >= 20 rows (1,878 destinations, 216,738 of 263,331 rows on all events).
- `reports/benchmark_results.csv`, the notebook-08 CSV export and the demo UI still show the 3-epoch model (section 10). Not changed here.
- Not computed: per-season coverage on a non-degenerate window; seed spread of the beyond-accuracy metrics.

## 12. Commands run (2026-10-11)

```
# top-20 lists and ranks of the SMLP4Rec checkpoints (no training; ~45 s)
C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/final_export_topk_smlp4rec.py

# lint / format
.venv/Scripts/ruff format scripts/final_results_summary.py scripts/final_export_topk_smlp4rec.py
.venv/Scripts/ruff check --fix scripts/final_results_summary.py scripts/final_export_topk_smlp4rec.py

# consolidated results (bootstrap included; ~100 s); output also printed to stdout
.venv/Scripts/python.exe scripts/final_results_summary.py

# assemble this file: replace each @@<title>@@ placeholder of a hand-written template with the
# matching "### <title>" block of results/week5_final/final_tables.md (session scratchpad script)
.venv/Scripts/python.exe <scratchpad>/assemble.py <scratchpad>/template.md
```

The tables in this file are pasted verbatim from `results/week5_final/final_tables.md` (heading level changed from ### to ####, no number edited). The script output is deterministic: a second full run gave identical tables. To refresh after a new run, rerun the two scripts and replace the tables.
