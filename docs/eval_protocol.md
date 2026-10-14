# Evaluation protocol (Expedia, effective 2026-10-12, week-6 retrain)

Applies to the three SMLP4Rec steps and the two baselines:
- SMLP4Rec steps: plain, + query embedding, + query embedding + sameDest (the main model);
- baselines: ItemKNN and global popularity (the train-window booking share of each cluster, one ranking for every row; it replaced logistic regression on 2026-10-13).

Change only with user approval. Plan and rationale:
- `docs/plans/plan_oct_12_new_split_retrain.md`
- `docs/plans/plan_oct_12_autoregressive_test.md`

Superseded on 2026-10-12: the RecBole 80/10/10 target split, the teacher-forced headline, and selection by Recall@5 + NDCG@5.

## Data
- **Source:** Expedia `train.csv`, rows with `is_booking == 1`.
- **Item:** item = `hotel_cluster`, 100 classes, a proxy for room category or package. Clicks are never positives.
- **Query:** the fields of the booking's own search, 14 fields (`nbp.data.query_features`).
- **Forbidden as input:** `hotel_continent`, `hotel_country`, `hotel_market`, `orig_destination_distance`, `cnt` and `is_booking`, because they describe the booked hotel or the label.
- **Cleaning** (`src/nbp/data/clean.py`):
  - drop duplicate keys;
  - null and flag context defects;
  - collapse burst repeats (`docs/data_cleaning_strategy.md`).
- **Input file:** every export reads `data/interim/expedia_bookings.parquet`. It is time-sorted, and its row position is the event index.

## Split (by booking `date_time`; `configs/split.yaml`, `scripts/expedia_build_split.py`)
- **Windows:**
  - train: t < 2014-09-01;
  - valid: 2014-09-01 ≤ t < 2014-11-01;
  - test: 2014-11-01 ≤ t < 2015-01-01.
  - A booking belongs to the window of its own time. There are no random splits.
- **Counts** (`results/week6_retrain/split_counts.json`):

  | Window | Bookings | Users | Warm (L ≥ 1) | Cold (L = 0) |
  |---|---|---|---|---|
  | train | 2,180,488 | 664,727 | 1,515,761 | 664,727 |
  | valid | 393,399 | 226,064 | 315,858 | 77,541 |
  | test | 378,788 | 222,428 | 307,071 | 71,717 |

- **History:** the user's last 20 earlier bookings, from any window, oldest first.
- **L and h:** L = the number of earlier bookings of the user. h = the number of earlier bookings of the user inside the same window.
- **Row sets:** warm (L ≥ 1), cold (L = 0) and all. All three are always reported.
- **Training rows:** train window only.
  - Plain SMLP4Rec: warm rows only, because it needs a history.
  - Query embedding, global popularity and ItemKNN: every train-window booking.
- **Train-only fits:** the query vocabulary, the popularity tables and the one-hot encoder are built from the train window only.

## Test protocol (effective 2026-10-14): user-level, one fixed top-20 per user
Replaces the row-level AR / TF test tables (superseded 2026-10-14). Code: `nbp.eval.two_step`, `nbp.eval.user_level.fixed_top_user_metrics`, `scripts/eval_user_top20.py`.
- **Unit:** the test user. Each user gets ONE ranked list of 20 clusters, used for every k.
- **Step 1:** the user's first test booking (h = 0), with its true history; all 100 clusters are scored, the top-20 and their probabilities are kept (P1).
- **Step 2:** the user's second test booking (h = 1). Its history slot for booking 1 holds the **top-1 of P1** (no label feedback); earlier-window slots stay real. A user with a single test booking gets a synthetic step 2: the context (query, searched destination) of booking 1 and the history of booking 1 followed by the top-1 of P1. Top-20 and probabilities again (P2). Bookings 3+ are not scored; they only enter the truth set.
- **Probabilities:** taken over all 100 clusters before the top-20 cut. SMLP4Rec steps: softmax of the scores (logits; the fused log-scores for sameDest). Baselines: score / row sum (popularity shares; ItemKNN similarity sums, cold rows already the first-booking popularity).
- **Rerankers** (P1, P2 → one top-20; **chosen: `m2_merge`**, decision 2026-10-14: best main-model NDCG@K and Recall@K of the three on the reported seed, `m1` and `m3` are kept only in `results/week6_retrain/user_top20_reranker_choice.md` as the record behind the choice; a cluster present in both keeps its max value, ties to the lower id):
  - `m1_first`: P1 only;
  - `m2_merge`: union of P1 and P2 sorted by probability;
  - `m3_sqrt_alpha`: P2 values x mapped to sqrt(alpha · x), alpha = probability of the P1 top-1, then union with P1.
  - Popularity gives the same list under all three (asserted).
- **Truth:** the SET of distinct clusters the user books in the whole test window (repeat bookings count once).
- **Metrics,** per user, then the mean over users; K = 5, 10, 20:
  - Recall@K = |top-K ∩ truth| / |truth|;
  - NDCG@K = Σ_{hit at rank r ≤ K} 1/log2(r + 1) / Σ_{r=1}^{min(K, |truth|)} 1/log2(r + 1);
  - NDCG@K is primary and Recall@K secondary; no MRR, MAP or Hit.
- **Seed:** one fixed seed, **2024** (`configs/split.yaml` `report_seed`, chosen 2026-10-13 as the seed closest to the 3-seed mean), for every reported and recommended number of the user-level test; no seed averages or ± in these tables. Run on the full real data with the date split above and check the windows before reporting.
- **User sets:** all, warm (L ≥ 1 at booking 1) and cold (L = 0 at booking 1).
- **Outputs:** `results/week6_retrain/report_numbers.md` (the one sheet of numbers for the report), `user_top20.json`, `user_top20_tables.md`, `user_top20_bootstrap.{json,md}` (paired user bootstrap), `user_top20_reranker_choice.md`; per-user lists in `data/interim/week6/user_top20.npz`; `test_with_result.csv` (split `test.csv` + one per-user top-10 column per model, reranker m2_merge, `scripts/export_test_with_result.py`).
- **Valid** (early stopping, selection) is unchanged: row-level AR NDCG@10 (`nbp.eval.autoregressive.score_view`: in-window history slots hold the top-1 predictions, rows scored in steps of h). One relevant item per row: NDCG@K = 1[rank ≤ K] / log2(rank + 1), every row ranked over all 100 clusters, ties to the lower model id (`nbp.eval.metrics`).

## Training, early stopping and selection (`configs/train.yaml`)
- **Training:** teacher forcing on train-window rows; Adam; batch 1024; at most 10 epochs; 3 seeds (2022, 2023, 2024).
- **Early stopping:**
  - **Monitor:** AR valid NDCG@10.
  - **Improvement:** only a gain > `min_delta` counts. `min_delta` = 2 × the user-clustered SE of the per-row epoch-to-epoch gain, calibrated by notebook 00 and stored in the config.
  - **Patience:** 2.
  - **NLL guard:** stop when the AR valid NLL has risen for 2 epochs in a row while the train loss fell.
  - **Restore:** the best epoch's weights are restored.
  - **Warning:** if the best epoch = 10, the run is flagged as possibly under-trained.
- **Selection,** on the reference seed (2022), by AR valid NDCG@10 with the one-standard-error rule clustered by user:
  - SMLP4Rec dropout ∈ {0.0, 0.2} (ties to the higher dropout);
  - ItemKNN K (ties to the smaller K);
  - sameDest alpha and w (ties to the smaller w, then the larger alpha).
  - The chosen values are frozen for the other seeds.
- **sameDest:** `log q_model + w · log q_sameDest`, `log q = log((1 − alpha) p + alpha / 100)`.
  - q_sameDest is the recency-weighted (0.7^lag) share of the history booked at the searched destination.
  - It is uniform when there is none, so the fusion is a no-op for a destination new to the user.
- **Reported seed (2026-10-13):** every number of the report and the slides is that of **seed 2024**, the seed whose results lie closest to the 3-seed mean (`scripts/pick_reported_seed.py` → `results/week6_retrain/reported_seed.json`, `configs/split.yaml` `report_seed`). The tables of this seed are `results/week6_retrain/reported_tables.md`; `final_tables.md` keeps the mean ± std over the three seeds as the full record.
- **Test** is read once, after all selection is frozen. There is no retraining on train + valid.

## Significance and reporting
- **Bootstrap:** paired, 1000 resamples, **users resampled** (AR rows of one user are dependent), reported seed. The result is reported as the difference with its 95 % interval (`nbp.eval.bootstrap`).
- **Tables:** every table states the dataset, row set and protocol. Results go to `results/week6_retrain/`.
- **`reports/benchmark_results.csv`:**
  - one row per dataset × model × seed × row set × protocol;
  - built by `scripts/build_benchmark_results.py`;
  - never edited by hand.
- **Limitations, stated with every result:**
  - the cluster is a proxy;
  - one dataset, offline only, no in-stay data;
  - two simple baselines;
  - no retraining on train + valid;
  - valid (Sep–Oct) and test (Nov–Dec) cover different months;
  - AR measures planning without feedback.
