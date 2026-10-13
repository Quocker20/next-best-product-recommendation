# Evaluation protocol (Expedia, effective 2026-10-12, week-6 retrain)

Applies to the three SMLP4Rec steps and the two baselines:
- SMLP4Rec steps: plain, + query embedding, + query embedding + sameDest (the main model);
- baselines: ItemKNN and logistic regression.

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
  - Query embedding, logistic regression and ItemKNN: every train-window booking.
- **Train-only fits:** the query vocabulary, the popularity tables and the one-hot encoder are built from the train window only.

## Test protocols
- **Primary: autoregressive (AR), no label feedback.** Inside an evaluated window (valid or test), the true cluster of a booking never enters any later history.
  - A history slot holding an earlier booking of the same window gets the **top-1 prediction** made for that booking.
  - Slots from earlier windows stay real.
  - The slot's destination is kept, because the searched destination is known at search time.
  - The row's own query stays visible.
  - Rows are scored in steps of h (`nbp.eval.autoregressive.score_view`).
  - A view must hold the whole window; warm and cold masks are applied only at metric time.
- **Secondary: teacher-forced (TF).** True histories. It is computed for every model on the same rows and reported next to AR, never mixed with it. It is the closer proxy of a deployed system, where past bookings are observed.
- **Sanity checks** (asserted in code):
  - logistic regression gives identical AR and TF ranks;
  - ItemKNN gives identical AR and TF ranks on rows with h = 0.

## Task, ranking, metrics
- **Task:** predict the cluster of a booking given the history and the query. Every row is ranked over all 100 clusters; there are no sampled negatives.
- **Metrics:** one relevant item per row, so with rank = position of the true cluster:
  - Recall@K = 1[rank ≤ K];
  - NDCG@K = 1[rank ≤ K] / log2(rank + 1), with ideal DCG = 1;
  - K = 5, 10, 20;
  - NDCG@K is primary and Recall@K secondary;
  - ties go to the lower model id (`nbp.eval.metrics`, unit-tested);
  - no MRR, MAP or Hit.
- **Slices:**
  - h (0, 1, 2, 3–5, 6+);
  - check-in month;
  - AR minus TF gap;
  - sameDest self-confirmation: the share of AR test rows whose top-1 equals an earlier predicted item at the searched destination.

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
  - logistic regression C (ties to the smaller C);
  - sameDest alpha and w (ties to the smaller w, then the larger alpha).
  - The chosen values are frozen for the other seeds.
- **sameDest:** `log q_model + w · log q_sameDest`, `log q = log((1 − alpha) p + alpha / 100)`.
  - q_sameDest is the recency-weighted (0.7^age) share of the history booked at the searched destination.
  - It is uniform when there is none, so the fusion is a no-op for a destination new to the user.
- **Test** is read once, after all selection is frozen. There is no retraining on train + valid.

## Significance and reporting
- **Bootstrap:** paired, 1000 resamples, **users resampled** (AR rows of one user are dependent), reference seed. The result is reported as the difference with its 95 % interval (`nbp.eval.bootstrap`).
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
