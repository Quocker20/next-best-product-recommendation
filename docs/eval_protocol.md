# Evaluation protocol (Expedia, effective 2026-10-08)

Applies to the main model (SMLP4Rec query token + sameDest) and the two baselines (ItemKNN, logistic regression). Change only with user approval.

## Data
- Expedia `train.csv`, `is_booking == 1`; item = `hotel_cluster` (100 classes, a proxy for room category / package). Clicks are never positives.
- Query = the fields of the target booking's own search. Forbidden as input: `hotel_continent`, `hotel_country`, `hotel_market`, `orig_destination_distance`, `cnt`, `is_booking` (they describe the booked hotel or the label).
- Cleaning rules (`src/nbp/data/clean.py`): drop duplicate keys; null + flag context defects; **collapse burst repeats** (2026-10-08; `docs/data_cleaning_strategy.md`). Every export and notebook reads `data/interim/expedia_bookings.parquet`; RecBole `timestamp` is the exact event index (row position, sorted by time, user, source row), so the split has no float32 ties. The uncollapsed numbers are kept as a sensitivity run in `results/week4_rebuild/with_burst/`.

## Split
- Global temporal 80/10/10 over next-booking targets (every booking except each user's first), ordered by `date_time` (RecBole `RS: [0.8, 0.1, 0.1]`, `order: TO`). No random splits.
- Sizes (collapsed data): train 1,710,952, valid 213,869 and test 213,869 warm targets. History = the user's previous bookings (max 20).
- **Warm rows:** L ≥ 1 prior bookings (the targets above). **Cold rows:** first booking of each user inside the valid / test window (L = 0; valid 50,769, test 49,462). **All events** = warm + cold (test 263,331).
- Warm rows are the headline set; cold and all-events are always reported next to it.

## Task and ranking
- Predict the cluster of the next booking given the history and the query. Full ranking over all 100 clusters; no sampled negatives.

## Metrics
- Recall@K and NDCG@K, K = 5, 10, 20, one relevant item per row (`nbp.eval.metrics`, unit-tested). No MRR / MAP / Hit.
- Beyond accuracy (stage D): intra-list diversity, popularity bias, coverage per destination and per season.
- Slices: history length L (1 | 2-4 | 5-9 | 10+), seen vs unseen user, known vs new destination, check-in month, trip context (solo / couple / family / group, `is_package`).

## Models and tuning
- Main model: `SMLPRECQuery` (`src/nbp/models/smlprec_query.py`), 2 layers, hidden 64, CE over 100 clusters, Adam; sameDest is a post-hoc fusion term `log q_model + w_s * log q_sameDest`, `log q = log((1 - alpha) * p + alpha / 100)`, weights and alpha chosen on valid by the one-standard-error rule.
- Baselines: ItemKNN (`implicit` cosine, history only, K picked on valid) and multinomial logistic regression (query features only, C picked on valid). ItemKNN has no score for cold rows (global-popularity fallback, stated in every table).
- Every model gets a small grid fixed up front, selected on valid by mean per-row Recall@5 + NDCG@5. Test is read once per stage.
- Seeds: main model 3 seeds (mean ± std); seed 2022 is the reference run. ItemKNN is deterministic.

## Significance
- Paired bootstrap over test rows (1000 resamples) for headline differences; report the difference and the 95 % CI (`nbp.eval.bootstrap`).

## Reporting
- Every table states dataset and row set. One row per dataset × model × seed × row set goes to `reports/benchmark_results.csv`, built by `scripts/build_benchmark_results.py` from `results/week4_rebuild/*.json`; never edited by hand.
- Limitations stated with every result: cluster is a proxy, one dataset, offline only, no in-stay data, two simple baselines, 3-epoch budget until stage C.
