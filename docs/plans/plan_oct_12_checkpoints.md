# Week-6 retrain — checkpoint notes for the user (2026-10-12)

The user asked for an uninterrupted run from T1 to T10, with a summary at each of the three planned checkpoints written here for a later check. Every number below is copied from the named output file, which is produced by code.

Plan: `docs/plans/plan_oct_12_new_split_retrain.md`.

## Checkpoint 1 — after T1 (date split)

**Source:** `results/week6_retrain/split_counts.json`, written by `scripts/expedia_build_split.py`.

| Window | Rule | Bookings | Share | Users | Warm (L ≥ 1) | Cold (L = 0) | Rows with h ≥ 1 | Max h | Mean history length |
|---|---|---|---|---|---|---|---|---|---|
| train | t < 2014-09-01 | 2,180,488 | 0.7385 | 664,727 | 1,515,761 | 664,727 | 1,515,761 (0.6951) | 96 | 3.901 |
| valid | 2014-09-01 ≤ t < 2014-11-01 | 393,399 | 0.1332 | 226,064 | 315,858 | 77,541 | 167,335 (0.4254) | 54 | 5.768 |
| test | 2014-11-01 ≤ t < 2015-01-01 | 378,788 | 0.1283 | 222,428 | 307,071 | 71,717 | 156,360 (0.4128) | 90 | 5.837 |

**Totals:** 2,952,675 bookings and 813,985 users, from 2013-01-07 00:00:50 to 2014-12-31 23:59:30. No booking falls after `test_end`.

**h buckets** (h = earlier bookings of the user inside the same window):

| Window | h = 0 | h = 1 | h = 2 | h = 3–5 | h ≥ 6 |
|---|---|---|---|---|---|
| valid | 226,064 | 77,545 | 35,620 | 36,838 | 17,332 |
| test | 222,428 | 74,957 | 32,989 | 33,093 | 15,321 |

**Checks (all passed)**
- The script compared 10,000 random histories against an independent per-user recomputation.
- The query codes equal `expedia_query_codes_clean.npz`.
- `dest` equals `ctx_dest_id` on every row (0 mismatches).

**Independent audit (T1):** PASS WITH MINOR ISSUES; no correctness bug, no leakage.
- The auditor recomputed every count in the table with pandas, on all rows; every value matched.
- It found 0 mismatches in 800 directly compared histories, including 300 rows whose history slides past 20 items.
- Two minor fixes were applied: type hints on `assign_window`, and a guard that windows never decrease in time order.
- One note for the data card: 4 pairs of bookings of the same user share the same second. For those, the parquet row order decides which one counts as history.

**Independent audits T2–T4:** all PASS WITH MINOR ISSUES.
- **T2:** the auditor compared the rollout against its own brute-force autoregressive implementation on 22,790 test rows: max score difference 0, top-1 mismatches 0.
- **Fixes applied after the audits:**
  - a guard that refuses a view that is not the whole window, since such a view would leak labels;
  - the bootstrap no longer holds every resample in memory;
  - resume refuses a different config;
  - a clear error when `min_delta` is not set;
  - tests added for the scorers.

## Checkpoint 2 — after T6 (`min_delta`) and the time estimate

**Source:** `results/week6_retrain/min_delta_calibration.json`, from notebook `00_min_delta_calibration`.

**Run:** query embedding, seed 2022, dropout 0.0, 3 epochs without early stopping, on all train-window rows (2,180,488).

| Epoch | Train loss | AR valid NDCG@10 (monitor) | AR valid NLL | TF valid NDCG@10 | Train-probe TF NDCG@10 | Train s | Eval s |
|---|---|---|---|---|---|---|---|
| 1 | 3.0649 | 0.4319 | 3.0217 | 0.4526 | 0.4815 | 523.6 | 35.3 |
| 2 | 2.7712 | 0.4399 | 2.9431 | 0.4608 | 0.4930 | 457.8 | 51.7 |
| 3 | 2.7150 | 0.4407 | 2.9515 | 0.4640 | 0.4967 | 438.6 | 42.3 |

**Result**
- Gain from epoch 2 to 3: +0.000835. Its user-clustered SE is 0.000350 (393,399 rows, 226,064 users).
- **`min_delta` = 2 × SE = 0.000699**, inside the bounds [1e-4, 5e-3]. It is written to `configs/train.yaml`.

**Reading** (calibration only; not a result)
- The AR monitor almost stops gaining after epoch 2.
- The AR valid NLL rises at epoch 3.
- AR sits about 0.02 below TF on valid. That is the cost of no label feedback.
- The generalisation gap (train probe minus TF valid) is about 0.03.
- Early stopping is therefore expected to stop the query-embedding runs after 3–5 epochs.

**Time estimate (measured)**
- Query embedding: about 7.5–8.7 min of training per epoch plus 0.6–0.9 min of evaluation, on 10 threads.
- The plain model trains on 1,515,761 rows, so it should be about 30 % faster per epoch.
- Workload with early stopping near epoch 4:
  - 4 query-embedding runs ≈ 2.5 h;
  - 4 plain runs ≈ 1.5–2 h;
  - the sameDest grid (65 configurations on AR valid) ≈ 0.5–1 h;
  - the baselines run in parallel.
- **Total ≈ 5–6 h; up to ≈ 12 h if every run goes to 10 epochs.**

**Audits:** T5 baselines and notebooks 01–04 passed with minor issues, all fixed.
- The auditor recomputed every cell of the 2 % smoke-run benchmark, plus the bootstrap and the test CSV top-10 recall, from the ranks files; the largest difference was 0.
- No leakage and no use of test data for selection were found.

## Checkpoint 3 — after T9 (test tables)

**Sources:**
- `results/week6_retrain/final_tables.md`, `final_benchmark.csv`, `bootstrap_ci.csv`, `protocol_gap.csv`, `slices_h.csv`, `samedest_diagnostic.json`, `csv_export.json` (notebook `04_compare`);
- the per-model jsons (notebooks 01–03).

**Setting:** Expedia, date split, test window (378,788 rows), mean ± sample std over 3 seeds. ItemKNN is deterministic.

### Headline: test, all rows, autoregressive (primary)

| Model | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|---|
| ItemKNN | 0.1445 | 0.1825 | 0.2248 | 0.3354 |
| SMLP4Rec plain | 0.1463 ± 0.0003 | 0.1844 ± 0.0005 | 0.2267 ± 0.0004 | 0.3381 ± 0.0009 |
| Logistic regression | 0.3246 ± 0.0065 | 0.3872 ± 0.0063 | 0.4315 ± 0.0057 | 0.6787 ± 0.0074 |
| + query embedding | 0.3817 ± 0.0029 | 0.4414 ± 0.0029 | 0.4804 ± 0.0024 | 0.7345 ± 0.0036 |
| + query embedding + sameDest (main) | 0.3901 ± 0.0029 | 0.4487 ± 0.0028 | 0.4872 ± 0.0024 | 0.7373 ± 0.0034 |

The teacher-forced numbers are in `final_tables.md`. For example, the main model's TF NDCG@10 (all rows) is 0.4803 ± 0.0029.

### Ladder (paired bootstrap, users resampled, seed 2022, test, all rows, NDCG@10, AR)

| Step | Difference | 95 % CI |
|---|---|---|
| plain − ItemKNN | +0.0017 | [+0.0011, +0.0023] |
| + query embedding − plain | +0.2604 | [+0.2590, +0.2618] |
| main − query embedding (the sameDest step) | +0.0072 | [+0.0068, +0.0076] |
| main − logistic regression | +0.0650 | [+0.0640, +0.0661] |

### AR minus TF (seed 2022, all rows, NDCG@10)

| Model | AR − TF |
|---|---|
| ItemKNN | −0.0439 |
| Plain | −0.0553 |
| Logistic regression | 0.0000 (as expected: it uses no history) |
| Query embedding | −0.0209 |
| Main | −0.0318 |

### Reading
- **Ladder:** the query embedding is the big step. sameDest adds a small, significant gain under AR (+0.0072) and a larger one under TF (+0.0181), because it relies on the real history.
- **Rows with h ≥ 1 under AR:** the history-based models lose their TF advantage. For example, the main model's NDCG@10 is 0.4493 at h ≥ 6 under AR, against 0.5721 under TF. The model that knows the search degrades gently.
- **Self-confirmation:** under AR, 12.10 % of the main model's test rows have a top-1 equal to an earlier predicted item at the same destination (11.86 % for the query embedding alone). NDCG@10 on those rows is not lower (0.4642 against 0.4502), so there is no sign of harmful self-reinforcement.

### Selection and early stopping (valid only)

| Model | Choice | Training outcome |
|---|---|---|
| Plain | dropout 0.0 | best epochs 2 / 2 / 3; stopped by patience / patience / NLL guard |
| Query embedding | dropout 0.2 (AR valid NDCG@10 0.4495 vs 0.4466) | best epochs 10 / 7 / 5; stopped by max_epochs / NLL guard / patience |
| sameDest | alpha 0.1, w 0.5 (one-SE pick; best was alpha 0.05, w 0.75; 9 configurations within one SE) | — |
| ItemKNN | K = 50 | — |
| Logistic regression | C = 1.0 | best pass 8 for all seeds |

**Warning:** for query embedding seed 2022 (dropout 0.2), the best epoch equals the cap of 10, so that run is possibly under-trained. The other two seeds stopped by themselves at epochs 7 and 5.

### CSV export
- **Folder:** `data/processed/expedia_date_split/csv/`, with train 2,180,488 / valid 393,399 / test 378,788 rows.
- **Columns:** the cluster column is `hotel_cluster`.
- **Test file:** `top10_main_ar` holds the main model's AR top-10 (seed 2022) and `truth_rank_main_ar` the rank of the true cluster. Recall@10 recomputed from the column is 0.7410, the same as the table.

### Final independent audit (T7–T9): PASS WITH MINOR ISSUES

**What was recomputed** (the auditor's own code, starting from the ranks files)
- All 78 rows of `final_benchmark.csv`, plus every mean ± std in `final_tables.md`.
- The h slices.
- The bootstrap differences, which match exactly. The bootstrap intervals agree within about 0.0002 when recomputed with an independent RNG.
- The test-CSV recall@10 (0.741019).
- `reports/benchmark_results.csv`.

**What was confirmed**
- No leakage, and no use of test data for selection.
- Every tuned value is the valid choice the one-SE rule gives.

**Notes for the report**
1. **Query embedding, seed 2022 (dropout 0.2):** it peaked at the 10-epoch cap, so it may be under-trained. It is also the best of the three seeds: main-model AR NDCG@10 0.4519 against 0.4464 and 0.4478.
2. **Logistic regression:** it stops at pass 8 while still improving by less than `min_delta`. That is consistent with a seed std (0.0063) that is high for a convex model, so it is slightly under-converged.
3. **sameDest on cold rows:** it is a no-op to 4 decimals. With seed 2022, one cold row in valid and one in test differ by one rank position (float noise between two scoring passes).

**Metadata fix**
- `results/week6_retrain/query_s*.json` and the run `metrics.json` files stored the base `hidden_dropout_prob` (0.0) in `model_config`, although the runs used 0.2.
- This was corrected from each file's own `hidden_dropout_prob` field, and the correction is noted in a `model_config_note` field. No result changed.
- `reports/benchmark_results.csv` now also carries `trained_on`.
