# Plan — date split, autoregressive evaluation, retrain (detailed, 2026-10-12)

**Status:** approved by the user on 2026-10-12. All six recommended defaults were accepted (section 0). Implementation has not started.

**Related plans:**
- Protocol rationale: `docs/plans/plan_oct_12_autoregressive_test.md`.
- This plan replaces the RecBole 80/10/10 split and the week-4/5 run notebooks.

## 0. Decisions (fixed)

| # | Decision |
|---|---|
| D1 | **Split by booking `date_time`:**<br>• train: before 2014-09-01<br>• valid: from 2014-09-01, before 2014-11-01<br>• test: from 2014-11-01, before 2015-01-01 |
| D2 | **Primary protocol: autoregressive (AR)** on valid and test. Inside the evaluated window, the user's next row gets the top-1 prediction appended, never the true cluster. Teacher-forced (TF) is computed for every model as the secondary protocol. |
| D3 | **Metrics:** NDCG@5/10/20 are primary; Recall@5/10/20 are secondary. Row sets: warm, cold and all events. |
| D4 | **Early stopping monitors AR valid NDCG@10.** This supersedes the "early stopping teacher-forced" wording of the morning of 2026-10-12. |
| D5 | **Hyperparameter selection:** on AR valid NDCG@10.<br>• Selected: the SMLP4Rec dropout grid {0.0, 0.2}, the ItemKNN K, the logistic regression C, and the sameDest alpha and weights.<br>• Run with seed 2022 only; the chosen values are reused for seeds 2023 and 2024. |
| D6 | **Seeds and epochs:** 3 seeds (2022, 2023, 2024) per trainable model, at most 10 epochs per seed, plus strict anti-overfit controls (section 5). |
| D7 | **No retraining on train + valid** before reading test. Test is read once, after all selection is frozen. |
| D8 | **Output folder:** `results/week6_retrain/`. Run folders go to `experiments/<date>_expedia_<model>_<tag>-s<seed>/`. |
| D9 | **Notebook `08` and the demo scripts are untouched.** They will not run until rewired, because their checkpoints are gone. |
| D10 | **Old notebooks** (`01c`, `01d_early_stop_threshold`, `01d_train_test_…`, `01e`, `07`, `07b`) are deleted only after the new notebooks reproduce their role (task T10). |

Models compared, all under AR (TF in a separate column):

| Name in tables | What | Trained on | Seeds |
|---|---|---|---|
| SMLP4Rec plain | history only (`src/models/smlprec.py`) | train-window warm rows (L ≥ 1) | 3 |
| + query embedding | `SMLPRECQuery` | train-window warm + cold rows (all events) | 3 |
| + query embedding + sameDest (main) | post-hoc fusion on the query-embedding model of the same seed | no training; alpha and weights tuned on AR valid | 3 (inherits) |
| ItemKNN | `implicit` cosine on the train user × cluster matrix; cold rows use first-booking popularity | train-window bookings | deterministic |
| Logistic regression | multinomial, one-hot query fields, SAGA passes with early stopping | train-window warm + cold rows | 3 |

## 1. Baseline facts (computed 2026-10-12, scratch script; T1 re-prints them from a repo script)

| Window | Bookings | Share | Users | Warm | Cold | AR rows with h ≥ 1 |
|---|---|---|---|---|---|---|
| train | 2,180,488 | 73.85 % | 664,727 | 1,515,761 | 664,727 | — |
| valid | 393,399 | 13.32 % | 226,064 | 315,858 | 77,541 | 167,335 (42.54 %) |
| test | 378,788 | 12.83 % | 222,428 | 307,071 | 71,717 | 156,360 (41.28 %) |

The data runs from 2013-01-07 to 2014-12-31 23:59:30, so no booking falls after the test end. h is the number of earlier bookings of the same user inside the same window.

## 2. Target architecture

```
.venv (pandas 3, pyarrow)                    smlp4rec venv (numpy 1.23, torch, RecBole as model base only)
-------------------------                    ---------------------------------------------------------
data/interim/expedia_bookings.parquet
        |  scripts/expedia_build_split.py
        v
data/processed/expedia_date_split/rows.npz  ─────────────►  notebooks 01_plain, 02_query_samedest
  (numpy only, no object arrays)                                 |  nbp.models.train / scorers
        |                                                         |  nbp.eval.autoregressive
        v                                                         v
notebook 03_baselines (nbp.baselines)          data/interim/week6/ranks_<model>_s<seed>.npz
        |                                                         |
        └──────────────► notebook 04_compare (nbp.eval.report) ◄──┘
                               v
             results/week6_retrain/*.json|csv|md, reports/benchmark_results.csv
```

- `rows.npz` is the single source of the split for all five models. RecBole's `create_dataset` and `data_preparation` are no longer used.
- `SMLPREC` and `SMLPRECQuery` get a stub dataset object that exposes only `num(item_field)`. `tests/test_smlprec_query.py` already uses this pattern.
- Model item ids: cluster c becomes id c + 1, and 0 is padding. Every score matrix keeps 101 columns, with column 0 set to -inf before ranking. Metrics use the true id.

## 3. `rows.npz` schema (one row per booking, in time order)

| Key | Shape / dtype | Meaning |
|---|---|---|
| `event` | (N,) int32 | event index = row position in `expedia_bookings.parquet` |
| `user` | (N,) int32 | factorized user id |
| `y` | (N,) int16 | model item id of the booking (cluster + 1) |
| `window` | (N,) int8 | 0 = train, 1 = valid, 2 = test |
| `L` | (N,) int32 | number of earlier bookings of the user (any window) |
| `h` | (N,) int32 | number of earlier bookings of the user inside the same window |
| `dest` | (N,) int32 | `srch_destination_id` (raw code, used by sameDest) |
| `seq` | (N, 20) int16 | true history, oldest first, right-padded with 0 (last 20 earlier bookings) |
| `seq_len` | (N,) int8 | min(L, 20) |
| `h_dest` | (N, 20) int32 | destination of each history item (0 on padding) |
| `h_event` | (N, 20) int32 | event index of each history item (-1 on padding); identifies items that fall in the same window |
| `codes` | (N, F) int32 | query codes (`nbp.data.query_features.build_query_codes`), the same as `expedia_query_codes_clean.npz` |
| `fields`, `dates` | str arrays | query field names; the three split dates (for provenance) |

**Size:** about 2.95 M rows × (20 × 2 + 20 × 4 + 20 × 4 bytes) ≈ 0.6 GB uncompressed, written with `savez_compressed`. If memory is tight, drop `h_event`: it can be rebuilt from `event`, `user` and `seq_len`.

## 4. Autoregressive rollout (`nbp.eval.autoregressive`)

```python
def rollout(
    rows: RowView,                 # the rows of ONE window (valid or test), any order
    score_fn: Callable[[Batch], np.ndarray],   # Batch -> (b, 101) scores, column 0 ignored
    max_len: int = 20,
    batch: int = 8192,
) -> RolloutResult:                # scores (n, 101) float32 in input order, top1 (n,), h (n,)
```

**Algorithm**
1. Sort the rows by (user, event).
2. For step s = 0, 1, …, max h:
   - Take every row with h == s.
   - Build its AR history from the true history, then overwrite the last `min(h, seq_len)` positions with the top-1 predictions of that user's previous `min(h, 20)` rows, in order.
   - Overwrite their `h_dest` with those rows' `dest`.
   - Score in batches, store the scores, and set top-1 = argmax over columns 1..100.
3. Positions holding history from earlier windows keep their true items. The 20-item window slides naturally, because an AR row has exactly as many history items as its TF twin.
4. If h ≥ 20, the whole window is predictions. If L = 0 (cold), the history is empty and the row's top-1 is still passed on.

`Batch` holds `seq`, `seq_len`, `h_dest`, `q_dest`, `codes_row` (the event index into `codes`) and `y`. `y` is passed through for metrics only; scorers never read it, and the tests assert this.

**Cost:** one forward pass per row, plus at most max-h sequential steps. Later steps hold few rows, so the total cost is close to one TF pass.

**`teacher_forced(rows, score_fn)`** applies the same batching to the true histories. It shares code with `rollout` (step 0 = TF), so the two can only differ where h ≥ 1.

## 5. Training and anti-overfitting (`nbp.models.train`)

```python
def fit(
    model, train: TrainRows, valid: RowView, score_fn_factory,
    cfg: TrainConfig, run_dir: Path, resume: bool = True,
) -> FitResult   # best_epoch, stop_reason, per_epoch rows, path to best.pth
```

**Each epoch**
1. One shuffled pass over the training rows. The shuffle uses a torch `Generator` seeded with `seed + epoch`.
2. Teacher-forced history on the training rows (unchanged training).
3. AR valid and TF valid scoring.
4. Log per epoch:
   - train loss (sum and per batch) and timing;
   - AR valid NDCG@5/10/20 and Recall@5/10/20;
   - TF valid NDCG@10;
   - valid NLL (AR);
   - the generalisation gap (train-sample NDCG@10 minus TF valid NDCG@10). The train sample is a fixed random 50 k train rows.

**Stopping rules, checked in this order after each epoch**
1. **Improvement:** an epoch is the new best only if AR valid NDCG@10 rises by more than `min_delta`.
2. **Patience:** stop after 2 epochs without an improvement.
3. **NLL guard:** stop if the AR valid NLL rises for 2 consecutive epochs while the train loss falls. The stop reason is logged as `nll_guard`.
4. **Ceiling:** at most 10 epochs. If the best epoch is 10, log `WARNING best epoch = max_epochs`, because the model may be under-trained. That is reported, not silently accepted.
5. **Restore:** reload `best.pth` before anything is scored for the report.

**`min_delta` calibration (task T6, once)**
- Run the query-embedding model with seed 2022 for 3 epochs, without stopping.
- Compute the paired per-row difference in AR valid NDCG@10 between epochs 2 and 3.
- Set `min_delta = 2 × cluster_se(diff, groups=user)`, using `nbp.hybrid.fusion.cluster_se`. The SE is clustered because AR rows of one user are dependent.
- Write it to `configs/train.yaml` together with the analysis json (`results/week6_retrain/min_delta_calibration.json`). Every model uses this value.

**Checkpoints**
- `best.pth` (weights) and `last.pth` (weights, optimizer, per-epoch rows, stopper state) are kept for exact resume. No `epochN.pth` files are kept.

**Run folder**
- `config_resolved.yaml`, `metrics.json`, `train.log` and `git_hash.txt` (`CLAUDE.md` §8).

**Regularisation grid** (seed 2022, AR valid NDCG@10; the one-SE rule picks the simpler value on near-ties)

| Model | Grid |
|---|---|
| SMLP4Rec plain | `hidden_dropout_prob` ∈ {0.0, 0.2} |
| + query embedding | `hidden_dropout_prob` ∈ {0.0, 0.2} |
| ItemKNN | K ∈ {5, 10, 20, 50, 100} |
| Logistic regression | C ∈ {0.1, 1.0, 10.0}, SAGA passes with the same early stopper (AR = TF for logistic regression) |
| sameDest | alpha ∈ {0.05, 0.1, 0.2, 0.3, 0.5} × w_s ∈ {0, 0.25, …, 3.0}, one weight per cell (destination known / new to the user); the cell is computed on the AR history |

Grid selection uses NDCG@10 per row with `one_se_pick` grouped by user. The old criterion, Recall@5 + NDCG@5, is dropped because NDCG is now the primary metric.

## 6. Tasks

Each task ends with green tests or printed checks before the next one starts. Code goes through `ruff` format and lint.

### T1 — Split builder (`.venv`)
- **Files:**
  - `configs/split.yaml`: dates, `max_len: 20`, seeds `[2022, 2023, 2024]`.
  - `src/nbp/data/split.py`: `assign_window(ts_ns, dates) -> int8[]` and `build_rows(df, max_len) -> dict`. Numpy and pandas only.
  - `scripts/expedia_build_split.py`: writes `rows.npz` and `results/week6_retrain/split_counts.json`. It also prints the section 1 table plus the h distribution.
- **Tests** in `tests/test_split.py`, against a hand-built toy frame with 3 users across 3 windows:
  - window edges: a booking exactly at 2014-09-01 00:00:00 goes to valid, and one at 2014-11-01 00:00:00 goes to test;
  - `L`, `h`, `seq`, `h_event` and `h_dest` match hand values;
  - the history slides after 20 items;
  - every history event is earlier than its row's event;
  - a cold row has `seq_len` 0.
- **Accept when:**
  - the counts equal section 1;
  - `codes` equal `expedia_query_codes_clean.npz` row for row;
  - for valid and test rows, `seq` equals the history a RecBole-style build from the full sequence gives (spot check of 10,000 rows).

### T2 — Rollout (numpy only, works in both venvs)
- **Files:** `src/nbp/eval/autoregressive.py` (`RowView`, `Batch`, `rollout` and `teacher_forced`, section 4).
- **Tests** in `tests/test_autoregressive.py`:
  1. Rows with h = 0 give AR scores identical to TF.
  2. **Leak probe:** a scorer that returns a high score for whatever cluster sits last in the history. Under AR, it never returns the true previous-row cluster unless that cluster equals the previous top-1.
  3. A scorer that ignores history gives AR = TF on every row (logistic regression stand-in).
  4. With h ≥ 20, the window holds only predictions; with h = 3 and seq_len = 20, the last 3 positions are predictions and the first 17 are true.
  5. Users with 1, 2 and 7 rows in the window, in shuffled input order, come back in input order.
  6. A cold first row gets an empty history and passes its top-1 on.
  7. `h_dest` of each appended item equals that row's `dest`.
- **Accept when:** all tests pass in both venvs.

### T3 — Report helpers
- **Files:**
  - `src/nbp/eval/report.py`:
    - `metrics_table(ranks: dict[model][protocol] -> (n,), masks) -> DataFrame`, with NDCG columns first;
    - `h_slices(rank, h)` for the buckets 0, 1, 2, 3–5, 6+;
    - `protocol_gap(rank_ar, rank_tf, groups)`;
    - `write_run_json(...)`.
  - `src/nbp/eval/bootstrap.py`: add an optional `groups` argument (resample users, not rows) and make the default for AR results user-clustered. The current row-level behaviour stays when `groups=None`.
- **Tests:** extend `tests/test_metrics.py` and add `tests/test_report.py`:
  - h buckets;
  - column order;
  - a cluster bootstrap with one row per user equals the row bootstrap given the same seed.

### T4 — Training loop (smlp4rec venv)
- **Files:**
  - `src/nbp/models/train.py`: `TrainConfig`, `fit` (section 5) and `StubDataset`.
  - `src/nbp/models/scorers.py`:
    - `plain_scorer(model)`;
    - `query_scorer(model, q_idx)`;
    - `samedest_scorer(base_scorer, alpha, wmap, base=0.7)`, which reuses `nbp.hybrid.same_dest` and `fusion`.
  - `configs/train.yaml`: `max_epochs: 10`, `patience: 2`, `min_delta` (filled in by T6), `nll_guard_epochs: 2`, `batch_size: 1024`, `lr: 1e-3`, the dropout grid.
  - `configs/smlprec_expedia.yaml`: remove the `eval_args` RS block, `epochs` and `stopping_step`; keep the model hyperparameters only.
- **Tests:**
  - `tests/test_train.py`, on a tiny synthetic task (5 users, 6 items, 3 epochs, CPU):
    - resume after epoch 2 gives the same per-epoch rows as an uninterrupted run;
    - the stopper honours `min_delta`, patience and the NLL guard;
    - `best.pth` is the best AR epoch;
    - no `epochN.pth` is written.
  - `tests/test_early_stop.py`: extend it for the NLL-guard helper.

### T5 — Baselines (`.venv`)
- **Files:**
  - `src/nbp/baselines/itemknn.py`: `fit_itemknn(train_rows, K)` and `itemknn_scorer(model, pop_first)`. The history count matrix is built from `Batch.seq`, so AR works unchanged.
  - `src/nbp/baselines/logreg.py`: `lr_frame` (ported from `07b`), `fit_logreg(X, y, C, seed, stopper)` and `logreg_scorer`.
  - `notebooks/hospitality/smlp4rec/03_baselines.ipynb`: the K and C grids on AR valid; logistic regression for seeds 2022, 2023 and 2024; ranks for both protocols, written to `data/interim/week6/`.
- **Tests:** `tests/test_baselines.py`:
  - the ItemKNN score on a 3-user toy equals a hand-computed cosine product;
  - `lr_frame` uses no forbidden column.
- **Accept when (asserts in the notebook):**
  - logistic regression AR ranks == TF ranks on every valid and test row;
  - ItemKNN AR == TF on h = 0 rows.

### T6 — `min_delta` calibration (smlp4rec venv)
- Short query-embedding run (seed 2022, dropout 0.0, 3 epochs, no stopping), using the T4 code.
- Writes `min_delta_calibration.json` and fills `configs/train.yaml`.
- **Accept when:** the value and the SE are printed. A value below 1e-4 or above 5e-3 triggers a stop-and-report rather than silent use.

### T7 — SMLP4Rec plain (smlp4rec venv)
- `notebooks/hospitality/smlp4rec/01_plain.ipynb`. Training rows: train-window warm rows (L ≥ 1, seq_len ≥ 1).
- Dropout grid with seed 2022, then the chosen value for seeds 2023 and 2024. Per seed: `fit`, AR and TF ranks for valid and test.
- Output: `results/week6_retrain/plain_s<seed>.json`.
- Cold rows are scored by the model with an empty history. That is out of distribution for plain and is stated as such in every table, the same as before.

### T8 — Query embedding + sameDest (smlp4rec venv)
- `notebooks/hospitality/smlp4rec/02_query_samedest.ipynb`.
- **Query embedding:**
  - query vocabulary from train-window rows only (`build_vocab(codes, train_rows)`);
  - training rows = all train-window rows, warm and cold;
  - dropout grid with seed 2022, then 3 seeds.
- **sameDest, per seed:**
  - the grid in section 5 on the AR valid rollout of that seed's query-embedding model;
  - the one-SE pick with seed 2022 fixes alpha and the weights for all seeds;
  - the per-seed result then uses those frozen values.
- **Diagnostics, written to `samedest_diagnostic.json`:**
  - the share of AR test rows whose top-1 equals an appended (predicted) item at the same destination;
  - NDCG@10 on those rows against the rest;
  - main minus query embedding, AR against TF.
- Outputs: `query_s<seed>.json` and `main_s<seed>.json`; ranks go to `data/interim/week6/`.

### T9 — Comparison and records (`.venv`)
- `notebooks/hospitality/smlp4rec/04_compare.ipynb` reads the run jsons and ranks.
- It writes to `results/week6_retrain/`:
  - `final_benchmark.csv`: model × seed × row set × protocol;
  - `final_tables.md`: NDCG columns first, mean ± std over seeds for AR and for TF;
  - `bootstrap_ci.csv`: user-clustered, seed 2022. Pairs:
    - main vs logistic regression;
    - main vs ItemKNN;
    - main vs query embedding;
    - query embedding vs plain;
    - AR minus TF per model;
  - `slices_h.csv`;
  - `protocol_gap.csv`.
- New `scripts/build_benchmark_results.py` rewrites `reports/benchmark_results.csv` with a `protocol` column.
- `scripts/verify_week6.py` re-computes every headline number from the ranks files and asserts equality with the tables.

### T10 — Docs and cleanup
- **Docs:**
  - `docs/eval_protocol.md`: rewrite the Split section with the dates and the T1 counts; early stopping on AR (D4); NDCG primary.
  - `CLAUDE.md`:
    - §4: the dates;
    - §6: split, AR primary, NDCG primary, seeds and epochs;
    - §7: new modules, notebooks, `results/week6_retrain`, and the RecBole venv now used only as the model base class;
    - §10: `src/nbp/baselines` is filled.
  - `PROGRESS.md`: one entry per finished task group.
- **Cleanup:** delete the old notebooks (D10) once T7, T8 and T5 reproduce their role. `08` and the demo scripts are left alone (D9).

## 7. Order, dependencies and checkpoints with the user

```
T1 ─► T2 ─► T3 ─► T4 ─► T6 ─► T7 ─┐
          └──────► T5 ────────────┤─► T9 ─► T10
                        T6 ─► T8 ─┘
```

- **After T1:** show the split counts and stop for a check. This is cheap and catches a wrong date rule early.
- **After T6:** show `min_delta` and the epoch-time estimate for 3 seeds × 2 models × up to 10 epochs before the long runs.
- **After T9:** show the AR and TF tables. Test is read once here.

## 8. Risks and how they are handled

| Risk | Handling |
|---|---|
| sameDest boosts its own predictions under AR | diagnostic in T8; main vs query embedding reported under both protocols |
| AR rows of one user are dependent, so row-level CIs come out too narrow | user-clustered bootstrap (T3); `cluster_se` for `min_delta` and the one-SE rule |
| Valid and test differ in season (Sep–Oct against Nov–Dec) | stated as a limitation; slices by check-in month kept in T9 |
| Best epoch = 10 (under-training) | warning logged and reported (section 5) |
| The smlp4rec venv cannot read Parquet | all model inputs come from `rows.npz` |
| Memory for `rows.npz` | compressed npz; drop `h_event` if needed (section 3) |
| The plain model scores cold rows out of distribution | stated in every table, as before |

## 9. Limitations to state with the results

- `hotel_cluster` is a proxy for room or package.
- One dataset; offline only.
- Two simple baselines.
- No retraining on train + valid.
- Valid and test cover different months.
- AR measures planning without feedback, not the deployed setting (TF is the closer proxy).

## 10. As built (2026-10-13): differences from the text above

- **Status:** T1-T10 were executed on 2026-10-12/13. The checkpoint notes are in `docs/plans/plan_oct_12_checkpoints.md`.
- **Name:** "query token" was renamed to "query embedding" everywhere (code, docs, history, slides) on the supervisor's request. The logic is unchanged.
- **Notebooks:** T6 got its own notebook, `00_min_delta_calibration`. The notebook list is `00`, `01_plain`, `02_query_samedest`, `03_baselines` and `04_compare`. The shared notebook code lives in `src/nbp/eval/runner.py`.
- **`build_rows`:** it takes arrays (user, item, dest, window), not a data frame.
- **`assign_window`:** it takes the three dates.
- **sameDest weight:** one weight w instead of one weight per cell. For a destination new to the user, the sameDest distribution is uniform, so a per-cell weight changes nothing. The w grid is 0 to 3 in steps of 0.25 (13 values).
- **Logistic regression features:** the one-hot of the same 14 query codes the query embedding reads (`rows.npz` `codes`), not the week-4 `lr_frame` columns.
- **ItemKNN:** `implicit` counts the item itself among its K neighbours. This is documented in `nbp.baselines.itemknn`.
- **One-SE tie rules:** ties go to the higher dropout, the smaller K, the smaller C, and the smaller w then the larger alpha.
- **Run folders:** they hold `metrics.json` (configs, commit, per-epoch rows, results), `per_epoch.json`, `train_config.json`, `train.log`, `best.pth` and `last.pth`. `CLAUDE.md` §8 was updated to match.
- **CSV export:** this was added at the user's request in task T9. The files are in `data/processed/expedia_date_split/csv/{train,valid,test}.csv`, and the cluster column is `hotel_cluster`. The test file adds:
  - `top10_main_ar`: the main model's autoregressive top-10, seed 2022, cluster ids 0-99;
  - `truth_rank_main_ar`: the rank of the true cluster.
- **Under-training warning:** for the query embedding with seed 2022 and dropout 0.2, the best epoch equals the cap of 10. This is reported, not hidden.
