# 05 Verification of the final results package (2026-10-11)

Independent check of the package produced tonight (`scripts/final_results_summary.py`,
`scripts/final_export_topk_smlp4rec.py`, `results/week5_final/*`, `docs/plans/final_prep/01..04`).

- Script: `scripts/verify_final_results.py` (project `.venv`). It does not import the summary script;
  it uses only `nbp.eval.metrics` (`recall_at_k`, `ndcg_at_k`) and `nbp.eval.bootstrap`.
- Per-row sources it reads: the notebooks' own saved ranks
  (`experiments/2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es/ranks_test_{warm,cold}.npz`,
  `experiments/2026-10-10_expedia_smlp4rec_plain-1e-clean-cold-es/ranks_test.npz`),
  `data/interim/baselines_es_ranks_test.parquet` (07b), `data/interim/final_topk_test.npz`
  (checkpoint re-scoring), and `data/interim/expedia_bookings.parquet` (warm / cold recomputed from
  each user's prior bookings). Nothing retrained, `data/raw/` not read.
- Output: `results/week5_final/verification_final.json` (every check with its detail).
- Result: **43 PASS, 0 FAIL, 1 NOTE** after the fixes below (before the fixes: 3 FAIL, of which one
  was a real error in `04_demo_options.md` and two were limitations of the doc-diff check itself).

Rebuild: `.venv/Scripts/python.exe scripts/verify_final_results.py` (about 1 min).

## 1. Checks

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | 01d saved warm / cold row ids = warm / cold events recomputed from bookings | PASS | 213,869 warm + 49,462 cold rows in the files |
| 2 | 07b baseline parquet covers exactly the test-window events, same warm / cold split | PASS | 263,331 rows |
| 3 | `final_topk_test.npz` ranks = notebooks' saved ranks | PASS | main 0.999996, query embedding 1.0, plain_all 1.0, plain_warm (warm) 0.999995 (one tie row each) |
| 4 | `final_topk_test.npz` n_prior = prior bookings recomputed | PASS | |
| 5 | Row counts all / cold / warm = 263,331 / 49,462 / 213,869 | PASS | also equal in `final_results.json` |
| 6 | Headline: 6 models x {all, cold, warm} x Recall / NDCG @5/10/20 recomputed = `final_benchmark.csv` | PASS | 108 values, 0 mismatches (tolerance = 4-decimal rounding) |
| 7 | The same 18 headline rows appear verbatim in `final_tables.md` | PASS | |
| 8 | Main model on cold rows is identical to the query embedding, rank by rank | PASS | the "identical to query embedding" label is correct |
| 9 | ItemKNN cold rows are one constant popularity ranking | PASS | the "NOT SCORED by the model" label is correct |
| 10 | Bootstrap main minus logistic regression (warm, cold; K = 5, 10) with seed 20261011 instead of 0 | PASS | same point differences; CI endpoints within 0.0004 of the package (MC error) |
| 11 | History-length slice recomputed = `slices_history_length.csv` | PASS | 174 values, 0 mismatches (includes plain_warm on warm buckets) |
| 12 | Beyond accuracy, main model, all rows: catalog coverage, exposure Gini (Lorenz formula, different from the package's), ARP@10 | PASS | 100 / 0.363 / 0.01396 both ways |
| 13 | Main model = early-stopping run in `experiments/2026-10-10_expedia_smlp4rec_query-1d-clean-cold-es/` | PASS | run JSON and cold JSON checkpoint paths point there; `best_epoch_by_valid` 8; `best.pth` epoch field 8 and weights equal `epoch8.pth` tensor by tensor; `metrics.json` test = run JSON test |
| 14 | sameDest weights in `final_topk_test.npz` = `tuned_on_valid` | PASS | alpha 0.2, w 0 (new destination) / 0.75 (known) |
| 15 | 01e plain and 07b baselines trained on the same rows as 01d | PASS | 713,754 first bookings + 1,710,952 warm targets in every case |
| 16 | Training-set labels: original code warm-only run labelled warm-only; every other model labelled "all events (warm + L=0)" and sourced from 01d ES / 01e ES / 07b | PASS | 18 headline rows |
| 17 | Cold-row handling labelled (main = query embedding; ItemKNN and warm-only plain = popularity fallback; 01e plain = one constant ranking) | PASS | `scored_by` column and `final_tables.md` |
| 18 | `01_results_summary.md` labels the warm-only original run "warm users only" | PASS | section 1 table; the tables say "warm rows only (L>=1)" (same meaning) |
| 19 | `01` prose "months 4-10 below 0.03 each" | PASS | max test share of months 4-10 = 0.0239 |
| 20 | `01`: every table row outside the hand-written sections 1 and 9 is verbatim in `final_tables.md` | PASS | |
| 21 | `01`: every prose number with >= 3 digits occurs in a source (`final_tables.md`, `final_results.json`, week-4 JSONs, week-5 CSVs) | PASS | the hand-read claims in the "Reading:" paragraph (ARP_ratio range, 23-24 / 65-74 clusters, 0.9159 vs 0.9335, lowest Gini) were also checked by eye against the tables |
| 22-35 | Spot checks of `02_report_toc.md`: 2,952,675 bookings, 813,985 users, 39.34 %, 48,010 burst repeats, 713,754 first bookings, 1,710,952 train targets, 50,769 valid cold, 33,702 parameters, 11,128 s, best epoch 8, JSD 0.439 bits, warm ItemKNN R@5 0.3036, warm LR NDCG@5 0.3354, warm main 0.6104 | PASS | all equal their source |
| 36 | `04_demo_options.md` section 0 table, warm "Main model Recall@5" | PASS after fix | was 0.5965, see section 2 |
| 37-41 | Other `04` spot checks: cold main 0.5252, warm LR 0.4969, cold ItemKNN 0.1318, first test event 2689344, last train event 2424705 | PASS | |
| 42 | `02` test window "mid-November to December" | PASS | `eda_clean.json` splits.test 2014-11-19 to 2014-12-31 |
| 43 | `reports/benchmark_results.csv` contains the early-stopping runs | NOTE | it does not (stale); see section 3 |

## 2. Mismatches found and fixes made

| File:line | Problem | Fix |
|---|---|---|
| `docs/plans/final_prep/04_demo_options.md:35` | Warm "Main model Recall@5" was 0.5965. That is the query embedding without sameDest; the main model (query embedding + sameDest, the model the demo runs) is 0.6104 (`final_benchmark.csv`). | Changed to 0.6104. |
| `docs/plans/final_prep/04_demo_options.md:41` | Consistency arithmetic used 0.5965: (1,500 x 0.5252 + 3,000 x 0.5965) / 4,500 = 0.573. | Recomputed with 0.6104: 0.582, which is still within about one binomial standard error (0.007) of the demo sample's 0.577; correction note added in the file. |
| `docs/plans/final_prep/02_report_toc.md` section 3 | Gap table still listed G1 (ES bootstrap CIs), G6 (slices) and G7 (RQ4 metrics) as missing; they now exist in `results/week5_final/`. | Added a one-line dated status note above the gap table; G2, G4, G5 marked still open. |

Not errors, recorded for the user:
- `results/week4_rebuild/smlprec_query_cold_L0_cold_train_es.json` field `run` says "checkpoint of notebook 01c", but its `checkpoint` is the 01d ES checkpoint (confirmed by check 13). Text label only; already noted in `01` section 10.
- Two popularity fallbacks are in use for cold rows: global popularity for the warm-only plain row (from the 01d cold JSON) and first-booking popularity for ItemKNN (07b). Both are labelled.
- `results/week4_rebuild/seeds_summary.json` (`paired_01d_minus_01c`): the keys `warm/qt_minus_logreg/*` and `cold/qt_minus_logreg/*` hold exactly the same values as `warm/qt/*` / `cold/qt/*`. It looks like a naming bug in the older aggregation script (`scripts/aggregate_seeds.py`), not part of tonight's package; the deck does not use these keys.
- The demo storyboard (`results/week5_final/demo_storyboard.json`, regenerated 01:29 by the demo agent) now uses the 07b baselines, as the tables do; the deck's demo slide reads this from the JSON's `models` field. The live demo (`scripts/demo_ui.py`) still uses notebook 07 baselines.
- Scripts produced tonight (`final_results_summary.py`, `final_export_topk_smlp4rec.py`) needed no change, so the summary was not re-run.

## 3. What the user still has to decide

1. **Logistic regression baseline: 07b (reported) or notebook 07.** Your rule (every model except the warm-only original trained on all events) points to 07b, which is what the package and the deck use. Notebook 07's logistic regression (C = 10, 3 SAGA passes) is stronger on the same rows: warm NDCG@5 gap +0.0893 instead of +0.1107 (`sensitivity_logreg_nb07.csv`). The deck footnotes this in the slide-12 notes and shows it as appendix A1. Decide whether to keep 07b with the footnote or to re-tune 07b (C grid above 1.0, or more passes) so the like-for-like baseline is not weaker than the earlier run.
2. **Season slice is degenerate.** Test bookings are made 2014-11-19 to 2014-12-31; 51.56 % of test check-ins are in December and 21.14 % in January. The deck says season is not tested. Options: rolling-origin split, or pooling valid + test for the slice (protocol change, needs your OK).
3. **Single seed for the early-stopping runs** (01d main, 01e plain, 07b logistic regression). CLAUDE.md section 6 asks for at least 3 seeds of the main model; only the 3-epoch 01d runs have seeds 2022-2024. Rerun 01d ES for seeds 2023 / 2024 (about 3 h each on CPU per `timing_seconds`), or state the limitation (the deck does).
4. **`reports/benchmark_results.csv` is stale**: no early-stopping rows; its logistic regression is notebook 07 and its main-model rows are the 3-epoch runs. Rebuild it from `results/week5_final/final_benchmark.csv` (or update `scripts/build_benchmark_results.py`); protocol-owned file, not changed here.
5. **First-bookings step on a mixed footing**: the "training with first bookings" effect is measured on the 3-epoch runs (01c vs 01d, 3 seeds), because 01c was not rerun with early stopping. The deck labels it.
6. **Label wording**: the tables say "warm rows only (L>=1)", `01` and the deck say "trained on warm users only". Both describe the same training set; pick one wording for the report.
7. Demo UI and notebook-08 export still use the 3-epoch checkpoint (`01` section 10, item 2); the storyboard uses the ES checkpoint.
