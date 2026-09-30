# Week 3 — SMLP4Rec pipeline-validation run on Expedia

Goal: check that the SMLP4Rec code runs end to end on Expedia bookings. **This is a pipeline check, not a benchmark result**: 1 epoch, untuned light config, CPU, no search context.

All numbers below are copied from `smlprec_expedia_run.json` (written by `scripts/run_smlprec_expedia.py`); the RecBole log is `smlprec_expedia_run.log`.

## What ran

| Step | File |
|---|---|
| Model (copied from the SMLP4Rec fork, minimal changes marked `# ADAPTED`) | `src/models/smlprec.py` |
| Bookings → RecBole `.inter` (format export only: user, cluster, unix time) | `scripts/expedia_to_recbole.py` → `data/interim/recbole/expedia/expedia.inter` |
| Config | `configs/smlprec_expedia.yaml` |
| Runner (train, validate, test, heuristics on the same test rows) | `scripts/run_smlprec_expedia.py` |

Environment: Python 3.11 venv `C:\Users\quoca\.venvs\smlp4rec` (torch CPU, numpy 1.23.5, pandas 1.5.3) with RecBole 1.0.1 installed editable from the patched `MLP4Rec` clone.

Changes to the model code (and nothing else):
- sequence length read from `MAX_ITEM_LIST_LENGTH` instead of a hard-coded 50;
- empty `selected_features` allowed (hotel clusters have no item attributes), so the item-feature layer is skipped.

Left as in the original: one SMLP block reused for every layer (shared weights), no residual connection.

## Setup

- Data: 3,000,693 bookings, 813,985 users, 100 clusters; 2,186,708 next-booking targets (every booking except each user's first).
- Split: global temporal 80/10/10 over targets, ordered by booking time. History = all earlier bookings of the user, capped at 20.

| Split | Targets | First target | Last target | Mean history length |
|---|---|---|---|---|
| train | 1,749,368 | 2013-01-07 | 2014-10-08 | 5.853 |
| valid | 218,670 | 2014-10-08 | 2014-11-19 | 7.391 |
| test | 218,670 | 2014-11-19 | 2015-01-01 | 7.255 |

- Model: 2 layers, hidden size 64, dropout 0.0, CE over all clusters, lr 0.001, batch 1024, 1 epoch, seed 2022; 33,702 parameters.
- Evaluation: full ranking over all 100 clusters (`mode: full`); valid metric NDCG@10.
- Time: data build 42.2 s, train + valid 156.9 s, test 6.4 s.

## Results (test)

| Ranker | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| Global popularity | 0.1478 | 0.2476 | 0.3925 |
| Repeat last cluster, then popularity | 0.2570 | 0.3439 | 0.4762 |
| **SMLP4Rec (1 epoch)** | **0.3350** | **0.4624** | **0.6291** |

SMLP4Rec test NDCG@5 / 10 / 20 = 0.2426 / 0.2836 / 0.3255; MRR@10 = 0.2290. Valid Recall@5 = 0.3325 (test is close, no sign of split mismatch).

## Reading the result

1. **The pipeline works.** Export, sequence build, temporal split, training, full-ranking evaluation and checkpoint reload all ran without errors in under 4 minutes on CPU.
2. **History alone carries signal.** On the same test rows SMLP4Rec beats repeat-last by +7.80 pp Recall@5 and global popularity by +18.72 pp.
3. **It is far below the week-2 context floors, as expected.** Destination popularity reached 53.07 % Recall@5 in week 2. This model sees only the past clusters, not the current search's destination, so it cannot use the strongest signal. That gap is exactly what the planned query token (current search context) must close.
4. **Not comparable to week-2 numbers yet.** Week 2 split all bookings at the 80th percentile of `date_time`; this run splits next-booking targets 80/10/10, and users' first bookings (cold start) are never test targets here. Test covers only users with at least one earlier booking.
5. **Recall@K equals Hit@K** because each test row has exactly one target.

## Known limits of this run

- 1 epoch, no tuning, no seeds beyond 2022 — numbers will move.
- No search context (destination, dates, party, package) and no cold-start hybrid.
- The SMLP4Rec source repo has no license file; the code is copied here for internal research only and labeled as an adaptation.
- A first run used timestamps in the wrong unit (pandas 3 parses to microseconds, so bookings within ~17 min tied). Fixed in `scripts/expedia_to_recbole.py` and rerun; the results above are from the fixed run.
