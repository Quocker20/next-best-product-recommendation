# Week 3 — SMLP4Rec pipeline-validation run on Expedia (3 epochs)

Goal: check that the SMLP4Rec code runs end to end on Expedia bookings and see what extra epochs do. **This is a pipeline check, not a benchmark result**: 3 epochs, untuned light config, CPU, history-only input (no search context).

All numbers below are copied from `smlprec_expedia_run.json` (written by `scripts/run_smlprec_expedia.py`); the RecBole log is `smlprec_expedia_run.log`. This folder was overwritten by the 3-epoch run; the earlier 1-epoch run gave the same epoch-1 numbers (Recall@5 0.335).

## What ran

| Step | File |
|---|---|
| Model (copied from the SMLP4Rec fork, minimal changes marked `# ADAPTED`) | `src/models/smlprec.py` |
| Bookings -> RecBole `.inter` (format export only: user, cluster, unix time) | `scripts/expedia_to_recbole.py` -> `data/interim/recbole/expedia/expedia.inter` |
| Config | `configs/smlprec_expedia.yaml` |
| Runner (train, valid + test after every epoch, heuristics on the same test rows) | `scripts/run_smlprec_expedia.py` |
| Slides (VN) | `reports/slides/week3/pipeline_validation_slides.html` / `.pdf` (`scripts/week3_pipeline_slides.py`) |

Environment: Python 3.11 venv `C:\Users\quoca\.venvs\smlp4rec` (torch CPU, numpy 1.23.5, pandas 1.5.3) with RecBole 1.0.1 installed editable from the patched `MLP4Rec` clone.

Changes to the model code (and nothing else):
- sequence length read from `MAX_ITEM_LIST_LENGTH` instead of a hard-coded 50;
- empty `selected_features` allowed (hotel clusters have no item attributes), so the item-feature layer is skipped.

Left as in the original: one SMLP block reused for every layer (shared weights), no residual connection.

## Setup

- Data: 813,985 users, 100 clusters, 2,186,708 next-booking targets (every booking except each user's first).
- Split: global temporal 80/10/10 over targets, ordered by booking time. History = all earlier bookings of the user, capped at 20.

| Split | Targets | Mean history length |
|---|---|---|
| train | 1,749,368 | 5.853 |
| valid | 218,670 | 7.391 |
| test | 218,670 | 7.255 |

- Model: 2 layers, hidden size 64, dropout 0.0, CE over all clusters, lr 0.001, batch 1024, 3 epochs, seed 2022; 33,702 parameters.
- Evaluation: full ranking over all 100 clusters (`mode: full`); valid metric NDCG@10; valid and test are scored after every epoch, the best epoch is picked on valid only.
- Time: data build 34.3 s, train + valid + per-epoch test 672.5 s, final test 7.3 s.
- Checkpoint (best epoch by valid, epoch 3): `C:\workspace\PROJECTS\next-best-product-recommendation\data\interim\recbole\saved\SMLPREC-Sep-30-2026_11-20-06.pth` (gitignored, under `data/interim/`).

## Results per epoch

| Epoch | Train loss | Train time | Valid Recall@5 | Valid NDCG@10 | Test Recall@5 | Test Recall@10 | Test Recall@20 | Test NDCG@10 |
|---|---|---|---|---|---|---|---|---|
| 1 | 6,686.1 | 194 s | 0.3325 | 0.2819 | 0.335 | 0.4624 | 0.6291 | 0.2836 |
| 2 | 6,631.4 | 212 s | 0.3347 | 0.2834 | 0.3359 | 0.4628 | 0.6292 | 0.2842 |
| 3 | 6,626.1 | 210 s | 0.3352 | 0.2838 | 0.3369 | 0.4645 | 0.6309 | 0.285 |

Best epoch on valid: 3/3. Test at the best epoch: Recall@5 / 10 / 20 = 0.3369 / 0.4645 / 0.6309, NDCG@5 / 10 / 20 = 0.2439 / 0.285 / 0.3269, MRR@10 = 0.2302.

## Results (test, best epoch) against the heuristics on the same rows

| Ranker | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| Global popularity | 0.1478 | 0.2476 | 0.3925 |
| Repeat last cluster, then popularity | 0.257 | 0.3439 | 0.4762 |
| **SMLP4Rec (3 epochs)** | **0.3369** | **0.4645** | **0.6309** |

## Reading the result

1. **The pipeline works.** Export, sequence build, temporal split, training, per-epoch full-ranking evaluation and checkpoint reload all ran without errors in about 11.9 minutes on CPU. Epoch 1 reproduces the earlier 1-epoch run exactly.
2. **History alone carries signal.** SMLP4Rec beats repeat-last by +7.99 pp and global popularity by +18.91 pp Recall@5.
3. **More epochs barely help.** Test Recall@5 moves +0.19 pp from epoch 1 to epoch 3 (valid Recall@5 0.3325 -> 0.3352) while train loss falls only 0.9 %. The curve is still slightly rising, so 3 epochs cannot show where it flattens, but the trend suggests the ceiling comes from the inputs (no search context) and the small, shared-weight model, not from training length. Extra epochs are not the main lever; the query token is.
4. **Far below the week-2 context floors, as expected.** Destination popularity reached 53.07 % Recall@5 in week 2; this model cannot see the current search's destination.
5. **Not comparable to week-2 numbers yet.** Week 2 split all bookings at the 80th percentile of `date_time` and included users with no history; this run splits next-booking targets 80/10/10, so users' first bookings (cold start) are never test targets.
6. Recall@K equals Hit@K because each test row has exactly one target.

## What the top-5 list is made of (epoch 3 checkpoint)

Computed by `behavior_stats` / `target_mix` in `scripts/run_smlprec_expedia.py` (also re-runnable alone with `--stats-only`); values are in `smlprec_expedia_run.json` under `top5_behavior` and `target_mix`. "Old" = the true next cluster was booked before by the user (includes the last booking); "new" = never booked before.

**What the user books next (share of target rows)**

| Books... | Train | Valid | Test |
|---|---|---|---|
| the last booked cluster | 14.77% | 14.62% | 14.87% |
| an older cluster (not the last) | 13.23% | 16.52% | 15.87% |
| an old cluster (incl. last) | 28.00% | 31.14% | 30.74% |
| a new cluster | 72.00% | 68.86% | 69.26% |

**What the model's top 5 does (all rows unless stated)**

| Statistic | Valid | Test |
|---|---|---|
| Top-1 is the last booked cluster | 82.74% | 83.04% |
| Last booked cluster is in the top 5 | 98.84% | 98.90% |
| Old booking (incl. last) hit in top 5 | 25.34% (of old rows: 81.39%) | 25.17% (of old rows: 81.89%) |
| New booking hit in top 5 | 8.17% (of new rows: 11.87%) | 8.52% (of new rows: 12.30%) |
| Top-5 slots that are not in the user's history | 45.60% | 46.21% |
| Global popularity top-5 on new-cluster rows (comparator) | 12.65% | 13.69% |

Check: old-hit + new-hit = 33.69% on test = Recall@5 (33.69%).

**Insight.** About 69% of next bookings go to a cluster the user has never booked. The model puts the last booked cluster at rank 1 in 83% of rows (repeat-last is right 14.87%, the same as its rank-1 accuracy), and 74.7% of its top-5 hits are old clusters. On new-cluster rows it hits 12.3%, below global popularity (13.7%). So this history-only model mostly re-surfaces past bookings; discovery of new clusters has to come from search context (destination prior, query token) and the hybrid.

## Known limits of this run

- 3 epochs, no tuning, one seed (2022).
- No search context (destination, dates, party, package) and no cold-start hybrid.
- The SMLP4Rec source repo has no license file; the code is copied here for internal research only and labeled as an adaptation.
- A first run used timestamps in the wrong unit (pandas 3 parses to microseconds, so bookings within ~17 min tied). Fixed in `scripts/expedia_to_recbole.py` and rerun; everything here is from the fixed data.
