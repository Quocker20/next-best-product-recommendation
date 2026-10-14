# Plan — user-level test with one fixed top-20 per user (2026-10-14)

Status: APPROVED by the user on 2026-10-14; implemented and run on the full real data. Supersedes the row-level autoregressive / teacher-forced **test** of `plan_oct_12_autoregressive_test.md` (autoregressive scoring stays for valid and early stopping). Protocol text: `docs/eval_protocol.md` (section "Test protocol"). Decision log: `PROGRESS.md`.

## What changed
Each test user gets ONE list of 20 clusters, fixed for the whole test window, instead of one ranking per booking.
1. Step 1: the user's first test booking, true history; top-20 and probabilities (P1).
2. Step 2: the second test booking with the P1 top-1 in place of booking 1 in the history (no label feedback); a user with one test booking reuses the context of booking 1 with that top-1 appended. Top-20 and probabilities (P2).
3. Rerank P1 and P2 into one top-20.
4. Recall@K and NDCG@K (K = 5, 10, 20), set-based, per user against the distinct clusters of all the user's test bookings; mean over users (`nbp.eval.user_level.fixed_top_user_metrics`). Standard definitions: a repeated booking of a cluster counts once.

## Decisions
- **Probabilities:** over all 100 clusters before the cut; softmax for SMLP4Rec, score / row sum for ItemKNN and popularity (user decision, 2026-10-14).
- **Rerankers compared:** `m1_first` (P1 only), `m2_merge` (union by probability, max per cluster), `m3_sqrt_alpha` (P2 mapped to sqrt(alpha·p), alpha = P1 top-1 probability, then merged).
- **Chosen reranker: `m2_merge`.** Main model, test, all users, seed 2024: NDCG@10 m1 0.4298 / m2 0.4474 / m3 0.4459; Recall@20 0.8308 / 0.8588 / 0.8537. Plain and ItemKNN prefer m1 by <= 0.005 NDCG@10; the comparison uses one reranker for every model.
- **Seed:** one fixed seed, 2024 (`report_seed`), for every reported and recommended number; no mean ± std across seeds. No retraining: the week-6 checkpoints are reloaded and the baselines refitted.
- **Data:** full real data, date split valid 2014-09-01, test 2014-11-01 to 2015-01-01 (test 378,788 bookings, 222,428 users; windows checked in the run).

## Code and outputs
`src/nbp/eval/two_step.py`, `src/nbp/eval/user_level.py`, `scripts/eval_user_top20.py` (parts baselines / smlp / report, two venvs), `scripts/export_test_with_result.py`; `results/week6_retrain/report_numbers.md` (single sheet for the report), `user_top20.json`, `user_top20_tables.md`, `user_top20_bootstrap.{json,md}`, `user_top20_reranker_choice.md` (m1 / m2 / m3 side by side); per-user lists `data/interim/week6/user_top20.npz`; `data/processed/expedia_date_split/csv/test_with_result.csv`. Tests: `tests/test_two_step.py`, `tests/test_user_level.py`.

## Reading the numbers
66.3 % of the test users have one test booking and 70.3 % book a single distinct cluster, so the user-level scores stay close to the row-level ones (NDCG@5 of the main model: 0.389 for one-booking users, 0.338 for two, 0.322 for three or more).

## Open items
Old row-level eval code removed only after notebooks 01-05 stop using it; a slice by number of test bookings; slides not yet rebuilt on these numbers.
