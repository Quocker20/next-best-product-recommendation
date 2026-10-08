# Replan: 8 Oct – 21 Oct 2026 (13 days left)

Written 2026-10-08 after notebook 01c was re-run with the final model (query token + sameDest). Revised the same day by user decision: **one main model, two classical baselines, data processing and tuning only.** Supersedes the Day 6-10 part of `plan_oct_01_14.md`; the first plan stays as history.

## 0. Scope decision (user, 2026-10-08)

- **Main model (locked): SMLP4Rec query token + sameDest.** All remaining effort goes to data processing and tuning this model.
- **Baselines: exactly two, both simple and classical, no rule-based baselines.**
  1. **ItemKNN** (item-item collaborative filtering on user x cluster).
  2. **Logistic regression** (destination + trip context, no history).
- Nothing else is run or planned: no popularity / repeat-last rules, no MF, item2vec, GRU4Rec / SASRec, LightGBM, AdaGIN, LLM rerank, plain-PyTorch rebuild or learned gate. The user will add goals back later if needed.
- Unchanged: Expedia only, current 80/10/10 RecBole split over next-booking targets, warm (L >= 1) and cold (L = 0) scored separately.

## 1. Where we are (all numbers: test, Expedia, one seed, 3 epochs)

Warm users (L >= 1, n = 218,670), NDCG@5 / Recall@5:

| Model | Role | NDCG@5 | Recall@5 |
|---|---|---|---|
| ItemKNN | baseline | 0.2195 | 0.3161 |
| Logistic regression (destination + context, no history) | baseline | 0.3561 | 0.5219 |
| SMLP4Rec plain (history only) | ablation of main model | 0.2439 | 0.3369 |
| SMLP4Rec query token | ablation of main model | 0.4209 | 0.5898 |
| **Query token + sameDest** | **main model** | 0.4475 | 0.6060 |
| Old hybrid (plain + prior + sameDest) | internal reference, not a baseline | 0.4539 | 0.6106 |

Findings so far:
- Destination and search context carry most of the signal; putting the search inside the model (query token) closes the gap to the history-only models.
- sameDest adds about +0.027 NDCG@5, all on the 30 % of rows whose destination is already in the history.
- Old hybrid beats the main model by 0.0065 NDCG@5 (bootstrap CI [-0.0074, -0.0056]). sameDest is zero on new-destination rows (70 % of warm test rows, 153,263), so the whole gap sits there: query token 0.5483 / 0.3746 vs old hybrid 0.5554 / 0.3842 (Recall@5 / NDCG@5). Only 2.4 % of test targets have an out-of-vocabulary destination (vocabulary = destinations with >= 5 train targets, 13,766 ids), so the gap is rare-destination quality, not only OOV. Tuning and data work should try to close it inside the model.
- sameDest is a post-hoc late-fusion term (`log q_model + w_s * log q_sameDest`, alpha 0.5, w_s 1.0 on known-destination rows, 0 on new), not an input of the network.
- Query inputs already in notebook 01c (14 fields): destination, destination type, check-in month, lead time, stay length, adults, children, rooms, package, mobile, channel, site, point-of-sale continent, user country. Shuffling the query drops NDCG@5 to 0.0864; masking the history leaves 0.3496.
- Training: train loss 5184 / 4632 / 4523 over 3 epochs, but valid NDCG@5 is flat from epoch 2 to 3 (0.4266). The retrain on 2026-10-08 reproduces the 2026-10-07 run exactly (same test metrics), so the seed is deterministic.
- Cold rows (L = 0, test 49,443): logistic regression 0.3454 / 0.5061 (NDCG@5 / Recall@5); ItemKNN cannot score them (falls back to global popularity, 0.0813 / 0.1317); the query token on an empty history and the prior are in `smlprec_query_cold_L0.json`.

## 2. Open items

| Item | Status |
|---|---|
| Data work: audit, cleaning, query-field features, rare-destination handling | Not systematically done |
| Model tuning: epochs (loss still falling at 3), hyperparameters, comparable budget for the two baselines | Not done |
| Seeds: mean +- std over >= 3 for the main model | One seed |
| Slices (main model + two baselines): season, trip context, user activity (L), known vs new destination | Only L and seen/unseen done |
| RQ4 metrics (diversity, popularity bias, coverage per destination / season) | Not computed |
| Final report + slides | Week-3 deck / report describe the discarded prior variant |

## 3. Plan

Rule: loop = change data or model -> benchmark on valid -> record row -> analyse. Test is read once at the end of each stage.

### A. Lock and record (8-9 Oct) — DONE 2026-10-08
- [x] `PROGRESS.md`: decision record.
- [x] `docs/eval_protocol.md`.
- [x] `reports/benchmark_results.csv` (15 rows, built by `scripts/build_benchmark_results.py` from the JSON files; rerun after every new result).
- [x] Commit and push.
- [x] Removed empty `src/nbp/priors/`; CLAUDE.md, README and docs rewritten for this scope.

### B. Data processing (9-12 Oct) — mostly DONE 2026-10-09
Done: burst collapse + flags (`docs/data_cleaning_strategy.md`), exports aligned to the cleaned parquet, leakage audit (`results/week4_rebuild/leakage_audit.json`), notebooks 01-07 rerun on the collapsed data, sensitivity runs. Results: `PROGRESS.md` (stage B results) and `reports/benchmark_results.csv`. Cold users in training: done in notebook 01d (better on warm, cold and all events; cold gap to logistic regression closed; see `PROGRESS.md`). Still open below: query-field ablations by retraining, `destinations.csv` latent features, rare-destination handling beyond the extra training rows.

- [ ] Align the RecBole exports (`expedia_to_recbole.py`, `expedia_query_to_recbole.py`) with the cleaned `expedia_bookings.parquet` (they read raw `train.csv` today), then rerun 01c and 07 on one source.
- [ ] Leakage audit of every query field and of the sameDest feature (the target row's own booking must never reach its inputs; checklist in `plan_oct_01_14.md`, Day 9).
- [ ] Data audit of the interim tables: duplicates, `srch_co < srch_ci`, missing `orig_destination_distance`, unknown destinations; report counts by code, fix in the scripted pipeline only.
- [ ] Query-field work, each tested as an ablation on valid (the 14 fields above are already in; do not redo them): per-field drop-one contribution; candidates not yet used: `destinations.csv` latent features d1..d149 (best lead for rare destinations, currently excluded), check-in weekday, user city / region (known at query time, excluded so far; check sparsity and leakage first), per-position context on the history bookings. `orig_destination_distance` and every `hotel_*` field stay excluded (describe the booked hotel).
- [ ] Rare-destination handling (source of the 0.0065 gap, on new-destination rows): frequency threshold (min count 5 now), destination-latent features, market backoff inside the model.
- [ ] Cold users (L = 0, 49,443 test rows): the query token on an empty history is not yet clearly above the destination prior on valid (0.5127 vs 0.529 Recall@5); decide the cold rule for the main model (query token alone, or with a prior) and compare with logistic regression. ItemKNN is reported as not applicable on cold rows (popularity fallback, stated in the table).

### C. Model optimisation (11-15 Oct)
- [ ] Epoch check: 3 vs 10 vs early stopping on valid.
- [ ] Small grid on valid: embedding size, layers, dropout, max sequence length, learning rate, weight decay, query-token fields. Fix the grid size up front; give ItemKNN (neighbours, shrinkage) and logistic regression (regularisation C) a comparable small grid.
- [ ] Re-tune the sameDest fusion weights per history bucket on valid for the final model.
- [ ] 3 seeds for the final configuration (about 16 min / seed on CPU, run overnight). ItemKNN is deterministic; logistic regression on 3 seeds only if the solver is stochastic.

### D. Evaluation (14-16 Oct)
- [ ] RQ1 slices for the main model and the two baselines: user activity (L buckets, single-booking users), season (check-in month), trip context (solo / couple / family / group, package), known vs new destination; bootstrap CIs.
- [ ] RQ4 metrics for the same three: intra-list diversity, popularity bias, coverage per destination and per season.

### E. Write-up (16-21 Oct)
- [ ] Rebuild report and slides from `results/` JSON only (no number typed by hand).
- [ ] Report the main model against two classical baselines. State limitations: cluster is a proxy for room category, one dataset, offline only, no in-stay data, only two simple baselines.
- [ ] Clean-up: stale scripts, `pyproject.toml` dependencies, CLAUDE.md §3 / §6 / §8 to match the new scope (with user OK).

## 4. Cut order if time slips
1. RQ4 coverage per destination / season (keep diversity + popularity bias).
2. Optional query-field ablations beyond party / season / days-ahead.
Never cut: leakage audit, epoch check, 3 seeds on the main model, season and trip-context slices.

## 5. Risks
- A gap of 0.0065 NDCG@5 may be below seed noise; the 3-seed run decides whether old hybrid vs main model is real.
- Feature engineering can leak: every new field goes through the audit before it counts.
- CPU time: 3 seeds x tuning grid; keep the grid small, run overnight.
- RecBole venv (Python 3.11, numpy 1.23) differs from `.venv`; run the neural pieces on the matching kernel.
- Logistic regression is already strong (0.52 Recall@5 with no history). If the main model's margin over it is small on some slice, report it as is.

## 6. Decisions needed from the user
None open.
