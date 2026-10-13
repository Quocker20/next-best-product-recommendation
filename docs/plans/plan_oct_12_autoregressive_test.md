# Plan — autoregressive test protocol (2026-10-12)

Status: APPROVED by the user on 2026-10-12 (supervisor's request). The autoregressive test is the **primary** protocol; teacher-forced stays as the secondary protocol for the report discussion. `docs/eval_protocol.md` and `CLAUDE.md` §6 updated the same day. The ladder plain → + query embedding → + sameDest and both baselines are compared under the primary protocol. Sections 2, 3 and 6 below still carry the original proposal wording; where they say "report both with teacher-forced as headline" the approved rule above wins.

## 1. What changes

Training is unchanged: teacher forcing, the same 80/10/10 global temporal split over next-booking targets, and the same early stopping.

Only scoring changes. Inside the test window, the true cluster of a test booking is never shown to the model. When a user has several test bookings t1 < t2 < t3:

| Test row | History fed today (teacher-forced) | History fed in the new protocol (autoregressive) |
|---|---|---|
| t1 | real bookings before the test window | same (nothing hidden yet) |
| t2 | real + **true t1** | real + **top-1 prediction for t1** |
| t3 | real + true t1 + true t2 | real + top-1(t1) + top-1(t2) |

- Every row is still ranked over all 100 clusters, and the full top-K list is scored against the true cluster.
- Only the top-1 of each row is appended to the history used by the user's next test row.
- Bookings before the test window (train and valid periods) stay real: they are the user's known past.
- The query embedding of each row stays the row's own search (destination, dates, party, package, channel). That information is known at search time, so it is not hidden.

Current code: RecBole builds every history from the full booking sequence before the split (`sequential_dataset.py` `data_augmentation`, then `build` splits by time), so today the test histories contain true test-period clusters. This was checked on 2026-10-12.

## 2. Is it a sound protocol?

It is sound as a **second protocol, next to the teacher-forced one, not as a replacement.**

What it measures: how well the model plans a user's future bookings when no feedback arrives after the test start. Equivalently, it measures multi-step forecasting with an information cut at the test boundary.

What it does not measure: the deployed setting. In production a booking that happened is observed, so the real history is available. Teacher forcing stays the closest offline proxy for "recommend the next booking".

Expected effects (to verify, not assumed):

- **Error compounding.** One wrong top-1 enters the history and can bias every later row of that user. If the top-1 hit rate (Recall@1) is p, most injected items are wrong whenever p < 0.5.
- **Exposure bias.** The model was trained only on true histories and never on its own predictions. A distribution shift is expected.
- **Self-reinforcement through sameDest.** A predicted cluster appended at the searched destination is boosted by sameDest on the user's next search at the same destination. The model then confirms its own guess. This is the biggest risk for the main model and must be reported separately.
- **The comparison shifts in favour of history-free models.**
  - Logistic regression uses no history, so its scores are identical under both protocols.
  - ItemKNN and every SMLP4Rec variant can only lose.
  - The RQ1 conclusion may therefore change, which is exactly why the protocol is worth running.
- **Most rows are unaffected.** A row changes only when the user has at least one earlier test booking. The share of such rows must be computed first; if it is small, the headline barely moves and the per-step slices carry the story.

## 3. Design decisions (proposed defaults)

| Decision | Default | Reason |
|---|---|---|
| What is appended | top-1 cluster (argmax, padding excluded) | user request; deterministic |
| Destination of an appended item (for sameDest) | the searched destination of that test row (`ctx_dest_id`) | known at search time, not a label |
| History window | last 20 items, as in training (`MAX_ITEM_LIST_LENGTH`) | same model input |
| Cold test rows (first booking in test, L = 0) | scored with an empty history; their top-1 is appended for the user's next row | consistent: the first booking is also hidden |
| Valid window | same autoregressive rule when tuning the sameDest weights and the ItemKNN K | tune and test under one protocol; early stopping stays teacher-forced (training is unchanged) |
| Teacher-forced results | kept and reported next to the new ones | reference, and the gap is the main finding |
| Seeds | 3 seeds for the main model, as in §6 | unchanged |

## 4. Metrics

Unchanged metrics: Recall@K and NDCG@K, K = 5, 10, 20. Full ranking. Warm, cold and all row sets. Paired bootstrap 95 % CI.

New in reports:

1. **Protocol gap**: autoregressive minus teacher-forced, per model, with a paired bootstrap on the same rows.
2. **Per-step slice**: metrics by h = number of predicted items already in the history (h = 0, 1, 2, 3–5, 6+). h = 0 rows must equal teacher-forced exactly, which doubles as a correctness check.
3. **Injected-history accuracy**: Recall@1 per step, i.e. the share of appended items that were right.
4. **sameDest self-reinforcement**:
   - share of rows where the top-1 equals an appended (predicted) item at the same destination;
   - accuracy on those rows against all other rows.
5. **Row counts**:
   - test rows with h ≥ 1;
   - users with ≥ 2 test bookings;
   - the h distribution.
   All counts are printed by code.

## 5. Implementation steps

1. **Counts first (no model).**
   - Script `scripts/expedia_test_step_counts.py` writes `results/<week>/test_step_counts.json`, computed on the new split once it is defined.
   - It holds rows by h, users with ≥ 2 test bookings, and h = 0 rows per row set.
   - Go / no-go: if the h ≥ 1 share is tiny, report it and keep the slice tables only.
2. **Shared rollout function** in `src/nbp/eval/autoregressive.py`. One function serves all models:
   - inputs: test rows sorted by (user, time), the real history before the test window, a `score_fn(seq, length, extra) -> (n, 100)` callback, and the window length;
   - loop: process step h = 0, 1, … across all users in one batch per step; take the top-1; shift it into each user's window; also roll the destination list for sameDest;
   - outputs: an (n, 100) score matrix aligned with the original row order, the appended items, and h per row.
3. **Unit tests** in `tests/test_autoregressive.py`, against hand-computed examples:
   - h = 0 rows reproduce teacher-forced scores;
   - true test clusters never reach the model (with a probe model that would score a leaked item);
   - the window drops the oldest item after 20;
   - several users with uneven step counts;
   - the cold user's first row has an empty history.
4. **Models.** Wire the rollout into the kept notebooks, scoring both protocols:
   - main and query-embedding: `01d`;
   - plain warm: `01`;
   - ItemKNN and logistic regression: `07b`, where logistic regression should show an identical result under both protocols as a sanity check.
5. **Tuning.**
   - Re-tune the sameDest weights and the ItemKNN K on valid, under the autoregressive rule applied to the valid window.
   - Read test once.
6. **Reporting.**
   - `docs/eval_protocol.md` gains a section "Autoregressive test (no feedback)".
   - Each result table states the protocol.
   - `reports/benchmark_results.csv` gains a `protocol` column.
   - `PROGRESS.md` entry.

## 6. Open questions for the user / mentor

- The new train / valid / test split strategy from the supervisor: this plan works for any time-ordered split, but step 1 must run on it.
- Top-1 only, or also a top-1 rollout without sameDest, to separate compounding from self-reinforcement? Proposed: yes, as a diagnostic row only.
- Does the report headline stay teacher-forced, with autoregressive as a robustness section, or the reverse?
