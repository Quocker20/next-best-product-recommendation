# Plan: SMLP4Rec with a query token (context-aware, no hybrid) — notebook 1c

Date: 2026-10-07. Status: draft, waiting for user go-ahead.

## Why

Notebook 01 feeds SMLP4Rec only `item_id` sequences (max 20 hotel clusters). `user_id` only groups bookings and `date_time` only orders/splits them. The 21 other raw columns, including everything about the current search, never reach the network (`configs/smlprec_expedia.yaml`: `load_col.inter = [user_id, item_id, timestamp]`, `selected_features = []`). A logistic regression on destination + trip context alone (nb07) beats plain SMLP4Rec (NDCG@5 0.354 vs 0.214 on all test events), so plain is a history-only model, not a context-aware recommender. This run puts the current search inside the model. The post-hoc prior / sameDest hybrid is **not** touched here.

## Scope

- One run: SMLP4Rec + query token, 3 epochs, same config as nb01 (2 layers, hidden 64, dropout 0, CE over 100 clusters, lr 1e-3, batch 1024, seed 2022).
- Same split as nb01 (RecBole global temporal 80/10/10 over next-booking targets). Warm rows only (L >= 1): 218,670 valid / 218,670 test. Cold (L = 0) rows are out of scope for 1c (the query token could serve them later; note as follow-up).
- No hybrid, no prior, no sameDest, no per-position context in this step.
- One seed. Dataset stated on every table: Expedia.

## Query features (raw column -> model input)

All are fields of the **target booking's own search** (known at query time). Forbidden, as in nb07: `hotel_continent`, `hotel_country`, `hotel_market`, `orig_destination_distance`, `user_location_region`, `user_location_city`, `cnt`, `is_booking`.

| Query field | Raw source | Encoding |
|---|---|---|
| destination | `srch_destination_id` | embedding of the id; vocabulary = destinations with >= 5 train targets, rest -> OOV id 0. `destinations.csv` (latent features d1..d149) is **not used** in 1c (user decision 2026-10-07) |
| destination type | `srch_destination_type_id` | embedding |
| check-in month | month of `srch_ci` | embedding (13 incl. unknown) |
| lead time | `srch_ci` - `date_time` (days) | bucket embedding (same bins as nb07 `LEAD_BINS`) |
| stay length | `srch_co` - `srch_ci` (nights) | clipped bucket embedding (nb07 `STAY_CLIP`) |
| adults, children, rooms | `srch_adults_cnt`, `srch_children_cnt`, `srch_rm_cnt` | clipped small-int embeddings |
| package | `is_package` | embedding (2) |
| channel, mobile | `channel`, `is_mobile` | embeddings |
| site, point-of-sale continent, user country | `site_name`, `posa_continent`, `user_location_country` | embeddings (user country as in nb07; train-count threshold) |

Values nulled by cleaning (zero adults, `srch_co < srch_ci`, missing `srch_ci`) map to an explicit "unknown" id, same convention as nb07. Derivation reuses the logic of `src/nbp/data/clean.py` (`party_type` not needed); do not re-derive differently.

## Architecture (new class, legacy `src/models/smlprec.py` stays untouched)

`src/nbp/models/smlprec_query.py`: `SMLPRECQuery`, reusing the legacy `SMLP` mixer block.

- History: `item_embedding(item_seq)` as today, positions 0..19 (RecBole right-pads, same layout as nb01 so the ablation is clean).
- Query token at fixed position 20: sum of the field embeddings above plus a learned `[MASK]` vector -> `(B, 1, H)`. Sequence length becomes 21 (`SMLP([1, 21, H])`).
- Output = hidden state at position 20 (fixed index, not `item_seq_len - 1`) -> LayerNorm -> dot with `item_embedding.weight` -> CE.
- Embedding init = existing `_init_weights` (normal 0.02).
- Parameter count printed and saved (nb01 plain has 33,702). The destination embedding will dominate the new count (~34.5k ids x 64); state it.

## Data (no edits to legacy scripts; rebuild policy)

1. New script `scripts/expedia_query_to_recbole.py`: one chunked pass over `data/raw/hospitality/expedia/train.csv`, `is_booking == 1`, **same row order as `expedia.inter`**, writes `data/interim/recbole/expedia_query/expedia_query.inter` with `user_id:token`, `item_id:token`, `timestamp:float` plus the query columns above as `:float` columns (float32 holds ids exactly below 2^24; the model casts to long). Plain ints: lead/stay buckets computed in the script.
2. RecBole's sequential augmentation copies every extra column to the target row (this is the query) and also builds a `<field>_list` history column for each (already seen with `srch_destination_id` in nb03/04). Per-position history lists are therefore created but unused in 1c; they cost memory (about 7 fields x 2.18M x 20 x 4 bytes ~ 1.2 GB as float32). If RAM is a problem, drop fields from `load_col` after the first check, or subset the history lists in the notebook.
3. Train-only vocabularies (destination, user country) are built in the notebook from `train_data` targets and passed to the model as lookup tensors, so valid/test-only ids map to OOV (RecBole's own vocabulary is built over all splits and would leave untrained embeddings).

## Notebook `notebooks/hospitality/smlp4rec/01c_train_test_smlp4rec_query.ipynb`

Copy the structure of nb01 (explicit train loop, per-epoch checkpoint, best epoch by valid NDCG@10 as nb01 config, metrics from `nbp.eval.metrics`). Sections:

1. Config + data (dataset `expedia_query`, `load_col` with the query fields); print split sizes.
2. **Split-identity check against nb01** (assert): same split sizes, same first/last target time, same `y` of valid and test rows, same history sequences (`item_id_list`). If row order or item-id mapping differs, stop.
3. Vocabulary from train targets; coverage table (share of valid/test targets with OOV destination / OOV user country).
4. Model + 3-epoch training; per-epoch valid and test Recall@K / NDCG@K (K = 5, 10, 20), full ranking over 100 clusters.
5. Results table: plain (nb01 numbers read from `results/week3_implementation/smlprec_expedia_run.json`) vs query-token model, same rows, plus global popularity and repeat-last. Slices as in nb06: seen / unseen users, L >= 1 only (warm). Paired bootstrap of the difference vs plain (1000 resamples, seed 0).
6. Diagnostic (cheap, same checkpoint, no training): rescore test with the query fields shuffled across rows, and with the history masked to padding. Shows how much the model relies on query vs history. Not a tuning step.
7. Save `results/week4_rebuild/smlprec_query_run.json` (config, params, per-epoch metrics, slices, bootstrap, coverage numbers) and a run folder under `experiments/`. Every number in later slides/reports comes from this file.

## Reusable code + tests

- `src/nbp/models/smlprec_query.py` + `tests/`: forward output shape `(B, 100+1)`, query changes the output while history is fixed, padding rows work, OOV id lookup maps unseen ids to 0.
- Feature builder (lead/stay bucket, month, clip, unknown handling) as a function in `src/nbp/data/` with a hand-computed test.

## Verification before trusting any number

- Leak check: the feature list contains none of the forbidden columns (assert, same style as nb07 `FORBIDDEN`).
- Query-shuffle diagnostic must drop to roughly the plain level; history-masked must still be well above popularity. If shuffle does not hurt, the query path is broken.
- Result "too good" check: compare against logistic regression (0.354 NDCG@5 all events, 0.365 seen) and the hybrid (0.435); a context-aware model far above the hybrid warrants a leakage review.
- Expected honest framing: the number answers "does putting the query inside SMLP4Rec beat history-only?", not "is it better than the hybrid".

## Risks / decisions

- **Destination id embedding**: ~34.5k destinations, thin signal per rare one; a 64-dim table adds ~2M parameters to a 34k-parameter model and 3 epochs may not train rare ids. Fallbacks if it overfits: destination-market backoff (modal `hotel_market` from train-period data only, as in nb03) or fewer hash buckets. Decide after the coverage table. `destinations.csv` d1..d149 stays out of scope for 1c.
- **Right-padding**: query at fixed position 20 sits after pad gaps for short histories. This is learnable (token mixing is position-specific) but, if short-history rows underperform, left-pad as an ablation.
- **CPU time**: nb01 trained in minutes per epoch; extra embedding lookups should be a modest increase. Time per epoch is recorded.
- **Memory** of the auto-generated history lists (above).
- Three epochs is a pipeline-validation budget, not a tuned result; state it in the notebook and in the result file.

## After 1c (not now)

Compare with the hybrid on the nb06 slices; then decide whether the prior / sameDest still add anything on top of the query-token model, whether per-position context helps, and whether the query token can replace the cold-start prior (L = 0).
