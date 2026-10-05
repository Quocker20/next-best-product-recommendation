# 10-Day Plan: restructure and optimize (Thu 1 Oct – Wed 14 Oct 2026)

Living document. After each day the plan is reviewed and edited; record every change in the revision log at the bottom. Final report and slides: 15–21 Oct.

## Goal

Rebuild the pipeline from scratch as modular plain-PyTorch code in the project's own environment (no single-file RecBole model, no separate RecBole venv), and implement the decided hybrid:

```
score(k) = w_m[L] * log p_SMLP4Rec(k | history)
         + w_p[L] * log p_prior(k | searched destination)
         + w_s[L] * sameDest(k)
```

- `L` = number of prior bookings of the user before the current query. Cold users (`L = 0`): prior only (`w_m = w_s = 0`).
- `p_prior`: destination cluster counts smoothed to the destination's market (m = 5), unseen destination falls back to global; built only from data before the evaluated period; never uses the event's own `hotel_market`.
- `sameDest(k)`: recency-weighted flag (0.7^age) that the user booked cluster `k` at the searched destination before.
- Weights per bucket of `L` (0 | 1 | 2-4 | 5-9 | 10+), tuned on validation only.

Decision record: `PROGRESS.md` (Decision 2026-09-30). Success bar: beat the no-learning same-destination rule on both Recall@5 and NDCG@5 (was MAP@5; metric focus changed 2026-10-02) on test (warm users and all events), paired-bootstrap interval not crossing zero. If the hybrid cannot, report that the learned part only adds recall depth and move the destination information inside the model (per-booking destination / same-destination flag + query token).

## Reference numbers to start from (current RecBole-era split, test, users with history)

| Ranker | Recall@5 | MAP@5 |
|---|---|---|
| Plain SMLP4Rec (history only) | 0.3369 | 0.2133 |
| Destination prior alone | 0.5298 | 0.3083 |
| Same-destination history rule | 0.5888 | 0.3880 |
| Late fusion (model + prior, w = 1.5) | 0.5919 | 0.3669 |

Cold users (prior only): 0.5123 Recall@5. Combined all events: fusion hybrid 0.5772, rule 0.5747. Source: `results/week3_implementation/smlprec_expedia_late_fusion.json` (merged to master 2026-10-02; re-run in `notebooks/hospitality/smlp4rec/03_destination_prior.ipynb`). These numbers change under the new split; the new baselines on the new split replace them.

## To confirm before Day 1

- [x] **Add `torch` (CPU build), `pyyaml`, `pytest` to `pyproject.toml` and the project `.venv`** (heavy dependency, CLAUDE.md section 10). RecBole venv and the MLP4Rec clone are retired after the Day 5 check.
- [ ] **Protocol change: event-level global time split** (every booking after the cut is a target, including each user's first booking; validation carved from the end of train time). Replaces the RecBole 80/10/10 split over targets, which cannot score cold users.
- [ ] **Colab T4** (optional, Days 7-9 only). Decide by Day 6 after a first GPU timing check. Device-agnostic code from Day 1.

## Why this order

1. Data and split first: every number depends on them.
2. Rules and baselines next: they set the bars, including the same-destination rule.
3. Model port, checked against the current checkpoint's result.
4. Hybrid: combines the model with the prior and the same-destination signal.
5. Context inside the model: the fallback if the hybrid cannot beat the rule.
6. Analysis and write-up last.

## Phase 1: Foundation (Days 1-5)

### Day 1 (Thu 1 Oct): environment and skeleton
- [x] Pin dependencies in `pyproject.toml`; install torch into `.venv`.
- [x] Create `src/nbp/` with `paths.py` (single path helper), `config.py` (YAML into dataclass), `seed.py`; create `tests/`, `configs/`, `experiments/`.
- [x] Device-agnostic training config (`device: auto`).
- [x] `ruff` and `pytest` wired in; one smoke test passing.
- Output: skeleton committed, `pytest` green.
- Done 2026-10-02: torch 2.14.1+cpu installed, `pip install -e ".[dev]"`, ruff check/format clean, 3 smoke tests pass. `resolve_device()` lives in `nbp.config` (no training config yet; `device: auto` in `configs/data.yaml`).

### Day 2 (Fri 2 Oct): data
- [x] `src/nbp/data/load.py`: one chunked pass over `train.csv`, bookings with all context columns, unit-safe timestamps. Output `data/interim/expedia_bookings.parquet`.
- [x] `src/nbp/data/clean.py`: drop duplicate keys only; rows with context defects are kept, the value is nulled and a `flag_*` set (decided 2026-10-05); bookings have no missing dates; standard schema (`user_id, item_id, timestamp, event_type, ctx_*`).
- [x] Script prints the counts that go into the data card (`scripts/expedia_build_bookings.py` -> `results/week4_rebuild/bookings_counts.json`).
- Output: cleaned bookings parquet + printed counts.
- Done 2026-10-05: 3,000,693 booking rows -> 3,000,685 after 8 duplicate keys dropped; 813,985 users, 2,360,713 (user, cluster) pairs, all matching the data card; 30 tests pass. Burst repeats flagged (48,010), collapse decision left to Day 3.

### Day 3 (Mon 5 Oct): split, sequences, protocol
- [ ] `src/nbp/data/split.py`: event-level time cut, validation carve-out, `L` per row.
- [ ] `src/nbp/data/sequences.py`: history per event (up to 20 clusters with each booking's destination), query destination; output `data/processed/*.parquet`.
- [ ] `docs/eval_protocol.md`: Recall@K and NDCG@K only (K = 5, 10, 20; Recall@5 primary); slices by repeat vs new cluster, by `L`, by destination support; paired bootstrap 95% intervals; 3 seeds for final tables.
- [ ] Tests for split and sequence functions against hand-made examples.
- Output: processed data, protocol document, passing tests.

### Day 4 (Tue 6 Oct): metrics and baselines
- [ ] `src/nbp/eval/metrics.py`: Recall and NDCG at 5, 10, 20 (done 2026-10-02); intra-list diversity; popularity bias. Each unit-tested against hand-computed cases.
- [ ] `src/nbp/eval/evaluate.py` (slices); `bootstrap.py` (paired CI) done 2026-10-02.
- [ ] `src/nbp/priors/destination.py`: smoothed destination prior, built only from data before the evaluated period.
- [ ] `src/nbp/baselines/`: global and per-destination popularity, repeat-last, ItemKNN, same-destination rule.
- Output: first rows of `reports/benchmark_results.csv` (the new bars on the new split).

### Day 5 (Wed 7 Oct): SMLP4Rec port
- [ ] `src/nbp/models/smlp_block.py`, `smlp4rec.py`: plain PyTorch, SMLP block logic from the fork.
- [ ] `src/nbp/train.py`: dataset and dataloader, CE over 100 clusters, early stopping on validation; writes `experiments/<date>_expedia_<model>_<tag>/` (config, `metrics.json`, log, git hash, checkpoint).
- [ ] **Check:** history-only port lands within about +-1 point Recall@5 of the current checkpoint's result on the same rows. If not, stop and debug before continuing.

**Milestone 1 (end of Day 5):** clean data and split; protocol document; baselines and rule bars on the new split; tested metrics; SMLP4Rec running without RecBole and passing the check; `PROGRESS.md` updated.

## Phase 2: Hybrid, optimization, analysis (Days 6-10)

### Day 6 (Thu 8 Oct): hybrid as decided
- [ ] `src/nbp/hybrid/score.py`: three-term score, cold users prior only.
- [ ] Grid tuning of the weights per bucket of `L` on validation.
- [ ] Comparison against the same-destination rule with bootstrap intervals on warm users, cold users, all events.
- [ ] Optional: first GPU timing check (Colab T4) to decide the Days 7-9 setup.
- Check: does it beat the rule on both Recall@5 and MAP@5?

### Day 7 (Fri 9 Oct): model fixes and context inside the model
- [ ] Architecture: separate weights per layer, residual connection, dropout.
- [ ] Per-position context: destination of each history booking and a same-destination-as-query flag; query token for the current search.
- [ ] Retrain with early stopping; plug the stronger model into the hybrid; rerun the Day 6 comparison.

### Day 8 (Mon 12 Oct): learned gate and comparators
- [ ] `src/nbp/hybrid/gate.py`: weights per bucket of `L` learned against the loss, model frozen.
- [ ] LightGBM baseline with candidates limited to the top 20-30 per query from the prior.
- [ ] AdaGIN: stretch goal; if skipped, state it.

### Day 9 (Tue 13 Oct): robustness and cost
- [ ] 3 seeds for final models, mean +- std.
- [ ] Ablation: model alone, + prior, + sameDest, + per-position context, + gate.
- [ ] RQ4 accuracy-versus-discovery curve (novelty weight sweep).
- [ ] Timing profile: data build, epoch, evaluation.
- [ ] Leakage audit of every prior and query field.

### Day 10 (Wed 14 Oct): write-up
- [ ] `reports/summary/week4/`: README, JSON outputs, figures.
- [ ] Rebuild slides from the JSON outputs.
- [ ] Draft RQ1 and RQ4 answers.
- [ ] Update `PROGRESS.md` and CLAUDE.md section 6.
- [ ] Remove RecBole scripts once the replacement is verified (ask before deleting).

**Milestone 2 (end of Day 10):** final benchmark table (baselines, rule, SMLP4Rec variants, hybrid, LightGBM) over 3 seeds with intervals, all sliced; a clear answer on whether the hybrid beats the same-destination rule; RQ1 and RQ4 drafts.

## Target layout

```
src/nbp/
  paths.py  config.py  seed.py
  data/      load.py clean.py split.py sequences.py
  priors/    destination.py
  models/    smlp_block.py smlp4rec.py
  hybrid/    score.py gate.py
  baselines/ popularity.py repeat.py itemknn.py same_dest_rule.py lgbm.py
  eval/      metrics.py evaluate.py bootstrap.py
  train.py   recommend.py
configs/     data.yaml smlp4rec.yaml hybrid.yaml baselines.yaml
tests/       test_metrics.py test_split.py test_sequences.py test_prior.py test_hybrid.py
experiments/ <date>_expedia_<model>_<tag>/        (gitignored)
data/interim/ data/processed/                     (gitignored)
docs/eval_protocol.md
reports/benchmark_results.csv  reports/summary/week4/
```

## Risks

- Days 1-3 overrun: cut the Day 8 comparators, not the Day 5 check.
- CPU time: about 3-4 min per epoch for the current model; 3 seeds x 4 variants x ~10 epochs is roughly 8 h. Run seeds overnight, cut to 2 variants, or use a Colab T4 for Days 7-9 (keep all final seeds on the same hardware; save checkpoints to Drive; upload processed parquet only, never the raw Kaggle data to shared locations).
- The hybrid only ties the rule: report as found; it becomes a real RQ1 finding.
- Scope: LLM rerank stays out of this window.

## Revision log

| Date | Change | Reason |
|---|---|---|
| 2026-09-30 | Plan created | Hybrid decision locked (SMLP4Rec + weighted prior + sameDest) |
| 2026-10-02 | Day 1 done | Deps installed, skeleton committed, pytest green; Day 2 data audit run in scratchpad, open data decisions listed in `PROGRESS.md` |
| 2026-10-02 | exp branches merged to master; week-3 runs re-done as notebooks (`notebooks/hospitality/smlp4rec/`), all numbers reproduced; metrics = Recall@K and NDCG@K only; `nbp.eval.metrics` / `bootstrap` added early | User request |
| 2026-10-05 | Day 2 done | Defect rows kept with nulled value + flag (not dropped); burst repeats flagged only; `.gitignore` `data/` anchored to `/data/` so `src/nbp/data/` is tracked |
