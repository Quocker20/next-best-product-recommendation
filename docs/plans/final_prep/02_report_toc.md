# Final report and slide deck: table of contents (draft, 2026-10-11)
> **User decision (2026-10-11): this table of contents is for the slide deck only. No written / docx report is needed now.** Section 2 (slide outline) is the deliverable structure; section 1 is kept as background for speaker notes and a possible later report.


Scope of the final deliverable (user, 2026-10-11): re-introduce the problem, the dataset, the method, the
**original** SMLP4Rec adaptation, **our improvements** (query embedding, sameDest, training with first
bookings), the benchmark against exactly **two** baselines (ItemKNN, logistic regression), and which parts
answer RQ1 and RQ4. RQ2 / RQ3 are out of scope and get one sentence. **Skipped on purpose:** old hybrid,
top-5 rerank rules, destination prior, behaviour split, LLM rerank, every other model (notebooks 02, 03,
05, the `late_fusion_*` / `dynamic_cap` / `mixed` runs). The old hybrid and the prior may appear only
in an appendix row if a reviewer asks.

Rules carried over from `CLAUDE.md`: every number is read from `results/` by a builder script, never typed
(§5, §9); every table states dataset and row set (§6); limitations are stated with the results (§3).
Numbers quoted below are only pointers to check the source; the builder must re-read them.

Report language: English here. Earlier decks (`reports/summary/week2..week4`) were written in Vietnamese,
and Vietnamese figures exist under `results/figures/vi/` and `results/figures/clean/vi/`; `docs_vi/`
holds Vietnamese copies of some docs. A Vietnamese edition is a translation pass, not a new structure.

---

## 0. Advice used for the structure (web, 2026-10-11)

- **Introduction as five questions** (problem, why it matters, why it is hard, why earlier work does not
  solve it, what we do and the limits) ending in a bullet list of contributions that doubles as the report
  outline. J. Widom, *Tips for Writing Technical Papers*: https://cs.stanford.edu/people/widom/paper-writing.html
- **Applied recsys evaluation must show tuned simple baselines** next to the neural model, with the same
  data, split and tuning budget; neural models often fail to beat well-tuned kNN or linear models.
  Ferrari Dacrema et al., RecSys 2019: https://arxiv.org/abs/1907.06902 ; best-practice questions for
  offline evaluation (Jannach et al., 2024): https://web-ainf.aau.at/pub/jannach/files/Other_best_practices-2024.pdf
  -> Our report puts the baseline tuning grids and the shared split in the method chapter, not in an appendix.
- **Assertion-evidence slides:** each headline is one full-sentence claim, supported by a visual (chart,
  diagram), not by bullets. M. Alley, Penn State: https://writing.engr.psu.edu/eye_presenting.pdf ;
  study: https://peer.asee.org/assertion-evidence-slides-appear-to-lead-to-better-comprehension-and-recall-of-more-complex-concepts.pdf
- **Short talk budget:** about one slide per minute, one idea per figure, simple tables only, most time on
  results (roughly intro 2-3 min, method 2-3, results 6-7, discussion 2-3, conclusion 1), plan answers to
  the two or three questions you know are coming.
  https://biology-assets.anu.edu.au/hosted_sites/Scott/how-to-give-a-good-15-minut.html ;
  https://ecampusontario.pressbooks.pub/scientificcommunication/chapter/talks/

The earlier decks already use assertion titles (e.g. `reports/summary/week4_rebuild/week4_clean_slides.html`);
keep that style and the inline-SVG builder pattern of `scripts/week4_clean_slides.py`.

---

## 1. Written report: table of contents

Headline model throughout: **SMLP4Rec + query embedding + sameDest, trained with first bookings, early
stopping** (notebook 01d, user decision 2026-10-10, `PROGRESS.md` "01d with early stopping"). Headline
row set: warm test rows; cold and all events next to it.

### Front matter
- **Abstract** (150-200 words): problem, data, model, the two baselines, headline warm and cold result,
  one limitation sentence. Source: `results/week4_rebuild/baselines_vs_final_es.json`.
- **Summary of contributions** (bullets, each pointing to a chapter): (1) cleaned, leakage-audited
  Expedia booking benchmark; (2) a runnable adaptation of SMLP4Rec on hotel clusters; (3) query embedding;
  (4) sameDest; (5) training with first bookings; (6) benchmark against ItemKNN and logistic regression
  with warm / cold rows; (7) RQ1 slices and RQ4 beyond-accuracy view.

### 1. Introduction
1.1 **Problem and context.** Next-best-product recommendation for a resort operator (Vinpearl × GSM,
topic C1): which room category / package to offer for the current search. Public data only; no Vinpearl
or GSM data exists in the project. Source: `docs/problem_statement_vi.md`, `CLAUDE.md` §1-2.
1.2 **Why it is hard.** Travel bookings are rare per user (39.34 % single-booking users), seasonal, and
driven by trip context (destination, party, dates). Source: `results/week4_rebuild/eda_clean.json`,
`docs/data_cards/expedia.md` (eda_clean block).
1.3 **Research questions and scope.** RQ1 (context-aware session model vs traditional CF on sparse,
seasonal, context-dependent data) and RQ4 (accuracy vs diversity / coverage) are active. RQ2 (food
delivery) and RQ3 (cross-sell Vinpearl ↔ GSM) are out of scope: no public linked data, and the
5-week budget went to one domain. One paragraph only. Source: `CLAUDE.md` §3, `PROGRESS.md`
"CURRENT DECISION" (2026-10-08).
1.4 **Contributions and outline.** The bullet list above, with chapter references.

### 2. Background and related work (short, 1-1.5 pages)
2.1 Collaborative filtering (item-kNN) and context features (multinomial logistic regression) as the
classical families. 2.2 Sequential / session recommenders; SMLP4Rec (Gao et al., TOIS 2024) as an
all-MLP sequence model. 2.3 Why tuned simple baselines matter (Ferrari Dacrema et al. 2019).
Source: `reports/summary/week2_methodology/methodology_selection_report.docx` (model choice
rationale), `docs/explainers/smlprec_explain.md`. **Needs:** full bibliographic entries (gap G13).

### 3. Data
3.1 **Expedia Hotel Recommendations.** Bookings only (`is_booking == 1`); clicks are never positives;
item = `hotel_cluster` (100 anonymous classes, a **proxy** for room category / package; no price).
Period 2013-01 to 2014-12. Source: `docs/data_cards/expedia.md`, `results/week1_dataset_selection/expedia_profile.json`.
3.2 **Why Expedia** (one paragraph; dataset scoring). Source: `docs/dataset_scores.md`, `docs/dataset_rubric.md`.
3.3 **Cleaning.** Drop duplicate keys; collapse burst repeats (48,010 rows); null and flag context
defects; flag session repeats and heavy users. Table of tiers. Source: `docs/data_cleaning_strategy.md`,
`results/week4_rebuild/data_audit.json`, `bookings_counts.json`; figure `results/figures/clean/en/eda_e1_cleaning.png`.
3.4 **What the data says** (only the facts that drive the design): user depth (`eda_e2_users.png`),
near-flat target (`eda_e3_target.png`), destination concentration and rare destinations
(`eda_e4_destinations.png`), repeats in history (`eda_e5_repeats.png`), information per field
(`eda_e6_fields.png`, destination dominates). Source: `results/week4_rebuild/eda_clean.json`.
3.5 **Input rules and leakage audit.** `hotel_*`, `orig_destination_distance`, `cnt` are never input;
user city / region and `destinations.csv` latents excluded. Audit: no forbidden column, 0 order
violations, destination alone is the only strong field. Source: `results/week4_rebuild/leakage_audit.json`,
`docs/eval_protocol.md` (Data).

### 4. Evaluation protocol
4.1 **Split.** Global temporal 80/10/10 over next-booking targets; sizes train 1,710,952 / valid
213,869 / test 213,869 warm targets; cold rows valid 50,769, test 49,462; all events test 263,331.
Source: `docs/eval_protocol.md`, `results/week4_rebuild/split_export_counts.json`,
`baselines_es_run.json` (`split_counts`); figure `eda_e7_split_composition.png`.
4.2 **Row sets.** Warm (L ≥ 1) headline; cold (L = 0, first observed booking, left-censored) and all
events reported next to it.
4.3 **Metrics.** Recall@K, NDCG@K, K = 5, 10, 20, full ranking over 100 clusters. Beyond accuracy
(RQ4): intra-list diversity, popularity bias, coverage per destination / season. Source:
`src/nbp/eval/metrics.py`, tests `tests/`.
4.4 **Tuning and selection.** Small grid per model, selected on valid; test read once; early stopping
on valid NDCG@10 (patience 2, min_delta 7e-4, max 10 epochs). Source: `configs/smlprec_expedia.yaml`,
`results/week4_rebuild/early_stop_threshold_analysis.txt`, `baselines_es_run.json` (`early_stop`, `selection`).
**Note:** `docs/eval_protocol.md` still says selection by Recall@5 + NDCG@5 (gap G3).
4.5 **Significance and seeds.** Paired bootstrap (1000 resamples) for headline differences; main model
3 seeds. Source: `src/nbp/eval/bootstrap.py`, `results/week4_rebuild/seeds_summary.json`.

### 5. Method
5.1 **Task formulation.** Input: user's previous bookings (max 20) + the current search; output: a
score for each of 100 clusters.
5.2 **Original model: SMLP4Rec adaptation ("plain").** Copied from the paper code, two marked
adaptations (sequence length from config; empty item-feature list because clusters have no attributes);
known issues kept (shared block across layers, no residual around it). History only, ignores the
search. 33,702 parameters. Source: `src/models/smlprec.py` (docstring), `docs/explainers/smlprec_explain.md`,
`results/week4_rebuild/smlprec_plain_cold_train_es_run.json` (`n_parameters`), notebook
`notebooks/hospitality/smlp4rec/01e_train_test_smlp4rec_plain_cold_train.ipynb`; week-3 slide
"Từ code gốc đến bản chạy với Expedia" in `reports/summary/week3_implementation/week3_report_slides_v3.html`.
5.3 **Improvement 1: query embedding.** One extra position at the end of the sequence: learned [MASK] +
sum of 14 field embeddings (destination, destination type, check-in month, lead, stay, adults,
children, rooms, package, mobile, channel, site, point-of-sale continent, user country); the hidden
state at that position scores the catalogue; train-only vocabulary (min count 5). Source:
`src/nbp/models/smlprec_query.py`, `src/nbp/data/query_features.py`, `scripts/expedia_query_to_recbole.py`,
`results/week4_rebuild/smlprec_query_cold_train_es_run.json` (`query_fields`, `query_vocab_min_count`,
`n_parameters`, `n_query_parameters`), plan `docs/plans/plan_oct_07_query_embedding.md`.
5.4 **Improvement 2: sameDest.** Recency-weighted score for clusters the user booked before at the
searched destination; post-hoc late fusion `log q_model + w_s · log q_sameDest`, alpha and weights
picked on valid by the one-SE rule; zero on new-destination and cold rows. State plainly that it is a
fusion term, not a network input. Source: `src/nbp/hybrid/same_dest.py`, `src/nbp/hybrid/fusion.py`,
`smlprec_query_cold_train_es_run.json` (`samedest_hybrid.formula`, `tuned_on_valid`).
5.5 **Improvement 3: training with first bookings.** Adds the 713,754 first bookings of the train
window (empty history) to training, so the model learns the cold case. Source: notebook
`01d_train_test_smlp4rec_query_cold_train.ipynb`, `baselines_es_run.json` (`training_set`),
`PROGRESS.md` "Notebook 01d".
5.6 **Training.** CE over 100 clusters, Adam, hidden 64, 2 layers, early stopping; best epoch 8.
Source: `smlprec_query_cold_train_es_run.json` (`config`, `per_epoch`, `best_epoch_by_valid`,
`stop_reason`), `configs/smlprec_expedia.yaml`.
5.7 **Baselines.** ItemKNN (`implicit` cosine, history only, K grid on valid, cold rows fall back to
first-booking popularity); multinomial logistic regression (saga, query fields only, C grid and passes
on valid). Same training set as the main model. Source: notebook `07b_baselines_same_train.ipynb`,
`results/week4_rebuild/baselines_es_run.json` (`models`, `K_grid_valid`, `per_pass_valid`, `verification`).
A short diagram of the three inputs (history / query / both) helps here (gap G9).

### 6. Results
6.1 **Main benchmark (RQ1 headline).** One table: global popularity (reference row only), ItemKNN,
logistic regression, plain SMLP4Rec, query embedding, query embedding + sameDest; warm / cold / all events;
Recall@5/10/20, NDCG@5/10/20. Pointer values (warm, Recall@5 / NDCG@5): ItemKNN 0.3036 / 0.2089,
logistic regression 0.4969 / 0.3354, main model 0.6104 / 0.4461. Source:
`results/week4_rebuild/baselines_vs_final_es.json`. **Needs:** paired bootstrap CIs vs both
baselines for the ES checkpoint (gap G1) and a decision on which logistic-regression run is reported
(gap G2).
6.2 **Ablation ladder: what each improvement adds.** plain -> + query embedding -> + first bookings ->
+ sameDest. Source: `baselines_vs_final_es.json` (plain, query embedding, + sameDest at ES);
`results/week4_rebuild/seeds_summary.json` (`paired_01d_minus_01c`, first-bookings effect over 3
seeds, 3-epoch runs); `smlprec_query_cold_train_es_run.json` (`bootstrap_query_minus_plain_test`).
Caveat to print: the first-bookings step is measured on 3-epoch runs (01c was not rerun with early
stopping).
6.3 **Where the signal comes from.** Query shuffled vs history masked controls; per-field permutation
(destination dominates). Source: `smlprec_query_cold_train_es_run.json` (`diagnostics_test`,
`field_permutation_test`); `leakage_audit.json` (`single_field_strength`).
6.4 **sameDest: gain and its honest size.** Gain all on known-destination rows; sensitivity excluding
re-books within 1 h / 1 d (lower bound). Source: `smlprec_query_cold_train_es_run.json`
(`samedest_hybrid.test_slices_recall@5_ndcg@5`); `results/week4_rebuild/sensitivity_cold_train.json`
(3-epoch; ES version missing, gap G1); `seeds_summary.json` (gain stability).
6.5 **Cold rows (L = 0).** Query embedding on an empty history vs logistic regression vs ItemKNN fallback.
Source: `results/week4_rebuild/smlprec_query_cold_L0_cold_train_es.json`, `baselines_vs_final_es.json`;
3-epoch CI in `sensitivity_cold_train.json` (`cold_rows`).
6.6 **RQ1 slices** (main model and both baselines, warm test): history length L (1 | 2-4 | 5-9 | 10+),
known vs new destination, trip context (solo / couple / family / group, package), season (check-in
month; degenerate on test, see 7.x). Source today: only L and seen / unseen for plain and query embedding
(`smlprec_query_cold_train_es_run.json`, `slices_test`) and known / new destination for the main-model
variants (`samedest_hybrid.test_slices_recall@5_ndcg@5`). **Baseline slices, trip-context slices and
season slices do not exist** (gap G6).
6.7 **Stability.** Mean ± std over 3 seeds (3-epoch runs); ES run is seed 2022 only. Source:
`seeds_summary.json`, `smlprec_query_cold_train_es_run.json`. (gap G5)
6.8 **RQ4: accuracy vs diversity / coverage.** Intra-list diversity, popularity bias, coverage per
destination and per season for the three models. **Not computed** (gap G7). Context from EDA: the
target is near flat, so popularity bias has little room; coverage per destination is the informative
view. Source: `eda_clean.json`.

### 7. Discussion
7.1 **RQ1 answer.** A context-aware sequence model beats item-kNN by a wide margin and beats a
context-only linear model; most of the gain comes from putting the search inside the model; history
adds a smaller, real increment concentrated on users returning to a known destination. Tie to 6.1-6.6.
7.2 **RQ4 answer** (from 6.8; or stated as not answered if G7 is cut).
7.3 **What this means for Vinpearl** (careful): the search context (destination, party, dates) is the
main signal; repeat guests at the same resort are where history pays. No claim about Vinpearl data.
7.4 **Threats to validity:** season and time confounded in the temporal split (test window is
mid-November to December; check-in month JSD train vs test 0.439 bits, `eda_clean.json`); left-censored
"first booking"; sameDest partly re-books (sensitivity); early-stopping threshold chosen after seeing
the valid curve (valid only, `early_stop_threshold_analysis.txt`).

### 8. Limitations (required, CLAUDE.md §3; also one line under every results table)
- `hotel_cluster` is a proxy for room category / package (anonymous IDs, no price column).
- One dataset (Expedia); no cross-dataset check.
- Offline evaluation only; no online / A-B evidence.
- No in-stay service data exists publicly (spa, F&B, activities), so in-stay recommendation is not
  addressed.
- Only two simple baselines (ItemKNN, logistic regression); no other neural sequence model was run.
- Plus: season slice degenerate on the test window; ES run single seed; first booking is first
  *observed* booking.

### 9. Conclusion and next steps (half page)
Next steps listed, not done: per-position context on history bookings (paper's original design),
`destinations.csv` latents for rare destinations, a third baseline family if reopened (user decision),
Trivago as a session-level contingency, linking to Vinpearl data if ever available (labeled simulated
until then).

### Appendices
A. Reproducibility: environments (two venvs), commands, notebooks 01d / 01e / 07b / 08, git hash.
Source: `CLAUDE.md` §7, run JSONs (`git_commit`), `README.md`.
B. Full metric tables (K = 5, 10, 20, all row sets). Source: `reports/benchmark_results.csv` (needs the
ES rows, gap G4).
C. Tuning grids and per-epoch curves. Source: `baselines_es_run.json`, `smlprec_query_cold_train_es_run.json`
(`per_epoch`), `smlprec_plain_cold_train_es_run.json`.
D. Cleaning tiers and audit counts. Source: `docs/data_cleaning_strategy.md`, `data_audit.json`.
E. Burst-collapse sensitivity (with vs without). Source: `results/week4_rebuild/with_burst/`.

### RQ -> section map

| RQ | Status | Answered in | Evidence files |
|---|---|---|---|
| RQ1 context-aware session model vs traditional CF on sparse, seasonal, context-dependent data | Active | 6.1, 6.2, 6.3, 6.5, 6.6, 7.1 | `baselines_vs_final_es.json`, `smlprec_query_cold_train_es_run.json`, `smlprec_query_cold_L0_cold_train_es.json`, `seeds_summary.json`, slices (G6) |
| RQ1 sub-aspect "sparse" | Active | 6.5 cold rows, 6.6 L buckets | as above |
| RQ1 sub-aspect "seasonal" | Partly answerable | 6.6 season slice + 7.4 caveat | `eda_clean.json` (test window), slice (G6) |
| RQ1 sub-aspect "trip context" | Active | 6.3 field permutation, 6.6 trip-context slice | `field_permutation_test`, slice (G6) |
| RQ4 accuracy vs diversity / coverage | Active, not yet computed | 6.8, 7.2 | (G7) |
| RQ2 food delivery | Out of scope | 1.3 (one sentence) | `CLAUDE.md` §3 |
| RQ3 cross-sell Vinpearl ↔ GSM | Out of scope | 1.3 (one sentence) | `CLAUDE.md` §3 |

---

## 2. Slide deck outline (target 17 slides, about 18 min talk + Q&A)

Assertion titles below are drafts; numbers in them must be filled by the builder from the source file
(placeholders `{…}`), never typed. Time column in minutes.

| # | Assertion title (draft) | Visual / evidence | Source / figure | Min |
|---|---|---|---|---|
| 1 | Title: "Next-best-product recommendation for resort bookings on public Expedia data" | title, team, date, "public data only" | n/a | 0.3 |
| 2 | Resorts must pick, at search time, which room / package to show, from very few past bookings per guest | problem picture: search -> top-5 clusters; one line on Vinpearl context | `docs/problem_statement_vi.md` | 1.0 |
| 3 | We answer two questions: does a context-aware sequence model beat classical CF (RQ1), and at what diversity cost (RQ4) | RQ box; RQ2 / RQ3 greyed "out of scope" | `CLAUDE.md` §3 | 0.7 |
| 4 | Expedia gives {2,952,675} bookings from {813,985} users, but {39.34 %} of users booked only once | user-depth chart | `results/figures/clean/en/eda_e2_users.png`; `eda_clean.json` | 1.0 |
| 5 | The destination of the search says far more about the cluster than any other field | bits per field bar chart | `results/figures/clean/en/eda_e6_fields.png`; `eda_clean.json` | 1.0 |
| 6 | We removed only duplicate events: {48,010} burst repeats collapsed, defects flagged, no hotel field used as input | cleaning tiers table (small) + leakage audit tick list | `eda_e1_cleaning.png`; `docs/data_cleaning_strategy.md`; `leakage_audit.json` | 1.0 |
| 7 | Every model is scored on the same temporal split, full ranking over 100 clusters, warm and first-booking rows apart | timeline train / valid / test with row counts | `eda_e7_split_composition.png`; `split_export_counts.json`; `docs/eval_protocol.md` | 1.0 |
| 8 | The original SMLP4Rec reads only past bookings and ignores the current search | architecture diagram (plain) | **new figure needed** (EN; `results/figures/vi/fig12_smlp4rec_architecture.png` is the VI week-2 version) | 1.0 |
| 9 | Our query embedding puts the 14 search fields inside the model as one extra position | architecture diagram with query embedding highlighted | **new figure needed** (week-4 deck has an HTML slide "Query embedding: 14 trường…" to reuse) | 1.2 |
| 10 | sameDest adds what the user booked before at this destination; training with first bookings teaches the cold case | two small panels: sameDest formula / training-set bar (warm vs + first bookings) | **new figure needed**; `same_dest.py`; `baselines_es_run.json` (`training_set`) | 1.2 |
| 11 | Each step adds accuracy: plain {..} -> query embedding {..} -> + sameDest {..} NDCG@5 (warm test) | ablation ladder / waterfall bar | **new figure needed**; `baselines_vs_final_es.json`; first-bookings step from `seeds_summary.json` | 1.5 |
| 12 | The main model beats logistic regression by {Δ} and ItemKNN by {Δ} NDCG@5 on warm rows [CI] | grouped bars, 3 models × warm / cold / all, CI whiskers | **new figure needed**; `baselines_vs_final_es.json`; CIs (G1) | 1.5 |
| 13 | On first bookings the model ties or beats the context-only baseline, while ItemKNN cannot score them | cold-row bars | **new figure needed**; `smlprec_query_cold_L0_cold_train_es.json`, `baselines_vs_final_es.json` | 1.0 |
| 14 | History pays where guests return: sameDest lifts known-destination rows and leaves new ones unchanged | known vs new destination bars; L-bucket line | **new figure needed**; `samedest_hybrid.test_slices_recall@5_ndcg@5`, `slices_test`; baselines slices (G6) | 1.3 |
| 15 | The advantage holds across trip types {and seasons, if G6 pooled}; family / package slices | slice heat-table (models × slices) | **new figure needed** (G6) | 1.2 |
| 16 | Higher accuracy {does / does not} cost diversity or coverage | accuracy vs diversity scatter; coverage per destination | **new figure needed** (G7) | 1.2 |
| 17 | Conclusion: search context is the main signal, history a real but smaller increment; limits: proxy cluster, one dataset, offline, two simple baselines | 3 takeaways + limitation strip + next steps | text slide | 1.2 |

Total ≈ 19.3 min with all slides; for a 15-min slot cut 6 (fold leakage into 7), 10 (merge into 9),
15 (fold into 14), keeping ≈ 15.5 min. Backup slides (after 17, not timed): per-field permutation
(`field_permutation_test`), early-stopping curve (`per_epoch`, new figure), sameDest re-book
sensitivity (`sensitivity_cold_train.json`), 3-seed table (`seeds_summary.json`), tuning grids
(`baselines_es_run.json`), burst-collapse effect (`with_burst/`). Prepare answers for the obvious
questions: "is it leakage?" (backup: `leakage_audit.json`, shuffled controls), "why not SASRec /
GRU4Rec?" (scope decision 2026-10-08), "is the cluster a real product?" (proxy limitation).

---

## 3. Gaps: content the TOC needs that does not exist yet

> Status 2026-10-11 (verification pass, `05_verification_report.md`): G1 (bootstrap CIs on the ES run), G6 (slices for all models) and G7 (RQ4 metrics) now exist in `results/week5_final/` (`bootstrap_ci.csv`, `slices_*.csv`, `beyond_accuracy_*.csv`). G2 (which logistic regression), G4 (stale `reports/benchmark_results.csv`) and G5 (ES seeds) remain open.

| ID | Gap | Needed for | Suggested producer | Priority |
|---|---|---|---|---|
| G1 | Paired bootstrap CIs of the **ES** main model vs ItemKNN and vs logistic regression (warm, cold, all events) and the sameDest re-book sensitivity on the ES checkpoint. Today: `sensitivity_cold_train.json` is on the 3-epoch checkpoint; `baselines_vs_final_es.json` has no CIs; the ES run only bootstraps vs the old hybrid. | 6.1, 6.4, 6.5, slides 12-13 | extend notebook 07b or `scripts/expedia_sensitivity.py` with `nbp.eval.bootstrap` | must |
| G2 | **Baseline inconsistency:** logistic regression warm test Recall@5 / NDCG@5 is 0.5222 / 0.3568 in `basic_baselines.json` (nb 07, in `reports/benchmark_results.csv`) but 0.4969 / 0.3354 in `baselines_vs_final_es.json` (nb 07b, same training set as the main model, C = 1.0, pass 8, selected on valid NDCG@10). The report must use one and explain the difference (training set incl. first bookings, selection metric, pass budget). Decide with the user; check 07b's C grid (nb 07 picked C = 10 at the grid edge, `PROGRESS.md`). | 6.1, fairness of the benchmark | user decision + short investigation in 07b | must |
| G3 | `docs/eval_protocol.md` says model selection by Recall@5 + NDCG@5 and a 3-epoch budget; the final runs select by valid NDCG@10 with early stopping. Protocol text must be updated (protocol change: ask user). | 4.4 | doc edit after user OK | must |
| G4 | `reports/benchmark_results.csv` has no rows for the ES runs (01d ES, 01e plain ES, 07b baselines). | Appendix B, builder input | extend `scripts/build_benchmark_results.py` | must |
| G5 | Seeds 2023 / 2024 not rerun with early stopping; the 3-seed std covers only 3-epoch runs. Either rerun (the seed-2022 ES run logged 11,128 s train + eval, about 3 h per seed on CPU; `smlprec_query_cold_train_es_run.json`, `timing_seconds`; possibly less with the 7e-4 threshold applied from the start) or state that stability is shown on 3-epoch runs. | 6.7, CLAUDE.md §6 (≥ 3 seeds) | notebook 01d with `NBP_SEED`, `scripts/aggregate_seeds.py` | should |
| G6 | RQ1 slices for **all three models** on warm test: L buckets, known vs new destination, trip context (solo / couple / family / group from adults / children / rooms; `is_package`), season. Baseline slices and trip-context slices do not exist anywhere. Season slice is degenerate on test (`eda_clean.json`); needs a user decision: pool valid + test, or state the limitation. | 6.6, slides 14-15 | new slice script over per-row ranks (export via notebook 08 pattern) + `nbp.eval` slice helper with tests | must (season + trip context are "never cut" in the plan) |
| G7 | RQ4 metrics: intra-list diversity, popularity bias, coverage per destination / season for the three models. No code in `src/nbp/eval/`, no results. Diversity over anonymous clusters needs a similarity definition (e.g. item-embedding cosine or co-booking), to decide. | 6.8, 7.2, slide 16 | new `nbp.eval` functions + unit tests, then a script writing `results/<week>/rq4_*.json` | must (cut order allows dropping coverage, keep diversity + popularity bias) |
| G8 | Ablation ladder with all four steps on one footing: the "first bookings" step exists only as 01c vs 01d at 3 epochs (`seeds_summary.json`); 01c was not rerun with early stopping. Either accept the mixed footing with a caveat, or rerun 01c with ES. | 6.2, slide 11 | caveat or one 01c ES run | should |
| G9 | English figures: SMLP4Rec plain architecture, query-embedding architecture, sameDest / first-bookings schematic, ablation ladder, main benchmark bars with CIs, cold bars, slice charts, RQ4 chart, early-stopping curve. Only EDA figures exist in English (`results/figures/clean/en/`); the architecture figure exists only in VI (`results/figures/vi/fig12_smlp4rec_architecture.png`, old). | slides 8-16, report ch. 5-6 | final builder script (inline SVG like `scripts/week4_clean_slides.py`) or `scripts/final_report_figures.py` writing `results/figures/{en,vi}/` | must |
| G10 | Final report and deck builders do not exist (`scripts/week4_clean_slides.py` reads the 3-epoch JSONs). A new folder `reports/summary/week5_final/` and `results/week5_final/` would follow CLAUDE.md §7 naming. | all | new scripts | must |
| G11 | `PROGRESS.md` has no entry for notebooks 01e, 07b, 08 and their JSONs (`smlprec_plain_cold_train_es_run.json`, `baselines_es_run.json`, `baselines_vs_final_es.json`, `smlprec_query_cold_L0_cold_train_es.json`, `split_export_counts.json`); they are untracked in git. | traceability | PROGRESS entry + commit (user) | must |
| G12 | Demo UI (`scripts/demo_ui.py`) still loads the 3-epoch checkpoints; update if the demo is shown in the talk. | optional demo slide | script edit | could |
| G13 | Full references: SMLP4Rec (Gao et al., TOIS 2024), Expedia Kaggle dataset, `implicit`, scikit-learn, RecBole 1.0.1, Ferrari Dacrema et al. 2019. No bibliography file in the repo. | ch. 2, references | `docs/` or report builder | should |
| G14 | Vietnamese edition, if wanted: translate the report and use `results/figures/*/vi/` figures; `docs_vi/` has only part of the docs. | optional | translation pass | could |

Not gaps (exist and can be used as is): cleaning and audit (`docs/data_cleaning_strategy.md`,
`data_audit.json`, `leakage_audit.json`), EDA figures (`results/figures/clean/en/eda_e1..e8.png`),
main ES result table (`baselines_vs_final_es.json`), signal controls and per-field permutation
(`smlprec_query_cold_train_es_run.json`), known / new destination slice for the main model
(`samedest_hybrid.test_slices_recall@5_ndcg@5`), cold rows (`smlprec_query_cold_L0_cold_train_es.json`),
3-seed stability (`seeds_summary.json`), early-stopping threshold analysis
(`early_stop_threshold_analysis.txt`).
