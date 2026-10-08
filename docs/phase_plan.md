# Plan and deliverables

Active plan: `plan_oct_08_replan.md` (in `docs/plans/`). Earlier plans are history.

## Current scope (2026-10-08)
One dataset (Expedia), one main model (SMLP4Rec query token + sameDest), two classical baselines (ItemKNN, logistic regression). Work = data processing, model tuning, evaluation, report. See `CLAUDE.md` §1 for what is out of scope.

## Original mentor phases (reference only)
1 Theory + EDA · 2 Baselines + protocol · 3 Sequence models · 4 Context-aware ablation + LLM rerank · 5 Diversity trade-off + catalog schema · 6 Cross-sell + final report. Under the current scope only Phase 1 (done), the protocol, the main model with two baselines, the RQ1 / RQ4 analysis and the final report + slides remain.

## Deliverables checklist
- [x] Dataset selection, EDA notebooks, data cards
- [x] `docs/eval_protocol.md`
- [ ] Cleaned and audited data, leakage audit (plan stage B)
- [ ] Tuned main model, 3 seeds, two baselines with comparable tuning (stage C)
- [x] `reports/benchmark_results.csv`
- [ ] RQ1 slices and RQ4 metrics (stage D)
- [ ] Final report + slides built from `results/` (stage E)
- [ ] Reproducible repo with README
