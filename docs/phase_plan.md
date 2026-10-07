# Phase plan and deliverables (moved from CLAUDE.md)

## Mentor's phase plan

| Phase | Work | Key outputs |
|-------|------|-------------|
| 1 | Theory + EDA | EDA notebooks, data cards, dataset scoring, chosen datasets |
| 2 | Baselines (Popularity, MF, item2vec) + evaluation protocol | Processed data, protocol doc, first benchmark table |
| 3 | Sequence models (GRU4Rec / SASRec) | Extended benchmark, RQ1 answer |
| 4 | Context-aware ablation + LLM rerank | Ablation results, rerank results with cost/latency, RQ2 answer |
| 5 | Diversity trade-off + catalog schema | Trade-off charts, RQ4 answer, unified catalog schema |
| 6 | Simulated cross-sell data + final report | Simulator, cross-sell results, RQ3 answer, demo, final report + slides |

Under the current scope (§1), the simulator / cross-sell / RQ3 part of Phase 6 is deferred; Phase 6 = final report + slides only. Phase 4's "context-aware ablation" runs on Expedia trip context (party, dates/season, package, destination, channel), not on food context. Phase 5's "unified catalog schema" covers packages/rooms only for now.

Work in loops: train → benchmark → analyze → answer RQ, adding to one shared results table. Do not train everything first and analyze at the end.

## Deliverables checklist

- [ ] Processed datasets + pipeline code + data cards
- [ ] Evaluation protocol doc
- [ ] Trained models (all families) with configs and metrics
- [ ] Benchmark results table + RQ1–RQ4 analysis
- [ ] Linked-ID simulator + simulated dataset
- [ ] Unified catalog schema (packages/rooms, rides, dishes) aligned with the roadmap's Integration Design section
- [ ] Vinpearl ↔ GSM cross-sell demo on simulated data
- [ ] Final benchmark report + presentation slides
- [ ] Reproducible repo with README
