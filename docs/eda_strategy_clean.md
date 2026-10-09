# EDA strategy for the cleaned Expedia data (2026-10-09)

Purpose: describe the data the models actually see (`data/interim/expedia_bookings.parquet`, burst repeats collapsed) and supply the facts that the report's data chapter and the stage-D slices need. The week-1 EDA (`results/week1_dataset_selection/`) described the raw file and stays valid for it; this pass does **not** repeat it. Script: `scripts/expedia_eda_clean.py` → `results/week4_rebuild/eda_clean.json`, figures in `results/figures/clean/{en,vi}/`, generated section in `docs/data_cards/expedia.md`.

Rule for scope: an analysis is in only if it feeds a named decision or a named report element. Everything else is out.

## In scope (question → analysis → what it feeds)

| # | Question | Analysis | Feeds |
|---|---|---|---|
| E1 | What did cleaning change? | Before / after table (rows, users, single-booking share, party mix, flags); distribution of the gap between a burst repeat and the booking it repeats | Data chapter; defence of the collapse decision |
| E2 | How sparse and how long are user histories? | Bookings per user (quantiles, shares of 1 / 5+ / 10+), user x cluster density, days between bookings, same-day share, history length L of the warm test rows by bucket | RQ1 slice "user activity"; why sequence models can help at all |
| E3 | What is the target like? | Cluster share (top-k, normalised entropy, Gini); Jensen-Shannon divergence train vs test and 2013 vs 2014 | Popularity-bias baseline for RQ4; drift statement |
| E4 | How concentrated and how rare are destinations? | Destinations, top-10 / top-100 share, cumulative curve, share of test targets whose destination has fewer than m train bookings (m = 1, 2, 5, 10, 20), cluster entropy given destination | Vocabulary threshold (min count 5), rare-destination work in stage C, known vs new destination slice |
| E5 | How much do users repeat? | By L bucket: share of targets whose cluster / destination / (destination, cluster) is already in the history | Rationale and ceiling of sameDest; RQ1 slice known vs new destination |
| E6 | Which query fields carry information about the target? | Held-out information gain in bits (fit on train, scored on valid, smoothed) for the 14 query fields and 3 candidate fields | Explains the feature-shuffle result; field choice in stage C |
| E7 | Does the test period look like the train period? | Train / valid / test composition: time range, month, party type, package, known-destination share, warm / cold share; JSD per field between train and test | Season and trip-context slices; drift caveat in the report |
| E8 | What is the time structure? | Monthly bookings, new users, active users; check-in month and lead time distribution | Season slice (check-in month); explains the volume jump in 2014 |

## Out of scope (and why)
- Hour-of-day, clicks, conversion: done in week 1, clicks are never positives.
- Raw-column distributions, correlation matrices, outlier plots: the cleaning audit (`data_audit.json`) already covers defects; the columns are categorical.
- Price, review text, hotel attributes: none exist; `hotel_*` fields are labels.
- User geography beyond country: the model uses country only; region / city only appear in E6 as candidates.
- Anything about other datasets.

## Outputs
- `results/week4_rebuild/eda_clean.json`: every number, computed by the script.
- `results/figures/clean/{en,vi}/eda_*.png`: eight figures, one per E1-E8 (E5 and E4 may share a figure only if the page stays readable).
- `docs/data_cards/expedia.md`: a generated section between the markers `eda_clean:start` and `eda_clean:end`; the older raw-file sections stay and are labelled as raw.
- `PROGRESS.md`: one entry with the findings that change a decision.
