# Dataset scoring rubric (Topic C1)

Purpose: pick one core dataset per domain (hospitality, food, ride) for the benchmark. Scores are only comparable **within a domain**. MovieLens is scored for the record but never competes for a core slot (CLAUDE.md §5).

This file is the **method only** — gates, criteria, weights, formula, decision thresholds. For a specific dataset's scores, the Step 0 interaction-definition table, the scoring sheet, notes, and the final proposed decision, see `docs/dataset_scores.md`. For the arithmetic (weighted sum → score → decision tier) and the `quick_profile()` helper, see `scripts/dataset_scoring.py`.

Score each dataset in three steps:

1. **Gates:** pass/fail. A dataset that fails any gate is not scored.
2. **Criteria:** 0–3 points each, weighted, summed to a score out of 100.
3. **Decision + pairwise check:** apply thresholds per domain, then check that the chosen pair of datasets can be linked by the cross-sell simulator (RQ3).

Inputs: the data cards in `docs/data_cards/` and the EDA notebooks. Every number used should be either taken from a data card or marked *provisional* (to be verified with `quick_profile()` on the cleaned interaction table) — the Step 0 mapping (per dataset) lives in `docs/dataset_scores.md` since it's a per-dataset decision, not part of the method.

## Step 1: Gates (pass/fail)

| Gate | Pass if |
|---|---|
| G1 License | Research/internal use is allowed (Kaggle competition rules, Yelp terms, UCI, TLC open data). |
| G2 Item ID | There is a recommendable thing (hotel cluster, item, restaurant, destination zone) with a stable ID. |
| G3 Loadable | The full file **or a user/session-level sample** (CLAUDE.md §7) loads on the dev machine within a few minutes. Datasets that need a chunked pass for the full file still pass if a sample fits. |
| G4 Interactions | Rows are user/session × item interactions, **or** trips/orders that can be turned into them under the Step 0 mapping. |
| G5 Downloaded | The raw files are present under `data/raw/` and reproducible via `scripts/download.py` (or its documented manual step). A dataset that is not on disk is not scored until it is. |

## Step 2: Criteria (0–3 each)

| # | Criterion | Weight | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| C1 | **Task fit** with the brief's task for that domain | ×3 | Different problem, cannot be reframed | Heavy reframing (e.g. predicting a country as a class; a task not in the mentor's plan) | Close with some adaptation (e.g. hotel cluster standing in for hotel) | Direct top-K next-item task |
| C2 | **User / session identity** | ×2 | No user or session ID | Only session ID, IDs for a minority of rows, **or a proxy actor** (driver, zone) — proxy actors are capped at 1 | User ID, but most users have one interaction | Stable user ID **and** sessions / repeat users |
| C3 | **Temporal info** | ×2 | No timestamps | Date only, or span < 1 month | Full timestamps, span < 1 year | Full timestamps, span ≥ 1 year **and** users/sessions with more than one event (so seasonality is visible in sequences, not just across users) |
| C4 | **History depth** (median interactions per actor or session under Step 0) | ×2 | ≤ 1 | 2 | 3–4 | ≥ 5 |
| C5 | **Context richness** (real, non-synthetic fields from the domain list below) | ×2 | None | 1 field | 2–3 fields | ≥ 4 fields |
| C6 | **Item metadata** (category, price, location, attributes) | ×1 | ID only | 1 attribute | 2–3 attributes | Rich attributes + category taxonomy |
| C7 | **Signal quality** | ×1 | Ratings/reviews only | Views/impressions only | Clicks | Bookings / orders / completed trips |
| C8 | **Feasibility** (size vs 3 dev weeks) | ×1 | Needs a cluster | Needs heavy sampling + GPU | Fits with user-level sampling | Fits in memory as is |
| C9 | **Cross-sell linkability** (real fields the simulator can use) | ×1 | None | City only | City + time | Real coordinates + timestamps + user attributes |
| C10 | **Data quality / cleanup cost** (from EDA findings) | ×1 | Core fields unusable (synthetic or missing) | Serious issues in core fields (large invalid share, unreliable flags, synthetic context) | Minor issues (few duplicates, outliers, small invalid share) | Clean |

Rules that apply across criteria:

- **Synthetic fields count as absent** for C5 and C9. Akeed's customer coordinates are 100 % outside the real Oman bounding box (see data card), so they are not counted as location context.
- **Proxy actors** (Porto `TAXI_ID`, NYC pickup zone) cap C2 at 1, and C4 is computed on the proxy but flagged as such in the sheet.
- C3 = 3 requires both the span and per-actor sequences; a dataset with one event per user gets at most 2 even with a multi-year span.

**Context fields for C5, by domain:**

| Domain | Fields to count |
|---|---|
| Hospitality | party size (adults/children), check-in/out dates, destination, device, price, search filters — *trip* context, not user profile (age, gender, language do not count) |
| Food | user location, restaurant location, order time, delivery time/ETA, city + date (to join weather), cuisine |
| Ride | origin coordinates, start time, day of week, trip duration, call type / stand |

## Step 3: Score, decide, then check the pair

**Formula:** Score = (C1×3 + C2×2 + C3×2 + C4×2 + C5×2 + C6 + C7 + C8 + C9 + C10) / 48 × 100 — computed by `scripts/dataset_scoring.py::score_dataset()`, not by hand.

| Score | Decision |
|---|---|
| ≥ 70 | **Core**: use it for the benchmark |
| 50–69 | **Secondary**: use it for one specific RQ or a robustness check |
| < 50 | **Drop**, or use only as context/reference |

**Tie-breakers, in order:**

1. The dataset that answers more RQs (see the "RQs it serves" column in `docs/dataset_scores.md`)
2. Higher C1 (task fit)
3. Smaller size (faster to iterate)

**Pairwise linkability check (RQ3).** Cross-sell is a property of a *pair* of datasets, which a per-dataset score cannot capture. After picking the core dataset per domain, check that each pair the simulator must link (hospitality ↔ food, ride ↔ food) shares at least one of: city / region, calendar time overlap or reusable time-of-day patterns, real coordinates. Record the shared fields in the "Link fields" table in `docs/dataset_scores.md`. If a pair shares nothing, pick the secondary dataset for one side or state the limitation in the report.
