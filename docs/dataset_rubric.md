# Dataset scoring rubric (Topic C1)

Purpose: pick one core dataset per domain (hospitality, food, ride) for the benchmark. Scores are only comparable **within a domain**. MovieLens is scored for the record but never competes for a core slot (CLAUDE.md §5).

Inputs: the data cards in `docs/data_cards/` and the EDA notebooks. Every number used below is either taken from a data card or marked *provisional* (to be verified with `quick_profile()` on the cleaned interaction table).

Score each dataset in three steps:

1. **Gates:** pass/fail. A dataset that fails any gate is not scored.
2. **Criteria:** 0–3 points each, weighted, summed to a score out of 100.
3. **Decision + pairwise check:** apply thresholds per domain, then check that the chosen pair of datasets can be linked by the cross-sell simulator (RQ3).

## Step 0: Interaction definition per dataset

`quick_profile()` assumes the standard interaction schema from CLAUDE.md §7 (`user_id`, `item_id`, `timestamp`). Raw files do not have it, so the mapping must be fixed first — otherwise C2/C4/C7 are not reproducible.

| Dataset | Actor (`user_id`) | Item (`item_id`) | Rows that count as an interaction | Sequence unit |
|---|---|---|---|---|
| Expedia | `user_id` | `hotel_cluster` | `is_booking == 1` (clicks kept as separate, weaker event_type) | user history ordered by `date_time` |
| Trivago 2019 | `session_id` (with `user_id` kept) | `reference` on item actions | `action_type` in {`clickout item`, `interaction item *`}; clickout is the target | session |
| Airbnb New User | `id` | `country_destination` | `country_destination != 'NDF'` | none (one event per user) |
| Akeed | `customer_id` | `vendor_id` | one row per order (`orders.csv`), after dropping duplicate `akeed_order_id` | customer history by `created_at` |
| Yelp | `user_id` | `business_id` | reviews (rating-only signal) | user history |
| Porto Taxi | `TAXI_ID` (**proxy actor** — driver, not passenger) | `ORIGIN_STAND`, or a grid cell of the trip end point | `CALL_TYPE == 'B'` for stands; all trips for grid destination | taxi history by `TIMESTAMP` |
| NYC TLC | `PULocationID` (**proxy actor** — a zone, not a person) | `DOLocationID` | one row per trip | none |
| MovieLens | `userId` | `movieId` | one row per rating | user history by `timestamp` |

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

**Formula:** Score = (C1×3 + C2×2 + C3×2 + C4×2 + C5×2 + C6 + C7 + C8 + C9 + C10) / 48 × 100

| Score | Decision |
|---|---|
| ≥ 70 | **Core**: use it for the benchmark |
| 50–69 | **Secondary**: use it for one specific RQ or a robustness check |
| < 50 | **Drop**, or use only as context/reference |

**Tie-breakers, in order:**

1. The dataset that answers more RQs (see the "RQs it serves" column)
2. Higher C1 (task fit)
3. Smaller size (faster to iterate)

**Pairwise linkability check (RQ3).** Cross-sell is a property of a *pair* of datasets, which a per-dataset score cannot capture. After picking the core dataset per domain, check that each pair the simulator must link (hospitality ↔ food, ride ↔ food) shares at least one of: city / region, calendar time overlap or reusable time-of-day patterns, real coordinates. Record the shared fields in the "Link fields" table below. If a pair shares nothing, pick the secondary dataset for one side or state the limitation in the report.

## Numbers to compute per dataset (for C2–C4, C8)

Run on the cleaned interaction table built under Step 0 (Parquet in `data/interim/`), not on the raw file.

```python
import pandas as pd

def quick_profile(df, user="user_id", item="item_id", ts="timestamp"):
    per_user = df.groupby(user).size()
    return {
        "rows": len(df),
        "users": df[user].nunique(),
        "items": df[item].nunique(),
        "sparsity": 1 - len(df) / (df[user].nunique() * df[item].nunique()),
        "median_inter_per_user": per_user.median(),
        "pct_users_ge5": (per_user >= 5).mean(),
        "time_span_days": (pd.to_datetime(df[ts]).max() - pd.to_datetime(df[ts]).min()).days if ts else None,
        "repeat_rate": 1 - df[[user, item]].drop_duplicates().shape[0] / len(df),
        "mem_mb": df.memory_usage(deep=True).sum() / 1e6,
    }
```

## Scoring sheet (first pass from data cards, 2026-09-17)

Values marked † are provisional: the data cards give means or single-interaction shares but not medians; confirm with `quick_profile()` once the interim tables exist. Gates: ✓ pass, ✗ fail.

| Dataset | G1–G5 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Score | Decision | RQs it serves | Main gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Expedia | ✓ | 2 | 3 | 3 | 1† | 3 | 2 | 3 | 2 | 2 | 3 | **79** | Core | RQ1 (seasonality, context), RQ4 | hotel cluster ≠ room/package; 38.8 % of bookers have one booking; 4 GB file needs user sampling |
| Trivago 2019 | ✓ | 3 | 2 | 1 | 2† | 3 | 3 | 2 | 2 | 2 | 3 | **77** | Core (session-model robustness) | RQ1 (session part), RQ4 | 6-day span — no seasonality; 56 % single-interaction users; clicks not bookings |
| Airbnb New User | ✓ | 1 | 2 | 1 | 0 | 1 | 2 | 3 | 3 | 1 | 2 | **44** | Drop | — | one booking per user, 12-class target, 58 % NDF, no trip context |
| Akeed | ✓ | 3 | 3 | 2 | 2† | 2 | 3 | 3 | 3 | 2 | 1 | **81** | Core | RQ2, RQ4, RQ3 (food side) | customer coordinates synthetic; 41 % `deliverydistance <= 0`; duplicate order/customer IDs; 100 vendors only |
| Yelp | G5 ✗ | – | – | – | – | – | – | – | – | – | – | – | Not scored | — | not downloaded (manual form, see `scripts/download.py`); rating-only signal would cap C7 at 0 |
| Porto Taxi | ✓ | 1 | 1 (proxy) | 3 | 3† (proxy) | 3 | 1 | 3 | 2 | 2 | 2 | **69** | Secondary | RQ3 (ride side, via coordinates + time), RQ4 | no passenger ID; destination prediction not in mentor's plan (CLAUDE.md §12); `DAY_TYPE` constant; `MISSING_DATA` unreliable |
| NYC TLC | ✓ | 1 | 0 | 1 | 0 | 3 | 1 | 3 | 3 | 2 | 2 | **46** | Drop (reference only) | zone-level demand context | no actor at all; 1 month; zone lookup table not pulled |
| MovieLens | ✓ | 3 | 3 | 3 | 3 | 0 | 2 | 0 | 3 | 0 | 3 | **73** | Prototype only | pipeline smoke test | no context, ratings only — never in benchmark conclusions |

Score arithmetic (for checking): Expedia 38/48, Trivago 37/48, Airbnb 21/48, Akeed 39/48, Porto 33/48, NYC 22/48, MovieLens 35/48.

### Notes on individual scores

- **Expedia C4 = 1†**: 3.0 M bookings over 814 k booking users (mean 3.7) with 38.8 % single-booking users suggests a median of 2. C8 = 2 because the full file exceeds RAM but a user-level sample fits and no GPU is needed.
- **Trivago C2 = 2**: sessions exist, but 56.4 % of users and 57.6 % of items appear once, and the 6-day window makes `user_id` weak across sessions. C4 = 2† counts all item actions per session (15.9 M rows over 911 k sessions, mean 17.5; median unknown). C7 = 2 because the strongest signal is a clickout, not a booking.
- **Airbnb C3 = 1**: `date_first_booking` is date-only and every user has exactly one booking, so no sequence exists.
- **Akeed C5 = 2**: order time, delivery timestamps, vendor category/tags count; customer coordinates do not (synthetic). C9 = 2 (vendor `city_id` + timestamps + gender/dob) rather than 3 for the same reason. C10 = 1 for synthetic coordinates, 41 % non-positive delivery distance, duplicate IDs.
- **Porto C1 = 1**: the natural task (destination from partial trajectory) is not in the mentor's phase plan and needs grid/POI discretisation. Scored on the taxi × stand reframing; C2/C4 are on a proxy actor.
- **NYC TLC**: kept as a zone-level demand/context reference only.

### Link fields for the RQ3 simulator (pairwise check)

| Pair | Shared fields | Status |
|---|---|---|
| Expedia (hospitality) ↔ Akeed (food) | timestamps (both full, non-overlapping years: 2013–14 vs 2019–20 — link by time-of-day / day-of-week patterns, not calendar date); user attributes (Expedia: `user_location_*`, party size; Akeed: gender, dob, language); no shared real coordinates | Linkable on time patterns + attributes; label as simulated (CLAUDE.md §2) |
| Porto (ride) ↔ Akeed (food) | timestamps (2013–14 vs 2019–20, same caveat); Porto has real coordinates, Akeed only vendor coordinates (customer coordinates synthetic) | Linkable on time patterns; spatial link is one-sided — state as limitation |

## Decision (proposed, pending mentor confirmation)

- Hospitality: **Expedia** core (seasonality, trip context). **Trivago** kept as a second core for session-based models (RQ1 session part) since it also clears 70; if time is short, drop Trivago first. Airbnb dropped.
- Food: **Akeed** core. Yelp not scored until downloaded; only revisit if Akeed's 100-vendor catalog proves too small for RQ4 coverage analysis.
- Ride: **Porto** secondary, scoped to the RQ3 ride-side simulator input; no ride-domain model benchmark unless CLAUDE.md §12 decides destination prediction stays in scope. NYC TLC reference only.
- Prototype: MovieLens.

Update CLAUDE.md §5 ("Chosen core datasets") and §12 once confirmed.
