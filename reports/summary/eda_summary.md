# EDA Summary — Dataset Survey and Scoring (Topic C1)

**Date**: 2026-09-17
**Scope**: 11 public datasets across four domains (hospitality, food, ride, prototype), evaluated for the Next-Best-Product Recommendation benchmark (Vinpearl × GSM, Topic C1).
**Sources**: `docs/data_cards/`, `notebooks/`, `docs/dataset_rubric.md`, `docs/dataset_scores.md`.

## 1. Methodology

Each dataset was profiled through an executed EDA notebook (`notebooks/<domain>/<dataset>_eda.ipynb`) following the 11-point checklist in `docs/eda_checklist.md`, then documented in a data card (`docs/data_cards/<dataset>.md`) recording source, license, schema, verified statistics, and a recommendation-task framing. Scoring applies a fixed rubric (`docs/dataset_rubric.md`): five pass/fail gates and ten weighted criteria (C1–C10, 0–3 points each) summed to a 0–100 score via `scripts/dataset_scoring.py`. Score thresholds: ≥70 Core, 50–69 Secondary, <50 Drop.

## 2. Dataset Inventory and Scores

| Domain | Dataset | Score | Decision | License |
|---|---|---|---|---|
| Hospitality | Expedia Hotel Recommendations | 79 | Core | Restricted, no open license (Kaggle competition rules) |
| Hospitality | Trivago RecSys 2019 | 77 | Core (session-model robustness) | Restricted/unverified (Kaggle mirror, license unknown) |
| Hospitality | Airbnb New User Bookings | 46 | Drop | Restricted, no open license (Kaggle competition rules) |
| Hospitality | Hotel Booking Demand | 46 | Drop (reference only) | CC BY 4.0 |
| Food | Akeed Restaurant Recommendation | 73 | Core | Restricted/unverified (Zindi terms, page no longer live) |
| Food | Instacart Market Basket Analysis | 69 | Secondary | CC0: Public Domain |
| Food | Yelp Open Dataset | 62 | Secondary | Restricted, research/personal use (Yelp Dataset License) |
| Ride | Porto Taxi Trajectory (ECML/PKDD 2015) | 69 | Secondary | CC BY 4.0 |
| Ride | Citi Bike Trip Data (2024-01) | 52 | Secondary | Open (NYCBS Data Use Policy) |
| Ride | NYC TLC Trip Records | 50 | Secondary (boundary — flag for mentor) | Open (NYC Open Data Law) |
| Prototype | MovieLens | 73 | Prototype only | Open (no license gate) |

Full scoring breakdown (per-criterion values, arithmetic, notes): `docs/dataset_scores.md`.

## 3. Cross-Cutting Findings

### 3.1 Absence of stable actor identity

Four datasets provide no true user/rider identity, only a proxy:

- Porto Taxi: `TAXI_ID` (driver, not passenger)
- NYC TLC: `PULocationID` (pickup zone)
- Citi Bike: `start_station_id` (dock station; `member_casual` is a two-value category, not an identifier)
- Hotel Booking Demand: no identifier at all; `country` used as a weak, aggregate proxy

This caps C2 (user/session identity) and C4 (history depth) at 0–1 for all four, and rules out personalized sequence modeling without reframing to a proxy-level (zone/station/aggregate) task.

### 3.2 Synthetic or invalid location fields

Akeed's location data is synthetic on both sides: customer coordinates are 100% outside the real Oman bounding box (lat 16–27, lon 51–60), and vendor coordinates likewise fail the same check (0/100 vendors inside the bbox; one vendor latitude of 205 is not a valid latitude value). `city_id`/`country_id` are constant across all 100 vendors with no name lookup. In contrast, Porto Taxi (99.69% of GPS points inside the real Porto bbox), Yelp (100% of business locations inside the real US/Canada bbox), and Citi Bike (100% inside the real NYC bbox) carry genuine coordinates.

### 3.3 Misleading or absent precomputed fields

Two cases where a field needed to assess data quality does not do what it would be expected to:

- NYC TLC: no `MISSING_DATA` or other data-quality flag column exists anywhere in the file (verified full 19-column schema, 2,964,624 rows). Bad rows are entirely unmarked and must be found by manual filtering: 60,371 rows (2.04%) have `trip_distance == 0`, 37,448 rows have negative `fare_amount`, and 56 rows have `tpep_dropoff_datetime` before `tpep_pickup_datetime`.
- Yelp: `user.json.review_count` is a platform-wide lifetime statistic (mean 23.39 over the full 1,987,897-row file, no sampling) and does not reflect activity within this dataset export. The actual per-user activity, computed from a full pass over `review.json` (6,990,280 rows), shows 1,136,008 of 1,987,929 users (57.1%) with only 1 review ever — the two counts diverge sharply and only `review.json` should be used for history-depth or cold-start calculations.

### 3.4 Heavy single-interaction concentration

Cold-start prevalence is high across most datasets: Trivago (56.4% of users, 57.6% of items single-interaction), Akeed (36.1% of customers single-order), Yelp (57.1% of users single-review, dataset-wide median of 1). Instacart is the sole counter-example: its own release already excludes users with fewer than 4 orders, yielding a floor on `order_number` of 4 and a median of 10 orders per user — no user-side cold start exists in this dataset.

### 3.5 License status is largely unresolved

Only five of eleven datasets carry a verified open license: Porto Taxi and Hotel Booking Demand (CC BY 4.0), Instacart (CC0), and NYC TLC / Citi Bike (open government/public data policy, no account gate). The remaining datasets — including both current food and hospitality core candidates, Akeed and Expedia — operate under Kaggle competition rules, Zindi terms, or an unverifiable Kaggle mirror license, none of which grant a confirmed right to redistribute or use commercially. This constrains what can be published in the final report and demo.

### 3.6 No dataset provides a ready-made next-best-item label except Instacart

Every dataset requires reconstructing an interaction/target definition from raw fields (documented per-dataset in `docs/dataset_scores.md`, Step 0): Expedia (click vs. booking), Trivago (clickout action), Akeed (order row after deduplication), Airbnb (destination), Yelp (review). Instacart is the exception: `order_products__prior/train.csv` carries a native `reordered` flag (58.97% reorder rate across 32,434,489 order lines), directly usable for RQ2's exploitation-versus-exploration framing without manual reframing.

### 3.7 Weather-join feasibility (tested against Open-Meteo, 2026-09-17)

Feasibility was tested live, not assumed. Trivago is feasible: its `city` field is a real, geocodable place name; a full chain (geocode → historical archive query) was executed successfully with no API key required. Akeed is infeasible: no real coordinate or resolvable place name exists anywhere in the dataset (see §3.2). No weather data has been joined into an interim or processed table; this is a feasibility probe only, per the project's phase plan (weather-context integration is Phase 4 work).

## 4. Domain-Level Assessment

**Hospitality**: Expedia (79) and Trivago (77) both clear the Core threshold and are proposed as a joint core (session-model robustness for Trivago). Airbnb (46) and Hotel Booking Demand (46) are dropped from the core benchmark; the latter's absence of any guest/booking identifier is a defining, unresolvable limitation, not a data-quality defect.

**Food**: Akeed (73) is the domain-fit core (restaurant/vendor ordering, matches RQ2 directly). Instacart (69) does not clear the Core threshold — it is grocery rather than restaurant/dish, has no real geographic field at all, and contains no absolute calendar date anywhere in the schema — only day-of-week, hour-of-day, and a `days_since_prior_order` field censored at 30 days. It is retained as a secondary dataset specifically for the RQ2 exploitation/exploration curve, where its native label and absence of cold start are a genuine advantage, not as a substitute for Akeed's role in RQ1/RQ3/RQ4. Yelp (62) is retained for RQ4 catalog-coverage analysis at scale.

**Ride**: no ride dataset clears the Core threshold; Porto Taxi (69) is the strongest, followed by Citi Bike (52) and NYC TLC (50, weakest of the three and sitting exactly on the Secondary/Drop boundary — treat with caution pending mentor review). All three require a proxy-actor reframing, and destination prediction is explicitly outside the mentor's current phase plan (open item, CLAUDE.md §12). Ride-domain use is scoped to the RQ3 cross-sell simulator input, not a standalone model benchmark, pending that decision.

**Prototype**: MovieLens (73) is reserved for pipeline smoke-testing only and does not enter benchmark conclusions, per project convention.

## 5. Open Items

- License status for Akeed, Trivago, Expedia, and Airbnb requires resolution before public report/demo distribution.
- Cross-domain simulator links (Expedia↔Akeed, Porto↔Akeed) share no calendar-overlapping timestamps; linking is restricted to day-of-week/hour-of-day pattern matching and must be labeled as simulated (CLAUDE.md §2).
- Ride-domain scope (whether destination prediction remains in the benchmark) is unresolved (CLAUDE.md §12).
- Final dataset selection remains pending mentor confirmation (CLAUDE.md §5, §12).
- Three scores were corrected after an arithmetic/rubric audit (2026-09-17): Akeed 81→73, Instacart 71→69 (Core→Secondary), NYC TLC 46→50 (Drop→Secondary, a boundary result). See `docs/dataset_scores.md` notes for the corrected criteria and rationale; NYC TLC's boundary score in particular should be confirmed with the mentor before relying on it.

## 6. References

- `docs/dataset_rubric.md` — scoring method (gates, criteria, formula)
- `docs/dataset_scores.md` — full scoring sheet, Step 0 interaction mapping, per-dataset notes, proposed decision
- `docs/data_cards/*.md` — per-dataset source, license, schema, verified statistics
- `notebooks/<domain>/<dataset>_eda.ipynb` — executed analysis underlying each data card
- `scripts/dataset_scoring.py` — scoring arithmetic implementation
