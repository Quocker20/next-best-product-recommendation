# Dataset scores (Topic C1)

Results of applying `docs/dataset_rubric.md` to every candidate dataset. See that file for the method (gates, criteria, weights, formula); see `scripts/dataset_scoring.py` for the scoring arithmetic and the `quick_profile()` helper. Inputs: the data cards in `docs/data_cards/` and the EDA notebooks.

Last updated: 2026-09-17.

## Step 0: Interaction definition per dataset

`quick_profile()` assumes the standard interaction schema from CLAUDE.md §7 (`user_id`, `item_id`, `timestamp`). Raw files do not have it, so the mapping must be fixed first — otherwise C2/C4/C7 are not reproducible.

| Dataset | Actor (`user_id`) | Item (`item_id`) | Rows that count as an interaction | Sequence unit |
|---|---|---|---|---|
| Expedia | `user_id` | `hotel_cluster` | `is_booking == 1` (clicks kept as separate, weaker event_type) | user history ordered by `date_time` |
| Trivago 2019 | `session_id` (with `user_id` kept) | `reference` on item actions | `action_type` in {`clickout item`, `interaction item *`}; clickout is the target | session |
| Airbnb New User | `id` | `country_destination` | `country_destination != 'NDF'` | none (one event per user) |
| Akeed | `customer_id` | `vendor_id` | one row per order (`orders.csv`), after dropping duplicate `akeed_order_id` | customer history by `created_at` |
| Instacart | `user_id` | `product_id` | one row per order line (`order_products__prior`/`__train`), `reordered` flag kept as a native label | user history by `order_number` (no absolute date) |
| Yelp | `user_id` | `business_id` | reviews (rating-only signal) | user history |
| Porto Taxi | `TAXI_ID` (**proxy actor** — driver, not passenger) | `ORIGIN_STAND`, or a grid cell of the trip end point | `CALL_TYPE == 'B'` for stands; all trips for grid destination | taxi history by `TIMESTAMP` |
| NYC TLC | `PULocationID` (**proxy actor** — a zone, not a person) | `DOLocationID` | one row per trip | none |
| Citibike | `start_station_id` (**proxy actor** — a dock station, not a person; no rider ID exists at all) | `end_station_id` | one row per trip | none |
| Hotel Booking Demand | `country` (**proxy actor**, weak — guest origin, not a person; no guest/booking ID exists at all) | `reserved_room_type` | one row per booking (all rows — no click/book distinction beyond `reservation_status`) | none |
| MovieLens | `userId` | `movieId` | one row per rating | user history by `timestamp` |

## Per-dataset context field inventory

Raw column → `ctx_*` mapping per CLAUDE.md §7; ✗ = not usable, synthetic counts as ✗.

| Dataset | ctx_timestamp | ctx_location | ctx_price | ctx_session_id | ctx_party_size | ctx_weather |
|---|---|---|---|---|---|---|
| Expedia | `date_time` | `user_location_*`, `hotel_*` (region IDs, not coordinates) | ✗ (no price column) | ✗ | `srch_adults_cnt`, `srch_children_cnt`, `srch_rm_cnt` | untested |
| Trivago 2019 | `timestamp` | `city` (text) | `prices` (pipe-separated, clickout rows only) | `session_id` | ✗ | **feasible** (geocode city → Open-Meteo archive, tested live 2026-09-17) |
| Airbnb New User | `date_account_created`, `timestamp_first_active` | ✗ (no coordinates on user rows; `countries.csv` has destination-level lat/lng only) | ✗ | ✗ (sessions.csv has no session_id, only `user_id` + `action`) | ✗ | untested |
| Akeed | `created_at` | ✗ synthetic — both customer AND vendor coordinates fail the real Oman bbox check (0/100 vendors inside it, one vendor lat=205 which isn't even a valid latitude); `city_id`/`country_id` are constant with no name lookup | `grand_total` | ✗ | ✗ | **infeasible** — no real coordinate or resolvable place name anywhere in the dataset (tested 2026-09-17) |
| Porto Taxi | `TIMESTAMP` | real lat/lon from `POLYLINE` | ✗ | ✗ (no session concept; `CALL_TYPE`/`ORIGIN_STAND` stand in) | ✗ | untested (real coords present, likely feasible — same pattern as Porto's own lat/lon → Open-Meteo call, not yet run) |
| NYC TLC | `tpep_pickup_datetime` | `PULocationID`/`DOLocationID` (zone IDs, not coordinates) | `fare_amount`, `total_amount` | ✗ | `passenger_count` | untested (zone ID → needs the TLC zone lookup table, not pulled, to get coordinates first) |

**Weather-join feasibility test (2026-09-17)**: ran live against Open-Meteo (no API key required). Tested Trivago (feasible — city text needs a one-time geocode step, then archive API returns hourly temperature/precipitation for any date) and Akeed (infeasible — confirmed both customer and vendor coordinates are synthetic garbage, `city_id` is a constant with no lookup, so there is no real location to key weather on at all). This is a feasibility probe only — no weather data has been joined into any interim/processed table (that build happens in Phase 2, per CLAUDE.md §4). See per-dataset notes in `docs/data_cards/trivago_2019.md` and `docs/data_cards/akeed.md`.

## Scoring sheet (first pass from data cards, 2026-09-17)

Values marked † are provisional: the data cards give means or single-interaction shares but not medians; confirm with `quick_profile()` once the interim tables exist. Gates: ✓ pass, ✗ fail.

| Dataset | G1–G5 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | Score | Decision | RQs it serves | Main gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Expedia | ✓ | 2 | 3 | 3 | 1† | 3 | 2 | 3 | 2 | 2 | 3 | **79** | Core | RQ1 (seasonality, context), RQ4 | hotel cluster ≠ room/package; 38.8 % of bookers have one booking; 4 GB file needs user sampling |
| Trivago 2019 | ✓ | 3 | 2 | 1 | 2† | 3 | 3 | 2 | 2 | 2 | 3 | **77** | Core (session-model robustness) | RQ1 (session part), RQ4 | 6-day span — no seasonality; 56 % single-interaction users; clicks not bookings |
| Airbnb New User | ✓ | 1 | 2 | 1 | 0 | 1 | 2 | 3 | 3 | 1 | 2 | **46** | Drop | — | one booking per user, 12-class target, 58 % NDF, no trip context |
| Akeed | ✓ | 3 | 3 | 2 | 1 | 2 | 3 | 3 | 3 | 0 | 1 | **73** | Core | RQ2, RQ4, RQ3 (food side) | customer coordinates synthetic; 41 % `deliverydistance <= 0`; duplicate order/customer IDs; 100 vendors only |
| Yelp | ✓ | 2 | 2 | 3 | 0 | 2 | 3 | 0 | 2 | 2 | 3 | **62** | Secondary | RQ4 (coverage/diversity at scale) | rating-only signal, not order/delivery behavior; median 1 review per user in-dataset (57.1% single-review); no delivery/ETA context — weak fit for RQ2's exploitation-vs-exploration framing |
| Instacart | ✓ | 3 | 3 | 0 | 3 | 1 | 2 | 3 | 2 | 0 | 3 | **69** | Secondary | RQ2 (exploitation/exploration — has a native `reordered` label) | grocery, not restaurant/dish; **no absolute calendar date anywhere** (only day-of-week/hour-of-day, capped `days_since_prior_order`) — cannot support seasonal analysis (RQ1) or calendar-based cross-domain linking |
| Porto Taxi | ✓ | 1 | 1 (proxy) | 3 | 3† (proxy) | 3 | 1 | 3 | 2 | 2 | 2 | **69** | Secondary | RQ3 (ride side, via coordinates + time), RQ4 | no passenger ID; destination prediction not in mentor's plan (CLAUDE.md §12); `DAY_TYPE` constant; `MISSING_DATA` unreliable |
| Citibike | ✓ | 1 | 0 | 2 | 0 | 3 | 1 | 3 | 3 | 2 | 3 | **52** | Secondary | RQ3 (ride side, real coords + time), RQ4 (context reference) | no rider ID at all (`member_casual` is a 2-value category, not identity); 1 month only; destination prediction not in mentor's plan (CLAUDE.md §12) |
| NYC TLC | ✓ | 1 | 0 | 2 | 0 | 3 | 1 | 3 | 3 | 2 | 2 | **50** | Secondary (boundary — flag for mentor) | zone-level demand context | no actor at all; 1 month; zone lookup table not pulled; score sits exactly on the Secondary/Drop line |
| Hotel Booking Demand | ✓ | 1 | 0 | 2 | 0 | 3 | 1 | 3 | 3 | 1 | 1 | **46** | Drop (reference only) | seasonality/cancellation reference | no guest/booking ID at all — not even a proxy actor with real recurrence; 26.8% full-row duplicates unresolvable |
| MovieLens | ✓ | 3 | 3 | 3 | 3 | 0 | 2 | 0 | 3 | 0 | 3 | **73** | Prototype only | pipeline smoke test | no context, ratings only — never in benchmark conclusions |

Score arithmetic (verified with `scripts/dataset_scoring.py`): Expedia 38/48, Trivago 37/48, Airbnb 22/48, Akeed 35/48, Yelp 30/48, Instacart 33/48, Porto 33/48, Citibike 25/48, NYC 24/48, Hotel Booking Demand 22/48, MovieLens 35/48.

### Notes on individual scores

- **Expedia C4 = 1†**: 3.0 M bookings over 814 k booking users (mean 3.7) with 38.8 % single-booking users suggests a median of 2. C8 = 2 because the full file exceeds RAM but a user-level sample fits and no GPU is needed.
- **Trivago C2 = 2**: sessions exist, but 56.4 % of users and 57.6 % of items appear once, and the 6-day window makes `user_id` weak across sessions. C4 = 2† counts all item actions per session (15.9 M rows over 911 k sessions, mean 17.5; median unknown). C7 = 2 because the strongest signal is a clickout, not a booking.
- **Airbnb C3 = 1**: `date_first_booking` is date-only and every user has exactly one booking, so no sequence exists.
- **Akeed C4 = 1**: `quick_profile()` on the deduplicated `orders.csv` (Step 0 mapping) gives a median of 2.0 orders/customer, which is the rubric's C4=1 tier (2), not the earlier provisional 2†. **C9 = 0** (corrected from 2): `vendors.csv`'s `city_id`/`country_id` are a constant (1.0) with no lookup file to resolve a real place name, and customer/vendor coordinates are synthetic (rubric rule: synthetic fields count as absent for C9) — there is no real field left to satisfy even the "City only" (C9=1) tier, matching the weather-join feasibility note above ("no real coordinate or resolvable place name anywhere in the dataset"). **C5 = 2**: order time, delivery timestamps, vendor category/tags count; customer coordinates do not (synthetic). C10 = 1 for synthetic coordinates, 41 % non-positive delivery distance, duplicate IDs. Net effect of the C4/C9 correction: score drops from 81 to 73 — still clears the Core threshold.
- **Porto C1 = 1**: the natural task (destination from partial trajectory) is not in the mentor's phase plan and needs grid/POI discretisation. Scored on the taxi × stand reframing; C2/C4 are on a proxy actor.
- **NYC TLC C3 = 2** (corrected from 1): pickup timestamps are full (`tpep_pickup_datetime`), and once the 18 out-of-range rows are excluded the span is one calendar month (Jan 2024) — the same span/precision profile as Citibike, which is already scored C3=2 below; scoring NYC TLC at 1 while Citibike scores 2 for an equivalent span was inconsistent. This raises the score from 46 to 50, which crosses from Drop into Secondary — a boundary result (exactly 50) that should be treated cautiously and flagged for the mentor rather than taken as a confident upgrade. Otherwise kept as a zone-level demand/context reference only.
- **Citibike** (alternative candidate found in the Thursday dataset search, downloaded 2026-09-17): open license (NYCBS Data Use Policy, no gate at all — the most frictionless download of any dataset here), genuine GPS (100% inside the real NYC bbox, unlike Akeed), and denser station coverage than Porto or NYC TLC (median 381 trips/station, only 0.6% stations with ≤5 trips). But like Porto and NYC TLC it has **no rider identity** (`member_casual` is a 2-value category, not an ID) — C2/C4 = 0. Scores 52, the strongest of the three ride datasets on a like-for-like "no real actor" basis, but destination prediction is still out of the mentor's phase plan (CLAUDE.md §12) regardless of which ride dataset is used. Kept as a secondary ride-context reference alongside Porto, ahead of NYC TLC.
- **Hotel Booking Demand** (alternative candidate found in the Thursday dataset search, downloaded 2026-09-17): CC BY 4.0, the cleanest license of any hospitality dataset here, and rich trip context (party size, dates, price, channel, meal, deposit type) rivaling Expedia. But **it has no guest or booking ID column at all** — not even the weak proxy recurrence that Porto's `TAXI_ID` or NYC's `PULocationID` have — so C2/C4 are both 0, not 1. The only usable "interaction" is an aggregate `country` × `reserved_room_type` proxy (population preference, not personalized). C10 = 1 for the unresolvable 26.8% full-row duplicate rate (a documented artifact of the public release, not our ETL). Confirms Expedia/Akeed remain the right core picks; this one is a same-tier alternative to NYC TLC, not an upgrade — kept as a seasonality/cancellation-prediction reference only.
- **Yelp C4 = 0**: `user.json`'s own `review_count` field is a platform-wide lifetime stat (mean 23.39, max 17,473, confirmed by a full chunked pass over all 1,987,897 rows) and does **not** match this dataset — the actual per-user activity inside `review.json` (the real interaction table) shows 1,136,008 of 1,987,929 users (57.1%) with only 1 review ever, confirmed by a full pass over all 6,990,280 reviews. Use `review.json`, never `user.json.review_count`, for any history-depth or cold-start calculation. C7 = 0 because signal is star ratings/reviews, not orders. C9 = 2: real business coordinates + timestamps exist, but no useful user-side attribute (location, demographics) for the simulator, capping below 3.
- **Instacart scores 69 (Secondary tier, one point below the Core threshold) — flagging honestly rather than just reporting the number (CLAUDE.md §10)**: this is real and reproducible (see arithmetic: 9+6+0+6+2+2+3+2+0+3=33/48), driven by genuinely excellent fundamentals — Instacart's own release already filters out users with <4 orders (`order_number` floor = 4, median 10, C2=3/C4=3), 32.4M order-lines give a native `reordered` label (58.97% reorder rate, directly usable for RQ2 without reframing, C7=3), and the data is essentially defect-free (0 duplicates, 6.03% missing that is 100% structural, C10=3). **C9 = 0** (corrected from 1): Instacart has no geographic field of any kind (no city, coordinates, or state anywhere in the schema), so it does not clear even the "City only" tier. But **C3 = 0**: there is no absolute calendar date anywhere in the dataset — only `order_dow`, `order_hour_of_day`, and a `days_since_prior_order` field capped/censored at 30 days. This is not a minor gap: it means Instacart cannot support RQ1's seasonality slicing at all, and can only be cross-domain-linked by day-of-week/hour-of-day pattern, never by calendar date (same limitation as the Expedia↔Akeed and Porto↔Akeed pairs, but total here rather than partial). C5 = 1 because none of the food-domain context fields the rubric asks for (user/restaurant location, delivery ETA, cuisine) exist — Instacart is grocery, not restaurant/dish, so it cannot answer RQ2's "location, meal time, weather, ETA" question, only the reorder-vs-explore half of it. **Recommendation: treat as Secondary, alongside Yelp** — use it specifically for the RQ2 exploitation/exploration curve (where its native label and lack of cold-start are a genuine advantage over Akeed), not as a replacement for Akeed's restaurant-domain fit or RQ4/RQ3 role.

### Link fields for the RQ3 simulator (pairwise check)

| Pair | Shared fields | Status |
|---|---|---|
| Expedia (hospitality) ↔ Akeed (food) | timestamps (both full, non-overlapping years: 2013–14 vs 2019–20 — link by time-of-day / day-of-week patterns, not calendar date); user attributes (Expedia: `user_location_*`, party size; Akeed: gender, dob, language); no shared real coordinates | Linkable on time patterns + attributes; label as simulated (CLAUDE.md §2) |
| Porto (ride) ↔ Akeed (food) | timestamps (2013–14 vs 2019–20, same caveat); Porto has real coordinates, Akeed only vendor coordinates (customer coordinates synthetic) | Linkable on time patterns; spatial link is one-sided — state as limitation |

## Decision (proposed, pending mentor confirmation)

- Hospitality: **Expedia** core (seasonality, trip context). **Trivago** kept as a second core for session-based models (RQ1 session part) since it also clears 70; if time is short, drop Trivago first. Airbnb dropped.
- Food: **Akeed** core (73, direct order/delivery signal, restaurant domain fit, matches RQ2's exploitation-vs-exploration framing). **Instacart** secondary (scores 69, just under the Core threshold — see notes above): use specifically for the RQ2 reorder-vs-explore curve, where its native `reordered` label and zero-cold-start users beat Akeed, but it cannot support RQ1 seasonality (no calendar date) or the "cuisine"/ETA half of RQ2 (it's grocery). **Yelp** secondary (62) — 150,346-business catalog and real GPS make it a good RQ4 coverage/diversity check, but its rating-only signal, near-zero repeat interactions (median 1 review/user), and lack of delivery/ETA context mean it cannot stand in for Akeed on RQ2.
- Ride: **Porto** secondary (69), scoped to the RQ3 ride-side simulator input; no ride-domain model benchmark unless CLAUDE.md §12 decides destination prediction stays in scope. **Citibike** (52) kept alongside Porto as a secondary context reference — denser stations and real GPS, but same no-rider-ID ceiling. **NYC TLC** (50, corrected from 46 — see C3 note above) also now falls in the Secondary band, but the score sits exactly on the Secondary/Drop boundary; treat as the weakest of the three and flag for mentor review before using beyond a reference role.
- Prototype: MovieLens.

Update CLAUDE.md §5 ("Chosen core datasets") and §12 once confirmed.
