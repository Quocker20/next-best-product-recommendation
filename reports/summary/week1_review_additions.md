# Week-1 Review Response — Computed Content for the Four Mentor Points

**Date**: 2026-09-21
**Scope**: Expedia (core dataset) + the five comparison datasets. Travel domain only, per CLAUDE.md §1.
**Purpose**: supply the measured content for the slides the mentor asked for. No slide file was edited; this document is the source the slides should be built from.

**Provenance rule (CLAUDE.md §7/§10)**: every number below was produced by a script in `scripts/` and read back from its JSON/CSV output. Nothing is estimated or recalled. The source file is named in each section.

## 0. What was executed

| Script | Work done | Output |
|---|---|---|
| `scripts/expedia_eda_detail.py` | Full chunked pass over `train.csv` (37,670,293 rows): per-field coverage, daily/hourly/weekday series, lead time, stay length, distance, user depth, cluster/destination concentration, segment cross-tabs | `expedia_eda_detail.json`, `expedia_daily_series.csv`, `expedia_field_table.csv` |
| `scripts/expedia_schema_table.py` | Second pass for value ranges + distinct counts; merges group labels and field descriptions | `expedia_schema_table.{csv,json,md}` |
| `scripts/expedia_timeseries.py` | Trend/weekday decomposition, robust anomaly detection, holiday matching, YoY | `expedia_timeseries.json`, `expedia_daily_anomalies.csv` |
| `scripts/expedia_anomaly_audit.py` | 17 defect checks, clicks vs bookings, check-in year histogram | `expedia_anomaly_audit.json` |
| `scripts/expedia_context_signal.py` | Cluster-distribution shift per context (JS divergence) + popularity probe on the locked temporal split | `expedia_context_signal.json` |
| `scripts/dataset_benchmark.py` | Like-for-like metrics for 6 datasets (Expedia, Trivago, Airbnb, Hotel Booking Demand, Akeed, Porto) | `dataset_benchmark.{json,csv}` |
| `scripts/trivago_profile.py` | Backup-dataset depth profile (rows/sessions/clickouts per user, item metadata) | `trivago_profile.json` |
| `scripts/report_figures.py` | Post-processing only: every ratio/share quoted below | `report_figures.json` |

All outputs are in `reports/summary/`. Total measured volume: 37.67M Expedia rows + 15.93M Trivago rows + 1.71M Porto rows + 468K rows across the three small datasets.

---

## 1. Mentor point 1 — Field documentation in table format

Full paste-ready table: **`reports/summary/expedia_schema_table.md`** (24 rows, columns: No. / Group / Field / Description / Coverage all / Coverage bookings / Distinct / Range / Sample values).

### 1.1 Seven feature groups

| Group | Fields | Role in the benchmark |
|---|---|---|
| Interaction / time | `date_time`, `is_booking`, `cnt` | Event label and the temporal-split key |
| User identity | `user_id` | The only personalisation key (1,198,786 distinct; 813,985 with ≥1 booking) |
| Trip context | `srch_ci`, `srch_co`, `srch_adults_cnt`, `srch_children_cnt`, `srch_rm_cnt`, `is_package` | RQ1 context slices: season, party, package |
| Search / destination | `srch_destination_id`, `srch_destination_type_id` | Strongest single predictor (see §3.14) |
| Hotel / target | `hotel_cluster` (TARGET), `hotel_continent`, `hotel_country`, `hotel_market` | 100-class target + its geography |
| User geography | `user_location_city/region/country`, `orig_destination_distance` | Origin side; the only field with real missingness |
| Channel / platform | `site_name`, `posa_continent`, `channel`, `is_mobile` | Acquisition context |

### 1.2 The table (measured)

| No. | Group | Field | Coverage (all) | Coverage (bookings) | Distinct | Range | Samples |
|----:|---|---|---:|---:|---:|---|---|
| 1 | Interaction / time | `cnt` | 100.00% | 100.00% | 104 | 1 .. 269 | 3, 1, 2 |
| 2 | Interaction / time | `date_time` | 100.00% | 100.00% | n/a | 2013-01-07 .. 2014-12-31 | 2014-08-11 07:46:59 |
| 3 | Interaction / time | `is_booking` | 100.00% | 100.00% | 2 | 0 .. 1 | 0, 1 |
| 4 | User identity | `user_id` | 100.00% | 100.00% | 1,198,786 | 0 .. 1,198,785 | 12, 93, 501 |
| 5 | Trip context | `is_package` | 100.00% | 100.00% | 2 | 0 .. 1 | 1, 0 |
| 6 | Trip context | `srch_adults_cnt` | 100.00% | 100.00% | 10 | 0 .. 9 | 2, 1, 3 |
| 7 | Trip context | `srch_children_cnt` | 100.00% | 100.00% | 10 | 0 .. 9 | 0, 2, 3 |
| 8 | Trip context | `srch_ci` | 99.88% | **100.00%** | 1,269 | 2012-02-15 .. **2558-03-15** | 2014-08-27 |
| 9 | Trip context | `srch_co` | 99.88% | **100.00%** | 1,262 | 2012-09-04 .. **2558-03-16** | 2014-08-31 |
| 10 | Trip context | `srch_rm_cnt` | 100.00% | 100.00% | 9 | 0 .. 8 | 1, 2, 3 |
| 11 | Search / destination | `srch_destination_id` | 100.00% | 100.00% | 59,455 | 0 .. 65,107 | 8250, 14984 |
| 12 | Search / destination | `srch_destination_type_id` | 100.00% | 100.00% | 10 | 0 .. 9 | 1, 6, 4 |
| 13 | Hotel / target | `hotel_cluster` | 100.00% | 100.00% | 100 | 0 .. 99 | 1, 80, 21 |
| 14 | Hotel / target | `hotel_continent` | 100.00% | 100.00% | 7 | 0 .. 6 | 2, 0, 3 |
| 15 | Hotel / target | `hotel_country` | 100.00% | 100.00% | 213 | 0 .. 212 | 50, 185, 151 |
| 16 | Hotel / target | `hotel_market` | 100.00% | 100.00% | 2,118 | 0 .. 2,117 | 628, 1457 |
| 17 | User geography | `orig_destination_distance` | **64.10%** | **66.17%** | 8,495,289 | 0.01 .. 12,407.90 | 2234.26, 913.19 |
| 18 | User geography | `user_location_city` | 100.00% | 100.00% | 50,447 | 0 .. 56,508 | 48862, 35390 |
| 19 | User geography | `user_location_country` | 100.00% | 100.00% | 237 | 0 .. 239 | 66, 195, 69 |
| 20 | User geography | `user_location_region` | 100.00% | 100.00% | 1,008 | 0 .. 1,027 | 348, 442, 189 |
| 21 | Channel / platform | `channel` | 100.00% | 100.00% | 11 | 0 .. 10 | 9, 3, 2 |
| 22 | Channel / platform | `is_mobile` | 100.00% | 100.00% | 2 | 0 .. 1 | 0, 1 |
| 23 | Channel / platform | `posa_continent` | 100.00% | 100.00% | 5 | 0 .. 4 | 3, 4, 1 |
| 24 | Channel / platform | `site_name` | 100.00% | 100.00% | 45 | 2 .. 53 | 2, 30, 37 |

Descriptions per field are in `expedia_schema_table.md` (kept out of this summary for width).

**Insight**: 21 of 24 columns are 100 % populated across all 37.67M rows; on the 3.00M booking rows used for training, **23 of 24 are 100 % populated** and `orig_destination_distance` is the single incomplete field (66.17 %). The distance field is therefore the only one that needs a missing-value policy — everything else can be used as-is with no imputation.

**Insight (defect visible only in the range column)**: `srch_ci` ranges to **2558-03-15**. Quantified in §3.16: the far-future check-in dates exist exclusively on click rows (47 rows after 2016, 6 of them after 2050); booking rows top out at check-in year 2016 with a maximum lead time of 498 days. This is the clearest evidence for the "train on bookings, treat clicks as a weaker auxiliary signal" rule.

---

## 2. Mentor point 2 — Quantitative dataset benchmark

Source: `reports/summary/dataset_benchmark.{json,csv}` — every cell measured from the raw files in one pass per dataset.

| Metric | **Expedia** (chosen) | Trivago 2019 (backup) | Airbnb New User | Hotel Booking Demand | Akeed (food) | Porto Taxi (ride) |
|---|---|---|---|---|---|---|
| Rows scanned | **37,670,293** | 15,932,992 | 213,451 | 119,390 | 135,303 | 1,710,670 |
| Columns | 24 | 12 | 16 | 32 | 26 | 9 |
| File size (MB) | 4,070.4 | 2,100.8 | 24.9 | 16.9 | 22.9 | 1,942.8 |
| Users (distinct) | **1,198,786** (813,985 with a booking) | 730,803 | 213,451 | **none** | 27,445 | **none** (448 drivers) |
| Items (target space) | 100 clusters | 289,506 hotels | 12 countries | 10 room types | 100 vendors | undefined (GPS point) |
| Target interactions | **3,000,693 bookings** | 1,586,586 clickouts | 88,908 first bookings | 119,390 rows | 135,303 orders | 1,710,670 trips |
| Interactions / user (median) | **2.0** (mean 3.69) | 5.0 rows (mean 21.8) | 1.0 by construction | n/a | 2.0 (mean 4.93) | n/a |
| Users with ≥2 interactions | **61.22 %** | 80.43 % (rows) / 16.45 % (sessions) | **0 %** | n/a | 63.86 % | n/a |
| Time span | **2013-01-07 → 2014-12-31 (723 days)** | 2018-11-01 → 2018-11-06 (**5 days**) | 2010-01-01 → 2014-06-30 (1,641 d) | 2014-10-17 → 2017-09-14 (1,063 d) | 2019-05-10 → 2020-02-29 (295 d) | 2013-07-01 → 2014-06-30 (364 d) |
| Overall missing rate (cells) | **1.51 %** | 22.74 % | 6.40 % | 3.39 % | 27.18 % | 16.44 % |
| Columns with 0 missing | **21 / 24** | 9 / 12 | 13 / 16 | 28 / 32 | 11 / 26 | 6 / 9 |
| Worst column | `orig_destination_distance` 35.90 % | `current_filters` 92.76 % | `date_first_booking` 58.35 % | `company` 94.31 % | `promo_code` 96.82 % | `ORIGIN_CALL` 78.68 % |
| Sparsity (user × item) | **96.31 %** | 99.99925 % | 96.53 % | n/a | 95.07 % | n/a |
| Interactions per item | **30,006.9** | 5.48 | 7,409 | 11,939 | 1,353 | n/a |
| Signal type | **Booking** (clicks separable) | Clickout only | First-booking country | Reservation record | Delivered order | Trajectory |
| Blocking defect | none | 5-day span | 1 event/user | no guest id; 26.80 % duplicate rows | synthetic coordinates; 41.10 % orders with distance ≤ 0 | no rider identity |

### 2.1 Why Expedia wins, per metric

- **Span**: 723 days vs Trivago's 5 — **144.6× longer**. Seasonality (RQ1) is measurable in Expedia and structurally impossible in Trivago.
- **Density per item**: 30,006.9 bookings per cluster vs Trivago's 5.48 clickouts per hotel. Full-catalog ranking over 100 clusters is statistically well-supported; ranking 289,506 hotels from 1.59M clickouts is not, without a candidate-set restriction.
- **Signal quality**: Expedia's target is a completed booking. Trivago's is a clickout; Airbnb's is a first-destination country (1 per user, 58.35 % of users never book); Hotel Booking Demand has no user key at all.
- **Cleanliness**: 1.51 % overall missing cells, 21/24 columns complete — the lowest defect load of the six. Akeed (27.18 %) and Trivago (22.74 %) are an order of magnitude worse.
- **Repeat behaviour**: 61.22 % of booking users have ≥2 bookings — enough history for CF/MF and sequence models. Airbnb has 0 % by construction; Trivago has 16.45 % of users with more than one session.

### 2.2 What Trivago still wins (the reason it stays the contingency)

Source: `reports/summary/trivago_profile.json`.

- 927,142 items carry text property tags (median 15 tags, max 112, 157 distinct properties) — directly usable as LLM-rerank context, which Expedia's anonymous cluster ids cannot provide.
- Every clickout ships its impression list (median 25 items) and displayed prices (median 82, p10 29, p90 233) — a ready-made candidate set and the price field Expedia lacks.
- 34,752 real `"City, Country"` strings and 55 platforms, vs Expedia's integer-coded geography.

---

## 3. Mentor point 3 + 4 — Visual EDA: chart specs, data and insight

Each chart below lists the plot, the exact data source, and the insight line to print beneath it (mentor point 4). Priority **P1** = must be on the slides; **P2** = if space allows.

### 3.1 (P1) Daily transaction volume, 2013-01-07 → 2014-12-31
- **Plot**: line chart, 724 daily points, two series (bookings; clicks on a secondary axis), anomaly days marked.
- **Data**: `expedia_daily_series.csv` (date, clicks, bookings) and `expedia_daily_anomalies.csv` (expected, robust_z, is_anomaly, holiday).
- **Numbers**: mean 4,144.6 bookings/day, median 3,479, std 1,687.6 (CV 0.407), min 1,538 on 2013-12-25, max 9,246 on 2014-12-01.
- **Insight**: Volume nearly doubles across the two years — 1,024,389 bookings in 2013 vs 1,976,304 in 2014 (**+92.9 %**). A temporal split therefore trains on a materially smaller and different-mix period than it tests on; this is a real distribution shift to report, not a bug.

### 3.2 (P1) Weekly periodicity
- **Plot**: bar chart of booking share by weekday, with the weekday factor overlaid.
- **Data**: `expedia_timeseries.json` → `periodicity`.
- **Numbers**: Mon 16.34 %, Tue 16.28 %, Wed 16.25 %, Thu 14.83 %, Fri 13.93 %, Sun 12.00 %, Sat 10.36 %. Weekend share 22.36 %. Peak/trough ratio 1.577. Autocorrelation: lag-1 0.897, **lag-7 0.975**, lag-364 0.556.
- **Insight**: The dominant cycle is weekly, not annual (lag-7 autocorrelation 0.975 vs lag-364 0.556). Any time-aware feature must encode day-of-week; a model that only sees month will miss the strongest rhythm in the data.

### 3.3 (P1) Hour-of-day profile and conversion
- **Plot**: dual line — bookings by hour and conversion rate (bookings / all events) by hour.
- **Data**: `report_figures.json` → `hour_of_day`, `hour_highlights`.
- **Numbers**: booking peak at **11:00** (6.53 % of the day's bookings), trough at **03:00** (0.91 %) — a 7.2× swing. Clicks peak later, at **18:00**. Conversion is highest at **08:00 (8.77 %)** and lowest at **00:00 (6.71 %)**.
- **Insight**: Browsing and buying peak at different hours — evening traffic is the highest-volume but not the highest-converting. Hour-of-day is therefore an independent context feature, not a proxy for traffic volume.

### 3.4 (P1) Monthly volume, 2013 vs 2014 (YoY)
- **Plot**: grouped bars (12 months × 2 years) + YoY growth line.
- **Data**: `expedia_timeseries.json` → `monthly_bookings`, `yoy_2014_vs_2013_by_month`.
- **Numbers**: growth rises monotonically through the year, from +46.2 % (January) to **+165.3 % (December)**. Monthly volume runs 68,894 (2013-01) to 202,426 (2014-10).
- **Insight**: Growth is a trend, not a seasonal effect: the 2014 curve sits above 2013 in every month and the gap widens. Booking-date seasonality in this file is confounded with platform growth — the seasonality claim in the report must be based on **check-in month** (§3.5), not on booking-date volume.

### 3.5 (P1) Check-in seasonality, overall and by party type
- **Plot**: line chart of check-in-month share, one line for overall plus four segment lines (solo / couple / family / group).
- **Data**: `expedia_eda_detail.json` → `checkin_month_share`, `checkin_month_share_by_party`.
- **Numbers**: overall peak August 10.75 %, trough February 5.36 % (**2.0× swing**). By segment: families put **28.45 %** of their check-ins in July+August vs **16.41 %** for solo travellers (1.73×); solo peaks in October (11.11 %).
- **Insight**: The season is not one curve — it is four. Families are a summer-school-holiday segment, solo travellers an autumn segment. This is the concrete case for context-aware models over a single global seasonality feature, and the slice the RQ1 result table should report.

### 3.6 (P1) Conversion rate over time
- **Plot**: line chart, daily booking rate with a 28-day rolling mean.
- **Data**: `expedia_daily_series.csv` (`booking_rate`), `expedia_timeseries.json` → `conversion_rate`.
- **Numbers**: mean 8.38 %; 2013 mean 9.17 % vs 2014 mean 7.60 % (**−17.1 % relative**). Daily range 5.57 % (2014-12-28) to 14.25 % (2013-02-06). By weekday: Fri 9.08 % highest, Sun 7.49 % lowest.
- **Insight**: Click volume grew faster than bookings, so the click:booking ratio drifts across the two years. Any model that mixes clicks and bookings as one signal inherits this drift; keeping `event_type` explicit (CLAUDE.md §7 schema) is what prevents it.

### 3.7 (P1) Anomaly detection on the daily series
- **Plot**: the §3.1 line chart with flagged days highlighted; a companion table of the top spikes and drops.
- **Data**: `expedia_timeseries.json` → `anomalies`; per-day detail in `expedia_daily_anomalies.csv`.
- **Method** (state it on the slide): observed ÷ (7-day centred trend × weekday factor), robust z = (residual − median) / (1.4826 × MAD), flag |z| > 3.5.
- **Numbers**: **27 of 724 days flagged (3.73 %), 14 of them fall on a programmatically computed US public holiday.**

  | Type | Date | Weekday | Bookings | vs expected | z | Holiday |
  |---|---|---|---:|---:|---:|---|
  | Drop | 2013-12-25 | Wed | 1,538 | −33.9 % | −9.55 | Christmas Day |
  | Drop | 2013-12-31 | Tue | 2,113 | −25.2 % | −7.11 | New Year's Eve |
  | Drop | 2014-12-25 | Thu | 4,346 | −25.2 % | −7.10 | Christmas Day |
  | Drop | 2014-01-01 | Wed | 2,172 | −24.1 % | −6.81 | New Year's Day |
  | Drop | 2013-12-24 | Tue | 1,776 | −23.1 % | −6.52 | Christmas Eve |
  | Spike | 2014-12-27 | Sat | 5,237 | +25.1 % | +7.08 | — (post-Christmas Saturday) |
  | Spike | 2013-12-28 | Sat | 2,005 | +24.4 % | +6.88 | — (post-Christmas Saturday) |
  | Spike | 2013-12-27 | Fri | 2,545 | +20.1 % | +5.66 | — |
  | Spike | 2014-12-01 | Mon | 9,246 | +17.1 % | +4.81 | Cyber Monday |
  | Spike | 2013-12-02 | Mon | 3,336 | +14.2 % | +3.99 | Cyber Monday |

- **Insight**: Anomalies are calendar-driven and repeat across both years — demand collapses on the holiday itself and rebounds 2–3 days later. The single largest day in the dataset is Cyber Monday 2014 (9,246 bookings). A holiday-calendar flag is a cheap, legitimate context feature; without it the model will systematically over-predict on 25 December.

### 3.8 (P1) Booking-frequency distribution (cold start)
- **Plot**: bar chart of users by booking count bucket, with the cumulative booking share annotated.
- **Data**: `expedia_eda_detail.json` → `user_depth`.
- **Numbers**: 1 booking **38.78 %** (315,679 users), 2 bookings 20.65 %, 3–4 19.14 %, 5–10 15.19 %, 11–20 4.44 %, 21+ 1.80 %. Median 2, p90 8, p99 27. The **top 10 % of users hold 42.37 % of all bookings**.
- **Insight**: Almost 60 % of users have at most two bookings, so a single averaged accuracy number is dominated by users a CF model can barely fit. The benchmark must report the activity slices separately — and 30.14 % of the test-period users never appear in the training period at all (§3.14).

### 3.9 (P2) Lead time (booking date → check-in)
- **Plot**: histogram over the fixed buckets.
- **Data**: `expedia_eda_detail.json` → `lead_time_days`.
- **Numbers**: median **16 days**, mean 36.2, p90 98, p99 256. Same-day 6.52 %, 1–3 days 16.16 %, 15–30 days 17.37 %, 180+ days 3.08 %. 617 bookings (0.02 %) have a check-in before the search date.
- **Insight**: Half of all bookings are made within 16 days of arrival, but the tail runs past six months — the same catalogue is being shopped on two very different horizons. Lead time is a candidate context feature for the Phase-4 ablation, and the 617 negative-lead rows are a filter rule for the cleaning script.

### 3.10 (P2) Stay length
- **Plot**: histogram over nights.
- **Data**: `expedia_eda_detail.json` → `stay_length_nights`.
- **Numbers**: median **2 nights**, mean 2.43, p90 5, p99 10. 1 night 43.10 %, 2 nights 22.50 %, 3 nights 14.52 %, 8–14 nights 2.11 %, >30 nights 6 bookings.
- **Insight**: The dominant product is a 1–2 night city stay (65.6 % combined), not a week-long resort holiday. This is the sharpest mismatch with the Vinpearl use case and must be stated as a proxy limitation next to the `hotel_cluster` caveat.

### 3.11 (P2) Origin–destination distance
- **Plot**: histogram over distance bands, with the missing share called out as a separate bar.
- **Data**: `expedia_eda_detail.json` → `orig_destination_distance`.
- **Numbers**: missing on **33.83 %** of bookings. Among the rest: median 805.8 miles, mean 1,688.7, p90 5,035.0, max 12,199.2. Bands: <50 mi 7.47 %, 50–200 mi 15.59 %, 200–500 mi 18.35 %, 1,000–2,000 mi 16.78 %, 5,000+ mi 10.17 %. 21,422 bookings (0.71 %) sit under 1 mile.
- **Insight**: Trip distance is trimodal-ish — local (<200 mi, 23.1 %), domestic (200–2,000 mi, 49.0 %) and long-haul (>2,000 mi, 27.9 %) — so it carries real signal, but a third of rows would need imputation. Recommendation: use it as an optional feature group in the Phase-4 ablation and report results with and without it, rather than imputing silently.

### 3.12 (P2) Cluster popularity curve
- **Plot**: sorted bar/line of all 100 cluster shares (descending), cumulative line overlaid.
- **Data**: `expedia_eda_detail.json` → `cluster_popularity.share_sorted_desc` (100 values).
- **Numbers**: top cluster 4.03 %, top-10 22.87 %, top-20 38.54 %, bottom-50 25.43 %. Most/least popular ratio **49.1×**. Most popular cluster id 91 (120,972 bookings); least id 74 (2,465).
- **Insight**: The catalogue is skewed but not extreme — no cluster exceeds 4.03 %, so a popularity baseline is weak (Recall@1 = 4.04 %, §3.14) and catalog coverage over 100 items is near-trivial. This confirms the CLAUDE.md §3 note: report intra-list diversity and popularity bias for RQ4, not coverage.

### 3.13 (P2) Destination concentration and package mix
- **Plot**: cumulative share curve over destinations (log x), plus a second panel: monthly `is_package` share over the 24 months.
- **Data**: `expedia_eda_detail.json` → `destinations`, `monthly_package_share`.
- **Numbers**: 36,933 destinations appear in bookings; top-10 cover 11.51 %, top-100 35.38 %, top-1,000 72.27 %; **10,789 destinations (29.21 %) have exactly one booking**. Package share falls from 20.75 % (2013-01) to **9.42 % (2014-12)**, a 54.6 % relative decline, peaking at 21.78 % in 2014-01.
- **Insight**: The destination space has a long, thin tail — per-destination models need a back-off rule for the 29 % of destinations seen once. Separately, the package mix halves over the two years: a context feature whose base rate drifts. Ablations on `is_package` must be read against the split date, and the model should not be allowed to learn the 2013 package prior and apply it to 2014.

### 3.14 (P1) Does context actually help? Popularity probe on the locked split
- **Plot**: horizontal bar chart of Recall@5 per rule, with the global-popularity bar as the reference line.
- **Data**: `expedia_context_signal.json` → `popularity_probe`; lifts in `report_figures.json`.
- **Protocol** (identical to CLAUDE.md §8): temporal split at the 80th percentile of booking `date_time` = **2014-09-29 12:14:32.6**, 2,400,554 train / 600,139 test bookings, full ranking over all 100 clusters, single run, no tuning. **This is an EDA probe, not the Phase-2 benchmark.**

  | Rule | Recall@1 | Recall@5 | Lift @5 vs global |
  |---|---:|---:|---:|
  | Global popularity | 4.04 % | 14.06 % | 1.00× |
  | Popularity by check-in month | 4.04 % | 13.99 % | **0.99×** |
  | Popularity by party type | 4.04 % | 14.14 % | 1.01× |
  | Repeat-last-cluster | 6.02 % | 16.82 % | 1.20× |
  | Popularity by `hotel_market` | 14.67 % | 45.38 % | 3.23× |
  | **Popularity by `srch_destination_id`** | **19.03 %** | **53.07 %** | **3.77×** |
  | Popularity by destination × month | 17.28 % | 46.55 % | 3.31× |
  | Popularity by destination × party | 18.75 % | 51.59 % | 3.67× |
  | Popularity by destination × package | 19.03 % | 53.08 % | 3.78× |

  Coverage: 99.37 % of test rows have their destination present in the training table; 69.86 % of test rows belong to a user with training history.
- **Insight 1**: Geography is the dominant signal — conditioning popularity on the searched destination alone lifts Recall@5 from 14.06 % to 53.07 % (3.77×). Any baseline table that reports only global popularity understates the bar the learned models must clear.
- **Insight 2 (honest negative result)**: Season and party type, used alone as count tables, add **nothing** (0.99× and 1.01×), and splitting destination further by month makes it **worse** (53.07 % → 46.55 %, −12.3 %) because the per-cell counts fragment. Context in this dataset is not usable by naive conditioning; it needs models that share statistical strength across cells (MF/embedding/sequence models). That is precisely the RQ1 hypothesis, now with a measured motivation rather than an assumption.

### 3.15 (P2) Where context actually moves the distribution
- **Plot**: bar chart of mean Jensen-Shannon divergence (bits) between the conditioned and global `hotel_cluster` distribution, one bar per context field.
- **Data**: `expedia_context_signal.json` → `distribution_shift`, `destination_mix_shift_by_checkin_month`.
- **Numbers**: destination 0.4076 bits (mean over 78 destinations with ≥5,000 bookings; max 0.8439), hotel market 0.3953, package flag 0.0728 (package rows) vs 0.0043 (non-package), party 0.0139, check-in month **0.0019**. The destination *mix* itself shifts with the season by 0.0376 bits — **19.4× more than the cluster mix does**.
- **Insight**: Seasonality operates through *where* people go, not *which cluster* they pick once the destination is fixed. That is why month adds nothing on top of destination (§3.14) and why the seasonal RQ1 slice should be evaluated as a destination-mix effect.

### 3.16 (P1) Data-quality audit: clicks vs bookings
- **Plot**: grouped bar chart, defect rate on all rows vs on booking rows (log scale).
- **Data**: `expedia_anomaly_audit.json` (17 checks over all 37.67M rows).

  | Check | All rows | Booking rows |
  |---|---:|---:|
  | `srch_ci` missing | 47,083 (0.1250 %) | **0 (0 %)** |
  | `srch_co` missing | 47,084 (0.1250 %) | **0 (0 %)** |
  | Check-in before the search date | 8,457 (0.0225 %) | 617 (0.0206 %) |
  | Check-in after 2016 | 47 (0.0001 %) | **0** |
  | Lead time > 365 days | 45,131 (0.1198 %) | 1,817 (0.0606 %) |
  | Check-out before check-in | 798 (0.0021 %) | **3** |
  | Check-out = check-in (0 nights) | 144,804 (0.3844 %) | **1** |
  | Stay > 30 nights | 18,005 (0.0478 %) | 6 |
  | Zero adults | 70,979 (0.1884 %) | 5,239 (0.1746 %) |
  | Zero rooms | 859 (0.0023 %) | 250 (0.0083 %) |
  | Adults > 4 | 767,354 (2.0370 %) | 48,268 (1.6086 %) |
  | Guests > 4 × rooms | 766,826 (2.0356 %) | 35,606 (1.1866 %) |
  | `cnt` > 50 | 136 (0.0004 %) | **0** |
  | Distance missing | 13,525,001 (35.9036 %) | 1,015,179 (33.8315 %) |
  | Distance < 1 mile | 178,722 (0.4744 %) | 21,422 (0.7139 %) |

  Check-in year histogram, all rows: 2012 × 11, 2013 × 9,853,622, 2014 × 23,427,909, 2015 × 4,338,918, 2016 × 2,703, then 24 rows scattered over 2017–2030 and **2 rows in 2057, 5 in 2557, 1 in 2558**. Booking rows: 2012 × 1, 2013 × 947,598, 2014 × 1,861,675, 2015 × 191,327, 2016 × 92 — and nothing beyond. Maximum lead time: 198,409 days on click rows vs **498 days** on booking rows; maximum stay 516 nights vs **129**.
- **Insight**: Every extreme defect lives on the click side. The booking subset has zero missing dates, zero impossible check-in years, one zero-night stay and three reversed date pairs out of 3,000,693 rows. This is the quantified version of "the booking core is clean" — and the reason the benchmark trains on `is_booking == 1` while clicks stay a separately labelled auxiliary signal.

### 3.17 (P2) Segment cross-tab
- **Plot**: small multiples — for each party type: share of bookings, package rate, median lead time, mobile share.
- **Data**: `expedia_eda_detail.json` → `segments`.
- **Numbers**:

  | Segment | Share of bookings | Package rate | Median lead time | Mobile share |
  |---|---:|---:|---:|---:|
  | Couple | 43.98 % | 14.60 % | 19 d | 11.94 % |
  | Solo | 28.91 % | 15.30 % | **9 d** | 6.63 % |
  | Family | 18.82 % | **8.92 %** | 22 d | 10.40 % |
  | Group (3+ adults) | 8.12 % | 12.18 % | 24 d | 9.55 % |
  | Unknown (0 adults) | 0.17 % | **93.46 %** | 45 d | 10.11 % |

  Overall: mobile 9.92 %, package 13.67 %. Channel 9 carries 58.95 % of bookings; site 2 carries 65.55 %; hotel continent 2 carries 60.43 %.
- **Insight 1**: Party type changes behaviour on every axis — solo travellers book 13 days later than groups and use mobile half as often as couples. These are the natural RQ1 context slices.
- **Insight 2 (data artefact worth a line on the slide)**: the 5,239 zero-adult bookings are 93.46 % packages with a 45-day median lead time — they are not ordinary rooms but an artefact of the package booking flow. They should be labelled, not silently dropped, before the party-type slice is reported.

---

## 4. Verification of the numbers already on the week-1 slides

All re-measured against this run. ✓ = confirmed exactly.

| Slide | Claim | Status |
|---|---|---|
| 1, 7 | 37,670,293 rows; 34,669,600 clicks (92.03 %) / 3,000,693 bookings (7.97 %) | ✓ |
| 2, 7 | 813,985 booking users; 1,198,786 users in file | ✓ |
| 3, 7 | Sparsity 97.10 % (user × cluster pairs); 36,933 destinations; 100/100 clusters | ✓ (note: pair-based sparsity 97.10 % from the data card; row-based sparsity is 96.31 % — state which definition the slide uses) |
| 3, 5, 7 | 38.78 % of bookers have 1 booking; median 2.0, mean 3.686, p90 8.0 | ✓ |
| 3 | Check-in seasonality 5.4 % (Feb) → 10.7 % (Aug) | ✓ (5.36 % / 10.75 %) |
| 3, 7 | Party mix 44.0 / 28.9 / 18.8 / 8.1 % | ✓ (43.98 / 28.91 / 18.82 / 8.12 %) |
| 4 | Split cut 2014-09-29; 2.4M train / 600K test; 30.1 % unseen test users | ✓ (2,400,554 / 600,139 / 30.14 %) |
| 5, 6 | Distance missing 35.90 % (file) / 33.83 % (bookings) | ✓ |
| 6 | Check-in/out missing 0.12 % on clicks, 0 on bookings | ✓ (0.1250 %, 0) |
| 6 | Check-out before check-in: 798 file / 3 bookings; zero adults 5,239 (0.17 %) | ✓ |
| 6, 7 | Top cluster 4.0 %, top-10 22.9 %; package 13.7 % | ✓ (4.03 %, 22.87 %, 13.67 %) |
| 8 | Trivago 15.9M rows, 910,683 sessions, 1.58M clickouts, 289K items, 927K items with metadata | ✓ (15,932,992 / 910,683 / 1,586,586 / 289,506 / 927,142) |
| 8 | "56 % of users appear only once" | **Needs wording**: 56.43 % of clicking users have exactly **one clickout**; 83.55 % have exactly one **session**; only 19.57 % have a single **row**. Use the clickout definition. |
| 8 | "57.6 % of items single-interaction" | **Not reproduced**: measured on clickouts, 40.32 % of the 289,506 clicked items have exactly one clickout. The 57.6 % figure needs its definition restated or the slide corrected. |
| 5 | Distance min 0.0 (data card) | Schema scan prints 0.01 after 2-dp rounding; 178,722 rows (0.47 %) are below 1 mile. Quote the <1-mile share instead of the min. |

## 5. Suggested slide additions (no file was edited)

1. **New slide — "Schema in detail"**: the §1.2 table split over two slides by feature group, with the §1 insight lines.
2. **New slide — "Why Expedia, in numbers"**: the §2 comparison table, plus §2.1 as the takeaway strip and §2.2 as the contingency note.
3. **New slides — "Visual EDA"** (3–4 slides): P1 charts §3.1–3.8, §3.14, §3.16, each with its insight line printed underneath.
4. **Revision to existing slide 6**: replace the qualitative anomaly bullets with the §3.16 clicks-vs-bookings table.
5. **Revision to existing slide 4**: add the §3.14 probe table as the measured baseline floor the Phase-2 models must beat.
