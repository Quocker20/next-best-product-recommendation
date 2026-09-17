# Data Card — Hotel Booking Demand

- **Domain**: hospitality
- **Source**: Kaggle ([jessemostipak/hotel-booking-demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)), originally Antonio, Almeida & Nunes (2019), "Hotel booking demand datasets", *Data in Brief*, Vol. 22
- **License**: **CC BY 4.0** (Creative Commons Attribution 4.0 International), stated on the Kaggle dataset page — verified live 2026-09-17. Usability 10/10. Sharing/adaptation permitted with attribution. Cleanest license of any hospitality dataset in this project so far.
- **Location**: `data/raw/hospitality/hotel_booking_demand/`
- **EDA notebook**: `notebooks/hospitality/hotel_booking_demand_eda.ipynb`

## Files (verified `wc -l`)
| file | size | rows | cols |
|---|---|---|---|
| hotel_bookings.csv | 1.25 MB (downloaded zip) | 119,390 | 32 |

## Schema (verified via full-file `dtypes`)
`hotel` str (City Hotel / Resort Hotel), `is_canceled` int64, `lead_time` int64, `arrival_date_year` int64, `arrival_date_month` str, `arrival_date_week_number` int64, `arrival_date_day_of_month` int64, `stays_in_weekend_nights` int64, `stays_in_week_nights` int64, `adults` int64, `children` float64, `babies` int64, `meal` str, `country` str, `market_segment` str, `distribution_channel` str, `is_repeated_guest` int64, `previous_cancellations` int64, `previous_bookings_not_canceled` int64, `reserved_room_type` str, `assigned_room_type` str, `booking_changes` int64, `deposit_type` str, `agent` float64, `company` float64, `days_in_waiting_list` int64, `customer_type` str, `adr` float64 (price/night), `required_car_parking_spaces` int64, `total_of_special_requests` int64, `reservation_status` str, `reservation_status_date` str

## Verified stats (from executed notebook)
- **No booking ID or guest ID column exists anywhere in this file** — the dataset's defining limitation. `is_repeated_guest` is only a pre-computed boolean (3.19% of rows), not an identifier; it cannot reconstruct which rows belong to the same guest.
- **26.8% full-row duplicates** (31,994 / 119,390) — a documented public-release quirk (the original authors stripped any PMS booking key), not a join/ETL artifact on our side. Cannot deduplicate with confidence.
- Missing %: `company` 94.31% (mostly "no company involved"), `agent` 13.69%, `country` 0.41%, all other 29 columns 0%.
- `hotel`: City Hotel 79,330 (66.4%), Resort Hotel 40,060 (33.6%).
- `reservation_status`: Check-Out 75,166 (63.0%), Canceled 43,017 (36.0%), No-Show 1,207 (1.0%). `is_canceled` rate 37.0%.
- `adr` (price/night): mean 101.83, std 50.54, 1 negative row (-6.38), 1,959 rows (1.6%) at 0, 1 extreme outlier at 5,400.
- 180 rows have `adults == children == babies == 0` — invalid zero-occupancy bookings.
- `country`: 177 unique codes, 40.9% Portugal (PRT) — strong domestic skew (hotel is in Portugal).
- `arrival_date_year`: 2015–2017. `reservation_status_date` range: 2014-10-17 to 2017-09-14 (~3 years), clear Jul/Aug seasonality.
- `reserved_room_type`: 10 categories (A dominant at 72.0%); 12.5% of bookings have `assigned_room_type != reserved_room_type`.
- `agent`: 333 unique values (13.7% missing). `company`: 352 unique values (94.3% missing).

## Recommendation framing
- **No real user/session actor exists.** The only usable "recommendation core" is a weak, aggregate proxy: `country` (guest origin, 177 values) × `reserved_room_type` (10 categories) — a population-preference matrix, not a personalized next-best-item task. No user history, no sequences, no cold-start structure (there is no actor to be cold on).
- Rich trip-level context (arrival date, lead_time, party size, meal, market_segment, distribution_channel, deposit_type, special requests) and a real completed-vs-cancelled-vs-no-show target, comparable in richness to Expedia — but the total absence of a guest/booking ID caps its usefulness for this project's core task far below Expedia or Akeed.
- Best used as: a context/cancellation-prediction reference or a coarse room-type popularity-by-origin check, not a core recommendation benchmark dataset.
