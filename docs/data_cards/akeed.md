# Data Card — Akeed Restaurant Recommendation Challenge

- **Domain**: food
- **Source**: Kaggle mirror ([darisdzakwanhoesien2/akeed-restaurant-recommendation-challenge](https://www.kaggle.com/datasets/darisdzakwanhoesien2/akeed-restaurant-recommendation-challenge)), original on Zindi
- **License**: **Restricted / unverified.** Originally released under Zindi's competition-specific Competition Rules (per [zindi.africa/terms](https://zindi.africa/terms), each dataset's use is bound by "the relevant Competition Rules" layered on Zindi's general Terms of Use — no blanket open license). The original Akeed competition page is no longer live on Zindi to re-verify exact terms; the Kaggle mirror does not state a license either. Treat as research/non-commercial only pending re-confirmation from Zindi.
- **Location**: `data/raw/food/akeed/`
- **EDA notebook**: `notebooks/food/akeed_eda.ipynb`

## Files (verified `ls -la`)
| file | size | rows | cols |
|---|---|---|---|
| orders.csv | 22,861,045 B | 135,303 | 26 |
| train_customers.csv | 2,055,222 B | 34,674 | 8 |
| train_locations.csv | 3,186,518 B | 59,503 | 5 |
| vendors.csv | 46,826 B | 100 | 59 |
| test_customers.csv | 578,908 B | — | — |
| test_locations.csv | 895,487 B | — | — |
| SampleSubmission.csv | 33,039,050 B | — | — |

## Schema (verified via full-file `dtypes`)
**orders.csv**: `akeed_order_id` float64, `customer_id` str, `item_count` float64, `grand_total` float64, `payment_mode` int64, `promo_code` str, `vendor_discount_amount` float64, `promo_code_discount_percentage` float64, `is_favorite` str, `is_rated` str, `vendor_rating` float64, `driver_rating` float64, `deliverydistance` float64, `preparationtime` float64, `delivery_time` str, `order_accepted_time` str, `driver_accepted_time` str, `ready_for_pickup_time` str, `picked_up_time` str, `delivered_time` str, `delivery_date` str, `vendor_id` int64, `created_at` str, `LOCATION_NUMBER` int64, `LOCATION_TYPE` str, `CID X LOC_NUM X VENDOR` str

**train_customers.csv**: `akeed_customer_id` str, `gender` str, `dob` float64, `status` int64, `verified` int64, `language` str, `created_at` str, `updated_at` str

**train_locations.csv**: `customer_id` str, `location_number` int64, `location_type` str, `latitude` float64, `longitude` float64

**vendors.csv** (59 cols, key ones): `id` int64, `latitude` float64, `longitude` float64, `vendor_category_en` str, `delivery_charge` float64, `serving_distance` float64, `vendor_rating` float64, `prepration_time` int64, `commission` float64, `rank` int64, `vendor_tag` str, `country_id` float64, `city_id` float64 — plus 7-day opening-time columns (`<day>_from_time1/2`, `<day>_to_time1/2`).

## Verified stats (from executed notebook)
- customer x vendor: 27,445 customers, 100 vendors, 71,484 unique (customer, vendor) pairs, sparsity 97.40%.
- Vendor cold-start: 0 vendors with ≤5 orders (0.0%) — small vendor pool, all well-used. Customer cold-start: 9,919 customers (36.1%) with only 1 order.
- Duplicates: 0 full-row dup, **81 duplicate `akeed_order_id`**, **151 duplicate `akeed_customer_id`** in customers table.
- `deliverydistance <= 0`: 55,613 rows (41.1% of orders) — not a minor edge case.
- `grand_total <= 0`: 683 rows.
- **Location coordinates: 100% (59,503/59,503) fall outside the real Oman bounding box** (lat 16–27, lon 51–60) — confirmed synthetic/jittered GPS, not real, e.g. sampled point lat≈1.68 lon≈-78.79.
- **Vendor coordinates are also synthetic**: 0/100 vendors fall inside the real Oman bbox (lat range observed: -1.79 to 205.24 — the max is not a valid latitude at all). `vendors.csv` has a single constant `city_id=1.0`/`country_id=1.0` for all 100 rows, with no lookup file to resolve it to a real place name. **Weather join tested infeasible for Akeed**: neither coordinates nor city give a real, resolvable location (tested 2026-09-17 — see `docs/dataset_rubric.md` weather-join note).
- `created_at` range: 2019-05-10 20:59:58 to 2020-02-29 23:52:14.
- 3,276 `customer_id` values in orders.csv not present in train_customers.csv (test-set customers, train/test boundary).
- 0 `vendor_id` values in orders.csv missing from vendors.csv.

## Recommendation framing
- No labeled target column — task is reconstructing customer x vendor interaction (matches original: predict LOCATION x VENDOR pair likelihood).
- Most "native" rec dataset pulled — direct match for this project's goal, but needs real cleanup (dup IDs, distance≤0 rows, synthetic coords) before modeling.
