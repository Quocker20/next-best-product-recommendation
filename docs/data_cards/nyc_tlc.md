# Data Card — NYC TLC Yellow Taxi (2024-01)

- **Domain**: ride
- **Source**: [nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (open, no login)
- **Location**: `data/raw/ride/nyc_tlc/2024-01/yellow_tripdata_2024-01.parquet`
- **EDA notebook**: `notebooks/ride/nyc_tlc_eda.ipynb`

## File (verified `ls -la`)
| file | size | rows | cols |
|---|---|---|---|
| yellow_tripdata_2024-01.parquet | 49,961,641 B | 2,964,624 | 19 |

## Schema (verified via parquet metadata, full file)
`VendorID` int32, `tpep_pickup_datetime` datetime64[us], `tpep_dropoff_datetime` datetime64[us], `passenger_count` float64, `trip_distance` float64, `RatecodeID` float64, `store_and_fwd_flag` str, `PULocationID` int32, `DOLocationID` int32, `payment_type` int64, `fare_amount` float64, `extra` float64, `mta_tax` float64, `tip_amount` float64, `tolls_amount` float64, `improvement_surcharge` float64, `total_amount` float64, `congestion_surcharge` float64, `Airport_fee` float64

## Verified stats (from executed notebook)
- 0 full-row duplicates.
- 56 rows with `tpep_dropoff_datetime < tpep_pickup_datetime` (bad rows).
- 260 unique `PULocationID`, 261 unique `DOLocationID`.
- Negative `fare_amount`: 37,448 rows. Negative `total_amount`: 35,504 rows.
- `trip_distance == 0`: 60,371 rows. `trip_distance > 100 mi`: 59 rows.
- `passenger_count == 0`: 31,465 rows.
- Pickup date range: 2002-12-31 22:59:39 to 2024-02-01 00:01:15 — 18 rows (0.00%) fall outside the stated Jan-2024 month, including one extreme outlier dated 2002-12-31.
- `VendorID` values observed: {1, 2, 6}. `payment_type` values observed: {0, 1, 2, 3, 4}. `RatecodeID` values observed: {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0}.

## No rider/driver ID
- Anonymized — no user identifier exists in this file. Rec-context reframed as pickup-zone → dropoff-zone demand.
- Zone-pair (PULocationID, DOLocationID) sparsity: 61.19%. 21 pickup zones (8.1%) have ≤5 trips.

## Recommendation framing
- Not a classic user-item CF fit — best used as zone-level demand signal (e.g. next-zone/destination prediction), not personalized recommendation.
