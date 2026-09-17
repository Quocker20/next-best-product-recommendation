# Data Card — Porto Taxi Trajectory (ECML/PKDD 2015)

- **Domain**: ride
- **Source**: [UCI archive](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015), originally Kaggle ECML/PKDD 2015 challenge
- **Location**: `data/raw/ride/porto_taxi/`
- **EDA notebook**: `notebooks/ride/porto_taxi_eda.ipynb`

## Files (verified `ls -la`)
| file | size | rows | cols |
|---|---|---|---|
| train.csv | 1,942,848,724 B | 1,710,670 | 9 |
| Porto_taxi_data_test_partial_trajectories.csv | 447,386 B | — | — |
| solution_challengeII.csv | 3,898 B | — | — |
| solution_fixed.csv | 8,499 B | — | — |

## Schema (verified via full-file `dtypes`, 1,710,670 rows loaded)
`TRIP_ID` int64, `CALL_TYPE` str, `ORIGIN_CALL` float64, `ORIGIN_STAND` float64, `TAXI_ID` int64, `TIMESTAMP` int64 (unix epoch seconds), `DAY_TYPE` str, `MISSING_DATA` bool, `POLYLINE` str (JSON-array-like string of `[lon, lat]` pairs, one per 15s GPS sample)

Derived in notebook: `n_points` (= count of `[` in POLYLINE − 1), `trip_duration_sec` (= max(n_points−1, 0) × 15).

## Verified stats (from executed notebook)
- 448 unique `TAXI_ID`. 3 full-row duplicates, **81 duplicate `TRIP_ID`**.
- Missing %: `ORIGIN_CALL` 78.7%, `ORIGIN_STAND` 52.9%, all other columns 0%.
- `ORIGIN_CALL` missing perfectly aligned with `CALL_TYPE != 'A'` (0 mismatches either direction).
- `ORIGIN_STAND`: **11,302 rows with `CALL_TYPE == 'B'` still missing `ORIGIN_STAND`** — not perfectly structural.
- `MISSING_DATA == True`: only 10 rows. But empty `POLYLINE` (n_points==0): 5,901 rows. Zero-duration trips not flagged: 36,508 rows — the flag under-reports real gaps by ~590x.
- `trip_duration_sec` (n=1,710,670): mean 716.4s, median 600s, max 58,200s (16.2h). `n_points` mean 48.76, max 3,881.
- Pickup range: 2013-07-01 00:00:53 to 2014-06-30 23:59:56 (full year).
- `DAY_TYPE`: only 1 unique value (`'A'`) across all rows — zero information.
- Pickup GPS points outside real Porto bbox (lon -8.75 to -8.55, lat 41.05 to 41.25): 5,265 / 1,710,670 (0.31%) — genuine coordinates, not synthetic.

## Recommendation framing (reframed, no rider ID)
- This is driver/dispatch data, no passenger identifier. Rec-context reframed as `TAXI_ID` x `ORIGIN_STAND` (`CALL_TYPE == 'B'` rows only): 439 taxis, 63 stands, 20,811 pairs, sparsity 24.75%. 0 stands with ≤5 trips (0.0% cold).
- No given target; destination and trip duration are both derived from `POLYLINE` itself, need feature engineering.
