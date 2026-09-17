# Data Card — Citi Bike Trip Data (2024-01)

- **Domain**: ride
- **Source**: [citibikenyc.com/system-data](https://citibikenyc.com/system-data) (direct S3 download, `s3.amazonaws.com/tripdata`), governed by the NYCBS Data Use Policy
- **License**: **Open / public**, per the NYCBS Data Use Policy — "we invite developers, engineers, statisticians, artists, academics and other interested members of the public to use the data" — verified live 2026-09-17. No account, no ToS click-through required, direct monthly CSV/ZIP download.
- **Location**: `data/raw/ride/citibike/202401/`
- **EDA notebook**: `notebooks/ride/citibike_eda.ipynb`
- **Note**: publisher splits any month over 1M rows into multiple CSVs (`_1.csv`, `_2.csv` here); both loaded and concatenated for this EDA (1.89M rows total, fits fully in memory).

## Files (verified `wc -l`)
| file | size | rows |
|---|---|---|
| 202401-citibike-tripdata_1.csv | 195,372,814 B | 1,000,000 |
| 202401-citibike-tripdata_2.csv | 173,662,194 B | 888,085 |

## Schema (verified via full-file `dtypes`, 1,888,085 rows loaded)
`ride_id` str, `rideable_type` str (`electric_bike`/`classic_bike`), `started_at` str (parses to datetime, sub-second precision), `ended_at` str, `start_station_name` str, `start_station_id` str, `end_station_name` str, `end_station_id` str, `start_lat` float64, `start_lng` float64, `end_lat` float64, `end_lng` float64, `member_casual` str (`member`/`casual`)

## Verified stats (from executed notebook, full 1,888,085-row load)
- **No rider identifier anywhere in this file** — `member_casual` is only a 2-value category (member 89.0% / casual 11.0%), not an identity. No way to build a per-rider history or sequence.
- 0 full-row duplicates, 0 duplicate `ride_id`.
- Missing %: `end_station_*`/`end_lat`/`end_lng` 0.29%, `start_station_*`/`start_lat`/`start_lng` 0.06% — both are dockless start/end trips (real operational case, e-bikes can be locked outside a formal dock), not data errors.
- `rideable_type`: electric_bike 64.3%, classic_bike 35.7%.
- `duration_sec` (derived `ended_at - started_at`): min exactly 60.0s (confirms the publisher's own documented <60s filter is already applied), median 464.5s, mean 690.9s, max 90,030s (25h). 2,422 trips (0.13%) exceed 3 hours — outliers to flag.
- **100% of GPS coordinates fall inside the real NYC bounding box** (lat 40.4–41.0, lon -74.3 to -73.6) — genuine coordinates, unlike Akeed's synthetic ones.
- `started_at` range: 2023-12-31 to 2024-01-31 (one pulled month, same scope limitation as this project's NYC TLC pull).
- Station-pair core: 2,223 unique start stations, 2,185 unique end stations, 381,346 unique (start, end) pairs, sparsity 92.15%. Station-side cold-start is minimal — only 0.6% of stations have ≤5 trips (median 381 trips/station) — far denser than Porto's 63 taxi stands or NYC TLC's 260 zones.

## Recommendation framing
- Same reframing needed as Porto Taxi / NYC TLC: station-to-station demand, not personalized recommendation, since there is no rider identity to build history/sequences on. Destination prediction is explicitly not in the mentor's phase plan (CLAUDE.md §12) — same caveat as Porto.
- Best used as: a ride-domain feasibility/context reference alongside Porto and NYC TLC. Real coordinates and richer per-trip context (bike type, member type, precise duration/time) make it a stronger *context* dataset than NYC TLC, but the total lack of rider identity means it cannot anchor a personalized ride-recommendation benchmark on its own.
