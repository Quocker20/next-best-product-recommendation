# Data Card — Expedia Hotel Recommendations

- **Domain**: hospitality
- **Source**: Kaggle competition ([expedia-hotel-recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations)) — required account rule-accept to pull
- **Location**: `data/raw/hospitality/expedia/`
- **EDA notebook**: `notebooks/hospitality/expedia_eda.ipynb`
- **Note**: `train.csv` is 4.07GB / 37.7M rows — biggest dataset pulled, exceeds available RAM (~3GB free at analysis time), analyzed via chunked streaming pass (`chunksize=2_000_000`, dtype-optimized). All stats below come from that full pass over every row, not a sample.

## Files (verified `ls -la` + `wc -l`)
| file | size | rows | cols |
|---|---|---|---|
| train.csv | 4,070,445,781 B | 37,670,293 | 24 |
| destinations.csv | 138,159,416 B | 62,106 | 150 |
| test.csv | 276,554,476 B | — | — |
| sample_submission.csv | 31,756,066 B | — | — |

## Schema (verified via 100,000-row preview read, no mixed-type risk on these columns)
`date_time` str, `site_name` int64, `posa_continent` int64, `user_location_country` int64, `user_location_region` int64, `user_location_city` int64, `orig_destination_distance` float64, `user_id` int64, `is_mobile` int64, `is_package` int64, `channel` int64, `srch_ci` str, `srch_co` str, `srch_adults_cnt` int64, `srch_children_cnt` int64, `srch_rm_cnt` int64, `srch_destination_id` int64, `srch_destination_type_id` int64, `is_booking` int64, `cnt` int64, `hotel_continent` int64, `hotel_country` int64, `hotel_market` int64, `hotel_cluster` int64 (target, 0–99)

`destinations.csv`: `srch_destination_id` int64 + `d1`...`d149` float64 (149 latent dims from hotel-review text)

## Verified stats (full chunked pass, 37,670,293 rows)
- 1,198,786 unique users. 45 unique `site_name` values.
- Missing %: `orig_destination_distance` 35.90%, `srch_ci` 0.12%, `srch_co` 0.12%, all other 21 columns 0.00%.
- Approx (hash-based) duplicate (user_id, date_time, srch_destination_id, hotel_cluster) keys: 3,165 (0.0084%).
- `is_booking` split: 0 (click) = 34,669,600 rows (92.03%), 1 (booking) = 3,000,693 rows (7.97%).
- `srch_adults_cnt == 0`: 70,979 rows (0.19%). `srch_co < srch_ci` (invalid date order): 798 rows (0.0021%).
- `orig_destination_distance`: min 0.0, max 12,407.90, mean 1,970.1 (n=24,145,292 non-null).
- `hotel_cluster` distribution across all 100 classes: most popular cluster 2.77% share, least popular 0.13% share.
- `date_time` range: 2013-01-07 00:00:02 to 2014-12-31 23:59:59.

## Recommendation framing
- Recommendation core (bookings only, `is_booking==1`): 813,985 users, 100/100 clusters represented, 2,360,713 unique (user, cluster) pairs, sparsity 97.10%. 315,679 users (38.8%) have only 1 booking.
- `hotel_cluster` prediction from user + search context — classic multi-class rec task, but click-vs-book distinction must be resolved first (don't feed raw click rows as positive interactions).
