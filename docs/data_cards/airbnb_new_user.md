# Data Card — Airbnb New User Bookings

- **Domain**: hospitality
- **Source**: Kaggle competition ([airbnb-recruiting-new-user-bookings](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)) — required account rule-accept to pull
- **Location**: `data/raw/hospitality/airbnb_new_user/`
- **EDA notebook**: `notebooks/hospitality/airbnb_eda.ipynb`

## Files (verified `ls -la`)
| file | size | rows | cols |
|---|---|---|---|
| train_users_2.csv | 24,853,881 B | 213,451 | 16 |
| sessions.csv | 631,646,292 B | 10,567,737 | 6 |
| age_gender_bkts.csv | 11,905 B | 420 | 5 |
| countries.csv | 632 B | 10 | 7 |
| test_users.csv | 6,763,170 B | — | — |
| sample_submission_NDF.csv | 931,451 B | — | — |

## Schema (verified via full-file `dtypes`)
**train_users_2.csv**: `id` str, `date_account_created` str, `timestamp_first_active` int64, `date_first_booking` str, `gender` str, `age` float64, `signup_method` str, `signup_flow` int64, `language` str, `affiliate_channel` str, `affiliate_provider` str, `first_affiliate_tracked` str, `signup_app` str, `first_device_type` str, `first_browser` str, `country_destination` str (target)

**sessions.csv**: `user_id` str, `action` str, `action_type` str, `action_detail` str, `device_type` str, `secs_elapsed` float64

**age_gender_bkts.csv**: `age_bucket` str, `country_destination` str, `gender` str, `population_in_thousands` float64, `year` float64

**countries.csv**: `country_destination` str, `lat_destination` float64, `lng_destination` float64, `distance_km` float64, `destination_km2` float64, `destination_language` str, `language_levenshtein_distance` float64

## Verified stats (from executed notebook)
- Target `country_destination` distribution: NDF 58.35%, US 29.22%, other 4.73%, FR 2.35%, IT 1.33%, GB 1.09%, ES 1.05%, CA 0.67%, DE 0.50%, NL 0.36%, AU 0.25%, PT 0.10%.
- 0 full-row duplicates, 0 duplicate `id`.
- `age`: min 1.0, max 2014.0 (birth-year typo), mean 49.67, std 155.67 (n=125,461 non-null). 2,345 rows >100, 779 rows >1000, 57 rows <14.
- Missing %: `date_first_booking` 58.3%, `age` 41.2%, `first_affiliate_tracked` 2.8%, all other columns 0%.
- `date_first_booking` missing exactly 100.00% when `country_destination == NDF`, exactly 0.00% otherwise — fully structural.
- `account_created` date range: 2010-01-01 to 2014-06-30. `first_active` range: 2009-03-19 04:32:55 to 2014-06-30 23:58:24.
- `sessions.csv` covers 135,483 / 213,451 train users (63.5%). 47.60% of session rows have a `user_id` not present in train_users_2.csv.
- Top session actions: show (2,768,278), index (843,699), search_results (725,226), personalize (706,824), search (536,057).
- Among 88,908 users who did book: 11 distinct destinations (all except NDF).

## Recommendation framing
- `country_destination` prediction IS a next-best-item (destination) recommendation task — strong direct fit, closest conceptually to Akeed among all datasets pulled.
