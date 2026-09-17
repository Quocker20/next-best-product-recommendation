# Data Card — Yelp Open Dataset

- **Domain**: food
- **Source**: Kaggle mirror ([yelp-dataset/yelp-dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset), published by Yelp, Inc. itself), official at [yelp.com/dataset/download](https://www.yelp.com/dataset/download)
- **License**: Governed by Yelp's Dataset License / Terms of Use (`Dataset_User_Agreement.pdf`, included in the download). Research/personal use; no confirmed open-redistribution right.
- **Location**: `data/raw/food/yelp/`
- **EDA notebook**: `notebooks/food/yelp_eda.ipynb`
- **Note**: `review.json` (5.3GB, 6.99M rows) and `user.json` (3.3GB, 1.99M rows) exceed available RAM (~3GB free at analysis time), analyzed via chunked streaming passes (`chunksize=500_000`), text-heavy columns (`text`, `friends`) dropped immediately. `business.json`, `tip.json`, `checkin.json` loaded fully.

## Files (verified `ls -la` + `wc -l`)
| file | size | rows |
|---|---|---|
| yelp_academic_dataset_business.json | 118,863,795 B | 150,346 |
| yelp_academic_dataset_checkin.json | 286,958,945 B | 131,930 |
| yelp_academic_dataset_review.json | 5,341,868,833 B | 6,990,280 |
| yelp_academic_dataset_tip.json | 180,604,475 B | 908,915 |
| yelp_academic_dataset_user.json | 3,363,329,011 B | 1,987,897 |

## Schema (verified via first-line JSON parse + full-file dtypes)
**business.json**: `business_id` str, `name` str, `address` str, `city` str, `state` str, `postal_code` str, `latitude` float64, `longitude` float64, `stars` float64, `review_count` int64, `is_open` int64, `attributes` dict/null, `categories` str (comma-separated), `hours` dict/null

**review.json**: `review_id` str, `user_id` str, `business_id` str, `stars` int64 (1–5), `useful` int64, `funny` int64, `cool` int64, `text` str, `date` str (parses to datetime64[us])

**user.json**: `user_id` str, `name` str, `review_count` int64, `yelping_since` str, `useful`/`funny`/`cool` int64, `elite` str (comma-separated years), `friends` str (comma-separated user ids), `fans` int64, `average_stars` float64, 11x `compliment_*` int64 columns

**tip.json**: `user_id` str, `business_id` str, `text` str, `date` str, `compliment_count` int64

**checkin.json**: `business_id` str, `date` str (comma-separated list of ALL check-in timestamps for that business, not one row per check-in)

## Verified stats (from executed notebook, full data — chunked passes cover every row)
- Missing %: business `hours` 15.45%, `attributes` 9.14%, `categories` 0.07% (structural — business didn't report); all other columns across all 5 files are 0% missing.
- Duplicates: 0 everywhere — 0 full-row dup in business.json, 0 duplicate `business_id`, 0 approx-duplicate `review_id`, 0 approx-duplicate `user_id`.
- `business.stars`/`review.stars` fully clean (1–5 scale, no invalid values). 0 `review.json` rows reference a `business_id` missing from business.json.
- All business locations fall inside the real US/Canada bbox (0.00% outliers) — genuine GPS. States observed: AB (Canada), AZ, CA, CO, DE, FL, HI, ID, IL, IN, LA, MA, MI, MO, MT, NC, NJ, NV, PA, SD, TN, TX, UT, VI, VT, WA, XMS.
- **`business.review_count` minimum is 5** (mean 44.9, max 7,568) — Yelp's public release already filters out businesses with very few reviews. Result: **0.0% of businesses have ≤2 reviews** — effectively no business-side cold-start, unlike every other dataset in this project.
- **ID mismatch**: review.json has 1,987,929 distinct `user_id` values but user.json has only 1,987,897 rows — 32 more users appear in the review log than in the user table. Check before joining.
- Date ranges: review.date spans 2005-02-16 to 2022-01-19; user.yelping_since spans 2004-10-12 to 2022-01-19 (consistent with each other).
- Top business categories: Restaurants (52,268), Food (27,781), Shopping (24,395), Home Services (14,356), Beauty & Spas (14,292).

## Recommendation framing
- Core interaction: `user_id` x `business_id` from review.json — 1,987,929 users, 150,346 businesses, 6,745,760 unique pairs, sparsity 99.9977%. 1,136,008 users (57.1%) have only 1 review ever — heavy user-side cold-start (similar magnitude to Trivago), but no business-side cold-start (curated minimum of 5 reviews per business).
- No single labeled target — `review.stars` as a rating target, or user x business as implicit interaction, same reconstructed-from-interaction pattern as Akeed/Airbnb/Trivago.
- Biggest dataset by uncompressed size in this project (9.3GB across 5 files), but smaller in row count (6.99M reviews) than Expedia (37.7M) or Trivago (15.9M).
