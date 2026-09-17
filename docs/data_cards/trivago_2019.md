# Data Card — Trivago RecSys Challenge 2019

- **Domain**: hospitality
- **Source**: Kaggle mirror ([phhasian0710/trivago-recsys](https://www.kaggle.com/datasets/phhasian0710/trivago-recsys)), original at recsys.trivago.cloud
- **Location**: `data/raw/hospitality/trivago_2019/`
- **EDA notebook**: `notebooks/hospitality/trivago_eda.ipynb`
- **Note**: `train.csv` is 2.1GB / 15.9M rows — exceeds available RAM (~3GB free at analysis time), analyzed via chunked streaming pass (`chunksize=1_000_000`), not a full in-memory load. All stats below come from that full pass over every row, not a sample.

## Files (verified `ls -la` + `wc -l`)
| file | size | rows | cols |
|---|---|---|---|
| train.csv | 2,100,807,496 B | 15,932,992 | 12 |
| item_metadata.csv | 257,901,183 B | 927,142 | 2 |
| test.csv | 534,912,813 B | — | — |
| submission_popular.csv | 53,035,911 B | — | — |

## Schema (verified via 100,000-row preview read + full-pass column presence)
**train.csv**: `user_id` str, `session_id` str, `timestamp` int64 (unix epoch sec), `step` int64, `action_type` str, `reference` str, `platform` str, `city` str, `device` str, `current_filters` str, `impressions` str (pipe-separated item ids), `prices` str (pipe-separated)
**item_metadata.csv**: `item_id` int, `properties` str (pipe-separated tags)

## Verified stats (full chunked pass, 15,932,992 rows)
- 730,803 unique users, 910,683 unique sessions, 34,752 unique cities.
- Timestamp range: 2018-11-01 00:00:08 to 2018-11-06 23:59:59 — only 6 days, a snapshot not a full history.
- `action_type` counts: interaction item image 11,860,750, clickout item 1,586,586, filter selection 695,917, search for destination 403,066, change of sort order 400,584, interaction item info 285,402, interaction item rating 217,246, interaction item deals 193,794, search for item 152,203, search for poi 137,444.
- Missing %: all 9 measured columns (user_id, session_id, timestamp, step, action_type, reference, platform, city, device) are 0.0% missing. `current_filters`/`impressions`/`prices` excluded from this pass (not measured — memory budget), known sparse-by-design (populated only on clickout rows).
- Approx (hash-based) duplicate (user_id, session_id, step) keys: 657 (0.0041%).
- Clickouts per user: count 717,774, mean 2.21, std 2.77, min 1, median 1, max 284. Only 2 users exceed 200 clickouts.
- `reference` on `clickout item` rows: 0 non-numeric values (100% are item ids in that context).
- 204 / 289,506 clicked items (0.07%) not found in item_metadata.csv.

## Recommendation framing
- Recommendation core (clickout only): 717,774 users, 289,506 items, 1,306,901 unique pairs, sparsity 99.9994% — largest/sparsest dataset in this project. 166,857 items (57.6%) and 405,038 users (56.4%) are single-interaction — heaviest cold-start of all datasets analyzed.
- No train-side label — target reconstructed from held-out `clickout item` rows in `test.csv` (item ranking task), same pattern as Akeed/Airbnb.
