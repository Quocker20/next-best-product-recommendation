# Data Card — MovieLens (ml-latest-small)

- **Domain**: prototype
- **Source**: [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/) (open, no license gate)
- **Location**: `data/raw/prototype/movielens/ml-latest-small/`
- **EDA notebook**: `notebooks/prototype/movielens_eda.ipynb`

## Files (verified `ls -la`)
| file | size | rows |
|---|---|---|
| ratings.csv | 2,483,723 B | 100,836 |
| movies.csv | 494,431 B | 9,742 |
| tags.csv | 118,660 B | 3,683 |
| links.csv | 197,979 B | 9,742 |
| README.txt | 8,342 B | — |

## Schema (verified via full-file `dtypes`)
**ratings.csv**: `userId` int64, `movieId` int64, `rating` float64, `timestamp` int64
**movies.csv**: `movieId` int64, `title` str, `genres` str (pipe-separated)
**tags.csv**: `userId` int64, `movieId` int64, `tag` str, `timestamp` int64
**links.csv**: `movieId` int64, `imdbId` int64, `tmdbId` float64 (has NaNs, hence float not int)

## Verified stats (from executed notebook)
- 610 unique users, 9,724 rated movies.
- Rating scale 0.5–5 (half-star steps), no invalid values (`between(0.5,5).all() == True`).
- 0 full-row duplicates, 0 duplicate (userId, movieId) keys, 0 duplicate movieId in movies.csv.
- Referential integrity: 100% of `movieId` in ratings/tags exist in movies.csv.
- User-item sparsity: 98.30%. 4,744 items (48.8%) have ≤2 ratings (cold-start).
- `links.tmdbId` missing 0.08% (this is the only column with any missing values across all 4 files) — non-core, safe to ignore.

## Recommendation framing
- Classic user-item CF setup: `userId` x `movieId`, `rating` as candidate target (regression/ranking).
- Smallest, cleanest dataset pulled — used as baseline to prototype pipeline before scaling to bigger domains.
