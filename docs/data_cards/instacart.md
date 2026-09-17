# Data Card — Instacart Market Basket Analysis

- **Domain**: food
- **Source**: Kaggle mirror ([psparks/instacart-market-basket-analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)), originally The Instacart Online Grocery Shopping Dataset 2017
- **License**: **CC0: Public Domain**, stated on the Kaggle dataset page — verified live 2026-09-17. Cleanest license of any food dataset in this project (Akeed and Yelp are both restricted/research-only).
- **Location**: `data/raw/food/instacart/`
- **EDA notebook**: `notebooks/food/instacart_eda.ipynb`
- **Note**: `order_products__prior.csv` is 577.6MB / 32.4M rows — analyzed via chunked passes (`chunksize=2-3M`), not a full in-memory load.

## Files (verified `wc -l`)
| file | size | rows | cols |
|---|---|---|---|
| order_products__prior.csv | 577,550,706 B | 32,434,489 | 4 |
| orders.csv | 108,968,645 B | 3,421,083 | 7 |
| order_products__train.csv | 24,680,147 B | 1,384,617 | 4 |
| products.csv | 2,166,953 B | 49,688 | 4 |
| aisles.csv | 2,603 B | 134 | 2 |
| departments.csv | 270 B | 21 | 2 |

## Schema (verified via full-file `dtypes`)
**orders.csv**: `order_id` int64, `user_id` int64, `eval_set` str (`prior`/`train`/`test`), `order_number` int64, `order_dow` int64, `order_hour_of_day` int64, `days_since_prior_order` float64
**order_products__prior.csv / __train.csv**: `order_id` int64, `product_id` int64, `add_to_cart_order` int64, `reordered` int64 (0/1)
**products.csv**: `product_id` int64, `product_name` str, `aisle_id` int64, `department_id` int64
**aisles.csv**: `aisle_id` int64, `aisle` str (134 rows). **departments.csv**: `department_id` int64, `department` str (21 rows)

## Verified stats (from executed notebook, full chunked pass over 32.4M rows)
- `orders.csv`: 3,421,083 orders, 206,209 users. `eval_set`: prior 3,214,874 / train 131,209 / test 75,000. 0 full-row duplicates, 0 duplicate `order_id`.
- Missing: `days_since_prior_order` 6.03% — **100% structural**, exactly aligned with `order_number == 1` (a user's first order has no prior gap to measure).
- `days_since_prior_order` is capped/censored at 30: 369,323 rows (11.5% of non-null) sit exactly at 30 — a documented Instacart quirk (real gaps beyond 30 days are truncated, not measured exactly).
- **No cold-start users at all**: `order_number` per user has a hard floor of 4 — Instacart's own release already excludes users with fewer than 4 orders. Median 10 orders/user, mean 16.6, max 100. 88.4% of users have ≥5 total orders.
- No absolute calendar dates anywhere — only relative/cyclical fields (`order_dow`, `order_hour_of_day`, `days_since_prior_order`). Cannot compute a real seasonal span or align to other domains by calendar date, only by day-of-week/hour-of-day pattern.
- `order_products__prior.csv` (32,434,489 rows): **reorder rate 58.97%** — a ready-made label for exploitation (reorder) vs exploration (new product), no manual reframing needed. Items per order: median 8, mean 10.09, max 145.
- Item side: 49,688 products (0 missing, 0 dup), 134 aisles, 21 departments — includes one placeholder department "missing" (id 21, 1,258 products) and one placeholder aisle "missing" (id 100), i.e. genuinely unclassified products, not a data error.
- User × product interaction core: 206,209 users, 49,677 distinct products actually ordered, 13,307,953 unique (user, product) pairs, **sparsity 99.87%**. Item-side long tail is mild: 6.4% of products have ≤5 total orders across the whole dataset (far less extreme than Trivago's 57.6% or Akeed's near-total single-vendor concentration).

## Recommendation framing
- `user_id` × `product_id`, with `reordered` as a genuine positive/label signal — the closest fit in this whole project to RQ2's stated framing (exploitation-vs-exploration, reorder vs new item), stronger on this specific axis than Akeed (which requires deriving a repeat-vendor signal manually from `orders.csv` + duplicate-ID cleanup).
- Main gap vs Akeed for RQ2 fit: this is grocery, not restaurant/dish ordering — no "cuisine" or delivery-ETA context (both explicitly named in RQ2), and no absolute date means no calendar-season analysis and only pattern-based (not date-based) cross-domain simulator linking.
