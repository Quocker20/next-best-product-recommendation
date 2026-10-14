# User-level test metrics: fixed top-20 per user (two steps + rerank `m2_merge`), seed 2024

Expedia, full test window (2014-11-01 to 2014-12-31), mean over test users (set-based Recall@k / NDCG@k, `docs/eval_protocol.md`). SMLP4Rec rows: seed 2024; ItemKNN and popularity are deterministic.

Users: users 222,428, warm_users 150,711, cold_users 71,717, users_with_one_test_booking 147,471, mean_distinct_clusters_per_user 1.523, share_users_with_more_than_20_distinct 0.000

## all users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1519 | 0.1920 | 0.2385 | 0.2080 | 0.3224 | 0.4900 |
| + query embedding | 0.3783 | 0.4426 | 0.4893 | 0.5113 | 0.6952 | 0.8590 |
| + query embedding + sameDest | 0.3840 | 0.4474 | 0.4936 | 0.5154 | 0.6968 | 0.8588 |
| ItemKNN | 0.1462 | 0.1852 | 0.2330 | 0.1984 | 0.3096 | 0.4815 |
| Global popularity | 0.1020 | 0.1356 | 0.1774 | 0.1459 | 0.2425 | 0.3939 |

## warm users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1833 | 0.2286 | 0.2789 | 0.2470 | 0.3757 | 0.5544 |
| + query embedding | 0.3923 | 0.4566 | 0.5026 | 0.5230 | 0.7047 | 0.8640 |
| + query embedding + sameDest | 0.4007 | 0.4635 | 0.5089 | 0.5288 | 0.7067 | 0.8637 |
| ItemKNN | 0.1777 | 0.2226 | 0.2731 | 0.2387 | 0.3662 | 0.5453 |
| Global popularity | 0.1082 | 0.1427 | 0.1866 | 0.1524 | 0.2504 | 0.4077 |

## cold users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.0859 | 0.1151 | 0.1537 | 0.1259 | 0.2104 | 0.3549 |
| + query embedding | 0.3487 | 0.4134 | 0.4612 | 0.4868 | 0.6751 | 0.8485 |
| + query embedding + sameDest | 0.3490 | 0.4137 | 0.4614 | 0.4873 | 0.6758 | 0.8485 |
| ItemKNN | 0.0801 | 0.1066 | 0.1489 | 0.1138 | 0.1908 | 0.3474 |
| Global popularity | 0.0888 | 0.1208 | 0.1579 | 0.1323 | 0.2259 | 0.3649 |
