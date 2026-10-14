# Report numbers: user-level test, seed 2024, reranker m2_merge

Single version of every number for the report. Expedia, full real data, date split (valid 2014-09-01, test 2014-11-01 to 2015-01-01). Each test user has one fixed top-20 from two prediction steps (`docs/eval_protocol.md`); Recall@K and NDCG@K are set-based per user, then averaged over users. SMLP4Rec rows use seed 2024 only; ItemKNN and popularity are deterministic. No seed averages, no ±.

Test users 222,428 (warm 150,711, cold 71,717); 147,471 (66.3%) have one test booking; 1.52 distinct clusters per user on average.

## Test metrics, all users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1519 | 0.1920 | 0.2385 | 0.2080 | 0.3224 | 0.4900 |
| + query embedding | 0.3783 | 0.4426 | 0.4893 | 0.5113 | 0.6952 | 0.8590 |
| + query embedding + sameDest | 0.3840 | 0.4474 | 0.4936 | 0.5154 | 0.6968 | 0.8588 |
| ItemKNN | 0.1462 | 0.1852 | 0.2330 | 0.1984 | 0.3096 | 0.4815 |
| Global popularity | 0.1020 | 0.1356 | 0.1774 | 0.1459 | 0.2425 | 0.3939 |

## Test metrics, warm users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1833 | 0.2286 | 0.2789 | 0.2470 | 0.3757 | 0.5544 |
| + query embedding | 0.3923 | 0.4566 | 0.5026 | 0.5230 | 0.7047 | 0.8640 |
| + query embedding + sameDest | 0.4007 | 0.4635 | 0.5089 | 0.5288 | 0.7067 | 0.8637 |
| ItemKNN | 0.1777 | 0.2226 | 0.2731 | 0.2387 | 0.3662 | 0.5453 |
| Global popularity | 0.1082 | 0.1427 | 0.1866 | 0.1524 | 0.2504 | 0.4077 |

## Test metrics, cold users

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.0859 | 0.1151 | 0.1537 | 0.1259 | 0.2104 | 0.3549 |
| + query embedding | 0.3487 | 0.4134 | 0.4612 | 0.4868 | 0.6751 | 0.8485 |
| + query embedding + sameDest | 0.3490 | 0.4137 | 0.4614 | 0.4873 | 0.6758 | 0.8485 |
| ItemKNN | 0.0801 | 0.1066 | 0.1489 | 0.1138 | 0.1908 | 0.3474 |
| Global popularity | 0.0888 | 0.1208 | 0.1579 | 0.1323 | 0.2259 | 0.3649 |

## sameDest gain (main minus + query embedding), paired bootstrap, 95 % CI

| User set | users | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| all | 222,428 | +0.0057 [+0.0053, +0.0061] | +0.0048 [+0.0045, +0.0052] | +0.0043 [+0.0040, +0.0046] | +0.0041 [+0.0036, +0.0046] | +0.0016 [+0.0012, +0.0020] | -0.0002 [-0.0005, +0.0000] |
| warm | 150,711 | +0.0083 [+0.0078, +0.0089] | +0.0069 [+0.0065, +0.0074] | +0.0063 [+0.0058, +0.0067] | +0.0058 [+0.0052, +0.0064] | +0.0020 [+0.0015, +0.0025] | -0.0003 [-0.0006, +0.0000] |
| cold | 71,717 | +0.0003 [+0.0000, +0.0005] | +0.0004 [+0.0002, +0.0005] | +0.0002 [+0.0001, +0.0003] | +0.0005 [-0.0001, +0.0010] | +0.0007 [+0.0002, +0.0012] | +0.0001 [-0.0003, +0.0004] |
| warm, destination known | 28,200 | +0.0405 [+0.0377, +0.0434] | +0.0347 [+0.0323, +0.0371] | +0.0315 [+0.0293, +0.0338] | +0.0279 [+0.0253, +0.0307] | +0.0120 [+0.0101, +0.0140] | +0.0011 [-0.0001, +0.0023] |
| warm, destination new | 122,511 | +0.0009 [+0.0006, +0.0012] | +0.0005 [+0.0003, +0.0008] | +0.0004 [+0.0002, +0.0007] | +0.0007 [+0.0003, +0.0012] | -0.0003 [-0.0007, +0.0001] | -0.0006 [-0.0009, -0.0003] |

## Query embedding gain (+ query embedding minus plain), paired bootstrap, 95 % CI

| User set | users | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|---|
| all | 222,428 | +0.2264 [+0.2246, +0.2280] | +0.2506 [+0.2491, +0.2521] | +0.2507 [+0.2493, +0.2521] | +0.3034 [+0.3011, +0.3054] | +0.3727 [+0.3706, +0.3749] | +0.3689 [+0.3669, +0.3709] |
| warm | 150,711 | +0.2091 [+0.2071, +0.2107] | +0.2279 [+0.2261, +0.2295] | +0.2237 [+0.2220, +0.2252] | +0.2760 [+0.2733, +0.2783] | +0.3290 [+0.3262, +0.3314] | +0.3096 [+0.3073, +0.3118] |
| cold | 71,717 | +0.2628 [+0.2599, +0.2658] | +0.2982 [+0.2955, +0.3011] | +0.3075 [+0.3050, +0.3103] | +0.3609 [+0.3570, +0.3647] | +0.4646 [+0.4606, +0.4686] | +0.4936 [+0.4896, +0.4974] |
| warm, destination known | 28,200 | +0.1017 [+0.0976, +0.1060] | +0.1125 [+0.1091, +0.1162] | +0.1086 [+0.1054, +0.1120] | +0.1374 [+0.1320, +0.1428] | +0.1670 [+0.1621, +0.1721] | +0.1502 [+0.1459, +0.1545] |
| warm, destination new | 122,511 | +0.2338 [+0.2317, +0.2358] | +0.2545 [+0.2527, +0.2565] | +0.2502 [+0.2484, +0.2520] | +0.3079 [+0.3050, +0.3107] | +0.3663 [+0.3634, +0.3693] | +0.3463 [+0.3433, +0.3491] |

Bootstrap: 1000 draws, users resampled, same draw for both models. A difference is significant when its interval excludes 0.

## Superseded (do not quote in the report)

- Row-level test tables: `reported_tables.md`, `final_tables.md`, `final_benchmark.csv`, `protocol_gap.csv`, `bootstrap_ci.csv`, `results_slices.json`, `slices_*.csv`, `reports/benchmark_results.csv` (autoregressive / teacher-forced, per booking; still valid for the valid window and for history in notebooks 01-05).
- `user_top20_reranker_choice.md`: the three rerankers side by side, the record behind the choice of m2_merge.
- Mean ± std over 3 seeds (`final_tables.md`) and `plan_oct_12_checkpoints.md`: historical.
- Slides `reports/summary/week6_retrain/slides_week6_vi.html` still show the row-level numbers.
