# Reranker selection record, seed 2024, all 222,428 test users

Chosen: `m2_merge`. The other two rerankers are kept here only as the comparison behind the choice.

## m1_first

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1530 | 0.1937 | 0.2399 | 0.2106 | 0.3268 | 0.4935 |
| + query embedding | 0.3640 | 0.4254 | 0.4704 | 0.4948 | 0.6712 | 0.8304 |
| + query embedding + sameDest | 0.3692 | 0.4298 | 0.4746 | 0.4982 | 0.6727 | 0.8308 |
| ItemKNN | 0.1491 | 0.1900 | 0.2358 | 0.2045 | 0.3219 | 0.4867 |
| Global popularity | 0.1020 | 0.1356 | 0.1774 | 0.1459 | 0.2425 | 0.3939 |

## m2_merge

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1519 | 0.1920 | 0.2385 | 0.2080 | 0.3224 | 0.4900 |
| + query embedding | 0.3783 | 0.4426 | 0.4893 | 0.5113 | 0.6952 | 0.8590 |
| + query embedding + sameDest | 0.3840 | 0.4474 | 0.4936 | 0.5154 | 0.6968 | 0.8588 |
| ItemKNN | 0.1462 | 0.1852 | 0.2330 | 0.1984 | 0.3096 | 0.4815 |
| Global popularity | 0.1020 | 0.1356 | 0.1774 | 0.1459 | 0.2425 | 0.3939 |

## m3_sqrt_alpha

| Model | ndcg@5 | ndcg@10 | ndcg@20 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|---|---|---|
| SMLP4Rec (plain) | 0.1502 | 0.1884 | 0.2359 | 0.2049 | 0.3136 | 0.4845 |
| + query embedding | 0.3758 | 0.4395 | 0.4860 | 0.5069 | 0.6890 | 0.8525 |
| + query embedding + sameDest | 0.3831 | 0.4459 | 0.4914 | 0.5140 | 0.6936 | 0.8537 |
| ItemKNN | 0.1462 | 0.1851 | 0.2327 | 0.1982 | 0.3094 | 0.4805 |
| Global popularity | 0.1020 | 0.1356 | 0.1774 | 0.1459 | 0.2425 | 0.3939 |
