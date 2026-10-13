"""Report helpers and the cluster bootstrap against hand-computed values."""

import numpy as np
import pytest

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.report import h_bucket, h_slices, metric_columns, metrics, row_set_masks


def test_column_order_ndcg_first():
    assert metric_columns((5, 10)) == ["ndcg@5", "ndcg@10", "recall@5", "recall@10"]


def test_metrics_by_hand():
    rank = np.array([1, 3, 6])
    m = metrics(rank, ks=(5,))
    assert m["n"] == 3
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["ndcg@5"] == pytest.approx((1 + 1 / np.log2(4)) / 3)  # 1/log2(2) = 1, 1/log2(4) = 0.5


def test_row_sets_and_h_buckets():
    m = row_set_masks(np.array([0, 1, 4]))
    assert m["warm"].tolist() == [False, True, True] and m["cold"].tolist() == [True, False, False]
    assert h_bucket(np.array([0, 1, 2, 3, 5, 6, 40])).tolist() == [
        "0",
        "1",
        "2",
        "3-5",
        "3-5",
        "6+",
        "6+",
    ]
    s = h_slices(np.array([1, 1, 9]), np.array([0, 0, 7]), ks=(5,))
    assert s["0"]["recall@5"] == 1.0 and s["6+"]["recall@5"] == 0.0 and s["1"]["n"] == 0


def test_cluster_bootstrap_with_one_row_per_group_equals_row_bootstrap():
    rng = np.random.default_rng(1)
    a, b = rng.integers(1, 30, 200), rng.integers(1, 30, 200)
    row = paired_bootstrap(a, b, k=10, n_boot=200, seed=3)
    grp = paired_bootstrap(a, b, k=10, n_boot=200, seed=3, groups=np.arange(200))
    assert row == grp


def test_cluster_bootstrap_diff_is_row_mean_and_ci_wider_for_correlated_groups():
    # 20 groups of 10 identical rows: the cluster interval must be wider than the row interval
    a = np.repeat(np.arange(1, 21) % 7 + 1, 10)
    b = np.repeat((np.arange(1, 21) * 3) % 9 + 1, 10)
    g = np.repeat(np.arange(20), 10)
    row = paired_bootstrap(a, b, k=5, n_boot=500, seed=0)
    grp = paired_bootstrap(a, b, k=5, n_boot=500, seed=0, groups=g)
    assert row["ndcg@5"]["diff"] == grp["ndcg@5"]["diff"]
    width = lambda r: r["ndcg@5"]["ci95"][1] - r["ndcg@5"]["ci95"][0]
    assert width(grp) > width(row)
