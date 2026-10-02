import math

import numpy as np
import pytest

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import ndcg_at_k, rank_in_list, recall_at_k, summarize, target_rank


def test_target_rank_basic_and_ties():
    scores = np.array(
        [
            [0.1, 0.9, 0.5, 0.3],  # true 2: only 0.9 above -> rank 2
            [0.4, 0.4, 0.4, 0.4],  # true 2, all tied: columns 0, 1 win the tie -> rank 3
            [-np.inf, 0.2, 0.1, 0.7],  # true 3: best -> rank 1
        ]
    )
    y = np.array([2, 2, 3])
    assert target_rank(scores, y).tolist() == [2, 3, 1]
    assert target_rank(scores, y, chunk=1).tolist() == [2, 3, 1]


def test_rank_in_list_hit_and_miss():
    top = np.array([[7, 3, 5], [1, 2, 4]])
    assert rank_in_list(top, np.array([5, 9])).tolist() == [3, 4]


def test_recall_and_ndcg_hand_computed():
    rank = np.array([1, 3, 6, 11])
    assert recall_at_k(rank, 5) == pytest.approx(0.5)
    assert recall_at_k(rank, 10) == pytest.approx(0.75)
    # NDCG@5 = (1/log2(2) + 1/log2(4)) / 4 = (1 + 0.5) / 4
    assert ndcg_at_k(rank, 5) == pytest.approx(0.375)
    assert ndcg_at_k(rank, 10) == pytest.approx((1 + 0.5 + 1 / math.log2(7)) / 4)


def test_summarize_keys_and_mask():
    rank = np.array([1, 3, 6, 11])
    s = summarize(rank, mask=np.array([True, True, False, False]), ks=(5,))
    assert s == {"n": 2, "recall@5": 1.0, "ndcg@5": 0.75}


def test_paired_bootstrap_identical_is_zero():
    rank = np.array([1, 2, 7, 3, 12])
    res = paired_bootstrap(rank, rank, k=5, n_boot=50)
    assert res["recall@5"] == {"diff": 0.0, "ci95": [0.0, 0.0]}
    assert res["ndcg@5"]["diff"] == 0.0


def test_paired_bootstrap_sign():
    a, b = np.array([1, 1, 1, 1]), np.array([9, 9, 9, 9])
    res = paired_bootstrap(a, b, k=5, n_boot=50)
    assert res["recall@5"]["diff"] == 1.0
    assert res["ndcg@5"]["ci95"] == [1.0, 1.0]


def test_compare_nested_matches_common_numeric_leaves():
    from nbp.eval.compare import compare_nested

    ref = {"a": {"recall@5": 0.5, "name": "x"}, "per": [{"x": 1}], "timing": {"t": 3.0}}
    new = {"a": {"recall@5": 0.75, "extra": 1.0}, "per": [{"x": 1}], "timing": {"t": 9.0}}
    assert compare_nested(ref, new) == [("a.recall@5", 0.5, 0.75, 0.25), ("per.0.x", 1.0, 1.0, 0.0)]
