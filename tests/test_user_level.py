import numpy as np
import pytest

from nbp.eval.user_level import user_anchor_and_truth, user_set_metrics


def ranked_with(positions: dict, n: int = 10) -> list:
    """A ranked list of n items where each given item sits at its 1-based position; other slots are filler."""
    out = [None] * n
    for item, pos in positions.items():
        out[pos - 1] = item
    filler = iter(range(1000, 2000))
    return [x if x is not None else next(filler) for x in out]


def test_matches_teammate_notebook_examples():
    # values printed by the teammate's Metrics.ipynb (recall@5, ndcg@5)
    cases = [
        ({42, 49}, {42: 5, 49: 27}, 0.5, 0.2372),
        ({97}, {97: 4}, 1.0, 0.4307),
        ({42, 94}, {42: 1, 94: 4}, 1.0, 0.8772),
        ({38}, {38: 23}, 0.0, 0.0),
    ]
    for truth, pos, rec, nd in cases:
        top = np.array([ranked_with({c: p for c, p in pos.items() if p <= 10})])
        got = user_set_metrics(top, [truth], k=5)
        assert got["recall@5"] == pytest.approx(rec, abs=1e-4)
        assert got["ndcg@5"] == pytest.approx(nd, abs=1e-4)


def test_single_target_reduces_to_row_level():
    top = np.array([ranked_with({7: 2}), ranked_with({7: 8})])
    got = user_set_metrics(top, [{7}, {7}], k=5)
    assert got["recall@5"] == pytest.approx(0.5)
    assert got["ndcg@5"] == pytest.approx((1 / np.log2(3)) / 2)


def test_anchor_is_earliest_event_and_truth_is_all_items():
    users = np.array([5, 9, 5, 5, 9])
    ts = np.array([30, 10, 10, 20, 10])  # user 5: earliest is index 2; user 9: tie -> lower index 1
    items = np.array([1, 2, 3, 3, 4])
    uid, anchor, truths = user_anchor_and_truth(users, ts, items)
    assert uid.tolist() == [5, 9]
    assert anchor.tolist() == [2, 1]
    assert truths == [{1, 3}, {2, 4}]
