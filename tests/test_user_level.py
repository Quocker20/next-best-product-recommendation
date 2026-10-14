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


def test_fixed_top_user_metrics_example():
    from nbp.eval.user_level import fixed_top_user_metrics

    # top-5 {A..E}=0..4 ; bookings A,B,C,D,C,E,A,Z,Z,Z -> truth set of 6, 5 hit
    top = np.array([[0, 1, 2, 3, 4] + list(range(10, 25))])
    truth = [{0, 1, 2, 3, 4, 99}]
    got = fixed_top_user_metrics(top, truth, ks=(5, 10, 20))
    assert abs(got["recall@5"] - 5 / 6) < 1e-12
    assert got["recall@20"] == got["recall@10"] == got["recall@5"]
    disc = [1 / np.log2(r + 1) for r in range(1, 6)]
    assert abs(got["ndcg@5"] - sum(disc) / sum(disc)) < 1e-12  # k=5: all top-5 hit, ideal = 5 hits
    # k=20: ideal = 6 slots, dcg = 5 hits at ranks 1..5
    d6 = sum(1 / np.log2(r + 1) for r in range(1, 7))
    assert abs(got["ndcg@20"] - sum(disc) / d6) < 1e-12


def test_fixed_top_matches_user_set_metrics():
    from nbp.eval.user_level import fixed_top_user_metrics

    rng = np.random.default_rng(0)
    top = np.stack([rng.permutation(50)[:20] for _ in range(30)])
    truths = [set(rng.choice(50, size=int(rng.integers(1, 12)), replace=False).tolist()) for _ in range(30)]
    got = fixed_top_user_metrics(top, truths)
    for k in (5, 10, 20):
        ref = user_set_metrics(top, truths, k=k)
        assert abs(got[f"recall@{k}"] - ref[f"recall@{k}"]) < 1e-12
        assert abs(got[f"ndcg@{k}"] - ref[f"ndcg@{k}"]) < 1e-12


def test_slide_example_appendix_a2():
    """The worked example of appendix A2 of the week-6 deck: list 7, 3, 12, 40, 9; truth {3, 9, 55}; K = 5."""
    from nbp.eval.user_level import fixed_top_user_metrics

    top = np.array([[7, 3, 12, 40, 9] + list(range(100, 115))])
    got = fixed_top_user_metrics(top, [{3, 9, 55}], ks=(5,))
    assert abs(got["recall@5"] - 2 / 3) < 1e-12
    dcg = 1 / np.log2(3) + 1 / np.log2(6)
    ideal = 1 + 1 / np.log2(3) + 0.5
    assert abs(got["ndcg@5"] - dcg / ideal) < 1e-12
    assert round(got["ndcg@5"], 3) == 0.478
