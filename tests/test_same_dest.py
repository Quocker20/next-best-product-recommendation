"""sameDest scores against hand-computed examples."""

import numpy as np
import pytest

from nbp.hybrid.same_dest import same_dest_scores

# one user, history oldest -> newest: cluster 5 at destination 100, cluster 6 at 200, cluster 7 at 100
SEQ = np.array([[5, 6, 7, 0]])
LEN = np.array([3])
H_DEST = np.array([[100, 200, 100, 0]])
Q = np.array([100])


def test_decay_recent_counts_more():
    s = same_dest_scores(SEQ, LEN, H_DEST, Q, n_items=10, base=0.7)[0]
    # ages: cluster 5 -> 2, cluster 6 -> 1 (other destination), cluster 7 -> 0
    assert s[7] == pytest.approx(1.0)
    assert s[5] == pytest.approx(0.49)
    assert s[6] == 0.0


def test_normalized_is_share_of_whole_history():
    s = same_dest_scores(SEQ, LEN, H_DEST, Q, n_items=10, base=0.7, normalize=True)[0]
    total = 0.49 + 0.7 + 1.0  # the other-destination booking still counts in the total
    assert s[5] == pytest.approx(0.49 / total)
    assert s[7] == pytest.approx(1.0 / total)
    assert s.sum() <= 1.0


def test_repeated_cluster_adds_up():
    seq = np.array([[5, 5, 0]])
    s = same_dest_scores(seq, np.array([2]), np.array([[100, 100, 0]]), Q, n_items=10, base=0.7)[0]
    assert s[5] == pytest.approx(0.7 + 1.0)


def test_padding_and_other_rows_are_ignored():
    seq = np.array([[5, 0, 0], [8, 9, 0]])
    h_dest = np.array([[100, 100, 100], [100, 300, 0]])
    s = same_dest_scores(seq, np.array([1, 2]), h_dest, np.array([100, 100]), n_items=10, base=0.7)
    assert s[0, 0] == 0.0  # padding id never scores although its destination field equals the query
    assert s[0, 5] == pytest.approx(1.0)
    assert s[1, 8] == pytest.approx(0.7) and s[1, 9] == 0.0


def test_values_beyond_the_history_length_never_change_the_score():
    """Future-proofing: whatever sits in padded positions (even a query-destination match) is ignored."""
    base = same_dest_scores(SEQ, LEN, H_DEST, Q, n_items=10, base=0.7)
    seq2 = np.array([[5, 6, 7, 9]])  # a "future" booking of cluster 9 after the history
    h2 = np.array([[100, 200, 100, 100]])  # at the query destination
    assert np.array_equal(base, same_dest_scores(seq2, LEN, h2, Q, n_items=10, base=0.7))
