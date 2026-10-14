"""Baselines against hand-computed values."""

import numpy as np
import pytest

pytest.importorskip("implicit")
pytest.importorskip("sklearn")

from nbp.baselines.itemknn import (
    fit_itemknn,
    history_counts,
    itemknn_scorer,
    user_item_matrix,
)
from nbp.baselines.logreg import fit_encoder, logreg_proba
from nbp.baselines.popularity import fit_popularity, popularity_scorer
from nbp.eval.autoregressive import Batch


def _ui():
    # users 0, 1, 2; model ids 1..3 (column 0 = padding)
    return user_item_matrix(np.array([0, 0, 1, 1, 2]), np.array([1, 2, 1, 3, 2]), 4)


def test_cosine_similarity_by_hand():
    sim = fit_itemknn(_ui(), k=4).toarray()
    # columns: id1 = [1, 1, 0], id2 = [1, 0, 1], id3 = [0, 1, 0]
    assert sim[1, 2] == pytest.approx(1 / 2)
    assert sim[1, 3] == pytest.approx(1 / np.sqrt(2))
    assert sim[2, 3] == pytest.approx(0.0)
    assert sim[0].sum() == 0


def test_history_counts_ignore_padding():
    h = history_counts(np.array([[1, 1, 3, 0]]), np.array([3]), 4).toarray()
    assert h.tolist() == [[0, 2, 0, 1]]


def test_scorer_history_rows_and_cold_rows():
    sim = fit_itemknn(_ui(), k=4)
    pop = np.array([0, 0.4, 0.4, 0.2])
    pop_first = np.array([0, 0.1, 0.7, 0.2])
    b = Batch(
        seq=np.array([[2, 0], [0, 0]]),
        seq_len=np.array([1, 0]),
        h_dest=np.zeros((2, 2), int),
        dest=np.zeros(2, int),
        event=np.arange(2),
    )
    s = itemknn_scorer(sim, pop, pop_first, tie_eps=1e-6)(b)
    expected = sim.toarray()[2] + 1e-6 * pop  # history = one booking of id 2
    assert np.allclose(s[0], expected)
    assert np.allclose(s[1], pop_first)


def test_logreg_proba_columns_follow_model_ids():
    from sklearn.linear_model import LogisticRegression

    codes = np.array([[1, 5], [2, 5], [1, 6], [2, 6]] * 5)
    y = np.array([3, 7, 3, 7] * 5)
    enc = fit_encoder(codes)
    m = LogisticRegression(max_iter=200).fit(enc.transform(codes), y)
    p = logreg_proba(m, enc.transform(codes[:2]), 10)
    assert p.shape == (2, 10)
    assert np.allclose(p.sum(1), 1) and (p[:, [0, 1, 2, 4, 5, 6, 8, 9]] == 0).all()
    assert p[0, 3] > p[0, 7] and p[1, 7] > p[1, 3]


def test_logreg_features_are_the_query_codes_and_never_hotel_fields():
    from nbp.data.query_features import FORBIDDEN_EXACT, FORBIDDEN_PREFIX, QUERY_FIELDS, RAW_COLS

    assert not [c for c in RAW_COLS if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIX)]
    enc = fit_encoder(np.zeros((3, len(QUERY_FIELDS)), int))
    assert len(enc.categories_) == len(QUERY_FIELDS)  # one block per query field, nothing else


def test_popularity_share_and_same_ranking_for_every_row():
    # model ids 1..3 (column 0 = padding): counts 2, 1, 1 out of 4 bookings
    share = fit_popularity(np.array([1, 1, 2, 3]), 4)
    assert share.tolist() == [0.0, 0.5, 0.25, 0.25]
    b = Batch(
        seq=np.array([[2, 0], [0, 0], [3, 1]]),
        seq_len=np.array([1, 0, 2]),
        h_dest=np.zeros((3, 2), int),
        dest=np.zeros(3, int),
        event=np.arange(3),
    )
    s = popularity_scorer(share)(b)
    assert s.shape == (3, 4)
    assert (s == share[None, :]).all()  # history, cold or warm: nothing changes the ranking
