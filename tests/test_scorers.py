"""sameDest fusion and the plain scorer against hand-computed values."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("recbole")

from nbp.eval.autoregressive import Batch
from nbp.hybrid.fusion import smoothed_log
from nbp.models import scorers as S


def _batch(seq, seq_len, h_dest, dest):
    return Batch(
        np.array(seq), np.array(seq_len), np.array(h_dest), np.array(dest), np.arange(len(seq))
    )


def test_samedest_fuse_by_hand():
    base = np.log(np.array([[1.0, 0.5, 0.3, 0.2]]))  # column 0 = padding, ignored
    b = _batch(
        [[2, 0]], [1], [[9, 0]], [9]
    )  # one earlier booking of id 2 at the searched destination
    out = S.samedest_fuse(base, b, alpha=0.1, w=2.0, recency=0.7)
    p = np.array([[0.5, 0.3, 0.2]])
    sd = np.array([[0.0, 1.0, 0.0]])  # id 2 holds all the history weight
    want = smoothed_log(p, 0.1) + 2.0 * smoothed_log(sd, 0.1)
    assert np.isneginf(out[0, 0])
    assert np.allclose(out[:, 1:], want, atol=1e-6)


def test_samedest_leaves_new_destination_ranking_unchanged():
    base = np.log(np.array([[1.0, 0.5, 0.3, 0.2]]))
    b = _batch([[2, 0]], [1], [[8, 0]], [9])  # history at another destination
    out = S.samedest_fuse(base, b, alpha=0.1, w=3.0, recency=0.7)
    assert np.argsort(-out[0, 1:]).tolist() == [0, 1, 2]


def test_plain_scorer_cold_rows_share_one_ranking():
    cfg = {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": 4,
        "device": "cpu",
        "n_layers": 1,
        "n_heads": 1,
        "hidden_size": 8,
        "hidden_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "selected_features": [],
        "pooling_mode": "mean",
        "loss_type": "CE",
        "seed": 0,
    }
    m = S.build_plain(cfg, 6).eval()
    b = _batch(
        [[0, 0, 0, 0], [0, 0, 0, 0], [3, 2, 0, 0]], [0, 0, 2], np.zeros((3, 4), int), [1, 1, 1]
    )
    s = S.plain_scorer(m)(b)
    assert np.allclose(s[0], s[1]) and not np.allclose(s[0], s[2])
    with pytest.raises(ValueError):
        S.plain_loss(np.zeros((1, 4), int), np.array([0]), np.array([1]))
