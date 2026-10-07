import pytest

torch = pytest.importorskip("torch")  # torch before anything that imports pandas (Windows c10.dll)
pytest.importorskip("recbole")

from nbp.models.smlprec_query import ROW_ID, SMLPRECQuery  # noqa: E402

S, H, N_ITEMS, F = 6, 8, 11, 3


class _Dataset:
    def num(self, field):
        return N_ITEMS


def _config():
    return {
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "NEG_PREFIX": "neg_",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": S,
        "device": "cpu",
        "n_layers": 2,
        "n_heads": 1,
        "hidden_size": H,
        "hidden_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "selected_features": [],
        "pooling_mode": "mean",
        "loss_type": "CE",
        "MODEL_INPUT_TYPE": None,
    }


def _model(n_rows=5):
    torch.manual_seed(0)
    q = torch.randint(0, 4, (n_rows, F))
    return SMLPRECQuery(_config(), _Dataset(), q, [4] * F).eval()


def _batch(model, rows):
    seq = torch.tensor([[1, 2, 3, 0, 0, 0]] * len(rows))
    return {
        model.ITEM_SEQ: seq,
        model.ITEM_SEQ_LEN: torch.tensor([3] * len(rows)),
        ROW_ID: torch.tensor(rows, dtype=torch.float32),
        model.POS_ITEM_ID: torch.tensor([4] * len(rows)),
    }


def test_output_shape_and_sequence_length():
    m = _model()
    assert m.layers[0] is not None
    out = m.full_sort_predict(_batch(m, [0, 1, 2]))
    assert out.shape == (3, N_ITEMS)


def test_query_changes_scores_with_history_fixed():
    m = _model()
    m.query_idx[0] = torch.tensor([0, 0, 0], dtype=torch.int32)
    m.query_idx[1] = torch.tensor([3, 2, 1], dtype=torch.int32)
    with torch.no_grad():
        out = m.full_sort_predict(_batch(m, [0, 1]))
    assert not torch.allclose(out[0], out[1])


def test_same_query_same_scores_and_loss_backprops():
    m = _model().train()
    m.query_idx[1] = m.query_idx[0]
    out = m.full_sort_predict(_batch(m, [0, 1]))
    assert torch.allclose(out[0], out[1])
    loss = m.calculate_loss(_batch(m, [0, 1, 2]))
    loss.backward()
    assert m.query_emb[0].weight.grad is not None and m.query_mask.grad is not None


def test_padding_only_history_still_scores():
    m = _model()
    b = _batch(m, [0])
    b[m.ITEM_SEQ] = torch.zeros(1, S, dtype=torch.long)
    assert torch.isfinite(m.full_sort_predict(b)).all()
