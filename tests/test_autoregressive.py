"""Autoregressive and teacher-forced scoring against hand-built windows."""

import numpy as np

from nbp.data.split import build_rows
from nbp.eval.autoregressive import RowView, score_view

N_COLS = 8  # model ids 0..7, 0 = padding


def _rows(user, item, dest, window, max_len=4):
    r = build_rows(
        np.array(user), np.array(item), np.array(dest), np.array(window, np.int8), max_len
    )
    r["event"] = np.arange(len(user))
    r["dest"] = np.array(dest)
    r["window"] = np.array(window, np.int8)
    return r


# user 1: train booking (item 0), then 3 test bookings (items 1, 2, 3); user 2: 1 test booking;
# user 3: 7 test bookings, first one cold.  Items are clusters (model id = item + 1).
USER = [1, 1, 3, 2, 1, 3, 3, 1, 3, 3, 3, 3]
ITEM = [0, 1, 5, 4, 2, 5, 6, 3, 2, 2, 2, 2]
DEST = [9, 9, 7, 8, 9, 7, 7, 9, 7, 7, 7, 7]
WIN = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


def _view(r):
    return RowView.from_rows(r, r["window"] == 2)


def last_item_scorer(b):
    """Scores the last history item highest; falls back to model id 1 on an empty history."""
    s = np.zeros((len(b.seq), N_COLS), np.float32)
    rows = np.arange(len(b.seq))
    last = np.where(b.seq_len > 0, b.seq[rows, np.maximum(b.seq_len - 1, 0)], 1)
    s[rows, last] = 1.0
    return s


def constant_scorer(b):
    """Ignores the history (a history-free baseline): always ranks model id 7 first."""
    s = np.tile(np.arange(N_COLS, dtype=np.float32), (len(b.seq), 1))
    return s + b.dest[:, None] * 0


def test_h0_rows_identical_under_both_protocols():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    ar, tf = score_view(v, last_item_scorer, True), score_view(v, last_item_scorer, False)
    h0 = v.h == 0
    assert np.array_equal(ar.scores[h0], tf.scores[h0])


def test_true_window_labels_never_reach_the_scorer():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    ar = score_view(v, last_item_scorer, True)
    # user 1: the train booking (model id 1) is true history; every later prediction repeats it
    u1 = np.flatnonzero(np.isin(v.event, [1, 4, 7]))
    assert ar.top1[u1].tolist() == [1, 1, 1]
    tf = score_view(v, last_item_scorer, False)
    assert tf.top1[u1].tolist() == [1, 2, 3]  # TF repeats the true previous booking


def test_history_free_scorer_is_identical_under_both_protocols():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    ar, tf = score_view(v, constant_scorer, True), score_view(v, constant_scorer, False)
    assert np.array_equal(ar.scores, tf.scores)


def test_window_shift_and_replaced_slot_count():
    r = _rows(USER, ITEM, DEST, WIN, max_len=4)
    v = _view(r)
    seen = {}

    def spy(b):
        for e, s in zip(b.event, b.seq):
            seen[int(e)] = s.tolist()
        return last_item_scorer(b)

    ar = score_view(v, spy, True)
    # user 3's last row (event 11, h = 6): the 4 slots are all predictions; cold first row -> id 1
    # is predicted forever after
    assert seen[11] == [1, 1, 1, 1]
    i11 = int(np.flatnonzero(v.event == 11)[0])
    assert ar.n_replaced[i11] == 4
    # user 1's last row (event 7, h = 2): slot 0 = true train booking, slots 1-2 = predictions
    assert seen[7][:3] == [1, 1, 1]
    i7 = int(np.flatnonzero(v.event == 7)[0])
    assert ar.n_replaced[i7] == 2


def test_uneven_users_and_input_order_preserved():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    perm = np.random.default_rng(0).permutation(len(v))
    shuffled = RowView(**{k: getattr(v, k)[perm] for k in v.__dataclass_fields__})
    a = score_view(v, last_item_scorer, True)
    b = score_view(shuffled, last_item_scorer, True)
    assert np.array_equal(a.scores[perm], b.scores)
    assert np.array_equal(a.top1[perm], b.top1)


def test_cold_first_row_empty_history_passes_its_prediction_on():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    seen = {}

    def spy(b):
        for e, s, n in zip(b.event, b.seq, b.seq_len):
            seen[int(e)] = (s.tolist(), int(n))
        return last_item_scorer(b)

    ar = score_view(v, spy, True)
    assert seen[2] == ([0, 0, 0, 0], 0)  # user 3's first booking: empty history
    assert seen[5][0][0] == ar.top1[int(np.flatnonzero(v.event == 2)[0])]


def test_destinations_of_history_slots_are_kept():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    seen = {}

    def spy(b):
        for e, d in zip(b.event, b.h_dest):
            seen[int(e)] = d.tolist()
        return last_item_scorer(b)

    score_view(v, spy, True)
    assert seen[7][:3] == [9, 9, 9]
    assert seen[11] == [7, 7, 7, 7]


def test_padding_column_is_minus_inf_and_never_top1():
    r = _rows(USER, ITEM, DEST, WIN)
    v = _view(r)
    ar = score_view(v, lambda b: np.ones((len(b.seq), N_COLS)), True)
    assert np.isneginf(ar.scores[:, 0]).all()
    assert (ar.top1 >= 1).all()


def test_sub_mask_view_is_rejected_under_ar():
    import pytest

    r = _rows(USER, ITEM, DEST, WIN)
    warm_only = (r["window"] == 2) & (r["L"] >= 1)
    v = RowView.from_rows(r, warm_only)
    with pytest.raises(ValueError):
        score_view(v, last_item_scorer, True)
    score_view(v, last_item_scorer, False)  # TF never reads predictions: any subset is fine


def test_empty_view():
    r = _rows(USER, ITEM, DEST, WIN)
    v = RowView.from_rows(r, np.zeros(len(USER), bool))
    assert len(score_view(v, last_item_scorer, True).top1) == 0
