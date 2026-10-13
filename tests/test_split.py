"""Date split and history arrays against hand-computed examples."""

import numpy as np
import pytest

from nbp.data.split import TEST, TRAIN, VALID, assign_window, build_rows

DATES = ("2014-09-01", "2014-11-01", "2015-01-01")


def test_window_edges_are_start_inclusive_end_exclusive():
    ts = np.array(
        [
            "2014-08-31T23:59:59",
            "2014-09-01T00:00:00",
            "2014-10-31T23:59:59",
            "2014-11-01T00:00:00",
            "2014-12-31T23:59:59",
            "2015-01-01T00:00:00",
        ],
        dtype="datetime64[s]",
    )
    assert assign_window(ts, *DATES).tolist() == [TRAIN, VALID, VALID, TEST, TEST, -1]


def test_bad_date_order_raises():
    with pytest.raises(ValueError):
        assign_window(
            np.array(["2014-01-01"], dtype="datetime64[D]"),
            "2014-11-01",
            "2014-09-01",
            "2015-01-01",
        )


# events in time order: user 7 books 4 times (train, train, valid, test), user 3 twice (valid, valid),
# user 5 once (test)
USER = np.array([7, 7, 3, 7, 3, 5, 7])
ITEM = np.array([10, 11, 20, 12, 21, 30, 13])
DEST = np.array([100, 101, 200, 100, 201, 300, 102])
WIN = np.array([TRAIN, TRAIN, VALID, VALID, VALID, TEST, TEST], dtype=np.int8)


def test_history_length_and_same_window_count():
    r = build_rows(USER, ITEM, DEST, WIN, max_len=20)
    assert r["L"].tolist() == [0, 1, 0, 2, 1, 0, 3]
    assert r["h"].tolist() == [
        0,
        1,
        0,
        0,
        1,
        0,
        0,
    ]  # user 7's valid and test bookings open new windows
    assert r["seq_len"].tolist() == [0, 1, 0, 2, 1, 0, 3]
    assert r["y"].tolist() == [11, 12, 21, 13, 22, 31, 14]


def test_history_items_destinations_and_events():
    r = build_rows(USER, ITEM, DEST, WIN, max_len=20)
    assert r["seq"][6, :4].tolist() == [
        11,
        12,
        13,
        0,
    ]  # user 7's last row: model ids of its 3 earlier bookings
    assert r["h_dest"][6, :4].tolist() == [100, 101, 100, 0]
    assert r["h_event"][6, :4].tolist() == [0, 1, 3, -1]
    assert r["seq"][4, :2].tolist() == [21, 0]
    assert r["h_event"][4, :2].tolist() == [2, -1]
    assert (r["seq"][[0, 2, 5]] == 0).all()  # cold rows: empty history
    assert (r["h_event"][[0, 2, 5]] == -1).all()


def test_every_history_event_is_earlier_than_its_row():
    r = build_rows(USER, ITEM, DEST, WIN, max_len=20)
    ev = r["h_event"]
    rows = np.arange(len(USER))[:, None]
    assert ((ev < rows) | (ev == -1)).all()


def test_window_slides_after_max_len():
    n = 6
    r = build_rows(np.zeros(n, int), np.arange(n), np.arange(n), np.zeros(n, np.int8), max_len=3)
    assert r["seq"][5].tolist() == [
        3,
        4,
        5,
    ]  # items 2, 3, 4 (model ids + 1): the last 3 earlier bookings
    assert r["h_event"][5].tolist() == [2, 3, 4]
    assert r["seq_len"].tolist() == [0, 1, 2, 3, 3, 3]
    assert r["h"].tolist() == [0, 1, 2, 3, 4, 5]
