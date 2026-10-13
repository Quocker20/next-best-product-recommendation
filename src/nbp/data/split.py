"""Date split of the bookings and the per-row history arrays used by every model.

One row per booking. A row belongs to the window of its own booking time (0 = train, 1 = valid,
2 = test). Its history is the user's earlier bookings (any window, the last `max_len` of them),
oldest first, right-padded with 0. Items are model ids: cluster c -> c + 1, 0 = padding.

Only numpy is used, so the module also imports in the SMLP4Rec venv (numpy 1.23).
"""

from __future__ import annotations

import numpy as np

TRAIN, VALID, TEST = 0, 1, 2
WINDOW_NAMES = {TRAIN: "train", VALID: "valid", TEST: "test"}


def assign_window(
    ts: np.ndarray,
    valid_start: str | np.datetime64,
    test_start: str | np.datetime64,
    test_end: str | np.datetime64,
) -> np.ndarray:
    """(n,) int8 window of each booking time.

    Args:
        ts: (n,) datetime64 (any unit) booking times.
        valid_start, test_start, test_end: datetime64-compatible bounds; the start of a window is
            inclusive, its end exclusive.
    Returns:
        0 = train (ts < valid_start), 1 = valid, 2 = test, -1 = at or after test_end.
    """
    ts = np.asarray(ts).astype("datetime64[ns]")
    vs, ts_, te = (np.datetime64(x, "ns") for x in (valid_start, test_start, test_end))
    if not vs < ts_ < te:
        raise ValueError("split dates must satisfy valid_start < test_start < test_end")
    out = np.full(len(ts), -1, dtype=np.int8)
    out[ts < te] = TEST
    out[ts < ts_] = VALID
    out[ts < vs] = TRAIN
    return out


def build_rows(
    user: np.ndarray, item: np.ndarray, dest: np.ndarray, window: np.ndarray, max_len: int
) -> dict:
    """Per-row history arrays for bookings given in event order (time order, ties already fixed).

    Args:
        user: (n,) int user id per booking.
        item: (n,) int cluster per booking (0..99); stored as model id item + 1.
        dest: (n,) int searched destination per booking.
        window: (n,) int8 window per booking (non-decreasing in time).
        max_len: history length S.
    Returns:
        dict of arrays, row i = booking i:
        ``y`` (n,) int16 model id; ``L`` (n,) int32 earlier bookings of the user;
        ``h`` (n,) int32 earlier bookings of the user in the same window;
        ``seq_len`` (n,) int8 = min(L, S); ``seq`` (n, S) int16 history model ids, oldest first;
        ``h_dest`` (n, S) int32 destination of each history item (0 on padding);
        ``h_event`` (n, S) int32 event index of each history item (-1 on padding).
    """
    user = np.asarray(user)
    n = len(user)
    if np.any(np.diff(np.asarray(window).astype(np.int64)) < 0):
        raise ValueError("bookings must be in time order: window decreases somewhere")
    order = np.argsort(user, kind="stable")  # per user, event (time) order is kept
    u_sorted = user[order]
    new_user = np.r_[True, u_sorted[1:] != u_sorted[:-1]]
    start = np.maximum.accumulate(np.where(new_user, np.arange(n), 0))
    pos = np.arange(n) - start  # position of the booking inside its user's sequence
    w_sorted = np.asarray(window)[order]
    new_group = new_user | np.r_[True, w_sorted[1:] != w_sorted[:-1]]
    g_start = np.maximum.accumulate(np.where(new_group, np.arange(n), 0))

    L = np.empty(n, np.int32)
    h = np.empty(n, np.int32)
    L[order] = pos
    h[order] = np.arange(n) - g_start
    seq_len = np.minimum(L, max_len).astype(np.int8)

    y = (np.asarray(item).astype(np.int64) + 1).astype(np.int16)
    dest = np.asarray(dest).astype(np.int32)
    seq = np.zeros((n, max_len), np.int16)
    h_dest = np.zeros((n, max_len), np.int32)
    h_event = np.full((n, max_len), -1, np.int32)
    # sorted position of each row and of its history item at slot p (oldest first)
    pos_s = pos
    len_s = np.minimum(pos_s, max_len)
    for p in range(max_len):
        ok = p < len_s
        src_sorted = (
            np.arange(n)[ok] - len_s[ok] + p
        )  # same user: the slice ends right before the row
        ev = order[src_sorted]
        rows = order[np.arange(n)[ok]]
        seq[rows, p] = y[ev]
        h_dest[rows, p] = dest[ev]
        h_event[rows, p] = ev
    return {
        "y": y,
        "L": L,
        "h": h,
        "seq_len": seq_len,
        "seq": seq,
        "h_dest": h_dest,
        "h_event": h_event,
    }
