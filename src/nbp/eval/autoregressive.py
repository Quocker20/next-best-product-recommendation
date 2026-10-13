"""Autoregressive (no label feedback) and teacher-forced scoring of one evaluation window.

Teacher-forced (TF): every row is scored with its true history.

Autoregressive (AR): inside the evaluated window the true cluster of a booking never enters a
history. A history slot that holds an earlier booking of the same window gets the top-1 prediction
made for that booking instead. Slots holding bookings of earlier windows keep their true item, and
the destination of every slot is kept (the searched destination is known at search time). Rows are
scored in steps h = 0, 1, ... (h = earlier bookings of the user inside the window), so the
predictions a row needs always exist before it is scored. Rows with h = 0 get the same input under
both protocols.

Scorers never see the label: `Batch` has no `y`. Only numpy is used (both venvs).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class RowView:
    """All rows of one window (any order; never a sub-mask such as warm rows only). Arrays come from `rows.npz` (see `nbp.data.split`).

    Attributes:
        event: (n,) global event index of each row.
        h: (n,) earlier bookings of the user inside this window.
        seq: (n, S) true history model ids, oldest first, 0 = padding.
        seq_len: (n,) real history length.
        h_dest: (n, S) destination of each history slot.
        h_event: (n, S) global event index of each history slot (-1 on padding).
        dest: (n,) searched destination of the row.
    """

    event: np.ndarray
    h: np.ndarray
    seq: np.ndarray
    seq_len: np.ndarray
    h_dest: np.ndarray
    h_event: np.ndarray
    dest: np.ndarray

    @classmethod
    def from_rows(cls, rows, mask: np.ndarray) -> RowView:
        """Select `mask` rows from a loaded `rows.npz` (or any mapping with the same keys)."""
        idx = np.flatnonzero(mask)
        return cls(
            event=np.asarray(rows["event"])[idx],
            h=np.asarray(rows["h"])[idx],
            seq=np.asarray(rows["seq"])[idx],
            seq_len=np.asarray(rows["seq_len"])[idx],
            h_dest=np.asarray(rows["h_dest"])[idx],
            h_event=np.asarray(rows["h_event"])[idx],
            dest=np.asarray(rows["dest"])[idx],
        )

    def __len__(self) -> int:
        return len(self.event)


@dataclass
class Batch:
    """Model input for b rows (no label).

    Attributes:
        seq: (b, S) int64 history model ids (true or predicted), oldest first, 0 = padding.
        seq_len: (b,) int64 history length.
        h_dest: (b, S) int64 destination of each history slot.
        dest: (b,) int64 searched destination.
        event: (b,) int64 global event index (to look up the query codes of the row).
    """

    seq: np.ndarray
    seq_len: np.ndarray
    h_dest: np.ndarray
    dest: np.ndarray
    event: np.ndarray


ScoreFn = Callable[[Batch], np.ndarray]


@dataclass
class Scored:
    """Scores of every row of a view, in the view's row order.

    Attributes:
        scores: (n, n_cols) float32 model scores; column 0 (padding) is -inf.
        top1: (n,) model id with the highest score (padding excluded).
        n_replaced: (n,) number of history slots that hold a prediction (0 under TF).
    """

    scores: np.ndarray
    top1: np.ndarray
    n_replaced: np.ndarray


def _score(score_fn: ScoreFn, b: Batch) -> np.ndarray:
    s = np.array(score_fn(b), dtype=np.float32, copy=True)
    s[:, 0] = -np.inf
    return s


def score_view(view: RowView, score_fn: ScoreFn, autoregressive: bool, batch: int = 8192) -> Scored:
    """Score every row of `view` under the AR or the TF protocol.

    Args:
        view: rows of one window.
        score_fn: Batch -> (b, n_cols) scores, column j = model id j; column 0 is ignored.
        autoregressive: True = AR protocol, False = TF protocol.
        batch: rows per scorer call.
    Returns:
        `Scored` aligned with the view's row order.
    """
    n = len(view)
    if n == 0:
        return Scored(np.empty((0, 0), np.float32), np.empty(0, np.int64), np.empty(0, np.int64))
    ev_sorted_idx = np.argsort(view.event, kind="stable")
    ev_sorted = view.event[ev_sorted_idx]
    # slot -> position of that booking inside this view (-1 = earlier window or padding)
    pos = np.searchsorted(ev_sorted, view.h_event)
    pos = np.minimum(pos, n - 1)
    in_view = (view.h_event >= 0) & (ev_sorted[pos] == view.h_event)
    slot_row = np.where(in_view, ev_sorted_idx[pos], -1)
    if autoregressive:
        # the view must hold the WHOLE window: every earlier in-window booking that is still inside
        # the history must be a row of the view, or its true label would stay in later histories
        expected = np.minimum(view.h, view.seq_len)
        if not np.array_equal(in_view.sum(1), expected):
            raise ValueError(
                "view does not contain the whole window (or h disagrees with the history): build it "
                "from the window mask only and apply warm / cold masks to the metrics afterwards"
            )

    scores = None
    top1 = np.zeros(n, np.int64)
    n_replaced = np.zeros(n, np.int64)
    steps = np.unique(view.h) if autoregressive else np.array([-1])
    for s in steps:
        rows_s = np.flatnonzero(view.h == s) if autoregressive else np.arange(n)
        for i in range(0, len(rows_s), batch):
            r = rows_s[i : i + batch]
            seq = view.seq[r].astype(np.int64)
            if autoregressive:
                src = slot_row[r]
                rep = src >= 0
                assert (view.h[src[rep]] < s).all(), (
                    "history row scored after the row that reads it"
                )
                seq[rep] = top1[src[rep]]
                n_replaced[r] = rep.sum(1)
            b = Batch(
                seq=seq,
                seq_len=view.seq_len[r].astype(np.int64),
                h_dest=view.h_dest[r].astype(np.int64),
                dest=view.dest[r].astype(np.int64),
                event=view.event[r].astype(np.int64),
            )
            sc = _score(score_fn, b)
            if scores is None:
                scores = np.empty((n, sc.shape[1]), np.float32)
            scores[r] = sc
            top1[r] = sc.argmax(1)
    return Scored(scores=scores, top1=top1, n_replaced=n_replaced)
