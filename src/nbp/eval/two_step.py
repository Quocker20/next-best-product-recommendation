"""User-level test protocol: one fixed top-20 list per user from two prediction steps (numpy only).

Per test user:
- step 1: the user's first test booking (h = 0), scored with its true history; keep the top-20
  clusters and their probabilities (P1);
- step 2: the user's second test booking (h = 1), whose history slot for booking 1 holds the top-1
  of P1 instead of the true cluster (no label feedback). A user with a single test booking gets a
  synthetic step 2: the context (query, searched destination) of booking 1 and the history of
  booking 1 followed by the top-1 of P1. Keep the top-20 again (P2);
- rerank P1 and P2 into one top-20 (`RERANKERS`), scored against the set of distinct clusters the
  user books in the whole test window (`nbp.eval.user_level.fixed_top_user_metrics`).

Probabilities are taken over all 100 clusters before the top-20 cut: softmax of the scores for the
SMLP4Rec models (logits, or the fused log-scores of sameDest), division by the row sum for the
non-negative baseline scores (popularity shares, ItemKNN similarity sums). Model ids are used
throughout (column 0 = padding, cluster c = column c + 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nbp.eval.autoregressive import Batch, RowView, ScoreFn, _score

TOP = 20


@dataclass
class UserPlan:
    """Rows of the two steps, one entry per test user (sorted by user id).

    Attributes:
        user: (u,) user ids.
        first: (u,) view position of the user's first test booking (h = 0).
        second: (u,) view position of the second test booking (h = 1), -1 if the user has one.
        truths: u sets of the distinct model ids the user books in the test window.
    """

    user: np.ndarray
    first: np.ndarray
    second: np.ndarray
    truths: list[set]


def user_plan(view: RowView, user: np.ndarray, y: np.ndarray) -> UserPlan:
    """Pick the two step rows and the truth set of every user of the window.

    Args:
        view: the whole test window.
        user: (n,) user id of each view row.
        y: (n,) true model id of each view row.
    """
    user, y = np.asarray(user), np.asarray(y)
    order = np.lexsort((view.h, user))  # by user, then h
    u_sorted = user[order]
    start = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
    end = np.r_[start[1:], len(order)]
    first = order[start]
    if not (view.h[first] == 0).all():
        raise ValueError("every user needs a booking with h = 0 in the window")
    has2 = (end - start) >= 2
    second = np.full(len(start), -1, np.int64)
    second[has2] = order[start[has2] + 1]
    if not (view.h[second[has2]] == 1).all():
        raise ValueError("the second booking of a user must have h = 1")
    truths = [set(y[order[s:e]].tolist()) for s, e in zip(start, end)]
    return UserPlan(user=u_sorted[start], first=first, second=second, truths=truths)


def _batch(view: RowView, r: np.ndarray) -> Batch:
    return Batch(
        seq=view.seq[r].astype(np.int64),
        seq_len=view.seq_len[r].astype(np.int64),
        h_dest=view.h_dest[r].astype(np.int64),
        dest=view.dest[r].astype(np.int64),
        event=view.event[r].astype(np.int64),
    )


def step2_batch(view: RowView, plan: UserPlan, top1: np.ndarray) -> Batch:
    """Inputs of step 2 for every user (aligned with `plan`).

    Args:
        view: the whole test window.
        plan: from `user_plan`.
        top1: (u,) top-1 model id of step 1 per user.
    """
    b = _batch(view, plan.first)  # single-booking users: context of booking 1
    S = b.seq.shape[1]
    has2 = plan.second >= 0

    # users with a real second booking: its own row, the booking-1 slot replaced by the step-1 top-1
    r2 = plan.second[has2]
    seq2 = view.seq[r2].astype(np.int64)
    slot = view.h_event[r2] == view.event[plan.first[has2]][:, None]
    if not (slot.sum(1) == 1).all():
        raise ValueError("booking 1 must appear exactly once in the history of booking 2")
    seq2[slot] = top1[has2]  # one slot per row, row-major order
    b.seq[has2] = seq2
    b.seq_len[has2] = view.seq_len[r2]
    b.h_dest[has2] = view.h_dest[r2]
    b.dest[has2] = view.dest[r2]
    b.event[has2] = view.event[r2]

    # single-booking users: append (top-1, destination of booking 1) to the history of booking 1
    one = np.flatnonzero(~has2)
    full = b.seq_len[one] >= S
    for arr, val in ((b.seq, top1[one]), (b.h_dest, b.dest[one])):
        sub = arr[one]
        sub[full] = np.roll(sub[full], -1, axis=1)  # drop the oldest slot when the history is full
        pos = np.minimum(b.seq_len[one], S - 1)
        sub[np.arange(len(one)), pos] = val
        arr[one] = sub
    b.seq_len[one] = np.minimum(b.seq_len[one] + 1, S)
    return b


def to_prob(scores: np.ndarray, kind: str) -> np.ndarray:
    """(b, n_cols) probabilities over the clusters, column 0 = 0.

    Args:
        scores: model scores, column 0 = -inf (padding).
        kind: "softmax" (logits / log-scores) or "sum" (non-negative scores, divided by the row sum).
    """
    s = np.asarray(scores, np.float64)[:, 1:]
    if kind == "softmax":
        p = np.exp(s - s.max(1, keepdims=True))
    elif kind == "sum":
        if (s < 0).any():
            raise ValueError("'sum' needs non-negative scores")
        p = s.copy()
    else:
        raise ValueError(kind)
    p /= p.sum(1, keepdims=True)
    return np.concatenate([np.zeros((len(p), 1)), p], 1)


def top_with_prob(prob: np.ndarray, k: int = TOP) -> tuple[np.ndarray, np.ndarray]:
    """(b, k) model ids sorted by probability (ties to the lower id) and their probabilities."""
    p = prob.copy()
    p[:, 0] = -1.0  # never pick the padding column
    ids = np.argsort(-p, axis=1, kind="stable")[:, :k]
    return ids, np.take_along_axis(p, ids, 1)


@dataclass
class TwoStep:
    """Top-20 of both steps per user (model ids and probabilities)."""

    ids1: np.ndarray
    p1: np.ndarray
    ids2: np.ndarray
    p2: np.ndarray


def score_two_steps(
    view: RowView, plan: UserPlan, score_fn: ScoreFn, kind: str, batch: int = 8192, k: int = TOP
) -> TwoStep:
    """Run steps 1 and 2 for every user of `plan`."""

    def run(make_batch, n):
        ids = np.empty((n, k), np.int64)
        pr = np.empty((n, k), np.float64)
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            ids[sl], pr[sl] = top_with_prob(to_prob(_score(score_fn, make_batch(sl)), kind), k)
        return ids, pr

    n = len(plan.user)
    ids1, p1 = run(lambda sl: _batch(view, plan.first[sl]), n)
    b2 = step2_batch(view, plan, ids1[:, 0])
    ids2, p2 = run(
        lambda sl: Batch(b2.seq[sl], b2.seq_len[sl], b2.h_dest[sl], b2.dest[sl], b2.event[sl]), n
    )
    return TwoStep(ids1, p1, ids2, p2)


def _merge(ids1, v1, ids2, v2, n_cols: int, k: int = TOP) -> np.ndarray:
    """Union of two scored lists, a cluster in both keeps its max value; top-k, ties to lower id."""
    full = np.full((len(ids1), n_cols), -1.0)
    rows = np.arange(len(ids1))[:, None]
    full[rows, ids1] = v1
    full[rows, ids2] = np.maximum(full[rows, ids2], v2)
    return np.argsort(-full, axis=1, kind="stable")[:, :k]


def rerank_first(t: TwoStep, n_cols: int, k: int = TOP) -> np.ndarray:
    """Method 1: the top-20 of step 1, step 2 ignored."""
    return t.ids1[:, :k].copy()


def rerank_merge(t: TwoStep, n_cols: int, k: int = TOP) -> np.ndarray:
    """Method 2: both lists merged by probability."""
    return _merge(t.ids1, t.p1, t.ids2, t.p2, n_cols, k)


def rerank_sqrt_alpha(t: TwoStep, n_cols: int, k: int = TOP) -> np.ndarray:
    """Method 3: step-2 probabilities x mapped to sqrt(alpha * x), alpha = step-1 top-1 probability,
    then merged with step 1 by value."""
    alpha = t.p1[:, :1]
    return _merge(t.ids1, t.p1, t.ids2, np.sqrt(alpha * t.p2), n_cols, k)


RERANKERS = {"m1_first": rerank_first, "m2_merge": rerank_merge, "m3_sqrt_alpha": rerank_sqrt_alpha}
