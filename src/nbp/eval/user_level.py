"""User-level, set-valued Recall@K / NDCG@K (the teammate's `Metrics.ipynb` protocol).

One prediction per user, scored against the SET of clusters the user books inside the evaluation window
(not one next booking per row, as `nbp.eval.metrics` does). Per user:

    recall@k = |top_k ∩ truth| / |truth|
    ndcg@k   = sum_{hit at rank r <= k} 1/log2(r+1)  /  sum_{r=1}^{min(k, |truth|)} 1/log2(r+1)

and the reported value is the mean over users. With |truth| = 1 both reduce to the row-level formulas.
"""

from __future__ import annotations

import math

import numpy as np


def user_anchor_and_truth(
    users: np.ndarray, ts: np.ndarray, items: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[set]]:
    """One anchor event and one truth set per user.

    Args:
        users: (n,) user id of every event of the evaluation window.
        ts: (n,) event time (any sortable dtype); ties are broken by the lower event index.
        items: (n,) item booked by the event.

    Returns:
        (user_ids, anchor, truths): sorted distinct user ids (u,); (u,) index of each user's earliest event
        (the event whose prediction stands for the user); one set of distinct items per user, over ALL the
        user's events in the window.
    """
    users, ts, items = np.asarray(users), np.asarray(ts), np.asarray(items)
    order = np.lexsort((np.arange(len(users)), ts, users))  # by user, then time, then row index
    u_sorted = users[order]
    start = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
    end = np.r_[start[1:], len(order)]
    user_ids = u_sorted[start]
    anchor = order[start]
    truths = [set(items[order[s:e]].tolist()) for s, e in zip(start, end)]
    return user_ids, anchor, truths


def user_set_metrics(top: np.ndarray, truths: list[set], k: int = 5) -> dict:
    """Mean user-level Recall@k and NDCG@k.

    Args:
        top: (u, >= k) item ids ranked best first, one row per user.
        truths: u sets of the items each user booked in the window.
        k: cutoff.

    Returns:
        {"users": u, "recall@k": ..., "ndcg@k": ...} (unrounded).
    """
    disc = [1.0 / math.log2(r + 1) for r in range(1, k + 1)]
    rec = nd = 0.0
    for ranked, truth in zip(np.asarray(top)[:, :k].tolist(), truths):
        hits = [c in truth for c in ranked]
        rec += sum(hits) / len(truth)
        dcg = sum(d for d, h in zip(disc, hits) if h)
        nd += dcg / sum(disc[: min(k, len(truth))])
    n = len(truths)
    return {"users": n, f"recall@{k}": rec / n, f"ndcg@{k}": nd / n}
