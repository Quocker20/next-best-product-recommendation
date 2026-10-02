"""Ranking metrics for next-item recommendation with one relevant item per row.

Every row has exactly one true item, so everything is computed from the rank of that item:

    Recall@K = 1[rank <= K]
    NDCG@K   = 1[rank <= K] / log2(rank + 1)      (ideal DCG = 1)

Metrics are means over rows. Only numpy is used, so the module also imports in the
RecBole venv (numpy 1.23).
"""

from __future__ import annotations

import numpy as np

KS: tuple[int, ...] = (5, 10, 20)


def target_rank(scores: np.ndarray, y: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Rank (1 = best) of the true item in each row of a full score matrix.

    Args:
        scores: (n, n_items) float scores; excluded columns (e.g. padding) set to -inf.
        y: (n,) int column index of the true item.
        chunk: rows per block (bounds memory).

    Returns:
        (n,) int64 ranks. Ties are broken by lower column index first (deterministic).
    """
    scores = np.asarray(scores)
    y = np.asarray(y, dtype=np.int64)
    idx = np.arange(scores.shape[1])[None, :]
    out = np.empty(len(y), np.int64)
    for i in range(0, len(y), chunk):
        s, t = scores[i : i + chunk], y[i : i + chunk]
        st = s[np.arange(len(t)), t][:, None]
        out[i : i + chunk] = 1 + (s > st).sum(1) + ((s == st) & (idx < t[:, None])).sum(1)
    return out


def rank_in_list(top: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank of the true item inside a fixed top-K list.

    Args:
        top: (n, K) item ids, best first.
        y: (n,) true item ids.

    Returns:
        (n,) int64 ranks in 1..K, or K + 1 when the true item is not in the list. Metrics
        computed from these ranks are valid only for cutoffs <= K.
    """
    top = np.asarray(top)
    pos = top == np.asarray(y)[:, None]
    return np.where(pos.any(1), pos.argmax(1) + 1, top.shape[1] + 1).astype(np.int64)


def recall_at_k(rank: np.ndarray, k: int) -> float:
    """Mean of 1[rank <= k]."""
    return float((np.asarray(rank) <= k).mean())


def ndcg_at_k(rank: np.ndarray, k: int) -> float:
    """Mean of 1[rank <= k] / log2(rank + 1)."""
    rank = np.asarray(rank, dtype=np.float64)
    return float(((rank <= k) / np.log2(rank + 1)).mean())


def summarize(
    rank: np.ndarray, mask: np.ndarray | None = None, ks: tuple[int, ...] = KS, digits: int = 4
) -> dict:
    """Recall@K and NDCG@K for each K, plus the row count.

    Args:
        rank: (n,) ranks from target_rank or rank_in_list.
        mask: optional (n,) bool row filter (a slice).
        ks: cutoffs.
        digits: rounding of the returned values.

    Returns:
        {"n": rows, "recall@K": ..., "ndcg@K": ...} for every K in ks.
    """
    rank = np.asarray(rank)
    if mask is not None:
        rank = rank[mask]
    out: dict = {"n": len(rank)}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(rank, k), digits)
    for k in ks:
        out[f"ndcg@{k}"] = round(ndcg_at_k(rank, k), digits)
    return out
