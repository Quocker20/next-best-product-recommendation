"""sameDest: how strongly the user's history says "rebook this cluster at this destination".

For one query (destination q) and every cluster k, the score adds a weight for each past booking
of cluster k that was made at destination q. The weight of a past booking depends on its age:

    weight = base ** age,   age = L - 1 - position   (age 0 = most recent booking of the L in the history)

With ``base = 0.7`` recent bookings count more. ``normalize=True`` divides by the sum of the
weights of ALL bookings in the history (same destination or not), so the score is the share of the
history's weight that falls on cluster k at the query destination, in [0, 1] for any L.

Only numpy is used, so the module also imports in the RecBole venv (numpy 1.23).
"""

from __future__ import annotations

import numpy as np


def same_dest_scores(
    seq: np.ndarray,
    length: np.ndarray,
    h_dest: np.ndarray,
    q_dest: np.ndarray,
    n_items: int,
    base: float = 0.7,
    normalize: bool = False,
) -> np.ndarray:
    """Recency-weighted same-destination history score for every (row, cluster).

    Args:
        seq: (n, S) int model item ids of the history, oldest first, padded with 0 on the right.
        length: (n,) int number of real bookings in each history (1..S).
        h_dest: (n, S) int destination of each history booking.
        q_dest: (n,) int destination of the current query.
        n_items: number of columns of the output (model item ids, padding id 0 included).
        base: weight of a booking is ``base ** age`` (age 0 = most recent).
        normalize: divide by the sum of ``base ** age`` over the whole history of that row.

    Returns:
        (n, n_items) float32; 0 for clusters never booked at the query destination.
    """
    seq = np.asarray(seq)
    length = np.asarray(length, dtype=np.int64)
    n, steps = seq.shape
    out = np.zeros((n, n_items), np.float64)
    total = np.zeros(n, np.float64)
    for j in range(steps):
        valid = j < length
        weight = np.where(valid, float(base) ** np.where(valid, length - 1 - j, 0), 0.0)
        total += weight
        match = valid & (h_dest[:, j] == q_dest)
        idx = np.flatnonzero(match)
        out[idx, seq[idx, j]] += weight[idx]
    if normalize:
        out /= np.maximum(total, 1e-12)[:, None]
    return out.astype(np.float32)
