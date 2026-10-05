"""User booking behaviour from the history, for the behaviour split of the hybrid weights.

Every feature is computed from the history that precedes the target (the model's input sequence),
never from the target, so it is known at query time.

Within-history repeat rates, over history positions 2..L (position 1 has nothing before it):

- ``dest_repeat``: share of bookings whose destination was already booked earlier in the history.
- ``cluster_repeat``: share of bookings whose cluster was already booked earlier (any destination);
  the behaviour split uses it as ``old_share`` ("rebooks more than it books new").
- ``pair_repeat``: share of bookings whose (destination, cluster) pair was already booked earlier,
  i.e. the user went back to the same destination and booked the same cluster again.

All three are 0 when L = 1.

``split_gain`` scores candidate thresholds as a one-split regression stump (reduction of the sum of
squared errors of an outcome), the evidence used to choose the cut on the train split.

Only numpy is used, so the module also imports in the RecBole venv (numpy 1.23).
"""

from __future__ import annotations

import numpy as np


def history_behaviour(seq: np.ndarray, length: np.ndarray, h_dest: np.ndarray) -> dict:
    """Within-history repeat rates of every row.

    Args:
        seq: (n, S) int item ids of the history, oldest first, padded with 0 on the right.
        length: (n,) int number of real bookings in each history (1..S).
        h_dest: (n, S) int destination of each history booking.

    Returns:
        {"dest_repeat", "cluster_repeat", "pair_repeat"}: (n,) float64 each, in [0, 1].
    """
    seq = np.asarray(seq)
    h_dest = np.asarray(h_dest)
    length = np.asarray(length, dtype=np.int64)
    n, steps = seq.shape
    dest = np.zeros(n)
    cluster = np.zeros(n)
    pair = np.zeros(n)
    for j in range(1, steps):
        valid = j < length
        same_d = h_dest[:, :j] == h_dest[:, [j]]
        same_c = seq[:, :j] == seq[:, [j]]
        dest += valid & same_d.any(1)
        cluster += valid & same_c.any(1)
        pair += valid & (same_d & same_c).any(1)
    denom = np.maximum(length - 1, 1)
    return {
        "dest_repeat": dest / denom,
        "cluster_repeat": cluster / denom,
        "pair_repeat": pair / denom,
    }


def split_gain(
    x: np.ndarray,
    y: np.ndarray,
    cuts: list[float],
    above_or_equal: bool = False,
    weight: np.ndarray | None = None,
) -> dict:
    """Reduction of the (weighted) sum of squared errors of y when rows are split at each cut of x.

    Args:
        x: (n,) feature.
        y: (n,) outcome (e.g. 0/1).
        cuts: candidate thresholds.
        above_or_equal: split as ``x >= cut`` instead of ``x > cut``.
        weight: optional (n,) row weights (e.g. bootstrap counts); default 1.

    Returns:
        {cut: {"gain", "share_high", "mean_low", "mean_high"}}; gain 0 when one side is empty.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)

    def sse(m):
        mu = np.average(y[m], weights=w[m])
        return float((w[m] * (y[m] - mu) ** 2).sum())

    everyone = w > 0
    total = sse(everyone)
    out = {}
    for c in cuts:
        high = (x >= c if above_or_equal else x > c) & everyone
        low = ~high & everyone
        share = float(w[high].sum() / w.sum())
        if not high.any() or not low.any():
            out[c] = {"gain": 0.0, "share_high": share, "mean_low": None, "mean_high": None}
            continue
        out[c] = {
            "gain": total - sse(high) - sse(low),
            "share_high": share,
            "mean_low": float(np.average(y[low], weights=w[low])),
            "mean_high": float(np.average(y[high], weights=w[high])),
        }
    return out
