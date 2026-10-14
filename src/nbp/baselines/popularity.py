"""Global popularity baseline: every row gets the same ranking, the train-window booking share.

No history and no query are read, so the ranking is identical for every row, for cold and warm
rows alike, and under both test protocols. Scores use model ids (column 0 = padding, share 0;
cluster c = column c + 1). Ties are broken by lower column index in `nbp.eval.metrics.target_rank`.
"""

from __future__ import annotations

import numpy as np

from nbp.eval.autoregressive import Batch, ScoreFn


def fit_popularity(y_train: np.ndarray, n_items: int) -> np.ndarray:
    """(n_items,) share of each model id among the train-window bookings (column 0 stays 0).

    Args:
        y_train: (n,) model ids of the train-window bookings.
        n_items: number of model ids including the padding id 0.
    """
    counts = np.bincount(np.asarray(y_train, np.int64), minlength=n_items).astype(np.float64)
    return counts / counts.sum()


def popularity_scorer(share: np.ndarray) -> ScoreFn:
    """ScoreFn that returns `share` for every row of the batch."""

    def score(b: Batch) -> np.ndarray:
        return np.tile(share[None, :], (len(b.event), 1))

    return score
