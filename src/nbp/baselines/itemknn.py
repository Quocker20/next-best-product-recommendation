"""ItemKNN baseline: item-item cosine similarity from the train window (`implicit`), history only.

Score of cluster j for a row = sum over the history items i (counts) of sim(i, j) with the K
nearest neighbours kept per item (`implicit.nearest_neighbours.CosineRecommender`). Rows with an
empty history (cold rows) get the first-booking popularity of the train window. Ties are broken by
train popularity: `tie_eps` x popularity share is added, far below any similarity step.
`implicit` keeps the item itself among its K neighbours (self-similarity 1), so K = k keeps k - 1
other items and every history item also votes for itself (a repeat-booking signal), as in the
earlier ItemKNN setup.
Scores use model ids (column 0 = padding, cluster c = column c + 1).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from implicit.nearest_neighbours import CosineRecommender

from nbp.eval.autoregressive import Batch, ScoreFn


def user_item_matrix(user: np.ndarray, y: np.ndarray, n_items: int) -> sp.csr_matrix:
    """(users, n_items) booking counts; `y` are model ids (column 0 stays empty)."""
    u = np.unique(user, return_inverse=True)[1]
    m = sp.csr_matrix(
        (np.ones(len(y), np.float32), (u, np.asarray(y, np.int64))), shape=(u.max() + 1, n_items)
    )
    m.sum_duplicates()
    return m


def fit_itemknn(ui: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """(n_items, n_items) item-item cosine similarity keeping the k nearest neighbours per item."""
    model = CosineRecommender(K=k)
    model.fit(ui, show_progress=False)
    return sp.csr_matrix(model.similarity)


def history_counts(seq: np.ndarray, seq_len: np.ndarray, n_items: int) -> sp.csr_matrix:
    """(b, n_items) counts of the real history slots of each row."""
    seq = np.asarray(seq)
    valid = np.arange(seq.shape[1])[None, :] < np.asarray(seq_len)[:, None]
    r, c = np.nonzero(valid)
    return sp.csr_matrix(
        (np.ones(len(r), np.float32), (r, seq[r, c].astype(np.int64))), shape=(len(seq), n_items)
    )


def itemknn_scorer(
    sim: sp.csr_matrix, pop_share: np.ndarray, pop_first_share: np.ndarray, tie_eps: float
) -> ScoreFn:
    """ScoreFn of ItemKNN.

    Args:
        sim: (n_items, n_items) similarity from `fit_itemknn`.
        pop_share: (n_items,) train-window booking share per model id (tie-break).
        pop_first_share: (n_items,) train-window first-booking share (score of cold rows).
        tie_eps: weight of the popularity tie-break.
    """
    n_items = sim.shape[0]

    def score(b: Batch) -> np.ndarray:
        s = (history_counts(b.seq, b.seq_len, n_items) @ sim).toarray().astype(np.float64)
        s += tie_eps * pop_share[None, :]
        cold = np.asarray(b.seq_len) == 0
        s[cold] = pop_first_share[None, :]
        return s

    return score
