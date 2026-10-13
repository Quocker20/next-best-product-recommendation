"""Paired bootstrap confidence intervals for metric differences between two rankers."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    k: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict:
    """Mean difference (a - b) of Recall@k and NDCG@k with a 95% percentile interval.

    Args:
        rank_a, rank_b: (n,) ranks of the true item for the same rows under rankers a and b.
        k: cutoff.
        n_boot: bootstrap resamples (same resample for a and b).
        seed: RNG seed.
        groups: optional (n,) cluster label per row (e.g. user id). None = rows are drawn with
            replacement; otherwise whole groups are drawn with replacement (cluster bootstrap),
            which is needed when rows of one group are dependent (autoregressive evaluation).

    Returns:
        {"recall@k": {"diff": d, "ci95": [lo, hi]}, "ndcg@k": {...}}, values rounded to 4 digits.
    """
    rank_a = np.asarray(rank_a, dtype=np.float64)
    rank_b = np.asarray(rank_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(rank_a)
    if groups is None:
        inv, n_groups = np.arange(n), n
    else:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        n_groups = int(inv.max()) + 1
    size = np.bincount(inv, minlength=n_groups).astype(np.float64)
    fns = (
        (f"recall@{k}", lambda r: (r <= k).astype(np.float64)),
        (f"ndcg@{k}", lambda r: (r <= k) / np.log2(r + 1)),
    )
    diffs = [fn(rank_a) - fn(rank_b) for _, fn in fns]
    tots = [np.bincount(inv, weights=d, minlength=n_groups) for d in diffs]
    means = np.empty((len(fns), n_boot))
    for b in range(n_boot):  # one draw at a time: memory stays O(n_groups)
        i = rng.integers(0, n_groups, n_groups)
        denom = size[i].sum()
        for j, tot in enumerate(tots):
            means[j, b] = tot[i].sum() / denom
    out = {}
    for j, (name, _) in enumerate(fns):
        out[name] = {
            "diff": round(float(diffs[j].mean()), 4),
            "ci95": [
                round(float(np.percentile(means[j], 2.5)), 4),
                round(float(np.percentile(means[j], 97.5)), 4),
            ],
        }
    return out
