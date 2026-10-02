"""Paired bootstrap confidence intervals for metric differences between two rankers."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    rank_a: np.ndarray, rank_b: np.ndarray, k: int = 5, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Mean difference (a - b) of Recall@k and NDCG@k with a 95% percentile interval.

    Args:
        rank_a, rank_b: (n,) ranks of the true item for the same rows under rankers a and b.
        k: cutoff.
        n_boot: bootstrap resamples (rows drawn with replacement, same rows for a and b).
        seed: RNG seed.

    Returns:
        {"recall@k": {"diff": d, "ci95": [lo, hi]}, "ndcg@k": {...}}, values rounded to 4 digits.
    """
    rank_a = np.asarray(rank_a, dtype=np.float64)
    rank_b = np.asarray(rank_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(rank_a)
    idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    out = {}
    for name, fn in (
        (f"recall@{k}", lambda r: (r <= k).astype(np.float64)),
        (f"ndcg@{k}", lambda r: (r <= k) / np.log2(r + 1)),
    ):
        d = fn(rank_a) - fn(rank_b)
        means = np.array([d[i].mean() for i in idx])
        out[name] = {
            "diff": round(float(d.mean()), 4),
            "ci95": [
                round(float(np.percentile(means, 2.5)), 4),
                round(float(np.percentile(means, 97.5)), 4),
            ],
        }
    return out
