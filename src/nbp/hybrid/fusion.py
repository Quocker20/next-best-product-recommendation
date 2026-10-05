"""SMLP4Rec + prior + sameDest hybrid: components on one smoothed log-probability scale, plus the selection rules.

Each component is a distribution over the K clusters, mixed with the uniform distribution
before the log (a "product of smoothed experts"):

    log q(k) = log((1 - alpha) * p(k) + alpha / K)

so every term lies in the same bounded range [log(alpha / K), log(1 - alpha + alpha / K)] and a
weight reads as relative trust. ``sameDest`` becomes the distribution of the user's recency-weighted
past bookings at the query destination; with no such booking it is uniform, a constant that does
not change the ranking (the term switches itself off on new-destination queries).

Selection helpers implement the one-standard-error rule (Breiman et al. 1984; Hastie, Tibshirani
and Friedman, ESL 2nd ed., section 7.10): among candidates whose score is within one standard
error of the best, take the simplest. Standard errors are clustered by user, because one user
contributes several rows.

Only numpy is used, so the module also imports in the RecBole venv (numpy 1.23).
"""

from __future__ import annotations

import numpy as np


def smoothed_log(p: np.ndarray, alpha: float) -> np.ndarray:
    """log((1 - alpha) * p + alpha / K) row-wise; p is (n, K) with rows summing to 1."""
    p = np.asarray(p, dtype=np.float64)
    k = p.shape[1]
    return np.log((1.0 - alpha) * p + alpha / k).astype(np.float32)


def same_dest_distribution(same_dest: np.ndarray) -> np.ndarray:
    """Row-normalized sameDest scores; uniform for rows without any same-destination booking.

    Args:
        same_dest: (n, K) non-negative scores (e.g. ``same_dest_scores`` restricted to the K clusters).

    Returns:
        (n, K) float64 distribution per row.
    """
    s = np.asarray(same_dest, dtype=np.float64)
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.maximum(tot, 1e-300), 1.0 / s.shape[1])


def cluster_se(x: np.ndarray, groups: np.ndarray) -> float:
    """Standard error of mean(x) with rows clustered by ``groups`` (e.g. user ids).

    Args:
        x: (n,) per-row values (e.g. a per-row metric difference between two rankers).
        groups: (n,) cluster label of each row.

    Returns:
        sqrt(G / (G - 1) * sum_g (sum_{i in g} (x_i - mean))^2) / n, G = number of clusters.
    """
    x = np.asarray(x, dtype=np.float64)
    _, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=x - x.mean())
    g = len(s)
    if g < 2:
        return 0.0
    return float(np.sqrt(g / (g - 1) * (s**2).sum()) / len(x))


def one_se_pick(scores: dict, groups: np.ndarray, complexity) -> tuple:
    """One-standard-error rule over candidates scored per row.

    Args:
        scores: {candidate: (n,) per-row score, higher is better}, same rows for every candidate.
        groups: (n,) cluster label of each row for the standard error.
        complexity: function candidate -> sortable key, smaller = simpler.

    Returns:
        (pick, best, eligible): the simplest candidate whose mean score is within one clustered
        standard error of the best candidate's (SE of the paired per-row difference), the best
        candidate, and the list of eligible candidates.
    """
    best = max(
        scores,
        key=lambda c: (float(np.mean(scores[c])), [-v for v in np.atleast_1d(complexity(c))]),
    )
    eligible = []
    for c, s in scores.items():
        d = np.asarray(s, dtype=np.float64) - scores[best]
        if d.mean() >= -cluster_se(d, groups):
            eligible.append(c)
    return min(eligible, key=complexity), best, eligible
