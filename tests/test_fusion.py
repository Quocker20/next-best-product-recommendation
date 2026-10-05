"""Hybrid fusion helpers against hand-computed examples."""

import numpy as np
import pytest

from nbp.hybrid.fusion import cluster_se, one_se_pick, same_dest_distribution, smoothed_log


def test_smoothed_log_bounds_and_values():
    p = np.array([[1.0, 0.0, 0.0, 0.0]])
    q = smoothed_log(p, alpha=0.2)[0]
    assert q[0] == pytest.approx(np.log(0.8 + 0.05))
    assert q[1] == pytest.approx(np.log(0.05))
    # alpha = 1 -> uniform, a constant row
    assert np.ptp(smoothed_log(np.array([[0.7, 0.2, 0.1]]), alpha=1.0)) == pytest.approx(0.0)


def test_same_dest_distribution():
    s = np.array([[0.0, 0.3, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]])
    d = same_dest_distribution(s)
    assert d[0] == pytest.approx([0, 0.75, 0.25, 0])
    assert d[1] == pytest.approx([0.25] * 4)  # no information -> uniform, no effect on the ranking


def test_cluster_se_matches_iid_when_one_row_per_group():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    se = cluster_se(x, np.arange(4))
    # sqrt(G/(G-1) * sum (x - mean)^2) / n = sqrt(4/3 * 0.75) / 4 = 0.25
    assert se == pytest.approx(0.25)


def test_cluster_se_grows_with_within_group_correlation():
    x = np.array([1.0, 1.0, 0.0, 0.0])
    assert cluster_se(x, np.array([0, 0, 1, 1])) > cluster_se(x, np.arange(4))


def test_one_se_pick_prefers_simpler_within_noise():
    groups = np.arange(6)
    scores = {
        "complex": np.array([1, 1, 1, 1, 1, 0], float),  # mean 0.833
        "simple": np.array([1, 1, 1, 1, 0, 0], float),  # mean 0.667, diff -0.167, SE of diff 0.167
        "bad": np.zeros(6),
    }
    order = {"simple": 0, "complex": 1, "bad": 2}
    pick, best, eligible = one_se_pick(scores, groups, complexity=lambda c: order[c])
    assert best == "complex"
    assert set(eligible) == {"complex", "simple"}
    assert pick == "simple"
