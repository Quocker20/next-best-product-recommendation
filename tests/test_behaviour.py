"""History behaviour rates and the stump split gain against hand-computed examples."""

import numpy as np
import pytest

from nbp.hybrid.behaviour import history_behaviour, split_gain

# rows, history oldest -> newest as (cluster, destination):
# 0: (5,100)                         L=1
# 1: (5,100) (6,200) (7,300)         new destination every time
# 2: (5,100) (6,100) (7,200)         back to 100 once, another cluster
# 3: (5,100) (6,200) (5,100) (5,100) same pair twice more; cluster 5 repeated twice
# 4: (5,100) (5,200)                 same cluster, other destination
SEQ = np.array([[5, 0, 0, 0], [5, 6, 7, 0], [5, 6, 7, 0], [5, 6, 5, 5], [5, 5, 0, 0]])
H_DEST = np.array(
    [[100, 0, 0, 0], [100, 200, 300, 0], [100, 100, 200, 0], [100, 200, 100, 100], [100, 200, 0, 0]]
)
LEN = np.array([1, 3, 3, 4, 2])


def test_rates():
    r = history_behaviour(SEQ, LEN, H_DEST)
    assert r["dest_repeat"] == pytest.approx([0, 0, 0.5, 2 / 3, 0])
    assert r["cluster_repeat"] == pytest.approx([0, 0, 0, 2 / 3, 1])
    assert r["pair_repeat"] == pytest.approx([0, 0, 0, 2 / 3, 0])


def test_padding_is_ignored():
    # padding positions share id 0 and destination 0: they must not count as repeats
    r = history_behaviour(np.array([[5, 0, 0]]), np.array([1]), np.array([[100, 0, 0]]))
    assert r["dest_repeat"][0] == 0 and r["pair_repeat"][0] == 0


def test_split_gain_hand_example():
    x = np.array([0.0, 0.0, 0.5, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    g = split_gain(x, y, [0.0, 0.5, 1.0])
    # total SSE = 4 * 0.25 = 1; split at > 0 separates perfectly
    assert g[0.0]["gain"] == pytest.approx(1.0)
    assert g[0.0]["mean_low"] == 0.0 and g[0.0]["mean_high"] == 1.0
    # > 0.5: high = {1.0}; low y = [0, 0, 1] SSE = 2/3
    assert g[0.5]["gain"] == pytest.approx(1 - 2 / 3)
    assert g[1.0]["gain"] == 0.0  # nothing above 1
    ge = split_gain(x, y, [1.0], above_or_equal=True)
    assert ge[1.0]["share_high"] == 0.25


def test_split_gain_weights_equal_duplicated_rows():
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 1.0])
    gw = split_gain(x, y, [0.0, 0.5], weight=np.array([2, 1, 0]))
    gd = split_gain(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), [0.0, 0.5])
    for c in (0.0, 0.5):
        assert gw[c]["gain"] == pytest.approx(gd[c]["gain"])
        assert gw[c]["share_high"] == pytest.approx(gd[c]["share_high"])
