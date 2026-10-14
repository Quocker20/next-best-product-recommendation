"""Two-step user-level protocol: step rows, step-2 history, probabilities and the three rerankers."""

import numpy as np

from nbp.eval.two_step import (
    RERANKERS,
    TwoStep,
    score_two_steps,
    step2_batch,
    to_prob,
    top_with_prob,
    user_plan,
)
from tests.test_autoregressive import DEST, ITEM, N_COLS, USER, WIN, _rows, _view


def _setup(max_len=4):
    r = _rows(USER, ITEM, DEST, WIN, max_len=max_len)
    v = _view(r)
    test = r["window"] == 2
    return r, v, np.array(USER)[test], r["y"][test]


def test_plan_rows_and_truth_sets():
    r, v, user, y = _setup()
    plan = user_plan(v, user, y)
    assert plan.user.tolist() == [1, 2, 3]
    assert v.event[plan.first].tolist() == [1, 3, 2]
    assert plan.second[1] == -1 and v.event[plan.second[[0, 2]]].tolist() == [4, 5]
    # model id = item + 1; truth = distinct clusters over ALL test bookings of the user
    assert plan.truths == [{2, 3, 4}, {5}, {6, 7, 3}]


def test_step2_history():
    r, v, user, y = _setup(max_len=4)
    plan = user_plan(v, user, y)
    top1 = np.array([7, 6, 5])
    b = step2_batch(v, plan, top1)
    # user 1: row of booking 2 (event 4); history = train item (id 1) + booking-1 slot -> top-1 7
    assert b.event[0] == 4 and b.seq[0, : b.seq_len[0]].tolist() == [1, 7]
    # user 2 (one test booking, cold): context of booking 1, history = [top-1]
    assert b.event[1] == 3 and b.dest[1] == 8
    assert b.seq_len[1] == 1 and b.seq[1, 0] == 6 and b.h_dest[1, 0] == 8
    # user 3: booking 2 (event 5), cold booking 1 -> history = [top-1]
    assert b.event[2] == 5 and b.seq[2, : b.seq_len[2]].tolist() == [5]


def test_step2_single_booking_full_history_drops_oldest():
    user = [1, 1, 1, 1]
    item = [0, 1, 2, 3]
    r = _rows(user, item, [9] * 4, [0, 0, 0, 2], max_len=3)
    v = _view(r)
    test = r["window"] == 2
    plan = user_plan(v, np.array(user)[test], r["y"][test])
    b = step2_batch(v, plan, np.array([7]))
    assert b.seq_len[0] == 3 and b.seq[0].tolist() == [2, 3, 7]


def test_probabilities():
    s = np.array([[-np.inf, 1.0, 2.0, 0.0]])
    p = to_prob(s, "softmax")
    assert p[0, 0] == 0 and abs(p.sum() - 1) < 1e-12 and np.argmax(p) == 2
    q = to_prob(np.array([[-np.inf, 1.0, 3.0, 0.0]]), "sum")
    assert np.allclose(q, [[0, 0.25, 0.75, 0]])
    ids, pr = top_with_prob(q, 2)
    assert ids.tolist() == [[2, 1]] and np.allclose(pr, [[0.75, 0.25]])


def test_rerankers_hand_computed():
    t = TwoStep(
        ids1=np.array([[1, 2, 3]]), p1=np.array([[0.5, 0.3, 0.1]]),
        ids2=np.array([[4, 2, 5]]), p2=np.array([[0.6, 0.35, 0.02]]),
    )
    assert RERANKERS["m1_first"](t, N_COLS, 3).tolist() == [[1, 2, 3]]
    # max per cluster: 4:.6, 1:.5, 2:.35, 3:.1, 5:.02
    assert RERANKERS["m2_merge"](t, N_COLS, 3).tolist() == [[4, 1, 2]]
    # sqrt(.5*x): 4:.548, 2:.418 (> .3), 5:.1 (ties 3:.1 -> lower id 3 first)
    assert RERANKERS["m3_sqrt_alpha"](t, N_COLS, 4).tolist() == [[4, 1, 2, 3]]


def test_constant_scorer_all_methods_equal():
    r, v, user, y = _setup()
    plan = user_plan(v, user, y)
    share = np.r_[0.0, np.arange(1, N_COLS)].astype(float)
    t = score_two_steps(v, plan, lambda b: np.tile(share, (len(b.event), 1)), "sum", k=3)
    out = [f(t, N_COLS, 3) for f in RERANKERS.values()]
    assert all(np.array_equal(out[0], o) for o in out) and out[0][0].tolist() == [7, 6, 5]
