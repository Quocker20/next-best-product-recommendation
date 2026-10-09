import numpy as np
import pytest

from nbp.models.early_stop import EarlyStopper, mean_nll


def test_stops_after_patience_and_keeps_best():
    st = EarlyStopper(patience=2, min_delta=0.001)
    flags = [st.update(e, s) for e, s in enumerate([0.40, 0.45, 0.4505, 0.449], start=1)]
    assert flags == [True, True, False, False]  # 0.4505 is within min_delta of 0.45
    assert st.best_epoch == 2 and st.should_stop


def test_improvement_resets_counter():
    st = EarlyStopper(patience=2, min_delta=0.0)
    for e, s in enumerate([0.1, 0.2, 0.15, 0.3], start=1):
        st.update(e, s)
    assert st.best_epoch == 4 and st.bad_epochs == 0 and not st.should_stop


def test_from_scores_matches_incremental():
    scores = [0.1, 0.3, 0.29, 0.31, 0.2]
    a = EarlyStopper(2, 0.005)
    for e, s in enumerate(scores, start=1):
        a.update(e, s)
    b = EarlyStopper.from_scores(scores, 2, 0.005)
    assert (a.best_epoch, a.bad_epochs, a.should_stop) == (
        b.best_epoch,
        b.bad_epochs,
        b.should_stop,
    )
    assert a.best_epoch == b.best_epoch == 4  # 0.31 beats 0.30 by 0.01 > min_delta
    assert b.bad_epochs == 1 and not b.should_stop


def test_non_finite_score_raises():
    with pytest.raises(ValueError):
        EarlyStopper(1).update(1, float("nan"))


def test_mean_nll_hand_computed():
    scores = np.array([[0.0, 0.0, 0.0, 0.0], [np.log(3.0), 0.0, -np.inf, 0.0]])
    # row 0 uniform over 4: nll = ln 4 ; row 1: probs 3/5, 1/5, 0, 1/5, true = 1 -> ln 5
    assert mean_nll(scores, np.array([2, 1])) == pytest.approx((np.log(4) + np.log(5)) / 2)
