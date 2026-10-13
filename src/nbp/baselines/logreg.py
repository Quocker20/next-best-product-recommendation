"""Logistic-regression baseline: multinomial over the 100 clusters, one-hot query fields, no history.

Features are the same 14 query fields the query-embedding model reads (`rows.npz` `codes`, built by
`nbp.data.query_features.build_query_codes`), one-hot encoded on the train window (unseen values
are ignored). Training is SAGA, one pass per `fit` call with `warm_start`, so passes act as epochs
and get the same early stopping as the neural models.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from nbp.eval.autoregressive import Batch, ScoreFn
from nbp.models.early_stop import EarlyStopper, mean_nll, nll_guard


def fit_encoder(codes_train: np.ndarray) -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", dtype=np.float64).fit(codes_train)


def fit_logreg(
    X,
    y: np.ndarray,
    c: float,
    seed: int,
    max_passes: int,
    patience: int,
    min_delta: float,
    nll_guard_passes: int,
    eval_fn: Callable[[LogisticRegression], dict],
    train_loss_fn: Callable[[LogisticRegression], float],
    log: Callable[[str], None] = print,
) -> tuple[LogisticRegression, list, EarlyStopper, str]:
    """SAGA passes with early stopping on `eval_fn(model)["monitor"]`.

    Args:
        X: (n, d) sparse one-hot train features; y: (n,) model ids of the train rows.
        c: inverse L2 strength; seed: solver random state.
        eval_fn: model -> {"monitor", "valid_nll", ...} on the valid window.
        train_loss_fn: model -> mean train log loss on a fixed probe sample (for the NLL guard).
    Returns:
        (model with the best pass's weights, per-pass rows, stopper, stop reason).
    """
    m = LogisticRegression(
        solver="saga", C=c, max_iter=1, tol=0.0, warm_start=True, random_state=seed
    )
    stopper = EarlyStopper(patience, min_delta)
    rows, best = [], None
    reason = "max_epochs"
    for p in range(1, max_passes + 1):
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        ev = eval_fn(m)
        row = {
            "epoch": p,
            "train_loss": train_loss_fn(m),
            "seconds": round(time.time() - t, 1),
            **ev,
        }
        if stopper.update(p, float(ev["monitor"])):
            best = (m.coef_.copy(), m.intercept_.copy())
        rows.append(row)
        log(
            f"C={c} seed={seed} pass {p}: valid monitor {ev['monitor']:.4f} nll {ev['valid_nll']:.4f} "
            f"| best {stopper.best_epoch}, no gain {stopper.bad_epochs}/{patience}"
        )
        if stopper.should_stop:
            reason = "patience"
            break
        if nll_guard(
            [r["valid_nll"] for r in rows], [r["train_loss"] for r in rows], nll_guard_passes
        ):
            reason = "nll_guard"
            break
    for r in rows:
        r["is_best"] = r["epoch"] == stopper.best_epoch
    m.coef_, m.intercept_ = best
    return m, rows, stopper, reason


def logreg_proba(model: LogisticRegression, X, n_items: int) -> np.ndarray:
    """(n, n_items) class probabilities in model-id columns (column 0 and unseen classes = 0)."""
    p = np.zeros((X.shape[0], n_items))
    p[:, model.classes_] = model.predict_proba(X)
    return p


def train_log_loss(model: LogisticRegression, X, y: np.ndarray, n_items: int) -> float:
    p = logreg_proba(model, X, n_items)
    return mean_nll(np.log(p[:, 1:] + 1e-300), np.asarray(y) - 1)


def logreg_scorer(
    model: LogisticRegression, enc: OneHotEncoder, codes: np.ndarray, n_items: int
) -> ScoreFn:
    """ScoreFn: log class probability of the row's query (looked up by event index); no history."""

    def score(b: Batch) -> np.ndarray:
        p = logreg_proba(model, enc.transform(codes[b.event]), n_items)
        return np.log(p + 1e-300)

    return score
