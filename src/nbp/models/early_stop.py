"""Early stopping on a validation score, plus the validation NLL used as an overfitting signal.

The stopper looks only at the monitored validation score (higher is better), never at test or
train loss. An epoch counts as an improvement only if it beats the best score by more than
`min_delta`; ties and noise-level gains do not move the best epoch, so the earlier, less-fitted
epoch is kept. State is rebuilt from the list of past scores, so resuming a run is exact.
"""

from __future__ import annotations

import numpy as np


class EarlyStopper:
    """Track the best epoch of a higher-is-better validation score.

    Args:
        patience: stop after this many consecutive epochs without an improvement > `min_delta`.
        min_delta: minimum gain over the best score that counts as an improvement (>= 0).
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1 or min_delta < 0:
            raise ValueError("patience must be >= 1 and min_delta >= 0")
        self.patience, self.min_delta = patience, min_delta
        self.best_epoch: int | None = None
        self.best_score = float("-inf")
        self.bad_epochs = 0
        self.last_epoch = 0

    def update(self, epoch: int, score: float) -> bool:
        """Register one epoch's validation score; return True if it is the new best."""
        if not np.isfinite(score):
            raise ValueError(f"non-finite validation score at epoch {epoch}: {score}")
        self.last_epoch = epoch
        if self.best_epoch is None or score > self.best_score + self.min_delta:
            self.best_epoch, self.best_score, self.bad_epochs = epoch, score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    @classmethod
    def from_scores(
        cls, scores: list[float], patience: int, min_delta: float = 0.0
    ) -> EarlyStopper:
        """Rebuild the state after epochs 1..len(scores) (used when resuming a run)."""
        stopper = cls(patience, min_delta)
        for epoch, score in enumerate(scores, start=1):
            stopper.update(epoch, score)
        return stopper


def nll_guard(valid_nll: list[float], train_loss: list[float], n: int) -> bool:
    """True if, over the last `n` epochs, the valid NLL rose every epoch while the train loss fell.

    Args:
        valid_nll: per-epoch validation NLL (epoch 1 first).
        train_loss: per-epoch mean training loss, same length.
        n: number of consecutive epoch-to-epoch steps that must show the pattern (>= 1).
    Returns:
        False while fewer than n + 1 epochs exist.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(valid_nll) != len(train_loss):
        raise ValueError("valid_nll and train_loss must have the same length")
    if len(valid_nll) < n + 1:
        return False
    v, t = np.asarray(valid_nll[-(n + 1) :]), np.asarray(train_loss[-(n + 1) :])
    return bool((np.diff(v) > 0).all() and (np.diff(t) < 0).all())


def mean_nll(scores: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the true item under softmax(scores).

    Args:
        scores: (n, n_items) logits; columns set to -inf (padding) get probability 0.
        y: (n,) true item index per row.
    Returns:
        Mean cross-entropy over rows. Unlike NDCG it keeps rising when the model becomes
        over-confident, so a valid NLL that turns up while train loss falls is the overfit signal.
    """
    s = scores.astype(np.float64)
    m = s.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(s - m).sum(axis=1))
    return float((lse - s[np.arange(len(y)), y]).mean())
