"""Model construction, training losses and `ScoreFn` adapters for the SMLP4Rec variants (SMLP4Rec venv).

Variants:
- plain: history-only SMLP4Rec (`nbp.models.smlprec` (fetched, see README));
- query embedding: `SMLPRECQuery`, the current search as one extra position;
- query embedding + sameDest: post-hoc fusion
  ``score = log q_model + w * log q_sameDest``, ``log q = log((1 - alpha) * p + alpha / K)``
  over the K = 100 clusters, sameDest = recency-weighted share of the history booked at the
  searched destination (`nbp.hybrid.same_dest`, normalised). For a destination new to the user
  the sameDest distribution is uniform, so the fusion leaves that row's ranking unchanged.

Scores have one column per model id (0 = padding, set to -inf by the evaluator).
"""

from __future__ import annotations

import numpy as np
import torch

from nbp.eval.autoregressive import Batch, ScoreFn
from nbp.hybrid.fusion import same_dest_distribution, smoothed_log
from nbp.hybrid.same_dest import same_dest_scores
from nbp.models.smlprec_query import SMLPRECQuery
from nbp.models.train import StubDataset
from nbp.models.smlprec import SMLPREC


def model_config(yaml_cfg: dict, overrides: dict | None = None) -> dict:
    """RecBole-style config dict for SMLPREC / SMLPRECQuery from configs/smlprec_expedia.yaml."""
    cfg = dict(yaml_cfg)
    cfg.update(overrides or {})
    return cfg


def build_plain(cfg: dict, n_items: int) -> SMLPREC:
    torch.manual_seed(cfg["seed"])
    return SMLPREC(cfg, StubDataset(n_items))


def build_query(
    cfg: dict, n_items: int, q_idx: torch.Tensor, field_sizes: list[int]
) -> SMLPRECQuery:
    torch.manual_seed(cfg["seed"])
    return SMLPRECQuery(cfg, StubDataset(n_items), q_idx, field_sizes)


def _t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).long()


def plain_loss(seq: np.ndarray, seq_len: np.ndarray, y: np.ndarray):
    """Loss on training rows for the plain model (rows must have seq_len >= 1)."""
    if not (np.asarray(seq_len) >= 1).all():
        raise ValueError("the plain model trains on rows with a history (seq_len >= 1) only")
    seq_t, len_t, y_t = _t(seq), _t(seq_len), _t(y)

    def loss(model: SMLPREC, idx: torch.Tensor) -> torch.Tensor:
        out = model.forward(seq_t[idx], len_t[idx])
        return model.loss_fct(out @ model.item_embedding.weight.T, y_t[idx])

    return loss


def query_loss(seq: np.ndarray, event: np.ndarray, y: np.ndarray):
    """Loss on training rows for the query-embedding model (empty histories allowed)."""
    seq_t, ev_t, y_t = _t(seq), _t(event), _t(y)

    def loss(model: SMLPRECQuery, idx: torch.Tensor) -> torch.Tensor:
        logits = model.scores(seq_t[idx], model.query_idx[ev_t[idx]].long())
        return model.loss_fct(logits, y_t[idx])

    return loss


def plain_scorer(model: SMLPREC) -> ScoreFn:
    """History-only scores. An empty history (cold row) reads position 0 (padding): one constant
    ranking for every cold row, out of the training distribution of the plain model."""

    @torch.no_grad()
    def score(b: Batch) -> np.ndarray:
        seq = _t(b.seq)
        out = model.forward(seq, torch.clamp(_t(b.seq_len), min=1))
        return (out @ model.item_embedding.weight.T).numpy()

    return score


def query_scorer(model: SMLPRECQuery) -> ScoreFn:
    """Query-embedding scores; the query codes are looked up by the row's event index."""

    @torch.no_grad()
    def score(b: Batch) -> np.ndarray:
        return model.scores(_t(b.seq), model.query_idx[_t(b.event)].long()).numpy()

    return score


def samedest_fuse(
    base_scores: np.ndarray, b: Batch, alpha: float, w: float, recency: float
) -> np.ndarray:
    """(b, n_cols) fused scores: log q_model + w * log q_sameDest, column 0 = -inf.

    Args:
        base_scores: (b, n_cols) logits of the base model (column 0 = padding).
        b: batch (history, destinations).
        alpha: smoothing of both distributions toward uniform over the clusters.
        w: weight of the sameDest term.
        recency: sameDest weight of a history booking = recency ** lag.
    """
    s = np.asarray(base_scores, dtype=np.float64)[:, 1:]
    s = s - s.max(1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(1, keepdims=True)
    sd = same_dest_scores(
        b.seq, b.seq_len, b.h_dest, b.dest, base_scores.shape[1], base=recency, normalize=True
    )
    out = smoothed_log(p, alpha) + w * smoothed_log(same_dest_distribution(sd[:, 1:]), alpha)
    return np.concatenate([np.full((len(out), 1), -np.inf, np.float32), out.astype(np.float32)], 1)


def samedest_scorer(base: ScoreFn, alpha: float, w: float, recency: float) -> ScoreFn:
    def score(b: Batch) -> np.ndarray:
        return samedest_fuse(base(b), b, alpha, w, recency)

    return score
