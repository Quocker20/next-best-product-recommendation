"""Shared evaluation steps of the week-6 notebooks (numpy only, both venvs).

- `load_rows`: rows.npz as a dict of arrays;
- `window_views`: one `RowView` per evaluation window (the whole window, as AR requires);
- `evaluate`: ranks of one scorer on valid and test under both protocols, plus the AR top-K list;
- `valid_eval_fn`: the per-epoch evaluation used by early stopping (AR valid NDCG@10 monitor,
  AR valid NLL, TF valid NDCG@10 and the train-probe NDCG@10 for the generalisation gap).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nbp.data.split import TEST, TRAIN, VALID
from nbp.eval.autoregressive import RowView, ScoreFn, score_view
from nbp.eval.metrics import ndcg_at_k, target_rank
from nbp.models.early_stop import mean_nll

WINDOWS = {"valid": VALID, "test": TEST}


def load_rows(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def window_views(rows: dict) -> dict[str, RowView]:
    return {name: RowView.from_rows(rows, rows["window"] == w) for name, w in WINDOWS.items()}


def window_index(rows: dict) -> dict[str, np.ndarray]:
    """Positions (into `rows`) of each evaluation window, in the row order of `window_views`.

    Equal to the global event index on the full data; on a user subsample use `rows["event"][pos]`.
    """
    return {name: np.flatnonzero(rows["window"] == w) for name, w in WINDOWS.items()}


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """(n, k) model ids sorted by score, ties to the lower id (same rule as `target_rank`)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    return order[:, :k]


def evaluate(
    score_fn: ScoreFn,
    views: dict[str, RowView],
    y: dict[str, np.ndarray],
    batch: int,
    keep_top: int = 0,
) -> dict:
    """Ranks of the true item per window and protocol.

    Args:
        score_fn: the model's scorer.
        views: {"valid": RowView, "test": RowView}.
        y: true model ids per window, aligned with the views.
        batch: rows per scorer call.
        keep_top: if > 0, also return the AR top-`keep_top` model ids per window.
    Returns:
        {"<window>_<ar|tf>": ranks (n,) int32, "<window>_ar_top": (n, keep_top) int16,
         "<window>_ar_replaced": (n,) int16 history slots that held a prediction}.
    """
    out = {}
    for name, v in views.items():
        for proto, ar in (("ar", True), ("tf", False)):
            sc = score_view(v, score_fn, autoregressive=ar, batch=batch)
            out[f"{name}_{proto}"] = target_rank(sc.scores, y[name]).astype(np.int32)
            if ar:
                out[f"{name}_ar_replaced"] = sc.n_replaced.astype(np.int16)
                if keep_top:
                    out[f"{name}_ar_top"] = top_k(sc.scores, keep_top).astype(np.int16)
    return out


def train_probe_view(rows: dict, n: int, seed: int) -> tuple[RowView, np.ndarray]:
    """A fixed random sample of train-window rows (TF only) for the generalisation gap."""
    idx = np.flatnonzero(rows["window"] == TRAIN)
    pick = np.sort(np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False))
    mask = np.zeros(len(rows["window"]), bool)
    mask[pick] = True
    return RowView.from_rows(rows, mask), rows["y"][pick]


def valid_eval_fn(
    make_scorer,
    valid: RowView,
    y_valid: np.ndarray,
    probe: RowView,
    y_probe: np.ndarray,
    k_monitor: int,
    batch: int,
):
    """Build the per-epoch `eval_fn(model)` for `nbp.models.train.fit`.

    `make_scorer(model)` returns the model's ScoreFn. The monitor is AR valid NDCG@k_monitor; the
    last AR valid ranks are kept on the returned function as `.last_ranks` (min_delta calibration).
    """

    def eval_fn(model) -> dict:
        sf = make_scorer(model)
        ar = score_view(valid, sf, autoregressive=True, batch=batch)
        r_ar = target_rank(ar.scores, y_valid)
        r_tf = target_rank(score_view(valid, sf, autoregressive=False, batch=batch).scores, y_valid)
        r_pr = target_rank(score_view(probe, sf, autoregressive=False, batch=batch).scores, y_probe)
        eval_fn.last_ranks = r_ar
        tf_nd = ndcg_at_k(r_tf, k_monitor)
        pr_nd = ndcg_at_k(r_pr, k_monitor)
        return {
            "monitor": ndcg_at_k(r_ar, k_monitor),
            "valid_nll": mean_nll(ar.scores[:, 1:], y_valid - 1),
            f"valid_tf_ndcg@{k_monitor}": tf_nd,
            f"train_probe_tf_ndcg@{k_monitor}": pr_nd,
            "generalisation_gap": pr_nd - tf_nd,
        }

    eval_fn.last_ranks = None
    return eval_fn


def subsample_users(rows: dict, frac: float, seed: int) -> dict:
    """Keep every row of a random `frac` of users (all windows), for smoke runs only.

    Event indices stay global, so query codes and history events still line up; whole users are
    kept, so every evaluation window stays complete for the autoregressive protocol.
    """
    users = np.unique(rows["user"])
    keep = np.random.default_rng(seed).choice(
        users, size=max(1, int(len(users) * frac)), replace=False
    )
    m = np.isin(rows["user"], keep)
    n = len(rows["user"])
    # `codes` stays whole: it is indexed by the global event index
    return {
        k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) and k != "codes" else v)
        for k, v in rows.items()
    }
