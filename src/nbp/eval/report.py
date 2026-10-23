"""Result tables shared by the evaluation notebooks: metrics per row set and protocol, h slices, gaps.

Row sets: warm (L >= 1), cold (L = 0), all. Protocols: "ar" (autoregressive, primary) and "tf"
(teacher-forced, secondary). NDCG columns come first because NDCG@K is the primary metric.
Only numpy is used (both venvs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nbp.eval.bootstrap import paired_bootstrap
from nbp.eval.metrics import KS, ndcg_at_k, recall_at_k

PROTOCOLS = ("ar", "tf")
ROW_SETS = ("all", "warm", "cold")
H_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 5, "3-5"),
    (6, None, "6+"),
)


def metric_columns(ks: tuple[int, ...] = KS) -> list[str]:
    """Column order of every result table: NDCG@K first, then Recall@K."""
    return [f"ndcg@{k}" for k in ks] + [f"recall@{k}" for k in ks]


def metrics(rank: np.ndarray, mask: np.ndarray | None = None, ks: tuple[int, ...] = KS) -> dict:
    """{"n", "ndcg@K"..., "recall@K"...} for the rows in `mask` (all rows if None), unrounded."""
    rank = np.asarray(rank)
    if mask is not None:
        rank = rank[mask]
    out: dict = {"n": len(rank)}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(rank, k) if len(rank) else float("nan")
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(rank, k) if len(rank) else float("nan")
    return out


def row_set_masks(L: np.ndarray) -> dict[str, np.ndarray]:
    """Masks of the row sets from the history length L (earlier bookings of the user)."""
    L = np.asarray(L)
    return {"all": np.ones(len(L), bool), "warm": L >= 1, "cold": L == 0}


def h_bucket(h: np.ndarray) -> np.ndarray:
    """(n,) bucket label of h (earlier bookings of the user inside the evaluated window)."""
    h = np.asarray(h)
    out = np.empty(len(h), dtype=object)
    for lo, hi, name in H_BUCKETS:
        m = h >= lo if hi is None else (h >= lo) & (h <= hi)
        out[m] = name
    return out


def h_slices(rank: np.ndarray, h: np.ndarray, ks: tuple[int, ...] = KS) -> dict[str, dict]:
    """Metrics per h bucket."""
    b = h_bucket(h)
    return {name: metrics(rank, b == name, ks) for _, _, name in H_BUCKETS}


def protocol_gap(
    rank_ar: np.ndarray,
    rank_tf: np.ndarray,
    groups: np.ndarray,
    ks: tuple[int, ...] = KS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """AR minus TF per metric with a user-clustered paired bootstrap 95 % interval."""
    out = {}
    for k in ks:
        out.update(paired_bootstrap(rank_ar, rank_tf, k=k, n_boot=n_boot, seed=seed, groups=groups))
    return out


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, obj: dict) -> None:
    """Write `obj` (numpy scalars and arrays allowed) as indented UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(obj), indent=2), encoding="utf-8")
