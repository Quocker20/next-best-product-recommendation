"""Late fusion of SMLP4Rec scores with a destination prior on Expedia (no retraining).

score(cluster) = log p_model(cluster | history) + w * log p_prior(cluster | searched destination)

- p_model: softmax over the 100 clusters from the saved 3-epoch plain checkpoint
  (path recorded in reports/summary/week3/smlprec_expedia_run.json). Weights are unchanged.
- p_prior: destination cluster counts smoothed toward the destination's market,
  p = (n_dest,k + m * p_market,k) / (n_dest + m) with m = 5; unseen destination -> global
  distribution. Market = modal hotel_market of that destination in the prior data, never the
  event's own hotel_market. Counts come only from bookings strictly before the last train target
  (minus a 128 s margin: RecBole stores timestamps as float32, resolution 128 s at 1.4e9), so no
  valid/test booking is counted in its own prior.
- w is chosen on VALID per history-length bucket (L = prior bookings: 1 | 2-4 | 5-9 | 10+) by
  Recall@5, then applied once on TEST. w = 0 is the plain model.

Same rows as the earlier runs (RecBole 80/10/10 temporal split over next-booking targets, users'
first bookings are never targets, so L >= 1 everywhere: cold users are not evaluated here).
Reference rows on the same split: destination prior alone, and "same-destination history first,
then prior" (recency 0.7^age), the week-2 heuristic.

Needs data/interim/recbole/expedia_dest/expedia_dest.inter (scripts/expedia_to_recbole.py
--with-destination).

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/late_fusion_destination_prior.py
Output: reports/summary/week3/smlprec_expedia_late_fusion.json
"""

import functools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The checkpoint stores the RecBole Config object, so PyTorch >= 2.6 needs weights_only=False.
torch.load = functools.partial(torch.load, weights_only=False)

import run_smlprec_expedia as base
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from src.models.smlprec import SMLPREC

RAW = ROOT / "data" / "raw" / "hospitality" / "expedia" / "train.csv"
PREV = base.OUT_DIR / "smlprec_expedia_run.json"
OUT_JSON = base.OUT_DIR / "smlprec_expedia_late_fusion.json"
M_SMOOTH = 5
EPS = 1e-6
TIME_MARGIN = 128  # seconds, float32 timestamp resolution
RECENCY = 0.7
W_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
BUCKETS = [("L=1", 1, 1), ("L=2-4", 2, 4), ("L=5-9", 5, 9), ("L>=10", 10, 10**9)]
N_CLUSTERS = 100


def build_prior_tables(cut_seconds: float) -> dict:
    """Cluster count tables from bookings with unix time < cut_seconds (train.csv, chunked)."""
    # pandas is imported here, after torch: on Windows, pandas-before-torch makes c10.dll fail.
    import pandas as pd

    cols = [
        "date_time",
        "is_booking",
        "srch_destination_id",
        "hotel_market",
        "hotel_cluster",
    ]
    dtypes = {c: "int32" for c in cols if c != "date_time"}
    parts = []
    for ch in pd.read_csv(RAW, usecols=cols, chunksize=2_000_000, dtype=dtypes):
        ch = ch[ch.is_booking == 1]
        ts = (
            pd.to_datetime(ch["date_time"]) - pd.Timestamp("1970-01-01")
        ) // pd.Timedelta(seconds=1)
        parts.append(
            ch.loc[
                ts.values < cut_seconds,
                ["srch_destination_id", "hotel_market", "hotel_cluster"],
            ]
        )
    b = pd.concat(parts, ignore_index=True)

    def table(key):
        t = b.groupby([key, "hotel_cluster"]).size().unstack(fill_value=0)
        return t.reindex(columns=range(N_CLUSTERS), fill_value=0)

    dest, mkt = table("srch_destination_id"), table("hotel_market")
    dm = b.groupby(["srch_destination_id", "hotel_market"]).size().reset_index(name="n")
    modal = dm.sort_values("n", ascending=False).drop_duplicates("srch_destination_id")
    glob = np.bincount(b["hotel_cluster"].values, minlength=N_CLUSTERS).astype(
        np.float64
    )
    return {
        "n_bookings": len(b),
        "dest": {int(d): r.astype(np.float64) for d, r in zip(dest.index, dest.values)},
        "market": {int(m): r.astype(np.float64) for m, r in zip(mkt.index, mkt.values)},
        "dest2mkt": dict(
            zip(
                modal["srch_destination_id"].astype(int),
                modal["hotel_market"].astype(int),
            )
        ),
        "global": glob / glob.sum(),
    }


def prior_vector(tables: dict, d: int) -> np.ndarray:
    """p(cluster | destination) over the 100 clusters, shape (100,)."""
    row = tables["dest"].get(d)
    if row is None:
        return tables["global"]
    pm = tables["market"][tables["dest2mkt"][d]]
    pm = pm / pm.sum()
    return (row + M_SMOOTH * pm) / (row.sum() + M_SMOOTH)


def prior_logp(
    tables: dict, dests: np.ndarray, cluster_ids: np.ndarray, n_items: int
) -> np.ndarray:
    """(n, n_items) log prior per row, columns = model item ids (padding column 0 stays 0)."""
    uniq, inv = np.unique(dests, return_inverse=True)
    table = np.stack([np.log(prior_vector(tables, int(d)) + EPS) for d in uniq]).astype(
        np.float32
    )
    out = np.zeros((len(dests), n_items), np.float32)
    out[:, cluster_ids] = table[inv]
    return out


def split_arrays(loader, config, cluster_ids) -> dict:
    """Inputs of one split: history ids/length/destinations, target, query destination."""
    ds = loader.dataset
    d = ds.inter_feat
    seq = d[ds.iid_field + config["LIST_SUFFIX"]].numpy()
    length = d[config["ITEM_LIST_LENGTH_FIELD"]].numpy()
    y = d[ds.iid_field].numpy()
    in_hist = np.zeros((len(y), int(seq.max()) + 1), bool)
    for j in range(seq.shape[1]):
        r = np.flatnonzero(j < length)
        in_hist[r, seq[r, j]] = True
    in_hist[:, 0] = False
    return {
        "seq": seq,
        "seq_t": d[ds.iid_field + config["LIST_SUFFIX"]],
        "length": length,
        "length_t": d[config["ITEM_LIST_LENGTH_FIELD"]],
        "y": y,
        "in_hist": in_hist,
        "q_dest": d["srch_destination_id"].numpy().astype(np.int64),
        "h_dest": d["srch_destination_id" + config["LIST_SUFFIX"]]
        .numpy()
        .astype(np.int64),
    }


@torch.no_grad()
def model_logp(model, arr) -> np.ndarray:
    """(n, n_items) log-softmax of model scores; padding column is -inf."""
    model.eval()
    outs = []
    for i in range(0, len(arr["y"]), 8192):
        sc = model.full_sort_predict(
            {
                model.ITEM_SEQ: arr["seq_t"][i : i + 8192],
                model.ITEM_SEQ_LEN: arr["length_t"][i : i + 8192],
            }
        )
        sc[:, 0] = float("-inf")
        outs.append(torch.log_softmax(sc, dim=1).numpy())
    return np.concatenate(outs)


def same_dest_history_scores(arr, prior: np.ndarray) -> np.ndarray:
    """Week-2 heuristic: past clusters booked at the query destination first (recency 0.7^age), then prior."""
    s = prior.copy()
    for j in range(arr["seq"].shape[1]):
        idx = np.flatnonzero(
            (j < arr["length"]) & (arr["h_dest"][:, j] == arr["q_dest"])
        )
        bonus = 1000.0 * RECENCY ** (arr["length"][idx] - 1 - j)
        cl = arr["seq"][idx, j]
        s[idx, cl] = np.maximum(s[idx, cl], prior[idx, cl] + bonus)
    return s


def ranks(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of the true cluster; ties broken by lower item id (stable)."""
    sc = torch.from_numpy(scores)
    sc[:, 0] = float("-inf")
    yy = torch.from_numpy(y)
    out = []
    for i in range(0, len(y), 8192):
        s, t = sc[i : i + 8192], yy[i : i + 8192]
        st = s.gather(1, t[:, None])
        idx = torch.arange(s.shape[1])[None, :]
        out.append(1 + (s > st).sum(1) + ((s == st) & (idx < t[:, None])).sum(1))
    return torch.cat(out).numpy()


def summarize(rank: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        rank = rank[mask]
    r4 = lambda x: round(float(x), 4)
    out = {"n": len(rank)}
    for k in (5, 10, 20):
        out[f"recall@{k}"] = r4((rank <= k).mean())
    out["ndcg@10"] = r4(((rank <= 10) / np.log2(rank + 1)).mean())
    out["mrr@10"] = r4(((rank <= 10) / rank).mean())
    return out


def bucket_masks(length: np.ndarray) -> dict:
    return {name: (length >= lo) & (length <= hi) for name, lo, hi in BUCKETS}


def fused(mlogp: np.ndarray, prior: np.ndarray, w) -> np.ndarray:
    """mlogp + w * prior; w is a scalar or an (n,) array."""
    w = np.asarray(w, np.float32)
    return mlogp + (w[:, None] if w.ndim else w) * prior


def main() -> None:
    base.WORK.mkdir(parents=True, exist_ok=True)
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    ckpt_path = Path(prev["checkpoint"])
    ckpt = torch.load(ckpt_path)
    config = Config(
        model=SMLPREC,
        dataset="expedia_dest",
        config_file_list=[str(base.CONFIG)],
        config_dict={
            "data_path": str(base.WORK),
            "checkpoint_dir": str(base.WORK / "saved_fusion"),
            "load_col": {
                "inter": ["user_id", "item_id", "timestamp", "srch_destination_id"]
            },
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    tok2id = dataset.field2token_id[dataset.iid_field]
    cluster_ids = np.array([tok2id[str(k)] for k in range(N_CLUSTERS)])
    n_items = dataset.item_num

    ts = train_data.dataset.inter_feat[config["TIME_FIELD"]]
    cut = float(ts.max()) - TIME_MARGIN
    print(f"timestamp dtype {ts.dtype}; prior cutoff (unix s) {cut:.0f}")
    tables = build_prior_tables(cut)
    print(
        f"prior built from {tables['n_bookings']} bookings, {len(tables['dest'])} destinations"
    )

    model = SMLPREC(config, dataset)
    model.load_state_dict(ckpt["state_dict"])
    arr = {
        "valid": split_arrays(valid_data, config, cluster_ids),
        "test": split_arrays(test_data, config, cluster_ids),
    }
    ml, pr = {}, {}
    for sp, a in arr.items():
        ml[sp] = model_logp(model, a)
        pr[sp] = prior_logp(tables, a["q_dest"], cluster_ids, n_items)
    t_prep = time.time() - t0

    # sanity: w = 0 must reproduce the earlier plain run (same weights, same rows)
    plain_prev = prev["per_epoch"][-1]
    sanity = {}
    for sp in ("valid", "test"):
        got = summarize(ranks(fused(ml[sp], pr[sp], 0.0), arr[sp]["y"]))["recall@5"]
        sanity[sp] = {
            "recomputed_recall@5": got,
            "earlier_run_recall@5": plain_prev[sp]["recall@5"],
        }
        print(
            f"sanity {sp}: plain recall@5 {got} vs earlier {plain_prev[sp]['recall@5']}"
        )

    # sweep on valid (selection) and test (information only)
    masks = {sp: bucket_masks(arr[sp]["length"]) for sp in arr}
    sweep = {sp: {} for sp in arr}
    for w in W_GRID:
        for sp in arr:
            rk = ranks(fused(ml[sp], pr[sp], w), arr[sp]["y"])
            sweep[sp][str(w)] = {
                "all": summarize(rk),
                **{b: summarize(rk, m) for b, m in masks[sp].items()},
            }
    chosen = {}
    for b, _, _ in BUCKETS:
        best = max(W_GRID, key=lambda w: (sweep["valid"][str(w)][b]["recall@5"], -w))
        chosen[b] = best
    best_global = max(
        W_GRID, key=lambda w: (sweep["valid"][str(w)]["all"]["recall@5"], -w)
    )

    def per_row_w(sp):
        w = np.zeros(len(arr[sp]["y"]), np.float32)
        for b, m in masks[sp].items():
            w[m] = chosen[b]
        return w

    rows = {}
    for sp, a in arr.items():
        y, in_hist = a["y"], a["in_hist"]
        old = in_hist[np.arange(len(y)), y]
        variants = {
            "plain_model (w=0)": fused(ml[sp], pr[sp], 0.0),
            "destination_prior_only": pr[sp] + 1e-6 * np.nan_to_num(ml[sp], neginf=0.0),
            "same_dest_history_then_prior": same_dest_history_scores(a, pr[sp]),
            f"fusion_global_w={best_global}": fused(ml[sp], pr[sp], best_global),
            "fusion_per_bucket_w": fused(ml[sp], pr[sp], per_row_w(sp)),
        }
        rows[sp] = {}
        for name, sc in variants.items():
            rk = ranks(sc, y)
            rows[sp][name] = {
                "all": summarize(rk),
                "old_target_rows": summarize(rk, old),
                "new_target_rows": summarize(rk, ~old),
                **{b: summarize(rk, m) for b, m in masks[sp].items()},
            }

    result = {
        "run": "week3 experiment: late fusion of SMLP4Rec log-probs with a destination prior, weight swept per history bucket",
        "formula": "score = log p_model + w * log p_prior(destination); w=0 is the plain model",
        "prior": {
            "smoothing_m": M_SMOOTH,
            "eps": EPS,
            "cutoff_unix_seconds": cut,
            "n_bookings": tables["n_bookings"],
            "n_destinations": len(tables["dest"]),
        },
        "checkpoint": str(ckpt_path),
        "w_grid": W_GRID,
        "chosen_w_per_bucket_on_valid_recall@5": chosen,
        "best_global_w_on_valid": best_global,
        "rows": {sp: len(arr[sp]["y"]) for sp in arr},
        "share_of_rows_per_bucket": {
            sp: {b: round(float(m.mean()), 4) for b, m in masks[sp].items()}
            for sp in arr
        },
        "sanity_plain_reproduces_earlier_run": sanity,
        "variants": rows,
        "sweep": sweep,
        "timing_seconds": {"prep": round(t_prep, 1)},
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("chosen w per bucket:", chosen, "| best global w:", best_global)
    for sp in ("valid", "test"):
        print(sp)
        for name, v in rows[sp].items():
            a = v["all"]
            print(
                f"  {name:38s} R@5 {a['recall@5']:.4f}  R@10 {a['recall@10']:.4f}  R@20 {a['recall@20']:.4f}  NDCG@10 {a['ndcg@10']:.4f}"
            )


if __name__ == "__main__":
    main()
