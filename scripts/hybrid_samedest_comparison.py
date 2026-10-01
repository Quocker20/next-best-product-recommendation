"""Four-way comparison on Expedia: plain, regional prior only, SMLP4Rec + prior, full hybrid.

    plain        : SMLP4Rec alone (history only)
    prior only   : regional (destination) prior alone
    fusion       : log p_SMLP4Rec + w_p * log p_prior                      (w_p tuned on valid)
    hybrid       : log p_SMLP4Rec + w_p * log p_prior + w_s * sameDest     (w_p, w_s tuned on valid)

w_m is fixed at 1 (only the ratios matter for ranking). sameDest(k) = sum over the user's past
bookings of cluster k made at the SAME destination as the current query of 0.7^age (age 0 = most
recent booking). Weights are chosen on VALID per history bucket (L = prior bookings: 1 | 2-4 |
5-9 | 10+) by Recall@5, plus a single global pair and a variant chosen by MAP@5; test is read
once. The no-learning same-destination rule is included as a reference row. Cold users (first
booking, L = 0) use the prior only in every variant except plain (plain has no cold handling:
global popularity is used there). Paired bootstrap (1000 resamples over test rows, seed 0)
gives 95% intervals for the key differences.

Reuses the prior, split arrays and cold-user windows of scripts/late_fusion_destination_prior.py
(same checkpoint, same rows; prior built only from bookings before the last train target).

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/hybrid_samedest_comparison.py
Output: reports/summary/week3/smlprec_expedia_hybrid_samedest.json
"""

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

import late_fusion_destination_prior as lf
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from src.models.smlprec import SMLPREC

OUT_JSON = lf.base.OUT_DIR / "smlprec_expedia_hybrid_samedest.json"
W_P = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
W_S = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
GRID = [(wp, ws) for wp in W_P for ws in W_S]
N_BOOT = 1000


def same_dest_matrix(arr: dict, n_items: int) -> np.ndarray:
    """(n, n_items) sum of 0.7^age over past bookings of the cluster at the query destination."""
    s = np.zeros((len(arr["y"]), n_items), np.float32)
    for j in range(arr["seq"].shape[1]):
        idx = np.flatnonzero(
            (j < arr["length"]) & (arr["h_dest"][:, j] == arr["q_dest"])
        )
        age = arr["length"][idx] - 1 - j
        s[idx, arr["seq"][idx, j]] += lf.RECENCY**age
    return s


def per_row(rank_by_combo: dict, chosen: dict, masks: dict, n: int) -> np.ndarray:
    """Assemble ranks where each bucket uses its own chosen (w_p, w_s)."""
    out = np.zeros(n, np.int16)
    for b, m in masks.items():
        out[m] = rank_by_combo[chosen[b]][m]
    return out


def best_combo(valid_rank: dict, mask: np.ndarray, metric: str, grid: list) -> tuple:
    """Grid pair with the best valid metric in the mask; ties go to smaller weights."""

    def score(c):
        rk = valid_rank[c][mask]
        v = (rk <= 5).mean() if metric == "recall" else ((rk <= 5) / rk).mean()
        return (-round(float(v), 6), c[0] + c[1])

    return min(grid, key=score)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Mean difference (a - b) of Recall@5 and MAP@5 with 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, fa, fb in (
        ("recall@5", (a <= 5).astype(np.float32), (b <= 5).astype(np.float32)),
        ("map@5", (a <= 5) / a, (b <= 5) / b),
    ):
        d = (fa - fb).astype(np.float32)
        n = len(d)
        means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        out[name] = {
            "diff": round(float(d.mean()), 4),
            "ci95": [
                round(float(np.percentile(means, 2.5)), 4),
                round(float(np.percentile(means, 97.5)), 4),
            ],
        }
    return out


def main() -> None:
    prev = json.loads(lf.PREV.read_text(encoding="utf-8"))
    ckpt = torch.load(Path(prev["checkpoint"]))
    config = Config(
        model=SMLPREC,
        dataset="expedia_dest",
        config_file_list=[str(lf.base.CONFIG)],
        config_dict={
            "data_path": str(lf.base.WORK),
            "checkpoint_dir": str(lf.base.WORK / "saved_fusion"),
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
    cluster_ids = np.array([tok2id[str(k)] for k in range(lf.N_CLUSTERS)])
    n_items = dataset.item_num

    ts = train_data.dataset.inter_feat[config["TIME_FIELD"]]
    tables = lf.build_prior_tables(float(ts.max()) - lf.TIME_MARGIN)
    model = SMLPREC(config, dataset)
    model.load_state_dict(ckpt["state_dict"])
    arr = {
        "valid": lf.split_arrays(valid_data, config, cluster_ids),
        "test": lf.split_arrays(test_data, config, cluster_ids),
    }
    ml, pr, sd, masks = {}, {}, {}, {}
    for sp, a in arr.items():
        ml[sp] = lf.model_logp(model, a)
        pr[sp] = lf.prior_logp(tables, a["q_dest"], cluster_ids, n_items)
        sd[sp] = same_dest_matrix(a, n_items)
        masks[sp] = lf.bucket_masks(a["length"])
    print(f"prep {time.time() - t0:.0f}s; prior from {tables['n_bookings']} bookings")

    # grid ranks for every (w_p, w_s) on both splits
    rk_grid = {sp: {} for sp in arr}
    for c in GRID:
        for sp, a in arr.items():
            sc = ml[sp] + c[0] * pr[sp] + c[1] * sd[sp]
            rk_grid[sp][c] = lf.ranks(sc, a["y"]).astype(np.int16)
    print(f"grid done {time.time() - t0:.0f}s")

    buckets = [b for b, _, _ in lf.BUCKETS]
    fusion_grid = [c for c in GRID if c[1] == 0.0]
    all_rows = np.ones(len(arr["valid"]["y"]), bool)
    chosen = {
        "fusion_global": {
            b: best_combo(rk_grid["valid"], all_rows, "recall", fusion_grid)
            for b in buckets
        },
        "hybrid_global": {
            b: best_combo(rk_grid["valid"], all_rows, "recall", GRID) for b in buckets
        },
        "hybrid_per_bucket": {
            b: best_combo(rk_grid["valid"], masks["valid"][b], "recall", GRID)
            for b in buckets
        },
        "hybrid_per_bucket_by_map": {
            b: best_combo(rk_grid["valid"], masks["valid"][b], "map", GRID)
            for b in buckets
        },
    }

    # reference rankers
    warm_rank = {sp: {} for sp in arr}
    for sp, a in arr.items():
        n = len(a["y"])
        warm_rank[sp]["plain"] = rk_grid[sp][(0.0, 0.0)]
        warm_rank[sp]["prior_only"] = lf.ranks(
            pr[sp] + 1e-6 * np.nan_to_num(ml[sp], neginf=0.0), a["y"]
        ).astype(np.int16)
        warm_rank[sp]["same_dest_rule (no model)"] = lf.ranks(
            lf.same_dest_history_scores(a, pr[sp]), a["y"]
        ).astype(np.int16)
        for name, ch in chosen.items():
            warm_rank[sp][name] = per_row(rk_grid[sp], ch, masks[sp], n)

    # cold users: first bookings in the valid / test windows, prior only
    import pandas as pd

    inter = pd.read_csv(lf.INTER_DEST, sep="\t").sort_values(
        ["user_id:token", "timestamp:float"], kind="stable"
    )
    first = inter.drop_duplicates("user_id:token", keep="first")
    t_train = float(ts.max())
    t_valid = float(valid_data.dataset.inter_feat[config["TIME_FIELD"]].max())
    windows = {
        "valid": (first["timestamp:float"] >= t_train)
        & (first["timestamp:float"] < t_valid),
        "test": first["timestamp:float"] >= t_valid,
    }
    glob_row = np.zeros(n_items, np.float32)
    glob_row[cluster_ids] = np.log(tables["global"] + lf.EPS)
    cold_rank = {}
    for sp, mask in windows.items():
        f = first[mask]
        y = np.array([tok2id[str(int(c))] for c in f["item_id:token"]])
        dests = f["srch_destination_id:float"].values.astype(np.int64)
        cold_rank[sp] = {
            "prior": lf.ranks(
                lf.prior_logp(tables, dests, cluster_ids, n_items), y
            ).astype(np.int16),
            "global": lf.ranks(np.tile(glob_row, (len(y), 1)), y).astype(np.int16),
        }

    labels = {
        "plain": "1. plain SMLP4Rec",
        "prior_only": "2. regional prior only",
        "fusion_global": "3. SMLP4Rec + prior (w_p)",
        "hybrid_per_bucket": "4. hybrid: + sameDest (per bucket, by Recall@5)",
        "hybrid_global": "4b. hybrid, one global (w_p, w_s)",
        "hybrid_per_bucket_by_map": "4c. hybrid, per bucket, chosen by MAP@5",
        "same_dest_rule (no model)": "ref. same-destination rule (no model)",
    }
    result = {
        "run": "week3 experiment: plain vs regional prior vs SMLP4Rec+prior vs hybrid with sameDest",
        "formula": "score = 1*log p_SMLP4Rec + w_p*log p_prior + w_s*sameDest; sameDest = sum 0.7^age of past bookings of the cluster at the query destination",
        "grids": {"w_p": W_P, "w_s": W_S},
        "chosen_weights_on_valid": {
            k: {b: list(v) for b, v in ch.items()} for k, ch in chosen.items()
        },
        "rows": {sp: len(arr[sp]["y"]) for sp in arr},
        "cold_rows": {sp: len(cold_rank[sp]["prior"]) for sp in arr},
        "warm": {},
        "all_events": {},
        "bootstrap_test_95ci": {},
    }
    for sp, a in arr.items():
        y, in_hist = a["y"], a["in_hist"]
        old = in_hist[np.arange(len(y)), y]
        result["warm"][sp] = {}
        result["all_events"][sp] = {}
        for key, label in labels.items():
            rk = warm_rank[sp][key].astype(np.int64)
            result["warm"][sp][label] = {
                "all": lf.summarize(rk),
                "old_target_rows": lf.summarize(rk, old),
                "new_target_rows": lf.summarize(rk, ~old),
                **{b: lf.summarize(rk, m) for b, m in masks[sp].items()},
            }
            cold = cold_rank[sp]["global" if key == "plain" else "prior"].astype(
                np.int64
            )
            result["all_events"][sp][label] = lf.summarize(np.concatenate([rk, cold]))
        result["all_events"][sp]["cold rows only: prior"] = lf.summarize(
            cold_rank[sp]["prior"].astype(np.int64)
        )

    comps = [
        ("hybrid_per_bucket", "same_dest_rule (no model)"),
        ("hybrid_per_bucket", "fusion_global"),
        ("hybrid_per_bucket", "prior_only"),
        ("fusion_global", "prior_only"),
        ("fusion_global", "same_dest_rule (no model)"),
        ("hybrid_per_bucket", "plain"),
    ]
    for a_key, b_key in comps:
        ra, rb = (warm_rank["test"][k].astype(np.int64) for k in (a_key, b_key))
        ca = cold_rank["test"]["global" if a_key == "plain" else "prior"].astype(
            np.int64
        )
        cb = cold_rank["test"]["global" if b_key == "plain" else "prior"].astype(
            np.int64
        )
        result["bootstrap_test_95ci"][f"{a_key} minus {b_key}"] = {
            "warm": paired_bootstrap(ra, rb),
            "all_events": paired_bootstrap(
                np.concatenate([ra, ca]), np.concatenate([rb, cb])
            ),
        }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("chosen weights (w_p, w_s) per bucket on valid:")
    for k, ch in chosen.items():
        print(" ", k, ch)
    for sp in ("valid", "test"):
        print(sp, "users with history")
        for label, v in result["warm"][sp].items():
            a = v["all"]
            print(
                f"  {label:52s} R@5 {a['recall@5']:.4f} R@10 {a['recall@10']:.4f} R@20 {a['recall@20']:.4f} NDCG@10 {a['ndcg@10']:.4f} MAP@5 {a['map@5']:.4f}"
            )
        print(sp, "all events (warm + cold)")
        for label, a in result["all_events"][sp].items():
            print(
                f"  {label:52s} R@5 {a['recall@5']:.4f} R@10 {a['recall@10']:.4f} R@20 {a['recall@20']:.4f} NDCG@10 {a['ndcg@10']:.4f} MAP@5 {a['map@5']:.4f}"
            )
    print("paired bootstrap, test, difference (95% CI):")
    for k, v in result["bootstrap_test_95ci"].items():
        for scope in ("warm", "all_events"):
            r, m = v[scope]["recall@5"], v[scope]["map@5"]
            print(
                f"  {k:62s} {scope:10s} R@5 {r['diff']:+.4f} {r['ci95']}  MAP@5 {m['diff']:+.4f} {m['ci95']}"
            )


if __name__ == "__main__":
    main()
