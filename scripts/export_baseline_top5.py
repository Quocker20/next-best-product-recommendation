"""Top-5 test predictions of the two baselines under the autoregressive protocol (project `.venv`).

Refits ItemKNN and logistic regression exactly as notebook 03 (hyper-parameters read from
`results/week6_retrain/`), scores the test window with the AR protocol and writes
`data/interim/week6/top5_baselines_s<seed>.npz` (keys: event, itemknn, logreg, rank_itemknn,
rank_logreg). Called by notebook `05_test_with_result` (the SMLP4Rec kernel has no `implicit`).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (torch before pandas-importing modules, Windows c10.dll)
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from nbp.baselines.itemknn import fit_itemknn, itemknn_scorer, user_item_matrix  # noqa: E402
from nbp.baselines.logreg import (  # noqa: E402
    fit_encoder,
    fit_logreg,
    logreg_proba,
    logreg_scorer,
    train_log_loss,
)
from nbp.data.split import TRAIN  # noqa: E402
from nbp.eval import runner as R  # noqa: E402
from nbp.eval.autoregressive import score_view  # noqa: E402
from nbp.eval.metrics import ndcg_at_k, target_rank  # noqa: E402
from nbp.models.early_stop import mean_nll  # noqa: E402
from nbp.paths import CONFIGS, INTERIM, PROCESSED  # noqa: E402

TOPN = 5
SPLIT = yaml.safe_load((CONFIGS / "split.yaml").read_text(encoding="utf-8"))
TRAIN_CFG = yaml.safe_load((CONFIGS / "train.yaml").read_text(encoding="utf-8"))
SEED = SPLIT["seeds"][0]
BATCH = int(TRAIN_CFG["eval_batch"])
K_MON = int(TRAIN_CFG["monitor"].split("@")[1])
RES = ROOT / "results" / "week6_retrain"
K_KNN = json.loads((RES / "itemknn.json").read_text(encoding="utf-8"))["k_chosen"]
C_LR = json.loads((RES / f"logreg_s{SEED}.json").read_text(encoding="utf-8"))["c_chosen"]

rows = R.load_rows(PROCESSED / "expedia_date_split" / "rows.npz")
codes = rows["codes"]
N_ITEMS = int(rows["y"].max()) + 1
views = R.window_views(rows)
idx = R.window_index(rows)
Y = {w: rows["y"][i].astype(np.int64) for w, i in idx.items()}
tr = np.flatnonzero(rows["window"] == TRAIN)
y_tr = rows["y"][tr].astype(np.int64)
out = {"event": rows["event"][idx["test"]]}


def run_ar(key: str, score_fn) -> None:
    sc = score_view(views["test"], score_fn, autoregressive=True, batch=BATCH)
    out[key] = (R.top_k(sc.scores, TOPN) - 1).astype(np.int16)  # model id -> hotel_cluster
    out[f"rank_{key}"] = target_rank(sc.scores, Y["test"]).astype(np.int32)
    print(f"{key}: AR test NDCG@{K_MON} {ndcg_at_k(out[f'rank_{key}'], K_MON):.4f}", flush=True)


# ItemKNN
pop_share = np.bincount(y_tr, minlength=N_ITEMS) / len(y_tr)
first = tr[rows["L"][tr] == 0]
pop_first_share = np.bincount(rows["y"][first], minlength=N_ITEMS) / len(first)
sim = fit_itemknn(user_item_matrix(rows["user"][tr], y_tr, N_ITEMS), K_KNN)
run_ar("itemknn", itemknn_scorer(sim, pop_share, pop_first_share, TRAIN_CFG["itemknn_tie_eps"]))

# Logistic regression (query fields only, no history)
enc = fit_encoder(codes[rows["event"][tr]])
X_tr = enc.transform(codes[rows["event"][tr]])
X_va = enc.transform(codes[rows["event"][idx["valid"]]])
probe, y_probe = R.train_probe_view(rows, TRAIN_CFG["train_probe_rows"], TRAIN_CFG["train_probe_seed"])
X_pr = enc.transform(codes[probe.event])


def lr_eval(m) -> dict:
    p = logreg_proba(m, X_va, N_ITEMS)
    r = target_rank(np.log(p + 1e-300), Y["valid"])
    return {"monitor": ndcg_at_k(r, K_MON), "valid_nll": mean_nll(np.log(p[:, 1:] + 1e-300), Y["valid"] - 1)}


lr, *_ = fit_logreg(
    X_tr, y_tr, C_LR, SEED, TRAIN_CFG["max_epochs"], TRAIN_CFG["patience"], TRAIN_CFG["min_delta"],
    TRAIN_CFG["nll_guard_epochs"], lr_eval, lambda m: train_log_loss(m, X_pr, y_probe, N_ITEMS),
)
run_ar("logreg", logreg_scorer(lr, enc, codes, N_ITEMS))

path = INTERIM / "week6" / f"top5_baselines_s{SEED}.npz"
np.savez_compressed(path, **out)
print("wrote", path)
