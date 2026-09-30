"""Week-3 pipeline-validation run: SMLP4Rec (src/models/smlprec.py) on Expedia bookings.

Runs RecBole 1.0.1 (patched MLP4Rec clone, installed editable in the smlp4rec venv)
with configs/smlprec_expedia.yaml: 1 epoch, global temporal split 80/10/10 over
next-booking targets, full ranking over the 100 clusters. For context it also scores
two heuristics on the same test rows: global popularity (train targets) and
repeat-last (the user's previous booked cluster first, then popularity).

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/run_smlprec_expedia.py
Outputs:
    reports/summary/week3/smlprec_expedia_run.json   (config, split sizes, metrics, timings)
    reports/summary/week3/smlprec_expedia_run.log    (RecBole log)
    data/interim/recbole/saved/                      (checkpoint, not committed)
"""

import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# RecBole 1.0.1 reloads its own checkpoints; PyTorch >= 2.6 defaults to weights_only=True.
torch.load = functools.partial(torch.load, weights_only=False)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from src.models.smlprec import SMLPREC

# recbole/model/sequential_recommender/sine.py enables anomaly detection on import.
torch.autograd.set_detect_anomaly(False)

CONFIG = ROOT / "configs" / "smlprec_expedia.yaml"
WORK = ROOT / "data" / "interim" / "recbole"
OUT_DIR = ROOT / "reports" / "summary" / "week3"
KS = (5, 10, 20)


def heuristic_recall(train_ds, test_ds, n_items: int) -> dict:
    """Recall@K on test targets for global popularity and repeat-last (+ popularity fill)."""
    iid = train_ds.iid_field
    pop = np.bincount(train_ds.inter_feat[iid].numpy(), minlength=n_items).astype(
        np.float64
    )
    pop[0] = -1  # padding id
    y = test_ds.inter_feat[iid].numpy()
    seq = test_ds.inter_feat[iid + train_ds.config["LIST_SUFFIX"]].numpy()
    length = test_ds.inter_feat[train_ds.config["ITEM_LIST_LENGTH_FIELD"]].numpy()
    last = seq[np.arange(len(seq)), length - 1]

    scores_pop = np.tile(pop, (len(y), 1))
    scores_rep = scores_pop.copy()
    scores_rep[np.arange(len(y)), last] = pop.max() + 1
    out = {}
    for name, sc in (
        ("global_popularity", scores_pop),
        ("repeat_last_then_popularity", scores_rep),
    ):
        rank = np.argsort(-sc, axis=1, kind="stable")
        out[name] = {
            f"recall@{k}": round(float((rank[:, :k] == y[:, None]).any(1).mean()), 4)
            for k in KS
        }
    return out


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)  # RecBole writes ./log and ./log_tensorboard relative to cwd

    config = Config(
        model=SMLPREC,
        dataset="expedia",
        config_file_list=[str(CONFIG)],
        config_dict={"data_path": str(WORK), "checkpoint_dir": str(WORK / "saved")},
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    t0 = time.time()
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    t_data = time.time() - t0

    model = SMLPREC(config, train_data.dataset).to(config["device"])
    logger.info(model)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    t1 = time.time()
    _, best_valid = trainer.fit(train_data, valid_data, saved=True)
    t_fit = time.time() - t1
    t2 = time.time()
    test_result = trainer.evaluate(test_data, load_best_model=True)
    t_test = time.time() - t2
    logger.info(f"test result: {test_result}")

    split_ts = {}
    for name, d in (("train", train_data), ("valid", valid_data), ("test", test_data)):
        ts = d.dataset.inter_feat[config["TIME_FIELD"]].numpy()
        split_ts[name] = {
            "targets": len(ts),
            "first_target_unix": int(ts.min()),
            "last_target_unix": int(ts.max()),
            "mean_history_len": round(
                float(
                    d.dataset.inter_feat[config["ITEM_LIST_LENGTH_FIELD"]]
                    .float()
                    .mean()
                ),
                3,
            ),
        }

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    result = {
        "run": "week3 pipeline validation - SMLP4Rec on Expedia bookings (research adaptation)",
        "dataset": "expedia (train.csv, is_booking == 1; item = hotel_cluster)",
        "git_commit": commit,
        "config": {
            k: config[k]
            for k in (
                "MAX_ITEM_LIST_LENGTH",
                "n_layers",
                "hidden_size",
                "hidden_dropout_prob",
                "loss_type",
                "epochs",
                "train_batch_size",
                "learning_rate",
                "seed",
                "eval_args",
                "topk",
                "valid_metric",
            )
        },
        "dataset_stats": {
            "users": int(dataset.user_num - 1),
            "items": int(dataset.item_num - 1),
            "sequence_targets": len(dataset.inter_feat),
        },
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "splits": split_ts,
        "timing_seconds": {
            "data_build": round(t_data, 1),
            "train_and_valid": round(t_fit, 1),
            "test": round(t_test, 1),
        },
        "valid": {k: round(float(v), 4) for k, v in best_valid.items()},
        "test": {k: round(float(v), 4) for k, v in test_result.items()},
        "heuristics_same_test_rows": heuristic_recall(
            train_data.dataset, test_data.dataset, dataset.item_num
        ),
    }
    (OUT_DIR / "smlprec_expedia_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    log_files = sorted((WORK / "log" / "SMLPREC").glob("*.log"), key=os.path.getmtime)
    if log_files:
        shutil.copy(log_files[-1], OUT_DIR / "smlprec_expedia_run.log")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
