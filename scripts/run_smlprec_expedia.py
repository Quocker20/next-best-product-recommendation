"""Week-3 pipeline-validation run: SMLP4Rec (src/models/smlprec.py) on Expedia bookings.

Runs RecBole 1.0.1 (patched MLP4Rec clone, installed editable in the smlp4rec venv)
with configs/smlprec_expedia.yaml: 3 epochs (valid and test scored after every epoch;
best epoch chosen on valid only), global temporal split 80/10/10 over
next-booking targets, full ranking over the 100 clusters. For context it also scores
two heuristics on the same test rows: global popularity (train targets) and
repeat-last (the user's previous booked cluster first, then popularity).

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/run_smlprec_expedia.py
    ... run_smlprec_expedia.py --stats-only   # recompute only the top-5 behaviour + target-mix stats from the
                                              # saved checkpoint and merge them into the JSON
Outputs:
    results/week3_implementation/smlprec_expedia_run.json   (config, split sizes, metrics, timings)
    results/week3_implementation/smlprec_expedia_run.log    (RecBole log)
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
OUT_DIR = ROOT / "results" / "week3_implementation"
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


def target_mix(loader, config) -> dict:
    """Which kind of cluster the user books next, on one split (fractions of target rows).

    last = same cluster as the user's most recent booking; older_not_last = booked before but
    not the most recent; old_incl_last = last + older_not_last; new = never booked before.
    Padding id 0 never equals a target, so padding cannot match.
    """
    ds = loader.dataset
    d = ds.inter_feat
    seq = d[ds.iid_field + config["LIST_SUFFIX"]].numpy()
    length = d[config["ITEM_LIST_LENGTH_FIELD"]].numpy()
    y = d[ds.iid_field].numpy()
    last = seq[np.arange(len(y)), length - 1] == y
    old = (seq == y[:, None]).any(1)
    r4 = lambda x: round(float(x), 4)
    return {
        "rows": len(y),
        "last": r4(last.mean()),
        "older_not_last": r4((old & ~last).mean()),
        "old_incl_last": r4(old.mean()),
        "new": r4((~old).mean()),
    }


def top5_popular(train_ds) -> np.ndarray:
    """Ids of the 5 most booked clusters among train targets (padding id excluded)."""
    ids = train_ds.inter_feat[train_ds.iid_field].numpy()
    pop = np.bincount(ids, minlength=int(ids.max()) + 1).astype(float)
    pop[0] = -1
    return np.argsort(-pop)[:5]


def behavior_stats(model, loader, config, pop_top5) -> dict:
    """What the top-5 list is made of, on one split (fractions of rows unless noted).

    "History" = every earlier booking of the user in the model input (includes the last
    booking). Keys: top1_is_last_booking, last_booking_in_top5, target_in_history (share of
    rows whose true next cluster is an old one), old_target_in_top5 (of all rows / of
    old-target rows), new_target (share of rows) and new_target_in_top5 (of all rows / of
    new-target rows), top5_slots_new_share (top-5 slots not in the user's history),
    global_popularity_top5_on_new_rows (Recall@5 of the global top-5 on new-target rows).
    """
    ds = loader.dataset
    d = ds.inter_feat
    seq = d[ds.iid_field + config["LIST_SUFFIX"]]
    length = d[config["ITEM_LIST_LENGTH_FIELD"]]
    y = d[ds.iid_field].numpy()
    seq_np, len_np = seq.numpy(), length.numpy()
    n = len(y)
    rows = np.arange(n)
    in_hist = np.zeros((n, int(seq_np.max()) + 1), bool)
    for j in range(seq_np.shape[1]):
        r = np.flatnonzero(j < len_np)
        in_hist[r, seq_np[r, j]] = True
    in_hist[:, 0] = False
    last = seq_np[rows, len_np - 1]
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, n, 8192):
            scores = model.full_sort_predict(
                {
                    model.ITEM_SEQ: seq[i : i + 8192],
                    model.ITEM_SEQ_LEN: length[i : i + 8192],
                }
            )
            scores[:, 0] = float("-inf")
            parts.append(torch.topk(scores, 5).indices.numpy())
    top5 = np.concatenate(parts)
    hit = (top5 == y[:, None]).any(1)
    old = in_hist[rows, y]
    r4 = lambda x: round(float(x), 4)
    return {
        "rows": n,
        "top1_is_last_booking": r4((top5[:, 0] == last).mean()),
        "last_booking_in_top5": r4((top5 == last[:, None]).any(1).mean()),
        "target_in_history": r4(old.mean()),
        "old_target_in_top5_of_all_rows": r4((hit & old).mean()),
        "old_target_in_top5_of_old_rows": r4(hit[old].mean()),
        "new_target": r4((~old).mean()),
        "new_target_in_top5_of_all_rows": r4((hit & ~old).mean()),
        "new_target_in_top5_of_new_rows": r4(hit[~old].mean()),
        "recall@5_check": r4(hit.mean()),
        "top5_slots_new_share": r4((~in_hist[rows[:, None], top5]).mean()),
        "global_popularity_top5_on_new_rows": r4(np.isin(y[~old], pop_top5).mean()),
    }


def stats_only() -> None:
    """Recompute behavior_stats from the checkpoint recorded in the result JSON."""
    path = OUT_DIR / "smlprec_expedia_run.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    ckpt = torch.load(result["checkpoint"])
    config = ckpt["config"]
    init_seed(config["seed"], config["reproducibility"])
    logging.disable(logging.CRITICAL)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    pop_top5 = top5_popular(train_data.dataset)
    model = SMLPREC(config, dataset)
    model.load_state_dict(ckpt["state_dict"])
    result["top5_behavior"] = {
        "checkpoint_epoch": result["best_epoch_by_valid"],
        "valid": behavior_stats(model, valid_data, config, pop_top5),
        "test": behavior_stats(model, test_data, config, pop_top5),
    }
    result["target_mix"] = {
        "train": target_mix(train_data, config),
        "valid": target_mix(valid_data, config),
        "test": target_mix(test_data, config),
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["top5_behavior"], indent=2))
    print(json.dumps(result["target_mix"], indent=2))


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
    per_epoch, cur = [], {}

    orig_train, orig_valid = trainer._train_epoch, trainer._valid_epoch

    def timed_train(*a, **k):
        t = time.time()
        loss = orig_train(*a, **k)
        cur["train_seconds"] = round(time.time() - t, 1)
        return loss

    def keep_valid(*a, **k):
        score, res = orig_valid(*a, **k)
        cur["valid"] = {m: round(float(v), 4) for m, v in res.items()}
        return score, res

    def after_epoch(epoch_idx, valid_score):
        """Test metrics of the current weights after every epoch (selection stays on valid)."""
        res = trainer.evaluate(test_data, load_best_model=False)
        row = {
            "epoch": epoch_idx + 1,
            "train_loss": round(float(trainer.train_loss_dict[epoch_idx]), 4),
            "train_seconds": cur["train_seconds"],
            "valid": cur["valid"],
            "test": {m: round(float(v), 4) for m, v in res.items()},
        }
        per_epoch.append(row)
        logger.info(
            f"epoch {epoch_idx + 1} test result (current weights): {row['test']}"
        )

    trainer._train_epoch, trainer._valid_epoch = timed_train, keep_valid
    t1 = time.time()
    _, best_valid = trainer.fit(
        train_data, valid_data, saved=True, callback_fn=after_epoch
    )
    t_fit = time.time() - t1
    best_epoch = max(
        per_epoch, key=lambda r: r["valid"][config["valid_metric"].lower()]
    )["epoch"]
    t2 = time.time()
    test_result = trainer.evaluate(test_data, load_best_model=True)
    t_test = time.time() - t2
    logger.info(f"test result: {test_result}")
    pop_top5 = top5_popular(train_data.dataset)
    top5_behavior = {
        "checkpoint_epoch": best_epoch,
        "valid": behavior_stats(model, valid_data, config, pop_top5),
        "test": behavior_stats(model, test_data, config, pop_top5),
    }
    logger.info(f"top-5 behaviour: {top5_behavior}")

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
        "run": "week3 pipeline validation, 3 epochs - SMLP4Rec on Expedia bookings (research adaptation)",
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
        "best_epoch_by_valid": best_epoch,
        "top5_behavior": top5_behavior,
        "target_mix": {
            "train": target_mix(train_data, config),
            "valid": target_mix(valid_data, config),
            "test": target_mix(test_data, config),
        },
        "checkpoint": str(trainer.saved_model_file),
        "per_epoch": per_epoch,
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
    if "--stats-only" in sys.argv:
        stats_only()
    else:
        main()
