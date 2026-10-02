"""Week-3 experiment: history-length-dependent cap on past clusters in the SMLP4Rec top-5.

Same data, config and seed as scripts/run_smlprec_expedia.py (3 epochs from scratch, model
unchanged). Only the post-retrieval step differs, and there is no fixed slot quota. Each row
has L = number of bookings the user made before the current query (the model's history length).
The top 5 is the 5 best-scoring clusters subject to a cap on how many may be "past" clusters
(clusters already in the user's history, including the last booking):

    L < 5   -> at most 1 past cluster in the top 5
    L >= 5  -> at most 2 past clusters in the top 5

"At most": the past clusters compete on score like every other candidate, so a row can end up
with fewer past clusters than the cap; the remaining slots go to the best-scoring novel clusters.
The rule does not touch training, so the weights equal the earlier run (same seed); the script
checks this against results/week3_implementation/smlprec_expedia_run.json. Valid and test are scored
after every epoch, both plain top 5 and capped top 5; the best epoch is chosen on valid capped
NDCG@5. The rule only applies to K = 5, so only @5 metrics are reported.

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/run_smlprec_expedia_dynamic_cap.py
Outputs (new files, earlier runs' files are not touched):
    results/week3_implementation/smlprec_expedia_dynamic_cap_run.json
    results/week3_implementation/smlprec_expedia_dynamic_cap_run.log
    data/interim/recbole/saved_dynamic_cap/          (checkpoint, not committed)
"""

import functools
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# RecBole 1.0.1 reloads its own checkpoints; PyTorch >= 2.6 defaults to weights_only=True.
torch.load = functools.partial(torch.load, weights_only=False)

import run_smlprec_expedia as base
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from src.models.smlprec import SMLPREC

K = 5
N_PAST_MAX = 2
L_THRESHOLD = 5
CAP_LOW, CAP_HIGH = 1, 2  # max past clusters when L < L_THRESHOLD / L >= L_THRESHOLD
PREV = base.OUT_DIR / "smlprec_expedia_run.json"
OUT_JSON = base.OUT_DIR / "smlprec_expedia_dynamic_cap_run.json"
OUT_LOG = base.OUT_DIR / "smlprec_expedia_dynamic_cap_run.log"
CKPT_DIR = base.WORK / "saved_dynamic_cap"


def split_arrays(loader, config) -> dict:
    """Model inputs and truth of one split.

    Returns seq (n, L) and length (n,) tensors, y (n,) numpy target ids, and in_hist
    (n, n_items) bool: True where the cluster appears in the user's history (padding id 0 False).
    """
    ds = loader.dataset
    d = ds.inter_feat
    seq = d[ds.iid_field + config["LIST_SUFFIX"]]
    length = d[config["ITEM_LIST_LENGTH_FIELD"]]
    y = d[ds.iid_field].numpy()
    seq_np, len_np = seq.numpy(), length.numpy()
    in_hist = np.zeros((len(y), int(seq_np.max()) + 1), bool)
    for j in range(seq_np.shape[1]):
        r = np.flatnonzero(j < len_np)
        in_hist[r, seq_np[r, j]] = True
    in_hist[:, 0] = False
    return {"seq": seq, "length": length, "y": y, "in_hist": in_hist}


def capped_top5(
    scores: torch.Tensor, in_hist: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Top-5 cluster ids under the history-length-dependent cap on past clusters.

    scores: (b, n_items) model scores, padding column already -inf. in_hist: (b, n_items) bool.
    length: (b,) L = prior bookings. Returns (b, 5) ids ordered by score (highest first).
    """
    neg = torch.tensor(float("-inf"))
    past_v, past_i = torch.topk(torch.where(in_hist, scores, neg), N_PAST_MAX, dim=1)
    novel_v, novel_i = torch.topk(torch.where(in_hist, neg, scores), K, dim=1)
    cap = torch.where(length < L_THRESHOLD, CAP_LOW, CAP_HIGH)
    allow_past = torch.arange(N_PAST_MAX)[None, :] < cap[:, None]
    cand_v = torch.cat([past_v.masked_fill(~allow_past, float("-inf")), novel_v], 1)
    cand_i = torch.cat([past_i, novel_i], 1)
    order = torch.topk(cand_v, K, dim=1).indices
    return torch.gather(cand_i, 1, order)


@torch.no_grad()
def both_top5(model, arr) -> tuple[np.ndarray, np.ndarray]:
    """(plain top-5, capped top-5) ids for every row of one split, current weights."""
    model.eval()
    plain, capped = [], []
    n = len(arr["y"])
    for i in range(0, n, 8192):
        scores = model.full_sort_predict(
            {
                model.ITEM_SEQ: arr["seq"][i : i + 8192],
                model.ITEM_SEQ_LEN: arr["length"][i : i + 8192],
            }
        )
        scores[:, 0] = float("-inf")
        plain.append(torch.topk(scores, K, dim=1).indices.numpy())
        capped.append(
            capped_top5(
                scores,
                torch.from_numpy(arr["in_hist"][i : i + 8192]),
                arr["length"][i : i + 8192],
            ).numpy()
        )
    return np.concatenate(plain), np.concatenate(capped)


def metrics(top5: np.ndarray, arr) -> dict:
    """Recall/NDCG/MRR@5, old vs new hit shares, past-slot count, by L (prior bookings) bucket."""
    y, in_hist = arr["y"], arr["in_hist"]
    rows = np.arange(len(y))
    pos = top5 == y[:, None]
    hit = pos.any(1)
    rank = pos.argmax(1) + 1
    old = in_hist[rows, y]
    low = arr["length"].numpy() < L_THRESHOLD
    r4 = lambda x: round(float(x), 4)
    return {
        "recall@5": r4(hit.mean()),
        "ndcg@5": r4((hit / np.log2(rank + 1)).mean()),
        "mrr@5": r4((hit / rank).mean()),
        "old_target_hit_of_all_rows": r4((hit & old).mean()),
        "new_target_hit_of_all_rows": r4((hit & ~old).mean()),
        "old_target_hit_of_old_rows": r4(hit[old].mean()),
        "new_target_hit_of_new_rows": r4(hit[~old].mean()),
        "avg_past_slots_in_top5": r4(in_hist[rows[:, None], top5].sum(1).mean()),
        "recall@5_L_lt_5": r4(hit[low].mean()),
        "recall@5_L_ge_5": r4(hit[~low].mean()),
        "share_of_rows_L_lt_5": r4(low.mean()),
    }


def main() -> None:
    base.WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(base.WORK)  # RecBole writes ./log and ./log_tensorboard relative to cwd
    config = Config(
        model=SMLPREC,
        dataset="expedia",
        config_file_list=[str(base.CONFIG)],
        config_dict={"data_path": str(base.WORK), "checkpoint_dir": str(CKPT_DIR)},
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    t0 = time.time()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    arrays = {
        "valid": split_arrays(valid_data, config),
        "test": split_arrays(test_data, config),
    }
    t_data = time.time() - t0

    model = SMLPREC(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    per_epoch, cur = [], {}
    orig_train = trainer._train_epoch

    def timed_train(*a, **k):
        t = time.time()
        loss = orig_train(*a, **k)
        cur["train_seconds"] = round(time.time() - t, 1)
        return loss

    def after_epoch(epoch_idx, valid_score):
        row = {
            "epoch": epoch_idx + 1,
            "train_loss": round(float(trainer.train_loss_dict[epoch_idx]), 4),
            "train_seconds": cur["train_seconds"],
        }
        for name in ("valid", "test"):
            plain, capped = both_top5(model, arrays[name])
            row[name] = {
                "plain": metrics(plain, arrays[name]),
                "capped": metrics(capped, arrays[name]),
            }
        per_epoch.append(row)
        logger.info(
            f"epoch {row['epoch']} plain vs capped @5: {json.dumps({n: {v: row[n][v]['recall@5'] for v in ('plain', 'capped')} for n in ('valid', 'test')})}"
        )

    trainer._train_epoch = timed_train
    t1 = time.time()
    trainer.fit(train_data, valid_data, saved=True, callback_fn=after_epoch)
    t_fit = time.time() - t1

    best = max(per_epoch, key=lambda r: r["valid"]["capped"]["ndcg@5"])
    best_plain = max(per_epoch, key=lambda r: r["valid"]["plain"]["ndcg@5"])
    prev = json.loads(PREV.read_text(encoding="utf-8"))["per_epoch"]
    same_as_prev = [
        abs(r["test"]["plain"]["recall@5"] - p["test"]["recall@5"]) < 1e-4
        for r, p in zip(per_epoch, prev)
    ]
    result = {
        "run": "week3 experiment: history-length-dependent cap on past clusters in top-5, 3 epochs - SMLP4Rec on Expedia bookings",
        "rule": "L = prior bookings of the user; top 5 = 5 best-scoring clusters with at most 1 past cluster if L < 5 and at most 2 past clusters if L >= 5 (no fixed quota, past clusters compete on score); past = cluster in the user history incl. last booking",
        "config": {
            k: config[k]
            for k in (
                "MAX_ITEM_LIST_LENGTH",
                "n_layers",
                "hidden_size",
                "hidden_dropout_prob",
                "epochs",
                "train_batch_size",
                "learning_rate",
                "seed",
            )
        },
        "weights_match_previous_run_per_epoch": same_as_prev,
        "timing_seconds": {
            "data_build": round(t_data, 1),
            "train_and_eval": round(t_fit, 1),
        },
        "checkpoint": str(trainer.saved_model_file),
        "per_epoch": per_epoch,
        "best_epoch_by_valid_capped_ndcg@5": best["epoch"],
        "best_epoch_by_valid_plain_ndcg@5": best_plain["epoch"],
        "final_test_at_best_capped_epoch": best["test"],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logs = sorted((base.WORK / "log" / "SMLPREC").glob("*.log"), key=os.path.getmtime)
    if logs:
        shutil.copy(logs[-1], OUT_LOG)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
