"""Recommend the next hotel cluster for one user with the saved SMLP4Rec checkpoint.

Input is a booking history (oldest first): either a real Expedia user_id (their bookings
are read from the RecBole export; the last booking is held out and shown as the truth) or
a comma-separated list of past hotel clusters. Output is the top-K clusters with softmax
probabilities. History-only model: it does not see the current search (destination, dates,
party), and cluster ids are anonymous (no hotel names). Week-3 checkpoint, 3 epochs.

Usage (smlp4rec venv, Python 3.11):
    python scripts/recommend_smlprec_expedia.py --user-id 12
    python scripts/recommend_smlprec_expedia.py --history 12,45,45,3 --top-k 10
Options: --checkpoint PATH (default: the one recorded in reports/summary/week3/smlprec_expedia_run.json)
"""

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The checkpoint stores the RecBole Config object, so PyTorch >= 2.6 needs weights_only=False.
torch.load = functools.partial(torch.load, weights_only=False)

from recbole.data import create_dataset
from recbole.utils import init_seed

from src.models.smlprec import SMLPREC

RESULT = ROOT / "reports" / "summary" / "week3" / "smlprec_expedia_run.json"
INTER = ROOT / "data" / "interim" / "recbole" / "expedia" / "expedia.inter"


def user_history(user_id: int) -> list[int]:
    """Booked clusters of one user, oldest first, from the RecBole export."""
    # pandas is imported here, after torch: on Windows, pandas-before-torch makes c10.dll fail.
    import pandas as pd

    df = pd.read_csv(INTER, sep="\t")
    df = df[df["user_id:token"] == user_id].sort_values(
        "timestamp:float", kind="stable"
    )
    return df["item_id:token"].tolist()


def load_model(checkpoint: Path):
    """Return (model, dataset, config) rebuilt from a RecBole checkpoint."""
    ckpt = torch.load(checkpoint)
    config = ckpt["config"]
    init_seed(config["seed"], config["reproducibility"])
    logging.disable(logging.CRITICAL)
    dataset = create_dataset(config)
    model = SMLPREC(config, dataset).to(config["device"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logging.disable(logging.NOTSET)
    return model, dataset, config


@torch.no_grad()
def recommend(
    model, dataset, config, history: list[int], top_k: int
) -> list[tuple[int, int, float]]:
    """Top-K clusters for one history (raw cluster ids, oldest first).

    Returns (rank, cluster, probability) rows, best first.
    """
    max_len = config["MAX_ITEM_LIST_LENGTH"]
    known = dataset.field2token_id[dataset.iid_field]
    ids = [known[str(c)] for c in history if str(c) in known][-max_len:]
    if not ids:
        raise ValueError(
            "no known cluster in the history; the model needs at least one booking"
        )
    seq = torch.zeros(1, max_len, dtype=torch.long)
    seq[0, : len(ids)] = torch.tensor(ids)
    inter = {
        model.ITEM_SEQ: seq.to(config["device"]),
        model.ITEM_SEQ_LEN: torch.tensor([len(ids)]).to(config["device"]),
    }
    scores = model.full_sort_predict(inter)[0]
    scores[0] = float("-inf")  # padding id
    probs = torch.softmax(scores, dim=0)
    top = torch.topk(probs, top_k)
    tokens = dataset.id2token(dataset.iid_field, top.indices.cpu().numpy())
    return [
        (r + 1, int(t), round(float(p), 4))
        for r, (t, p) in enumerate(zip(tokens, top.values))
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--user-id", type=int, help="real Expedia user_id")
    who.add_argument("--history", help="comma-separated past clusters, oldest first")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--checkpoint", type=Path, default=None)
    a = ap.parse_args()

    ckpt = a.checkpoint or Path(
        json.loads(RESULT.read_text(encoding="utf-8"))["checkpoint"]
    )
    truth = None
    if a.user_id is not None:
        full = user_history(a.user_id)
        if len(full) < 2:
            sys.exit(
                f"user {a.user_id} has {len(full)} booking(s); need at least 2 to test"
            )
        history, truth = full[:-1], full[-1]
    else:
        history = [int(x) for x in a.history.split(",")]

    model, dataset, config = load_model(ckpt)
    rec = recommend(model, dataset, config, history, a.top_k)

    print(f"checkpoint: {ckpt.name}")
    print(
        f"history (oldest -> newest, last {config['MAX_ITEM_LIST_LENGTH']} used): {history}"
    )
    print("rank  cluster  probability")
    for rank, cluster, prob in rec:
        print(f"{rank:>4}  {cluster:>7}  {prob:>11.4f}")
    if truth is not None:
        hit = [rank for rank, cluster, _ in rec if cluster == truth]
        where = f"rank {hit[0]}" if hit else f"not in top {a.top_k}"
        print(f"actual next booking: cluster {truth} -> {where}")


if __name__ == "__main__":
    main()
