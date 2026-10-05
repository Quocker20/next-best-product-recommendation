"""Print the min / max booking timestamp of the week-3 train / valid / test splits.

Rebuilds the RecBole 80/10/10 split exactly as scripts/run_smlprec_expedia.py does
(same config, same seed, no training) and reads the `timestamp` of every target row
in each split. Timestamps are unix seconds of the booking `date_time`; the script
prints them as UTC datetimes too.

Usage (smlp4rec venv, Python 3.11):
    C:/Users/quoca/.venvs/smlp4rec/Scripts/python.exe scripts/week3_split_time_range.py
"""

import functools
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# RecBole 1.0.1 reloads its own checkpoints; PyTorch >= 2.6 defaults to weights_only=True.
torch.load = functools.partial(torch.load, weights_only=False)

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from src.models.smlprec import SMLPREC

CONFIG = ROOT / "configs" / "smlprec_expedia.yaml"
WORK = ROOT / "data" / "interim" / "recbole"


def fmt(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    os.chdir(WORK)  # RecBole writes ./log relative to cwd
    config = Config(
        model=SMLPREC,
        dataset="expedia",
        config_file_list=[str(CONFIG)],
        config_dict={"data_path": str(WORK), "checkpoint_dir": str(WORK / "saved")},
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print(f"{'split':<6} {'rows':>10}  {'min timestamp':>14}  {'max timestamp':>14}  min (UTC)            max (UTC)")
    prev_max = None
    for name, loader in (("train", train_data), ("valid", valid_data), ("test", test_data)):
        ds = loader.dataset
        ts = ds.inter_feat[ds.time_field].numpy()
        lo, hi = ts.min(), ts.max()
        print(f"{name:<6} {len(ts):>10}  {lo:>14.0f}  {hi:>14.0f}  {fmt(lo)}  {fmt(hi)}")
        if prev_max is not None and lo < prev_max:
            print(f"  note: {name} min is before the previous split's max (overlap)")
        prev_max = hi


if __name__ == "__main__":
    main()
