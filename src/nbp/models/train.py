"""Training loop with strict early stopping, exact resume and per-epoch logging (SMLP4Rec venv).

`fit` is model-agnostic: the caller passes a loss on a batch of training row indices and an
evaluation function that scores the valid window. Per epoch:

1. one shuffled pass over the training rows (shuffle and dropout seeded by `seed` and the epoch, so
   a resumed run repeats an uninterrupted one exactly);
2. `eval_fn(model)` -> dict with at least `monitor` (autoregressive valid NDCG@10) and `valid_nll`;
3. stop rules: no gain > `min_delta` for `patience` epochs (`EarlyStopper`), or the valid NLL rising
   for `nll_guard_epochs` consecutive epochs while the train loss falls (`nll_guard`), or
   `max_epochs`.

Only `best.pth` (weights of the best epoch) and `last.pth` (resume state) are written.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nbp.models.early_stop import EarlyStopper, nll_guard


@dataclass
class TrainConfig:
    """Training hyperparameters (configs/train.yaml)."""

    max_epochs: int
    patience: int
    min_delta: float
    nll_guard_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int

    @classmethod
    def from_dict(cls, d: dict, seed: int) -> TrainConfig:
        keys = cls.__dataclass_fields__.keys() - {"seed"}
        if d.get("min_delta") is None:
            raise ValueError("min_delta is not set: run the min_delta calibration first")
        return cls(**{k: d[k] for k in keys}, seed=seed)


class StubDataset:
    """The only part of a RecBole dataset the SMLP4Rec models read: the item count."""

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    def num(self, field: str) -> int:
        return self.n_items


def git_hash(root: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def run_dir_for(experiments: Path, dataset: str, model: str, tag: str, today: str) -> Path:
    """`experiments/<date>_<dataset>_<model>_<tag>`; an existing folder with the same suffix is
    reused (so a run resumed on a later day keeps its folder and its `last.pth`)."""
    suffix = f"_{dataset}_{model}_{tag}"
    found = sorted(p for p in Path(experiments).glob(f"*{suffix}") if p.is_dir())
    return found[0] if found else Path(experiments) / f"{today}{suffix}"


@dataclass
class FitResult:
    best_epoch: int
    best_monitor: float
    stop_reason: str
    per_epoch: list
    best_path: Path


def fit(
    model: torch.nn.Module,
    n_train: int,
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    eval_fn: Callable[[torch.nn.Module], dict],
    cfg: TrainConfig,
    run_dir: Path,
    resume: bool = True,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Train `model` with early stopping on `eval_fn(model)["monitor"]`.

    Args:
        model: torch module (train / eval modes are set here).
        n_train: number of training rows; `loss_fn` receives int64 index batches into them.
        loss_fn: (model, idx) -> scalar loss of the rows `idx`.
        eval_fn: model -> dict with `monitor` (higher is better) and `valid_nll`; any other keys are
            logged. Called in eval mode under `torch.no_grad`.
        cfg: hyperparameters.
        run_dir: folder for `best.pth`, `last.pth` and `train.log`.
        resume: continue from `last.pth` if it exists.
        log: line printer (also appended to `train.log`).
    Returns:
        `FitResult`; the model is left with the best epoch's weights loaded.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"

    def say(msg: str) -> None:
        log(msg)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    per_epoch: list = []
    start = 1
    last = run_dir / "last.pth"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        saved = {k: v for k, v in ck.get("config", {}).items() if k != "max_epochs"}
        now = {k: v for k, v in asdict(cfg).items() if k != "max_epochs"}
        if saved != now:
            raise ValueError(f"{last} was trained with another config: {saved} != {now}")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        per_epoch, start = ck["per_epoch"], ck["epoch"] + 1
        say(f"resumed after epoch {ck['epoch']}")
    stopper = EarlyStopper.from_scores(
        [r["monitor"] for r in per_epoch], cfg.patience, cfg.min_delta
    )
    stop_reason = None
    if stopper.should_stop:
        stop_reason = "patience"
    elif per_epoch and nll_guard(
        [r["valid_nll"] for r in per_epoch],
        [r["train_loss"] for r in per_epoch],
        cfg.nll_guard_epochs,
    ):
        stop_reason = "nll_guard"

    for epoch in range(start, cfg.max_epochs + 1):
        if stop_reason:
            break
        t0 = time.time()
        model.train()
        torch.manual_seed(cfg.seed * 1000 + epoch)  # dropout stream of this epoch
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
        perm = torch.randperm(n_train, generator=gen)
        total, n_batches = 0.0, 0
        for i in range(0, n_train, cfg.batch_size):
            opt.zero_grad()
            loss = loss_fn(model, perm[i : i + cfg.batch_size])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}, batch {n_batches}"
                )
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        t_train = time.time() - t0
        model.eval()
        t1 = time.time()
        with torch.no_grad():
            ev = eval_fn(model)
        row = {
            "epoch": epoch,
            "train_loss": total / n_batches,
            "train_seconds": round(t_train, 1),
            "eval_seconds": round(time.time() - t1, 1),
            **ev,
        }
        is_best = stopper.update(epoch, float(ev["monitor"]))
        row["is_best"] = is_best
        per_epoch.append(row)
        if is_best:
            torch.save(model.state_dict(), run_dir / "best.pth")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "per_epoch": per_epoch,
                "config": asdict(cfg),
            },
            last,
        )
        say(
            f"epoch {epoch}: train loss {row['train_loss']:.4f} ({row['train_seconds']}s) | "
            f"valid monitor {ev['monitor']:.4f} nll {ev['valid_nll']:.4f} ({row['eval_seconds']}s) | "
            f"best {stopper.best_epoch}, no gain {stopper.bad_epochs}/{cfg.patience}"
        )
        if stopper.should_stop:
            stop_reason = "patience"
        elif nll_guard(
            [r["valid_nll"] for r in per_epoch],
            [r["train_loss"] for r in per_epoch],
            cfg.nll_guard_epochs,
        ):
            stop_reason = "nll_guard"
    if stop_reason is None:
        stop_reason = "max_epochs"
    if stopper.best_epoch == cfg.max_epochs:
        say(f"WARNING best epoch = max_epochs = {cfg.max_epochs}: possibly under-trained")
    for r in per_epoch:
        r["is_best"] = r["epoch"] == stopper.best_epoch
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False))
    model.eval()
    say(f"stop: {stop_reason}; best epoch {stopper.best_epoch} (monitor {stopper.best_score:.4f})")
    (run_dir / "per_epoch.json").write_text(json.dumps(per_epoch, indent=2), encoding="utf-8")
    (run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return FitResult(
        stopper.best_epoch, stopper.best_score, stop_reason, per_epoch, run_dir / "best.pth"
    )
