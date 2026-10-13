"""Training loop: early stopping, NLL guard, checkpoints and exact resume (tiny synthetic tasks)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # torch before anything that imports pandas (Windows c10.dll)

from nbp.models.early_stop import nll_guard
from nbp.models.train import TrainConfig, fit

X = torch.randn(64, 5, generator=torch.Generator().manual_seed(0))
Y = (X[:, 0] > 0).long()


def _cfg(**kw):
    base = dict(  # noqa: C408
        max_epochs=4,
        patience=2,
        min_delta=0.0,
        nll_guard_epochs=2,
        batch_size=16,
        learning_rate=0.05,
        weight_decay=0.0,
        seed=7,
    )
    base.update(kw)
    return TrainConfig(**base)


def _model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))


def _loss(model, idx):
    return torch.nn.functional.cross_entropy(model(X[idx]), Y[idx])


def _eval_real(model):
    logp = torch.log_softmax(model(X), 1)
    return {
        "monitor": float((logp.argmax(1) == Y).float().mean()),
        "valid_nll": float(-logp[range(64), Y].mean()),
    }


def test_resume_repeats_an_uninterrupted_run(tmp_path):
    full = fit(
        _model(),
        64,
        _loss,
        _eval_real,
        _cfg(max_epochs=4, patience=9),
        tmp_path / "a",
        log=lambda s: None,
    )
    fit(
        _model(),
        64,
        _loss,
        _eval_real,
        _cfg(max_epochs=2, patience=9),
        tmp_path / "b",
        log=lambda s: None,
    )
    resumed = fit(
        _model(),
        64,
        _loss,
        _eval_real,
        _cfg(max_epochs=4, patience=9),
        tmp_path / "b",
        log=lambda s: None,
    )
    keys = ("train_loss", "monitor", "valid_nll")
    assert [[r[k] for k in keys] for r in full.per_epoch] == [
        [r[k] for k in keys] for r in resumed.per_epoch
    ]


def _scripted(monitors, nlls):
    it = iter(zip(monitors, nlls))

    def ev(model):
        m, n = next(it)
        return {"monitor": m, "valid_nll": n}

    return ev


def test_patience_min_delta_and_best_weights_restored(tmp_path):
    model = _model()
    seen = {}
    ev = _scripted([0.10, 0.30, 0.305, 0.20], [1.0, 0.9, 0.9, 0.9])

    def ev_spy(m):
        out = ev(m)
        seen[len(seen) + 1] = {k: v.clone() for k, v in m.state_dict().items()}
        return out

    res = fit(
        model, 64, _loss, ev_spy, _cfg(min_delta=0.01, max_epochs=10), tmp_path, log=lambda s: None
    )
    # epoch 3 gains only 0.005 < min_delta: no improvement; epoch 4 drops: stop after 2 bad epochs
    assert res.best_epoch == 2 and res.stop_reason == "patience" and len(res.per_epoch) == 4
    assert [r["is_best"] for r in res.per_epoch] == [False, True, False, False]
    for k, v in model.state_dict().items():
        assert torch.equal(v, seen[2][k])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "best.pth" in names and "last.pth" in names
    assert not [n for n in names if n.startswith("epoch")]


def test_nll_guard_stops_training(tmp_path):
    # monitor keeps creeping up (patience never triggers) but the valid NLL rises twice in a row
    ev = _scripted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.0, 0.8, 0.9, 1.0, 1.1, 1.2])
    res = fit(
        _model(),
        64,
        _loss,
        ev,
        _cfg(max_epochs=6, learning_rate=0.01),
        tmp_path,
        log=lambda s: None,
    )
    tl = [r["train_loss"] for r in res.per_epoch]
    if all(np.diff(tl[1:4]) < 0):  # the guard needs the train loss to fall over the same epochs
        assert res.stop_reason == "nll_guard" and len(res.per_epoch) == 4


def test_nll_guard_rule():
    assert nll_guard([1.0, 1.1, 1.2], [0.5, 0.4, 0.3], 2)
    assert not nll_guard([1.0, 1.1, 1.0], [0.5, 0.4, 0.3], 2)
    assert not nll_guard([1.0, 1.1, 1.2], [0.5, 0.6, 0.3], 2)
    assert not nll_guard([1.0, 1.1], [0.5, 0.4], 2)
    with pytest.raises(ValueError):
        nll_guard([1.0], [1.0, 2.0], 1)


def test_resume_with_another_config_is_refused(tmp_path):
    fit(
        _model(),
        64,
        _loss,
        _eval_real,
        _cfg(max_epochs=1, patience=9),
        tmp_path,
        log=lambda s: None,
    )
    with pytest.raises(ValueError):
        fit(
            _model(),
            64,
            _loss,
            _eval_real,
            _cfg(max_epochs=2, patience=9, min_delta=0.5),
            tmp_path,
            log=lambda s: None,
        )
