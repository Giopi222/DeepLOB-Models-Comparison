import numpy as np
import pytest
import torch

from src.data import LOBWindows
from src.models import MODELS
from src.train import classification_metrics, predict, set_seed, train_model


def tiny_splits() -> tuple[LOBWindows, LOBWindows]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 40)).astype(np.float32)
    y = rng.integers(0, 3, size=500)
    return LOBWindows([(x[:300], y[:300])], 100), LOBWindows([(x[300:], y[300:])], 100)


def run(name: str, seed: int = 0) -> tuple[torch.nn.Module, dict]:
    train, val = tiny_splits()
    set_seed(seed)
    model = MODELS[name]()
    fit = train_model(model, train, val, epochs=2, patience=1, lr=1e-3, batch_size=64, seed=seed)
    return model, fit


@pytest.mark.parametrize("name", MODELS)
def test_training_smoke(name):
    model, fit = run(name)
    assert 1 <= fit["best_epoch"] <= fit["epochs"] <= 2
    assert np.isfinite(fit["history"][-1]["train_loss"])

    _, val = tiny_splits()
    loss, pred = predict(model, val)
    assert not model.training and np.isfinite(loss) and pred.shape == (len(val),)
    metrics = classification_metrics(val.labels, pred)
    assert np.array(metrics["confusion"]).sum() == len(val)


def test_same_seed_same_result():
    (_, a), (_, b) = run("deeplob", seed=1), run("deeplob", seed=1)
    assert [h["val_loss"] for h in a["history"]] == [h["val_loss"] for h in b["history"]]
