"""Shared training loop (identical for both models) and evaluation metrics."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F

from src.data import LOBWindows

LABELS = [0, 1, 2]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) and disable cuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model: nn.Module, data: LOBWindows, batch_size: int = 1024) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and predicted classes, in eval mode (dropout off, no gradients)."""
    model.eval()
    total, preds = 0.0, []
    for x, y in data.batches(batch_size):
        logits = model(x)
        total += F.cross_entropy(logits, y, reduction="sum").item()
        preds.append(logits.argmax(dim=1).cpu())
    return total / len(data), torch.cat(preds).numpy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro-F1, Cohen's kappa, per-class F1 and confusion matrix."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro")),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
        "f1_per_class": f1_score(y_true, y_pred, labels=LABELS, average=None).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def train_model(
    model: nn.Module,
    train: LOBWindows,
    val: LOBWindows,
    *,
    epochs: int,
    patience: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict:
    """Train with Adam + cross-entropy and early stopping on validation loss.

    The weights of the best validation epoch are restored before returning.
    The test set is never seen here.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    shuffle_gen = torch.Generator().manual_seed(seed)
    val_labels = val.labels
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=train.x.device)
        for x, y in train.batches(batch_size, shuffle=True, generator=shuffle_gen):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(y)

        val_loss, val_pred = predict(model, val)
        val_f1 = float(f1_score(val_labels, val_pred, labels=LABELS, average="macro"))
        train_loss = loss_sum.item() / len(train)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1_macro": val_f1}
        )
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
            f"| val macro-F1 {val_f1:.4f} | {time.perf_counter() - start:7.0f}s"
        )

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)
    return {
        "epochs": epoch,
        "best_epoch": best_epoch,
        "train_time_s": train_time,
        "history": history,
    }
