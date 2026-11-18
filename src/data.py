"""FI-2010 loading, leakage-free splitting and sliding-window batching.

Setup (as in DeepLOB, Zhang et al. 2018): train on days 1-7 (``Train_..._CF_7``), test on
days 8, 9, 10 (``Test_..._CF_7/8/9``). The last 20% of the training days is held out for
validation / early stopping.

Each raw file is a (149, N) matrix: rows 0-39 are the 10-level LOB snapshot
(ask price, ask volume, bid price, bid volume per level), rows 144-148 the labels for the
five horizons k = 10, 20, 30, 50, 100 events (1 = up, 2 = stationary, 3 = down).
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

# Mirror published by the DeepLOB authors (decimal-precision normalised FI-2010 subset).
DATA_URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"
)
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7"
TEST_FILES = [f"Test_Dst_NoAuction_DecPre_CF_{i}" for i in (7, 8, 9)]  # days 8, 9, 10
HORIZONS = (10, 20, 30, 50, 100)
CLASSES = ("up", "stationary", "down")  # label index 0, 1, 2
N_FEATURES = 40


def download_fi2010(data_dir: Path) -> None:
    """Download the FI-2010 subset and cache each file as a (N, 149) float32 ``.npy``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "data.zip"
    print(f"Downloading FI-2010 to {data_dir} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    for name in [TRAIN_FILE, *TEST_FILES]:
        txt = data_dir / f"{name}.txt"
        np.save(data_dir / f"{name}.npy", np.loadtxt(txt, dtype=np.float32).T)
        txt.unlink()


def load_fi2010(data_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the raw training array (days 1-7) and one raw array per test day."""
    if not all((data_dir / f"{n}.npy").exists() for n in [TRAIN_FILE, *TEST_FILES]):
        download_fi2010(data_dir)
    train = np.load(data_dir / f"{TRAIN_FILE}.npy")
    tests = [np.load(data_dir / f"{n}.npy") for n in TEST_FILES]
    return train, tests


class LOBWindows:
    """Sliding windows of ``window`` consecutive LOB snapshots over one or more segments.

    The sample ending at row ``t`` has input ``x[t - window + 1 : t + 1]`` and label ``y[t]``.
    Windows never cross segment boundaries (e.g. train/val split or two test days).
    """

    def __init__(self, segments: list[tuple[np.ndarray, np.ndarray]], window: int) -> None:
        ends, offset = [], 0
        for x, _ in segments:
            ends.append(np.arange(offset + window - 1, offset + len(x)))
            offset += len(x)
        self.x = torch.from_numpy(np.concatenate([x for x, _ in segments]))
        self.y = torch.from_numpy(np.concatenate([y for _, y in segments]))
        self.ends = torch.from_numpy(np.concatenate(ends))
        self.window = window

    def __len__(self) -> int:
        return len(self.ends)

    def to(self, device: torch.device | str) -> LOBWindows:
        self.x, self.y, self.ends = self.x.to(device), self.y.to(device), self.ends.to(device)
        return self

    @property
    def labels(self) -> np.ndarray:
        return self.y[self.ends].cpu().numpy()

    def get(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return inputs of shape (B, 1, window, 40) and labels of shape (B,)."""
        t = self.ends[idx.to(self.ends.device)]
        rows = t[:, None] + torch.arange(1 - self.window, 1, device=t.device)
        return self.x[rows].unsqueeze(1), self.y[t]

    def batches(
        self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None
    ):
        """Iterate over (x, y) mini-batches."""
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        for i in range(0, n, batch_size):
            yield self.get(order[i : i + batch_size])


def make_datasets(
    train_raw: np.ndarray,
    test_raws: list[np.ndarray],
    horizon: int = 50,
    window: int = 100,
    val_frac: float = 0.2,
    purge: int = 100,
) -> tuple[LOBWindows, LOBWindows, LOBWindows]:
    """Build train / validation / test windows without look-ahead.

    - Temporal split: the last ``val_frac`` of the training days is the validation set.
    - ``purge`` rows before the validation start are dropped, so no training label
      (which looks up to 100 events ahead) depends on validation prices.
    - Z-score statistics are fitted on the training rows only and applied to all splits.
    - Each test day is a separate segment, so no window mixes two days.
    """
    col = 144 + HORIZONS.index(horizon)  # columns 144-148 hold the five label horizons

    def split_xy(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return raw[:, :N_FEATURES], raw[:, col].astype(np.int64) - 1

    x, y = split_xy(train_raw)
    val_start = int(len(x) * (1 - val_frac))
    train_end = val_start - purge

    mean = x[:train_end].mean(axis=0, dtype=np.float64)
    std = x[:train_end].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0

    def norm(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    train = LOBWindows([(norm(x[:train_end]), y[:train_end])], window)
    val = LOBWindows([(norm(x[val_start:]), y[val_start:])], window)
    test = LOBWindows([(norm(xt), yt) for xt, yt in map(split_xy, test_raws)], window)
    return train, val, test
