import numpy as np
import torch

from src.data import LOBWindows, make_datasets


def synthetic_raw(n: int, seed: int = 0) -> np.ndarray:
    """Array in FI-2010 layout (N, 149): 40 LOB features, ..., 5 label columns in {1,2,3}."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, 149), dtype=np.float32)
    raw[:, :40] = rng.normal(size=(n, 40))
    raw[:, 144:] = rng.integers(1, 4, size=(n, 5))
    return raw


def test_window_contents_and_label_alignment():
    x = np.repeat(np.arange(10, dtype=np.float32)[:, None], 40, axis=1)  # value = row index
    ds = LOBWindows([(x, np.arange(10) % 3)], window=3)
    xb, yb = ds.get(torch.tensor([0, 7]))
    assert len(ds) == 8 and xb.shape == (2, 1, 3, 40)
    assert xb[0, 0, :, 0].tolist() == [0, 1, 2] and yb[0] == 2 % 3  # label of the last row
    assert xb[1, 0, :, 0].tolist() == [7, 8, 9] and yb[1] == 9 % 3


def test_windows_never_cross_segments():
    segments = [(np.full((5, 40), k, np.float32), np.full(5, k)) for k in range(3)]
    ds = LOBWindows(segments, window=3)
    xb, yb = ds.get(torch.arange(len(ds)))
    assert len(ds) == 3 * 3
    assert all((xw == yw).all() for xw, yw in zip(xb, yb, strict=True))  # one segment each


def test_split_is_temporal_and_purged():
    raw = synthetic_raw(1000)
    raw[:, 0] = np.arange(1000)  # feature 0 = time index (z-scoring keeps it affine)
    train, val, _ = make_datasets(raw, [synthetic_raw(300, 1)], window=10, purge=100)
    step = train.x[1, 0] - train.x[0, 0]

    def rows(d):
        return torch.round((d.x[:, 0] - train.x[0, 0]) / step).long().tolist()

    assert rows(train) == list(range(700))  # 800 train rows minus the 100-row purge
    assert rows(val) == list(range(800, 1000))
    assert torch.equal(train.y, torch.from_numpy(raw[:700, 147].astype(np.int64) - 1))


def test_normalisation_uses_training_rows_only():
    raw, test_raw = synthetic_raw(1000), synthetic_raw(300, 1)
    train, val, test = make_datasets(raw, [test_raw], window=10)
    assert torch.allclose(train.x.mean(0), torch.zeros(40), atol=1e-5)

    raw_shifted, test_shifted = raw.copy(), test_raw.copy()
    raw_shifted[800:, :40] += 1000.0  # change validation rows only
    test_shifted[:, :40] += 1000.0
    train2, val2, test2 = make_datasets(raw_shifted, [test_shifted], window=10)
    assert torch.equal(train.x, train2.x)  # statistics do not depend on val/test
    # val and test are both shifted by 1000 / train_std: the same train-fitted transform
    assert torch.allclose((val2.x - val.x).mean(0), (test2.x - test.x).mean(0))
