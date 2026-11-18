# DeepLOB vs. CNN-Transformer on FI-2010

[![CI](https://github.com/Giopi222/DeepLOB-Models-Comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/Giopi222/DeepLOB-Models-Comparison/actions/workflows/ci.yml)

Transformers replaced LSTMs in most sequence-modelling tasks. Does the same hold for Limit Order Book (LOB) data? This repo takes **DeepLOB** (Zhang et al., 2018), a CNN-LSTM for mid-price
prediction, swaps its LSTM for a **Transformer encoder**, and compares the two under a controlled
protocol: same CNN front-end, same data pipeline, same optimiser and training budget, matched
parameter count (~60k), several seeds.

**Result:** the Transformer is better by a small but consistent margin, +0.9 pt macro-F1 over seeds (every Transformer seed beats every DeepLOB seed), but it needs about 40% more training time.

## Task

Given the last 100 LOB snapshots (10 levels × ask/bid price and volume = 40 features), classify the
mid-price move over the next *k* = 50 events as **up / stationary / down**.   
FI-2010 labels compare the mean of the next *k* mid-prices with the current one (Ntakaris et al., 2017).

## Models

```text
             LOB window (1 × 100 × 40)
                        │
  conv 1×2 stride 1×2 (price + volume) → 2 × conv 4×1     40 → 20
  conv 1×2 stride 1×2 (ask + bid)      → 2 × conv 4×1     20 → 10
  conv 1×10 (all levels)               → 2 × conv 4×1     10 → 1
  inception: [1×1 → 3×1] [1×1 → 5×1] [maxpool 3×1 → 1×1]  → sequence 100 × 96
                        │
          ┌─────────────┴──────────────┐
   LSTM(64), last hidden state    linear 96 → 64 + sinusoidal positions
          │                       Transformer encoder, 1 layer, 4 heads, FFN 128
          │                       output at the last time step
   linear → 3 classes             linear → 3 classes

   DeepLOB: 60,947 parameters     CNN-Transformer: 59,155 parameters
```

- The CNN front-end is shared, so the only difference is the temporal block (*LSTM vs. attention*).
- DeepLOB follows the paper: 16 filters, inception with 32 filters per branch, Leaky ReLU (0.01), no
  batch norm. The PyTorch port reproduces the outputs of the original Keras model to within 1e-7
  when given the same weights.
- Attention is bidirectional (no causal mask). Every snapshot in the window comes before the label,
  so there is no look-ahead. The Transformer uses dropout 0.1, the standard setting; DeepLOB has
  none, as in the paper.

## Data and protocol

- **FI-2010** (Ntakaris et al., 2017): LOB data for 5 stocks traded on Nasdaq Nordic (Helsinki)
  over 10 trading days, licensed CC BY 4.0. `main.py` downloads the decimal-precision version
  (56 MB) from the mirror published by the DeepLOB authors.
- **Split** (DeepLOB "setup 2"): train on days 1–7, test on days 8–10. Validation, used for early
  stopping only, is the last 20% of the training rows. The 100 rows before it are dropped, so no
  training label depends on validation prices.
- **Normalisation**: per-feature z-score fitted on the training rows only, then applied to
  validation and test. Decimal-precision files are a pure rescaling of raw prices and volumes, so
  this is equivalent to z-scoring the raw data with training statistics.
- **Windows** never cross the train/val boundary or the boundary between two test days.
- **Class mix** (up / stationary / down): train 35.1 / 30.3 / 34.6%, validation
  31.6 / 37.0 / 31.4%, test 27.5 / 47.4 / 25.1%. Because the test set is stationary-heavy, the
  results report macro-F1 and Cohen's κ alongside accuracy.
- **Training** is identical for both models: cross-entropy loss, Adam with learning rate 1e-3,
  batch size 128, up to 50 epochs, early stopping on validation loss (patience 5) with the best
  weights restored, seeds 0–4.

## Reproduce

```bash
pip install -r requirements.txt
python main.py            # both models × 5 seeds; downloads FI-2010 on the first run
python main.py --help     # horizon, seeds, epochs, learning rate, ...
```

Each finished run is appended to `results/runs.json`; the aggregated table goes to
`results/summary.md`. Re-running skips completed runs, so an interrupted session can be resumed.
A GPU is strongly recommended. On a Tesla T4 the full run (2 models × 5 seeds) took about 1 h 45 min.
On a laptop CPU (Ryzen 7 5800U) a single epoch takes 8–13 minutes. On Google Colab with a GPU
runtime:

```bash
!git clone https://github.com/Giopi222/DeepLOB-Models-Comparison.git
%cd DeepLOB-Models-Comparison
!pip install -q -r requirements.txt
!python main.py           # add --out /content/drive/MyDrive/<dir> (Drive mounted) to keep
                          # results across runtime resets, so the resume works
```

Tests (shapes and parameter parity, leakage checks, training smoke test): `pip install pytest && pytest`.

## Results

Test days 8–10, horizon *k* = 50, Tesla T4. Values are mean ± std over seeds 0–4.

| Model | Params | Accuracy | Macro-F1 | Cohen's κ | Best epoch | Time / epoch | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLOB (CNN-LSTM) | 60,947 | 0.780 ± 0.003 | 0.762 ± 0.003 | 0.653 ± 0.004 | 10.4 ± 2.7 | 33 s | 8.5 ± 1.5 min |
| CNN-Transformer | 59,155 | **0.788 ± 0.003** | **0.771 ± 0.003** | **0.665 ± 0.005** | 12.8 ± 2.9 | 40 s | 12.0 ± 1.9 min |
| Constant "stationary" | – | 0.474 | 0.214 | 0.000 | – | – | – |

Per-class F1 (up / stationary / down): DeepLOB 0.730 / 0.842 / 0.714; CNN-Transformer
0.738 / 0.847 / 0.727. Confusion matrices are in [`results/summary.md`](results/summary.md) and the
per-epoch logs in [`results/train_log.txt`](results/train_log.txt).


## Limitations:

- Seeds capture training randomness only. The test set is a single 3-day period, so whether the
  gap holds across other stocks, periods and horizons is untested. Validation macro-F1 (about 0.68)
  is well below test macro-F1 (about 0.76) for both models: absolute levels depend heavily on the
  period, and the relative ranking is the more robust finding.
- FI-2010 is small and old: 5 stocks, 10 days in 2010, one venue. Conclusions may not transfer to
  other markets or periods.
- Labels are smoothed (mean of the next *k* mid-prices) and thresholded. Classification metrics are
  not trading performance: there are no transaction costs and no P&L.
- The files concatenate stocks (and, for training, days) without boundary markers, so under 2% of
  windows straddle a stock or day boundary. This affects both models equally.
- Hyperparameters are shared, not tuned per model. Transformers are often more sensitive to the
  learning rate and warm-up, so a small per-model sweep on validation is the natural next step.


## Repository layout

```text
main.py          experiment runner: train, evaluate, summarise (argparse)
src/data.py      download, purged temporal split, train-only z-score, sliding windows
src/models.py    shared CNN front-end, DeepLOB, CNN-Transformer
src/train.py     training loop with early stopping, metrics
tests/           model shapes, leakage checks, training smoke test
results/         summary of the reported run and per-epoch training logs
```

## References

- Zhang, Z., Zohren, S., Roberts, S. (2018). *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.*
- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., Iosifidis, A. (2017). *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.*
- Vaswani, A., et al. (2017). *Attention Is All You Need.* 
