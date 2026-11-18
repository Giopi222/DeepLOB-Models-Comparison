# Results: FI-2010, horizon k = 50

Test set: days 8-10, 139,290 windows. Mean ± std over seeds. Adam lr 0.001, batch 128, max epochs 50, early stopping on validation loss (patience 5). Hardware: Tesla T4.

| Model | Params | Seeds | Accuracy | Macro-F1 | Cohen's κ | Best epoch | Time / epoch (s) | Train time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepLOB (CNN-LSTM) | 60,947 | 5 | 0.780 ± 0.003 | 0.762 ± 0.003 | 0.653 ± 0.004 | 10.4 ± 2.7 | 33 ± 0 | 8.5 ± 1.5 |
| CNN-Transformer | 59,155 | 5 | 0.788 ± 0.003 | 0.771 ± 0.003 | 0.665 ± 0.005 | 12.8 ± 2.9 | 40 ± 0 | 12.0 ± 1.9 |

Test class mix (up / stationary / down): 27.5% / 47.4% / 25.1%. A constant 'stationary' predictor scores accuracy 0.474, macro-F1 0.214, κ 0.000.

Per-class F1 (mean over seeds) and confusion matrix (summed over seeds, row-normalised; rows = true class):

**DeepLOB (CNN-LSTM)**: F1 up 0.730, stationary 0.842, down 0.714

```
true \ pred      up   stat   down
up            0.709  0.166  0.125
stationary    0.068  0.862  0.070
down          0.127  0.169  0.704
```

**CNN-Transformer**: F1 up 0.738, stationary 0.847, down 0.727

```
true \ pred      up   stat   down
up            0.714  0.169  0.117
stationary    0.065  0.871  0.064
down          0.120  0.166  0.714
```
