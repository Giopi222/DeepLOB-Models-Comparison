"""DeepLOB (CNN-LSTM) vs CNN-Transformer on FI-2010: run the full comparison.

Downloads the data on first use, trains every (model, seed) pair with the same budget,
appends each finished run to <out>/runs.json (re-running skips completed runs) and writes
the aggregated table to <out>/summary.md.

    python main.py                          # both models, seeds 0-4, horizon k=50
    python main.py --seeds 0 --epochs 1     # quick end-to-end check
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from src.data import HORIZONS, load_fi2010, make_datasets
from src.models import MODELS, count_params
from src.train import classification_metrics, predict, set_seed, train_model

NAMES = {"deeplob": "DeepLOB (CNN-LSTM)", "transformer": "CNN-Transformer"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--horizon", type=int, default=50, choices=HORIZONS, help="label horizon k")
    p.add_argument("--epochs", type=int, default=50, help="maximum epochs per run")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    return p.parse_args()


def mean_std(values: list[float], fmt: str = ".3f") -> str:
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):{fmt}} ± {std:{fmt}}"


def summarize(runs: list[dict], y_test: np.ndarray, setup: dict) -> str:
    """Markdown summary: metrics as mean ± std over seeds, baseline, per-class detail."""
    device = runs[0]["device"] if runs else "n/a"
    lines = [
        f"# Results: FI-2010, horizon k = {setup['horizon']}",
        "",
        f"Test set: days 8-10, {len(y_test):,} windows. Mean ± std over seeds. "
        f"Adam lr {setup['lr']}, batch {setup['batch_size']}, max epochs {setup['epochs']}, "
        f"early stopping on validation loss (patience {setup['patience']}). Hardware: {device}.",
        "",
        "| Model | Params | Seeds | Accuracy | Macro-F1 | Cohen's κ | Best epoch "
        "| Time / epoch (s) | Train time (min) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        rs = [r for r in runs if r["model"] == name]
        if not rs:
            continue

        def test(key: str, rs: list[dict] = rs) -> list[float]:
            return [r["test"][key] for r in rs]

        lines.append(
            f"| {NAMES[name]} | {rs[0]['params']:,} | {len(rs)} | {mean_std(test('accuracy'))} "
            f"| {mean_std(test('f1_macro'))} | {mean_std(test('kappa'))} "
            f"| {mean_std([r['best_epoch'] for r in rs], '.1f')} "
            f"| {mean_std([r['train_time_s'] / r['epochs'] for r in rs], '.0f')} "
            f"| {mean_std([r['train_time_s'] / 60 for r in rs], '.1f')} |"
        )

    prior = np.bincount(y_test, minlength=3) / len(y_test)
    const = classification_metrics(y_test, np.ones_like(y_test))  # always "stationary"
    lines += [
        "",
        f"Test class mix (up / stationary / down): {' / '.join(f'{p:.1%}' for p in prior)}. "
        f"A constant 'stationary' predictor scores accuracy {const['accuracy']:.3f}, "
        f"macro-F1 {const['f1_macro']:.3f}, κ {const['kappa']:.3f}.",
        "",
        "Per-class F1 (mean over seeds) and confusion matrix (summed over seeds, "
        "row-normalised; rows = true class):",
    ]
    for name in MODELS:
        rs = [r for r in runs if r["model"] == name]
        if not rs:
            continue
        f1 = np.mean([r["test"]["f1_per_class"] for r in rs], axis=0)
        cm = np.sum([r["test"]["confusion"] for r in rs], axis=0)
        cm = cm / cm.sum(axis=1, keepdims=True)
        lines += [
            "",
            f"**{NAMES[name]}**: F1 up {f1[0]:.3f}, stationary {f1[1]:.3f}, down {f1[2]:.3f}",
            "",
            "```",
            "true \\ pred      up   stat   down",
            *(
                f"{cls:<12s} {row[0]:6.3f} {row[1]:6.3f} {row[2]:6.3f}"
                for cls, row in zip(("up", "stationary", "down"), cm, strict=True)
            ),
            "```",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = torch.cuda.get_device_name() if device.type == "cuda" else platform.processor()
    train, val, test = make_datasets(*load_fi2010(args.data_dir), horizon=args.horizon)
    train, val, test = train.to(device), val.to(device), test.to(device)
    print(f"windows: train {len(train):,} | val {len(val):,} | test {len(test):,} | {hardware}")

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    setup = {k: getattr(args, k) for k in ("horizon", "epochs", "patience", "lr", "batch_size")}

    for name in args.models:
        for seed in args.seeds:
            if any(r["model"] == name and r["seed"] == seed and r["setup"] == setup for r in runs):
                print(f"skipping {name} seed {seed}: already in {runs_path}")
                continue
            print(f"\n=== {NAMES[name]} | seed {seed} ===")
            set_seed(seed)
            model = MODELS[name]().to(device)
            fit = train_model(
                model,
                train,
                val,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
            )
            _, pred = predict(model, test)
            metrics = classification_metrics(test.labels, pred)
            print(
                f"test accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['f1_macro']:.4f} "
                f"| kappa {metrics['kappa']:.4f}"
            )
            runs.append(
                {
                    "model": name,
                    "seed": seed,
                    "setup": setup,
                    "params": count_params(model),
                    "device": hardware,
                    "torch": torch.__version__,
                    **fit,
                    "test": metrics,
                }
            )
            runs_path.write_text(json.dumps(runs, indent=1))

    current = [r for r in runs if r["setup"] == setup and r["model"] in args.models]
    summary = summarize(current, test.labels, setup)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
