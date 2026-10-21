"""Hyper-parameter selection on the VALIDATION split only (never the test set).

Grid-searches learning rate and epochs for the centralised MLP, averaged over
seeds, and selects by mean validation log loss. The chosen values are then
used unchanged for every centralised / local / federated experiment.

Usage: python -m fedmed.experiments.tune [--dataset wdbc]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from fedmed.clients.trainer import train
from fedmed.config import ExperimentCfg, TrainCfg
from fedmed.evaluation.metrics import evaluate_model
from fedmed.experiments.prepare import prepare
from fedmed.models.mlp import build_model
from fedmed.paths import RESULTS_DIR


LRS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
EPOCHS = [10, 20, 30, 50, 100]
BATCH_SIZE = 16


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="wdbc")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)
    lrs, epochs_grid, batch = LRS, EPOCHS, BATCH_SIZE
    rows = []
    for lr, epochs in itertools.product(lrs, epochs_grid):
        vals = []
        for seed in range(args.seeds):
            cfg = ExperimentCfg(seed=seed)
            cfg.data.dataset = args.dataset
            prep = prepare(cfg)
            tcfg = TrainCfg(lr=lr, batch_size=batch)
            model = build_model(prep.X_train.shape[1], cfg.model.hidden, seed)
            train(model, prep.X_train, prep.y_train, tcfg, epochs, seed)
            vals.append(evaluate_model(model, prep.X_val, prep.y_val))
        row = {
            "lr": lr, "epochs": epochs, "batch_size": batch,
            "val_log_loss_mean": float(np.mean([v["log_loss"] for v in vals])),
            "val_accuracy_mean": float(np.mean([v["accuracy"] for v in vals])),
            "val_roc_auc_mean": float(np.mean([v["roc_auc"] for v in vals])),
        }
        rows.append(row)
        print(row)
    best = min(rows, key=lambda r: r["val_log_loss_mean"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # absent in a fresh clone
    out = RESULTS_DIR / f"tuning_{args.dataset}.json"
    out.write_text(json.dumps({"selection": "min mean validation log loss", "seeds": args.seeds,
                               "best": best, "grid": rows}, indent=1))
    print("BEST", best, "->", out)


if __name__ == "__main__":
    main()
