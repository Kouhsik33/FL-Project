"""Binary classification metrics.

Positive class = 1 (malignant for WDBC, disease present for Heart Disease).
Precision, recall and F1 are computed for the positive class only (binary
averaging). Undefined values (e.g. precision with no predicted positives, or
ROC-AUC on a single-class shard) are reported as ``None`` rather than 0, so
that tiny client shards are not silently scored as failures.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from fedmed.models.mlp import MLP

METRIC_KEYS = (
    "accuracy", "precision", "recall", "specificity", "f1",
    "balanced_accuracy", "roc_auc", "log_loss",
)


def _div(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    n = len(y_true)
    out: dict = {"n": int(n), "n_pos": int(y_true.sum()), "threshold": threshold}
    if n == 0:
        out.update({k: None for k in METRIC_KEYS})
        out.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        return out
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    # F1 = 2TP / (2TP + FP + FN): defined (possibly 0) whenever there is any positive
    # label or prediction, even if precision is undefined because nothing was predicted positive.
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    bal = None if recall is None or specificity is None else (recall + specificity) / 2
    auc = float(roc_auc_score(y_true, prob)) if 0 < y_true.sum() < n else None
    eps = 1e-7
    p = np.clip(prob, eps, 1 - eps)
    log_loss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    out.update(
        accuracy=(tp + tn) / n, precision=precision, recall=recall, specificity=specificity,
        f1=f1, balanced_accuracy=bal, roc_auc=auc, log_loss=log_loss,
        tn=tn, fp=fp, fn=fn, tp=tp,
    )
    return out


@torch.no_grad()
def predict_proba(model: MLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    if len(X) == 0:
        return np.zeros(0)
    return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def evaluate_model(model: MLP, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    return compute_metrics(y, predict_proba(model, X), threshold)


def disparity(per_client: list[dict], metric: str) -> dict:
    """Best/worst/gap/std of a metric across clients, ignoring undefined values."""
    vals = [(c["client"], c[metric]) for c in per_client if c.get(metric) is not None]
    if not vals:
        return {"metric": metric, "n_defined": 0, "best": None, "worst": None, "gap": None, "std": None}
    scores = np.array([v for _, v in vals])
    best = max(vals, key=lambda t: t[1])
    worst = min(vals, key=lambda t: t[1])
    return {
        "metric": metric,
        "n_defined": len(vals),
        "n_undefined": len(per_client) - len(vals),
        "best": float(best[1]), "best_client": best[0],
        "worst": float(worst[1]), "worst_client": worst[0],
        "gap": float(best[1] - worst[1]),
        "std": float(scores.std()),
        "mean": float(scores.mean()),
    }


def clean_for_json(obj):
    """Replace NaN/inf with None recursively."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    return obj
