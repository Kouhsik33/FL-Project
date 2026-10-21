"""Sample-count-weighted Federated Averaging (McMahan et al., 2017).

    w_global = sum_k (n_k / sum_j n_j) * w_k      over participating clients k
"""

from __future__ import annotations

import numpy as np


def fedavg(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    if not results:
        raise ValueError("FedAvg needs at least one client result")
    total = sum(n for _, n in results)
    if total <= 0:
        raise ValueError("Total sample count must be positive")
    n_layers = len(results[0][0])
    if any(len(w) != n_layers for w, _ in results):
        raise ValueError("Clients returned different numbers of parameter arrays")
    agg = [np.zeros_like(results[0][0][i], dtype=np.float64) for i in range(n_layers)]
    for weights, n in results:
        for i, layer in enumerate(weights):
            agg[i] += (n / total) * layer.astype(np.float64)
    return [a.astype(np.float32) for a in agg]
