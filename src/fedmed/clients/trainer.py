"""Local training loop shared by centralised, local-only and federated training."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedmed.config import PrivacyCfg, TrainCfg
from fedmed.models.mlp import MLP
from fedmed.privacy.dp import make_private

warnings.filterwarnings("ignore", message="Secure RNG turned off")
warnings.filterwarnings("ignore", message="Using a non-full backward hook")


@dataclass
class TrainResult:
    n_samples: int
    epochs: int
    steps: int
    mean_loss: float
    train_time_s: float
    sample_rate: float | None = None  # DP only: Poisson sampling rate q
    noise_multiplier: float | None = None
    max_grad_norm: float | None = None


def train(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainCfg,
    epochs: int,
    seed: int,
    privacy: PrivacyCfg | None = None,
) -> TrainResult:
    """Train `model` in place. With `privacy.enabled`, uses Opacus DP-SGD."""
    if len(X) == 0:
        raise ValueError("Cannot train on an empty dataset")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    dp = privacy is not None and privacy.enabled
    sample_rate = None
    train_module = model
    if dp:
        _, train_module, optimizer, loader = make_private(
            model, optimizer, loader, privacy.noise_multiplier, privacy.max_grad_norm, privacy.secure_mode
        )
        sample_rate = 1.0 / len(loader)  # Opacus: q = 1 / number of batches per epoch
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_module.train()
    steps, loss_sum, loss_count = 0, 0.0, 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            # Poisson sampling may yield an empty batch; Opacus still performs a
            # noise-only step, which the accountant has already counted.
            out = train_module(xb)
            loss = loss_fn(out, yb) if len(xb) else out.sum() * 0.0
            loss.backward()
            optimizer.step()
            steps += 1
            if len(xb):
                loss_sum += float(loss) * len(xb)
                loss_count += len(xb)
    elapsed = time.perf_counter() - t0
    if dp:
        # Opacus wraps `model` in a GradSampleModule that shares its parameters;
        # remove the hooks so the plain module can be evaluated/serialised.
        train_module._close()
    mean_loss = loss_sum / loss_count if loss_count else math.nan
    return TrainResult(
        n_samples=len(X), epochs=epochs, steps=steps, mean_loss=mean_loss, train_time_s=elapsed,
        sample_rate=sample_rate,
        noise_multiplier=privacy.noise_multiplier if dp else None,
        max_grad_norm=privacy.max_grad_norm if dp else None,
    )
