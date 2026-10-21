"""Small multilayer perceptron and parameter (de)serialisation helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Binary classifier returning one logit per row.

    No batch normalisation (incompatible with per-sample gradients in DP-SGD)
    and no dropout regularisation, to keep the model compatible with Opacus and
    to avoid confusion with *client* dropout.
    """

    def __init__(self, n_features: int, hidden: list[int] | tuple[int, ...] = (32, 16)):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.n_features = n_features
        self.hidden = list(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(n_features: int, hidden: list[int], seed: int) -> MLP:
    torch.manual_seed(seed)
    return MLP(n_features, hidden)


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Parameters as a list of float32 arrays in state_dict order."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"Expected {len(keys)} arrays, got {len(weights)}")
    state = OrderedDict((k, torch.as_tensor(np.asarray(w, dtype=np.float32))) for k, w in zip(keys, weights))
    model.load_state_dict(state, strict=True)


def save_checkpoint(model: MLP, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": model.n_features,
            "hidden": model.hidden,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[MLP, dict]:
    blob = torch.load(path, map_location="cpu")
    model = MLP(blob["n_features"], blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    return model, blob.get("extra", {})
