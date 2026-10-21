"""Experiment configuration (YAML file + dotted command-line overrides)."""

from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataCfg:
    dataset: str = "wdbc"  # wdbc | heart_disease
    val_frac: float = 0.2
    test_frac: float = 0.2
    split_seed: int = 0  # fixes the global test set; keep constant across comparable runs


@dataclass
class PartitionCfg:
    strategy: str = "iid"  # iid | dirichlet | quantity_skew | natural
    n_clients: int = 5
    alpha: float | None = None  # concentration for dirichlet / quantity_skew
    min_train_per_client: int = 8


@dataclass
class ModelCfg:
    hidden: list[int] = field(default_factory=lambda: [32, 16])


@dataclass
class TrainCfg:
    lr: float = 0.005  # selected on validation data (results/tuning_*.json)
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 16
    local_epochs: int = 1  # per FL round
    epochs: int = 20  # centralised and local-only baselines (= rounds x local_epochs)
    threshold: float = 0.5  # decision threshold on P(positive)


@dataclass
class FederatedCfg:
    rounds: int = 20
    fraction_fit: float = 1.0  # fraction of *available* clients selected each round
    min_fit_clients: int = 2  # a round with fewer available clients is skipped
    dropout_rate: float = 0.0  # P(client unavailable at round start), independent per round
    midround_failure_rate: float = 0.0  # P(selected client crashes during local training)
    round_timeout: float | None = 600.0


@dataclass
class PrivacyCfg:
    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    accountant: str = "rdp"
    secure_mode: bool = False  # Opacus cryptographically secure RNG (slower; see docs)


@dataclass
class SecAggCfg:
    enabled: bool = False
    num_shares: float = 1.0  # float = fraction of selected clients
    reconstruction_threshold: float = 0.6  # float = fraction of num_shares
    max_weight: float = 1000.0  # must exceed the largest client sample count
    clipping_range: float = 8.0
    quantization_range: int = 4194304
    timeout: float | None = 600.0


@dataclass
class OutputCfg:
    runs_dir: str = "results/runs"
    save_checkpoints: bool = True
    tag: str = ""


@dataclass
class ExperimentCfg:
    name: str = "default"
    mode: str = "federated"  # centralized | local | federated
    seed: int = 0  # training seed (model init, batching, DP noise, dropout schedule)
    data: DataCfg = field(default_factory=DataCfg)
    partition: PartitionCfg = field(default_factory=PartitionCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    federated: FederatedCfg = field(default_factory=FederatedCfg)
    privacy: PrivacyCfg = field(default_factory=PrivacyCfg)
    secagg: SecAggCfg = field(default_factory=SecAggCfg)
    output: OutputCfg = field(default_factory=OutputCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, values: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        f = known[name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        kwargs[name] = _build(type(default), value or {}) if is_dataclass(default) else value
    return cls(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like section.key=value")
        key, value = item.split("=", 1)
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentCfg:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(apply_overrides(raw, overrides or []))


def config_from_dict(raw: dict[str, Any]) -> ExperimentCfg:
    return _build(ExperimentCfg, raw)
