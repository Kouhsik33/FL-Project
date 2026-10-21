"""Turn a config into concrete, leakage-safe arrays for every party."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedmed.config import ExperimentCfg
from fedmed.data.datasets import TabularDataset, load_dataset
from fedmed.data.partition import Partition, make_partition
from fedmed.data.preprocessing import Preprocessor
from fedmed.data.splits import GlobalSplit, make_split


@dataclass
class Prepared:
    dataset: TabularDataset
    split: GlobalSplit
    partition: Partition
    preprocessor: Preprocessor
    X_train: np.ndarray  # pooled, transformed (centralised baseline only)
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    clients: list[dict[str, np.ndarray]]  # per client: X_train, y_train, X_val, y_val, X_test, y_test

    def dataset_info(self) -> dict:
        ds = self.dataset
        return {
            "name": ds.name,
            "n_records": int(len(ds.y)),
            "n_features": ds.n_features,
            "feature_names": list(ds.X.columns),
            "positive_label": ds.positive_label,
            "negative_label": ds.negative_label,
            "n_positive": int(ds.y.sum()),
            "notes": ds.notes,
            "split_sizes": {k: int(len(v)) for k, v in self.split.as_dict().items()},
            "split_positive": {k: int(ds.y[v].sum()) for k, v in self.split.as_dict().items()},
        }


def prepare(cfg: ExperimentCfg) -> Prepared:
    ds = load_dataset(cfg.data.dataset)
    split = make_split(ds.y, ds.groups, cfg.data.val_frac, cfg.data.test_frac, cfg.data.split_seed)
    pcfg = cfg.partition
    partition = make_partition(
        split.as_dict(), ds.y, pcfg.strategy, pcfg.n_clients, seed=cfg.seed, alpha=pcfg.alpha,
        groups=ds.groups, min_train_per_client=pcfg.min_train_per_client,
    )
    X_raw = ds.X.to_numpy(dtype=np.float64)
    names = list(ds.X.columns)
    # Statistics come only from client *training* partitions (== pooled training split).
    pre = Preprocessor.fit_federated([X_raw[c["train"]] for c in partition.client_indices], names)
    clients = []
    for c in partition.client_indices:
        clients.append({
            f"{kind}_{pool}": arr
            for pool, idx in c.items()
            for kind, arr in (("X", pre.transform(X_raw[idx])), ("y", ds.y[idx].astype(np.float32)))
        })
    return Prepared(
        dataset=ds, split=split, partition=partition, preprocessor=pre,
        X_train=pre.transform(X_raw[split.train]), y_train=ds.y[split.train].astype(np.float32),
        X_val=pre.transform(X_raw[split.val]), y_val=ds.y[split.val].astype(np.float32),
        X_test=pre.transform(X_raw[split.test]), y_test=ds.y[split.test].astype(np.float32),
        clients=clients,
    )
