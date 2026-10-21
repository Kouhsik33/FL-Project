"""Reproducible partitioning of the global splits across simulated institutions.

A partition is defined by a client-proportion matrix ``P`` of shape
(n_classes, n_clients): ``P[c, k]`` is the share of class-``c`` records that
client ``k`` holds. The *same* matrix is applied to the training, validation
and test pools, so each simulated institution's local test shard follows its
training distribution, and the union of all test shards is exactly the global
test set.

Strategies
----------
iid            equal shares of every class (stratified round-robin)
dirichlet      label skew: P[c, :] ~ Dir(alpha), as in Hsu et al. (2019)
quantity_skew  client sizes ~ Dir(alpha), same label mix for every client
natural        one client per real collection site (Heart Disease only)

These partitions are simulations. They do not represent real hospital
populations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRATEGIES = ("iid", "dirichlet", "quantity_skew", "natural")


@dataclass
class Partition:
    strategy: str
    n_clients: int
    seed: int
    alpha: float | None
    proportions: np.ndarray  # (n_classes, n_clients)
    client_indices: list[dict[str, np.ndarray]]  # per client: pool -> global row indices
    client_names: list[str]

    def summary(self, y: np.ndarray) -> list[dict]:
        rows = []
        for k, pools in enumerate(self.client_indices):
            row = {"client": k, "name": self.client_names[k]}
            for pool, idx in pools.items():
                labels = y[idx]
                row[f"n_{pool}"] = int(len(idx))
                row[f"n_{pool}_pos"] = int(labels.sum())
                row[f"n_{pool}_neg"] = int(len(idx) - labels.sum())
                row[f"pos_frac_{pool}"] = float(labels.mean()) if len(idx) else float("nan")
            rows.append(row)
        return rows

    def to_dict(self, y: np.ndarray) -> dict:
        return {
            "strategy": self.strategy,
            "n_clients": self.n_clients,
            "seed": self.seed,
            "alpha": self.alpha,
            "proportions": self.proportions.tolist(),
            "client_names": self.client_names,
            "clients": self.summary(y),
        }


def _split_by_proportions(idx: np.ndarray, props: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Split shuffled ``idx`` into len(props) contiguous chunks with the given shares."""
    idx = rng.permutation(idx)
    cuts = np.round(np.cumsum(props)[:-1] * len(idx)).astype(int)
    return np.split(idx, cuts)


def _assign(
    pools: dict[str, np.ndarray], y: np.ndarray, P: np.ndarray, rng: np.random.Generator
) -> list[dict[str, np.ndarray]]:
    n_clients = P.shape[1]
    out: list[dict[str, list]] = [{pool: [] for pool in pools} for _ in range(n_clients)]
    for pool, pool_idx in pools.items():
        for c in range(P.shape[0]):
            cls_idx = pool_idx[y[pool_idx] == c]
            for k, chunk in enumerate(_split_by_proportions(cls_idx, P[c], rng)):
                out[k][pool].extend(chunk.tolist())
    return [{pool: np.sort(np.asarray(v, dtype=np.int64)) for pool, v in d.items()} for d in out]


def make_partition(
    pools: dict[str, np.ndarray],
    y: np.ndarray,
    strategy: str,
    n_clients: int,
    seed: int,
    alpha: float | None = None,
    groups: np.ndarray | None = None,
    min_train_per_client: int = 8,
    max_tries: int = 1000,
) -> Partition:
    """Partition the given index pools (e.g. train/val/test) across clients."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown partition strategy {strategy!r}; choose from {STRATEGIES}")
    n_classes = int(y.max()) + 1
    rng = np.random.default_rng(seed)

    if strategy == "natural":
        if groups is None:
            raise ValueError("The 'natural' strategy needs dataset groups (collection sites)")
        names = list(dict.fromkeys(groups.tolist()))  # stable order of first appearance
        clients = [
            {pool: np.sort(idx[groups[idx] == g]) for pool, idx in pools.items()} for g in names
        ]
        counts = np.array([[np.sum(y[c["train"]] == cls) for c in clients] for cls in range(n_classes)], float)
        P = counts / counts.sum(axis=1, keepdims=True)
        return Partition(strategy, len(names), seed, None, P, clients, names)

    if n_clients < 2:
        raise ValueError("Need at least two clients")
    names = [f"client_{k}" for k in range(n_clients)]

    for _ in range(max_tries):
        if strategy == "iid":
            P = np.full((n_classes, n_clients), 1.0 / n_clients)
        elif strategy == "dirichlet":
            if alpha is None or alpha <= 0:
                raise ValueError("dirichlet partitioning needs alpha > 0")
            P = rng.dirichlet(np.full(n_clients, alpha), size=n_classes)
        else:  # quantity_skew
            if alpha is None or alpha <= 0:
                raise ValueError("quantity_skew partitioning needs alpha > 0")
            P = np.tile(rng.dirichlet(np.full(n_clients, alpha)), (n_classes, 1))
        clients = _assign(pools, y, P, rng)
        if min(len(c["train"]) for c in clients) >= min_train_per_client:
            return Partition(strategy, n_clients, seed, alpha, P, clients, names)
    raise RuntimeError(
        f"Could not draw a {strategy} partition with >= {min_train_per_client} training "
        f"records per client after {max_tries} tries; increase alpha or reduce n_clients."
    )
