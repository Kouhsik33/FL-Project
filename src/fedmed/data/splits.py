"""Global train / validation / test split.

The split is made once per (dataset, split_seed) and is shared by every
experiment that uses that seed, so centralised, local-only and federated runs
are evaluated on the identical held-out test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class GlobalSplit:
    train: np.ndarray  # row indices into the full dataset
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def make_split(
    y: np.ndarray,
    groups: np.ndarray | None = None,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> GlobalSplit:
    """Stratified split by label (and by site when natural groups exist)."""
    idx = np.arange(len(y))
    strat = y.astype(str) if groups is None else np.char.add(groups.astype(str), y.astype(str))
    trval, test = train_test_split(idx, test_size=test_frac, stratify=strat, random_state=seed)
    val_rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        trval, test_size=val_rel, stratify=strat[trval], random_state=seed + 1
    )
    split = GlobalSplit(np.sort(train), np.sort(val), np.sort(test))
    assert not (set(split.train) & set(split.test)) and not (set(split.val) & set(split.test))
    return split
