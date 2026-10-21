"""Leakage-safe preprocessing: mean imputation followed by standardisation.

Statistics are computed from *sufficient statistics* (count, sum, sum of squares
of the non-missing values per feature). This lets the same transform be fitted

* centrally, on the pooled training split, or
* "federatedly", by summing per-client sufficient statistics from each client's
  training partition only.

Because the client training partitions are a disjoint cover of the pooled
training split, both routes give identical statistics (tested in
tests/test_preprocessing.py). Validation and test data are never used to fit.

Privacy note: the per-client statistics are aggregate values computed outside
the DP-SGD mechanism, so they are NOT covered by the reported (epsilon, delta).
See docs/architecture.md, "What is and is not protected".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SufficientStats:
    count: np.ndarray  # non-missing values per feature
    total: np.ndarray
    total_sq: np.ndarray

    @classmethod
    def from_array(cls, X: np.ndarray) -> "SufficientStats":
        X = np.asarray(X, dtype=np.float64)
        mask = ~np.isnan(X)
        Xz = np.where(mask, X, 0.0)
        return cls(mask.sum(axis=0).astype(np.float64), Xz.sum(axis=0), (Xz**2).sum(axis=0))

    def __add__(self, other: "SufficientStats") -> "SufficientStats":
        return SufficientStats(
            self.count + other.count, self.total + other.total, self.total_sq + other.total_sq
        )


@dataclass
class Preprocessor:
    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]

    @classmethod
    def from_stats(cls, stats: SufficientStats, feature_names: list[str]) -> "Preprocessor":
        if np.any(stats.count == 0):
            empty = [f for f, c in zip(feature_names, stats.count) if c == 0]
            raise ValueError(f"No observed training values for features {empty}")
        mean = stats.total / stats.count
        var = np.maximum(stats.total_sq / stats.count - mean**2, 0.0)
        std = np.sqrt(var)
        std[std < 1e-12] = 1.0  # constant feature: leave centred values unscaled
        return cls(mean=mean, scale=std, feature_names=list(feature_names))

    @classmethod
    def fit(cls, X_train: np.ndarray, feature_names: list[str]) -> "Preprocessor":
        return cls.from_stats(SufficientStats.from_array(X_train), feature_names)

    @classmethod
    def fit_federated(
        cls, client_train_arrays: list[np.ndarray], feature_names: list[str]
    ) -> "Preprocessor":
        stats = [SufficientStats.from_array(x) for x in client_train_arrays]
        total = stats[0]
        for s in stats[1:]:
            total = total + s
        return cls.from_stats(total, feature_names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isnan(X), self.mean, X)  # mean imputation
        return ((X - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "feature_names": self.feature_names,
        }
