"""Parsers for the raw UCI files.

Parsing only converts the raw text into a feature table, binary labels and (for
Heart Disease) the collection site. No statistics are computed here, so nothing
in this module can leak information from the test set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fedmed.paths import RAW_DIR

WDBC_BASE = [
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
]
# wdbc.names: columns 3-32 are mean, standard error and "worst" of each base feature.
WDBC_FEATURES = (
    [f"{b}_mean" for b in WDBC_BASE]
    + [f"{b}_se" for b in WDBC_BASE]
    + [f"{b}_worst" for b in WDBC_BASE]
)

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
HEART_SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}
# Dropped because they are mostly missing outside Cleveland (see docs/dataset_assessment.md).
HEART_DROPPED = ["slope", "ca", "thal"]


@dataclass
class TabularDataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # int64, 1 = positive class
    positive_label: str
    negative_label: str
    groups: np.ndarray | None = None  # natural institution label, if any
    categorical: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def load_wdbc(raw_dir: Path = RAW_DIR) -> TabularDataset:
    path = raw_dir / "wdbc" / "wdbc.data"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python scripts/download_data.py wdbc`")
    df = pd.read_csv(path, header=None, names=["id", "diagnosis", *WDBC_FEATURES])
    if df.isna().any().any():
        raise ValueError("Unexpected missing values in wdbc.data")
    if set(df["diagnosis"]) != {"M", "B"}:
        raise ValueError(f"Unexpected diagnosis codes: {set(df['diagnosis'])}")
    if df["id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in wdbc.data")
    y = (df["diagnosis"] == "M").to_numpy(dtype=np.int64)
    X = df[WDBC_FEATURES].astype(np.float64).reset_index(drop=True)
    return TabularDataset(
        name="wdbc",
        X=X,
        y=y,
        positive_label="malignant (M)",
        negative_label="benign (B)",
        notes=["The ID column is dropped; it is an identifier, not a feature."],
    )


def load_heart_disease(raw_dir: Path = RAW_DIR) -> TabularDataset:
    frames = []
    for site, fname in HEART_SITES.items():
        path = raw_dir / "heart_disease" / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run `python scripts/download_data.py heart_disease`"
            )
        part = pd.read_csv(path, header=None, names=HEART_COLUMNS, na_values="?")
        part["site"] = site
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    # Serum cholesterol of 0 mg/dl is physiologically impossible; the Switzerland
    # file records 0 for every patient and VA for some, which encodes "not measured".
    df.loc[df["chol"] == 0, "chol"] = np.nan
    y = (df["num"] > 0).to_numpy(dtype=np.int64)
    X = df.drop(columns=["num", "site", *HEART_DROPPED])
    # One-hot encode nominal codes using the fixed category sets documented in
    # heart-disease.names (data-independent, so no leakage).
    cats = {"cp": [1, 2, 3, 4], "restecg": [0, 1, 2]}
    for col, levels in cats.items():
        for lvl in levels:
            X[f"{col}_{lvl}"] = np.where(X[col].isna(), np.nan, (X[col] == lvl).astype(float))
        X = X.drop(columns=col)
    return TabularDataset(
        name="heart_disease",
        X=X.astype(np.float64).reset_index(drop=True),
        y=y,
        positive_label="heart disease present (num > 0)",
        negative_label="no heart disease (num = 0)",
        groups=df["site"].to_numpy(),
        categorical=[c for c in X.columns if c.startswith(("cp_", "restecg_"))] + ["sex", "fbs", "exang"],
        notes=[
            f"Dropped {HEART_DROPPED}: mostly missing outside Cleveland.",
            "chol == 0 treated as missing.",
            "Remaining missing values are mean-imputed with training statistics.",
        ],
    )


LOADERS = {"wdbc": load_wdbc, "heart_disease": load_heart_disease}


def load_dataset(name: str, raw_dir: Path = RAW_DIR) -> TabularDataset:
    try:
        return LOADERS[name](raw_dir)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(LOADERS)}") from exc
