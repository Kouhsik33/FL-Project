"""Data access for the dashboard: reads only saved experiment outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fedmed.experiments.report import aggregate, flatten


@dataclass
class RunStore:
    runs: dict[str, dict] = field(default_factory=dict)  # run_id -> summary
    errors: list[str] = field(default_factory=list)

    @property
    def table(self) -> pd.DataFrame:
        if not self.runs:
            return pd.DataFrame()
        return pd.DataFrame([flatten(s) for s in self.runs.values()]).sort_values("run_id")

    def federated_ids(self) -> list[str]:
        return [k for k, s in self.runs.items() if s["mode"] == "federated"]


def load_runs(runs_dir: Path) -> RunStore:
    store = RunStore()
    if not runs_dir.exists():
        store.errors.append(f"Runs directory {runs_dir} does not exist yet.")
        return store
    for path in sorted(runs_dir.glob("*/summary.json")):
        try:
            s = json.loads(path.read_text())
            if "run_id" not in s or "final" not in s:
                raise ValueError("missing required keys")
            store.runs[s["run_id"]] = s
        except Exception as exc:  # noqa: BLE001 - surface any unreadable file to the user
            store.errors.append(f"{path.parent.name}: unreadable summary ({exc})")
    return store


def aggregated_by_group(store: RunStore) -> pd.DataFrame:
    df = store.table
    if df.empty or "group" not in df or df["group"].isna().all():
        return pd.DataFrame()
    df = df[df["group"].fillna("") != ""]
    return aggregate(df) if not df.empty else pd.DataFrame()


def dp_sweep(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Utility vs epsilon for the controlled DP sweep (group C) only, one point per configuration.

    Other DP runs (different rounds, SecAgg, dataset) are separate experiments and
    must not be pooled into the same sigma point; see `other_dp_runs`.
    """
    dp = table[(table["dp"] == True) & (table["group"] == "C")]  # noqa: E712
    return (dp.groupby(["name", "mode", "sigma"])
              .agg(epsilon_mean=("epsilon", "mean"), mean=(f"test_{metric}", "mean"),
                   std=(f"test_{metric}", "std"), n_seeds=("seed", "count"))
              .reset_index().sort_values(["mode", "epsilon_mean"]))


def other_dp_runs(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    dp = table[(table["dp"] == True) & (table["group"] != "C")]  # noqa: E712
    return (dp.groupby(["group", "name", "dataset", "sigma"])
              .agg(epsilon_mean=("epsilon", "mean"), mean=(f"test_{metric}", "mean"),
                   std=(f"test_{metric}", "std"), n_seeds=("seed", "count")).reset_index())


def history_frame(summary: dict, split: str = "test") -> pd.DataFrame:
    rows = []
    for h in summary.get("history", []):
        m = h[split]
        rows.append({"round": h["round"], **{k: m.get(k) for k in
                     ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "log_loss")}})
    return pd.DataFrame(rows)


def participation_frame(summary: dict) -> pd.DataFrame:
    """Long table: round x client -> state (unavailable / available-not-selected / crashed / participated)."""
    n = len(summary["partition"]["clients"])
    rows = []
    for r in summary.get("rounds", []):
        for k in range(n):
            if k in r.get("participants", []):
                state = "participated"
            elif k in r.get("instructed_to_fail", []):
                state = "crashed mid-round"
            elif k in r.get("available", []):
                state = "available, not aggregated"
            else:
                state = "unavailable"
            rows.append({"round": r["round"], "client": k, "state": state, "round_status": r.get("status")})
    return pd.DataFrame(rows)


def per_client_frame(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(summary["final"]["client_test"])
