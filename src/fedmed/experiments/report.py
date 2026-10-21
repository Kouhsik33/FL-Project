"""Aggregate saved runs into tables (mean +/- std over seeds).

Usage: python -m fedmed.experiments.report
Writes results/report/runs.csv, results/report/aggregated.csv and results/report/tables.md.
Only runs listed as complete in results/suite_index.jsonl are included.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedmed.paths import RESULTS_DIR, RUNS_DIR

INDEX = RESULTS_DIR / "suite_index.jsonl"
OUT = RESULTS_DIR / "report"
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc"]


def flatten(summary: dict) -> dict:
    cfg = summary["config"]
    final = summary["final"]
    test = final["test"]
    row = {
        "run_id": summary["run_id"],
        "group": summary.get("tag"),
        "name": summary["name"],
        "mode": summary["mode"],
        "seed": summary["seed"],
        "dataset": cfg["data"]["dataset"],
        "partition": cfg["partition"]["strategy"],
        "alpha": cfg["partition"]["alpha"],
        "n_clients": len(summary["partition"]["clients"]),
        "dp": cfg["privacy"]["enabled"],
        "sigma": cfg["privacy"]["noise_multiplier"] if cfg["privacy"]["enabled"] else None,
        "secagg": cfg["secagg"]["enabled"],
        "dropout_rate": cfg["federated"]["dropout_rate"],
        "midround_failure_rate": cfg["federated"]["midround_failure_rate"],
        "rounds": cfg["federated"]["rounds"] if summary["mode"] == "federated" else None,
    }
    for m in METRICS:
        row[f"test_{m}"] = test.get(m)
    for m in ("f1", "recall", "accuracy"):
        d = final["disparity"].get(m, {})
        row[f"client_{m}_gap"] = d.get("gap")
        row[f"client_{m}_worst"] = d.get("worst")
    priv = summary.get("privacy")
    if priv:
        row["epsilon"] = priv.get("max_epsilon", priv.get("epsilon"))
        row["delta"] = priv.get("delta")
    comm = summary.get("communication")
    if comm:
        row["comm_training_bytes"] = comm["training_total_bytes"]
        row["comm_bytes_per_round"] = comm["training_total_bytes_per_aggregated_round"]
        row["model_payload_bytes"] = comm["model_payload_bytes"]
        row["comm_messages"] = sum(v["messages"] for k, v in comm["by_stage"].items() if k.startswith("train"))
    t = summary["timing"]
    row["train_time_s"] = t.get("train_time_s", t.get("client_train_time_s_total"))
    row["fl_wall_time_s"] = t.get("fl_wall_time_s")
    part = summary.get("participation")
    if part:
        row["rounds_aggregated"] = part["rounds_aggregated"]
        row["rounds_skipped"] = part["rounds_skipped"]
        row["simulated_failures"] = part["simulated_failures"]
    return row


def load_runs(index: Path = INDEX, runs_dir: Path = RUNS_DIR) -> pd.DataFrame:
    rows = []
    if not index.exists():
        return pd.DataFrame()
    for line in index.read_text().splitlines():
        rec = json.loads(line)
        if rec["status"] != "ok":
            continue
        path = runs_dir / rec["run_id"] / "summary.json"
        if path.exists():
            rows.append(flatten(json.loads(path.read_text())))
    return pd.DataFrame(rows).drop_duplicates("run_id")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "name"]
    num = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])
           and c not in ("seed", "alpha", "n_clients", "sigma", "dropout_rate", "midround_failure_rate", "rounds", "delta")]
    meta = df.groupby(keys).agg(
        mode=("mode", "first"), dataset=("dataset", "first"), partition=("partition", "first"),
        alpha=("alpha", "first"), n_clients=("n_clients", "first"), sigma=("sigma", "first"),
        secagg=("secagg", "first"), dropout_rate=("dropout_rate", "first"), n_seeds=("seed", "nunique"),
        precision_defined=("test_precision", "count"),
    )
    stats = df.groupby(keys)[num].agg(["mean", "std"])
    stats.columns = [f"{a}_{b}" for a, b in stats.columns]
    return meta.join(stats).reset_index()


def fmt(mean, std, digits=3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown(agg: pd.DataFrame) -> str:
    cols = [("test_accuracy", "Accuracy"), ("test_precision", "Precision"), ("test_recall", "Recall"),
            ("test_specificity", "Specificity"), ("test_f1", "F1"), ("test_roc_auc", "ROC-AUC")]
    extra = {
        "B": [("client_f1_gap", "Client F1 gap"), ("client_recall_gap", "Client recall gap")],
        "C": [("epsilon", "ε (max client)")],
        "D": [("rounds_aggregated", "Rounds aggregated"), ("simulated_failures", "Crashes")],
        "F": [("comm_bytes_per_round", "Bytes/round"), ("comm_messages", "Messages")],
        "A": [("comm_bytes_per_round", "Bytes/round")],
        "G": [("client_f1_gap", "Client F1 gap")],
    }
    out = ["# Experiment results (mean ± std over seeds; held-out test set)\n",
           "Generated by `python -m fedmed.experiments.report` from saved runs. "
           "All runs are simulations on public data.\n",
           "† Precision is undefined for a model that predicts no positives; the mean is over the "
           "(k/n) seeds where it is defined. F1 is defined (0) in that case.\n"]
    for group, sub in agg.groupby("group"):
        cs = cols + extra.get(group, [])
        out.append(f"\n## Group {group}\n")
        out.append("| Run | Mode | Seeds | " + " | ".join(c[1] for c in cs) + " |")
        out.append("|" + "---|" * (3 + len(cs)))
        for _, r in sub.iterrows():
            vals = []
            for c, _ in cs:
                m, s = r.get(f"{c}_mean"), r.get(f"{c}_std")
                big = c.startswith("comm") or c.startswith("rounds") or c == "simulated_failures"
                cell = fmt(m, s, 0 if big else 3) if m is not None else "n/a"
                if c == "test_precision" and r["precision_defined"] < r["n_seeds"]:
                    cell += f" †({int(r['precision_defined'])}/{r['n_seeds']})"
                vals.append(cell)
            out.append(f"| {r['name']} | {r['mode']} | {r['n_seeds']} | " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


ROUND_STUDY = [  # (label, 20-round run, 50-round run, 100-round run)
    ("FedAvg IID k5", "fedavg_iid_k5", "fedavg_iid_k5_r50", "fedavg_iid_k5_r100"),
    ("FedAvg Dirichlet 0.3 k5", "fedavg_dir0.3_k5", "fedavg_dir0.3_k5_r50", "fedavg_dir0.3_k5_r100"),
    ("FedAvg DP σ=1.0 k5", "fedavg_dp_sigma1.0_k5", "fedavg_dp_sigma1.0_k5_r50", None),
]


def round_study(df: pd.DataFrame) -> str:
    """Group H vs the 20-round runs, restricted to the seeds both have (seed-matched)."""
    h_seeds = sorted(df[df["group"] == "H"]["seed"].unique())
    if not h_seeds:
        return ""
    out = [f"\n## Group H vs 20 rounds (seed-matched: seeds {h_seeds})\n",
           "| Configuration | Rounds | Accuracy | Recall | F1 | ROC-AUC | ε (max client) | Bytes total | FL wall time (s) |",
           "|---|---|---|---|---|---|---|---|---|"]
    ref = df[(df["name"] == "central") & df["seed"].isin(h_seeds)]
    if not ref.empty:
        out.append("| Centralised (reference) | — | " + " | ".join(
            fmt(ref[c].mean(), ref[c].std()) for c in ("test_accuracy", "test_recall", "test_f1", "test_roc_auc"))
            + " | — | — | — |")
    for label, *names in ROUND_STUDY:
        for rounds, name in zip((20, 50, 100), names):
            sub = df[(df["name"] == name) & df["seed"].isin(h_seeds)]
            if name is None or sub.empty:
                continue
            eps = fmt(sub["epsilon"].mean(), sub["epsilon"].std(), 2) if "epsilon" in sub and sub["epsilon"].notna().any() else "—"
            out.append(f"| {label} | {rounds} | " + " | ".join(
                fmt(sub[c].mean(), sub[c].std()) for c in ("test_accuracy", "test_recall", "test_f1", "test_roc_auc"))
                + f" | {eps} | {sub['comm_training_bytes'].mean():,.0f} | {sub['fl_wall_time_s'].mean():.1f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    df = load_runs()
    if df.empty:
        raise SystemExit("No completed runs; run `python -m fedmed.experiments.suite` first")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(OUT / "aggregated.csv", index=False)
    (OUT / "tables.md").write_text(markdown(agg) + round_study(df))
    print(f"{len(df)} runs -> {OUT}")


if __name__ == "__main__":
    main()
