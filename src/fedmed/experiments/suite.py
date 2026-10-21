"""The controlled experiment suite (A-G). Each run executes in its own process.

Usage:
    python -m fedmed.experiments.suite                 # everything (resumable)
    python -m fedmed.experiments.suite --groups A C    # selected groups
    python -m fedmed.experiments.suite --seeds 2 --list

Completed runs are recorded in results/suite_index.jsonl and skipped on rerun.
All runs share data.split_seed=0, so every WDBC run is evaluated on the same
114-record test set (and every Heart Disease run on the same 184-record test set).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fedmed.paths import PROJECT_ROOT, RESULTS_DIR

INDEX = RESULTS_DIR / "suite_index.jsonl"
SIGMAS = (0.6, 1.0, 1.5, 2.0, 3.0)


def experiments() -> list[tuple[str, str, list[str]]]:
    """(group, run name, overrides) without the seed."""
    E: list[tuple[str, str, list[str]]] = []
    # A: centralised vs local-only vs FedAvg (IID) at 3 / 5 / 10 institutions.
    E.append(("A", "central", ["mode=centralized"]))
    for k in (3, 5, 10):
        E.append(("A", f"local_iid_k{k}", ["mode=local", f"partition.n_clients={k}"]))
        E.append(("A", f"fedavg_iid_k{k}", ["mode=federated", f"partition.n_clients={k}"]))
    # B: data heterogeneity (5 institutions); local-only alongside each partition.
    for label, ov in [
        ("dir1.0", ["partition.strategy=dirichlet", "partition.alpha=1.0"]),
        ("dir0.3", ["partition.strategy=dirichlet", "partition.alpha=0.3"]),
        ("dir0.1", ["partition.strategy=dirichlet", "partition.alpha=0.1"]),
        ("qskew0.5", ["partition.strategy=quantity_skew", "partition.alpha=0.5"]),
    ]:
        E.append(("B", f"fedavg_{label}_k5", ["mode=federated", *ov]))
        E.append(("B", f"local_{label}_k5", ["mode=local", *ov]))
    # C: differential privacy (record-level DP-SGD), FedAvg IID k=5 and centralised.
    for s in SIGMAS:
        dp = ["privacy.enabled=true", f"privacy.noise_multiplier={s}"]
        E.append(("C", f"fedavg_dp_sigma{s}_k5", ["mode=federated", *dp]))
        E.append(("C", f"central_dp_sigma{s}", ["mode=centralized", *dp]))
    # D: client availability / robustness.
    for k in (5, 10):
        for rate in (0.2, 0.5):
            E.append(("D", f"fedavg_drop{rate}_k{k}", ["mode=federated", f"partition.n_clients={k}",
                                                       f"federated.dropout_rate={rate}"]))
    E.append(("D", "fedavg_midfail0.2_k5", ["mode=federated", "federated.midround_failure_rate=0.2"]))
    # F: secure aggregation cost (and SecAgg+ with DP).
    for k in (5, 10):
        E.append(("F", f"fedavg_secagg_k{k}", ["mode=federated", f"partition.n_clients={k}",
                                               "secagg.enabled=true", "federated.min_fit_clients=3"]))
    E.append(("F", "fedavg_secagg_dp1.0_k5", ["mode=federated", "secagg.enabled=true", "federated.min_fit_clients=3",
                                              "privacy.enabled=true", "privacy.noise_multiplier=1.0"]))
    # H: more communication rounds (pre-registered after seeing that 20 rounds converge
    # slower than centralised training; reported as separate configs, not test-selected).
    for rounds in (50, 100):
        E.append(("H", f"fedavg_iid_k5_r{rounds}", ["mode=federated", f"federated.rounds={rounds}"]))
        E.append(("H", f"fedavg_dir0.3_k5_r{rounds}", ["mode=federated", f"federated.rounds={rounds}",
                                                       "partition.strategy=dirichlet", "partition.alpha=0.3"]))
    E.append(("H", "fedavg_dp_sigma1.0_k5_r50", ["mode=federated", "federated.rounds=50",
                                                 "privacy.enabled=true", "privacy.noise_multiplier=1.0"]))
    # G: secondary dataset, natural partition by collection site (4 real sites).
    hd = ["data.dataset=heart_disease"]
    E.append(("G", "hd_central", ["mode=centralized", *hd]))
    E.append(("G", "hd_local_natural", ["mode=local", *hd, "partition.strategy=natural"]))
    E.append(("G", "hd_fedavg_natural", ["mode=federated", *hd, "partition.strategy=natural"]))
    E.append(("G", "hd_fedavg_iid_k4", ["mode=federated", *hd, "partition.n_clients=4"]))
    E.append(("G", "hd_fedavg_natural_dp1.0", ["mode=federated", *hd, "partition.strategy=natural",
                                               "privacy.enabled=true", "privacy.noise_multiplier=1.0"]))
    return E


# Groups whose federated runs are cheap enough for 5 seeds; SecAgg uses 3.
SEEDS_FOR_GROUP = {"F": 3, "H": 3}


def load_index() -> dict[str, dict]:
    if not INDEX.exists():
        return {}
    rows = [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]
    return {r["key"]: r for r in rows if r["status"] == "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    done = load_index()
    todo = []
    for group, name, ov in experiments():
        if args.groups and group not in args.groups:
            continue
        for seed in range(min(args.seeds, SEEDS_FOR_GROUP.get(group, args.seeds))):
            key = f"{group}|{name}|s{seed}"
            if key not in done:
                todo.append((key, group, name, seed, ov))
    print(f"{len(todo)} runs to do ({len(done)} already complete)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # absent in a fresh clone
    if args.list:
        for t in todo:
            print(" ", t[0], " ".join(t[4]))
        return
    for i, (key, group, name, seed, ov) in enumerate(todo, 1):
        cmd = [sys.executable, "-m", "fedmed.experiments.run", f"name={name}", f"seed={seed}",
               f"output.tag={group}", *ov]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines()[::-1] if ln.startswith("[")), "")
        run_id = line[1:line.index("]")] if line.startswith("[") and "]" in line else None
        status = "ok" if proc.returncode == 0 and run_id else "failed"
        rec = {"key": key, "group": group, "name": name, "seed": seed, "overrides": ov,
               "run_id": run_id, "status": status, "seconds": round(time.time() - t0, 1)}
        with INDEX.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{i}/{len(todo)}] {status} {key} {rec['seconds']}s {line}", flush=True)
        if status == "failed":
            (RESULTS_DIR / "suite_failures.log").open("a").write(
                f"==== {key}\n{' '.join(cmd)}\n{proc.stdout[-3000:]}\n{proc.stderr[-5000:]}\n")


if __name__ == "__main__":
    main()
