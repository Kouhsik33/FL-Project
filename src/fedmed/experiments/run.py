"""Run one experiment (centralised, local-only or federated) and save its results.

Usage:
    python -m fedmed.experiments.run --config configs/default.yaml [section.key=value ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from fedmed.clients.trainer import train
from fedmed.config import ExperimentCfg, load_config
from fedmed.evaluation.metrics import clean_for_json, disparity, evaluate_model
from fedmed.evaluation.summaries import summarise_communication, summarise_privacy, read_jsonl
from fedmed.experiments.prepare import Prepared, prepare
from fedmed.models.mlp import build_model, save_checkpoint
from fedmed.paths import PROJECT_ROOT
from fedmed.privacy.dp import compute_epsilon

DISPARITY_METRICS = ("accuracy", "recall", "precision", "f1", "specificity")


def environment_info() -> dict:
    import flwr, opacus, ray, sklearn, torch  # noqa: E401

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "flwr": flwr.__version__,
        "opacus": opacus.__version__,
        "ray": ray.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count": os.cpu_count(),
    }


def make_run_dir(cfg: ExperimentCfg) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"{stamp}_{cfg.name}_{cfg.mode}_s{cfg.seed}_{digest}"
    run_dir = PROJECT_ROOT / cfg.output.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def per_client_eval(model, prep: Prepared, thr: float) -> list[dict]:
    return [{"client": k, **evaluate_model(model, c["X_test"], c["y_test"], thr)} for k, c in enumerate(prep.clients)]


def disparities(per_client: list[dict]) -> dict:
    return {m: disparity(per_client, m) for m in DISPARITY_METRICS}


def dp_info_single(cfg: ExperimentCfg, res) -> dict | None:
    if not cfg.privacy.enabled:
        return None
    eps = compute_epsilon(cfg.privacy.noise_multiplier, res.sample_rate, res.steps, cfg.privacy.delta)
    return {
        "guarantee": "record-level (add/remove one record), Opacus DP-SGD, RDP accountant",
        "noise_multiplier": cfg.privacy.noise_multiplier,
        "max_grad_norm": cfg.privacy.max_grad_norm,
        "delta": cfg.privacy.delta,
        "sample_rate": res.sample_rate,
        "steps": res.steps,
        "epsilon": eps,
    }


def run_centralized(cfg: ExperimentCfg, prep: Prepared, run_dir: Path) -> dict:
    model = build_model(prep.X_train.shape[1], cfg.model.hidden, cfg.seed)
    res = train(model, prep.X_train, prep.y_train, cfg.train, cfg.train.epochs, cfg.seed,
                cfg.privacy if cfg.privacy.enabled else None)
    thr = cfg.train.threshold
    if cfg.output.save_checkpoints:
        save_checkpoint(model, run_dir / "checkpoints" / "centralized.pt", {"mode": "centralized"})
    pcs = per_client_eval(model, prep, thr)
    privacy = dp_info_single(cfg, res)
    return {
        "final": {
            "val": evaluate_model(model, prep.X_val, prep.y_val, thr),
            "test": evaluate_model(model, prep.X_test, prep.y_test, thr),
            "client_test": pcs,
            "disparity": disparities(pcs),
        },
        "timing": {"train_time_s": res.train_time_s, "steps": res.steps},
        "privacy": privacy,
        "communication": None,
    }


def run_local(cfg: ExperimentCfg, prep: Prepared, run_dir: Path) -> dict:
    """Each institution trains alone on its own partition (no collaboration)."""
    thr = cfg.train.threshold
    per_client, train_times, privacy = [], [], []
    for k, c in enumerate(prep.clients):
        model = build_model(prep.X_train.shape[1], cfg.model.hidden, cfg.seed)
        res = train(model, c["X_train"], c["y_train"], cfg.train, cfg.train.epochs, cfg.seed + 1000 * k,
                    cfg.privacy if cfg.privacy.enabled else None)
        train_times.append(res.train_time_s)
        if cfg.output.save_checkpoints:
            save_checkpoint(model, run_dir / "checkpoints" / f"local_client_{k}.pt", {"mode": "local", "client": k})
        per_client.append({
            "client": k,
            "n_train": int(len(c["y_train"])),
            "own_test": evaluate_model(model, c["X_test"], c["y_test"], thr),
            "global_test": evaluate_model(model, prep.X_test, prep.y_test, thr),
            "val": evaluate_model(model, prep.X_val, prep.y_val, thr),
        })
        if cfg.privacy.enabled:
            privacy.append({"client": k, **dp_info_single(cfg, res)})
    own = [{"client": p["client"], **p["own_test"]} for p in per_client]

    def mean_of(key, metric):
        vals = [p[key][metric] for p in per_client if p[key][metric] is not None]
        return float(np.mean(vals)) if vals else None

    # "test" reports the unweighted mean over the local models on the global test set.
    keys = ("accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc", "log_loss")
    return {
        "final": {
            "test": {m: mean_of("global_test", m) for m in keys} | {"aggregation": "mean over local models"},
            "val": {m: mean_of("val", m) for m in keys} | {"aggregation": "mean over local models"},
            "client_test": own,
            "disparity": disparities(own),
            "local_models": per_client,
        },
        "timing": {"train_time_s": float(sum(train_times)), "per_client_train_time_s": train_times},
        "privacy": {"per_client": privacy, "max_epsilon": max(p["epsilon"] for p in privacy)} if privacy else None,
        "communication": None,
    }


def run_federated(cfg: ExperimentCfg, prep: Prepared, run_dir: Path) -> dict:
    from flwr.simulation import run_simulation

    from fedmed.clients.flower_client import build_client_app
    from fedmed.server.server_app import build_server_app

    data_dir = run_dir / "client_data"
    data_dir.mkdir()
    for k, c in enumerate(prep.clients):
        # Each simulated institution gets only its own training partition.
        np.savez(data_dir / f"client_{k}.npz", X_train=c["X_train"], y_train=c["y_train"])
    eval_data = {
        "val": (prep.X_val, prep.y_val),
        "test": (prep.X_test, prep.y_test),
        "client_test": [(c["X_test"], c["y_test"]) for c in prep.clients],
    }
    n_clients = len(prep.clients)
    server_app = build_server_app(cfg, run_dir, eval_data, prep.X_train.shape[1])
    client_app = build_client_app(cfg, run_dir)
    t0 = time.perf_counter()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=n_clients,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
            "init_args": {"num_cpus": min(n_clients, os.cpu_count() or 1), "include_dashboard": False},
        },
    )
    sim_wall = time.perf_counter() - t0
    shutil.rmtree(data_dir)  # don't leave copies of patient-level data in results/
    log_path = run_dir / "server_log.json"
    if not log_path.exists():
        raise RuntimeError("Simulation finished without a server log; see console output for errors")
    server = json.loads(log_path.read_text())
    client_log = read_jsonl(run_dir / "client_log.jsonl")
    comm_log = read_jsonl(run_dir / "comm_log.jsonl")
    history = server["eval_history"]
    if not history:
        raise RuntimeError("No evaluation history recorded")
    last = history[-1]
    rounds = server["rounds"]
    fit_records = [r for r in client_log if r.get("event") == "fit"]
    return {
        "final": {
            "round": last["round"],
            "val": last["val"],
            "test": last["test"],
            "client_test": last["client_test"],
            "disparity": disparities(last["client_test"]),
        },
        "history": history,
        "rounds": rounds,
        "participation": {
            "rounds_aggregated": sum(1 for r in rounds if r.get("aggregated")),
            "rounds_skipped": sum(1 for r in rounds if not r.get("aggregated")),
            "participation_count": {
                k: sum(1 for r in rounds if k in r.get("participants", [])) for k in range(n_clients)
            },
            "simulated_failures": sum(1 for r in client_log if r.get("event") == "simulated_failure"),
        },
        "timing": {
            "simulation_wall_time_s": sim_wall,
            "fl_wall_time_s": server["fl_wall_time_s"],
            "client_train_time_s_total": float(sum(r["train_time_s"] for r in fit_records)),
            "client_train_time_s_mean_per_fit": float(np.mean([r["train_time_s"] for r in fit_records]))
            if fit_records else None,
            "aggregation_time_s_total": float(sum(r.get("aggregation_time_s") or 0 for r in rounds)),
            "server_eval_time_s_total": float(sum(h["eval_time_s"] for h in history)),
            "network_time": "not measured (in-process simulation)",
        },
        "communication": summarise_communication(comm_log, rounds),
        "privacy": summarise_privacy(cfg, fit_records),
    }


RUNNERS = {"centralized": run_centralized, "local": run_local, "federated": run_federated}


def run(cfg: ExperimentCfg) -> Path:
    if cfg.mode not in RUNNERS:
        raise ValueError(f"mode must be one of {list(RUNNERS)}")
    started = datetime.now(timezone.utc).isoformat()
    prep = prepare(cfg)
    run_id, run_dir = make_run_dir(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    (run_dir / "partition.json").write_text(json.dumps(clean_for_json(prep.partition.to_dict(prep.dataset.y)), indent=1))
    body = RUNNERS[cfg.mode](cfg, prep, run_dir)
    summary = {
        "run_id": run_id,
        "name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "tag": cfg.output.tag,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "simulated": True,
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "dataset": prep.dataset_info(),
        "partition": prep.partition.to_dict(prep.dataset.y),
        "preprocessing": {"type": "mean imputation + standardisation",
                          "fitted_on": "union of client training partitions (== pooled training split)"},
        **body,
    }
    (run_dir / "summary.json").write_text(json.dumps(clean_for_json(summary), indent=1))
    t = summary["final"]["test"]
    print(f"[{run_id}] test acc={t['accuracy']:.4f} f1={t['f1']} recall={t['recall']} auc={t['roc_auc']}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. federated.rounds=10")
    args = parser.parse_args(argv)
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
