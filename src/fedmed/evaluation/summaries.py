"""Summaries built from the per-run JSONL logs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fedmed.config import ExperimentCfg
from fedmed.privacy.dp import compute_epsilon


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise_communication(comm_log: list[dict], rounds: list[dict]) -> dict:
    """Measured protobuf-serialised message sizes, split by message stage."""
    by_stage: dict[str, dict] = defaultdict(lambda: {"messages": 0, "downlink_bytes": 0, "uplink_bytes": 0})
    per_round: dict[str, dict] = defaultdict(lambda: {"downlink_bytes": 0, "uplink_bytes": 0})
    for rec in comm_log:
        s = by_stage[rec["stage"]]
        s["messages"] += 1
        s["downlink_bytes"] += rec["downlink_bytes"]
        s["uplink_bytes"] += rec["uplink_bytes"]
        if rec["stage"].startswith("train"):
            pr = per_round[str(rec["round"])]
            pr["downlink_bytes"] += rec["downlink_bytes"]
            pr["uplink_bytes"] += rec["uplink_bytes"]
    train_down = sum(v["downlink_bytes"] for k, v in by_stage.items() if k.startswith("train"))
    train_up = sum(v["uplink_bytes"] for k, v in by_stage.items() if k.startswith("train"))
    model_bytes = next((r["model_payload_bytes"] for r in rounds if r.get("model_payload_bytes")), None)
    n_agg = sum(1 for r in rounds if r.get("aggregated"))
    return {
        "measurement": "protobuf-serialised Flower message content (excl. gRPC/TLS/TCP framing)",
        "model_payload_bytes": model_bytes,
        "by_stage": dict(by_stage),
        "training_downlink_bytes": train_down,
        "training_uplink_bytes": train_up,
        "training_total_bytes": train_down + train_up,
        "training_total_bytes_per_aggregated_round": (train_down + train_up) / n_agg if n_agg else None,
        "per_round": dict(per_round),
        "model_level_downlink_bytes": sum(r.get("downlink_model_bytes") or 0 for r in rounds),
        "model_level_uplink_bytes": sum(r.get("uplink_model_bytes") or 0 for r in rounds),
    }


def summarise_privacy(cfg: ExperimentCfg, fit_records: list[dict]) -> dict | None:
    if not cfg.privacy.enabled:
        return None
    per_client: dict[int, dict] = {}
    for rec in sorted(fit_records, key=lambda r: r["round"]):
        per_client[rec["client_id"]] = rec  # last (cumulative) record per client
    clients = []
    for cid, rec in sorted(per_client.items()):
        # Independent recomputation from the logged protocol parameters.
        recomputed = compute_epsilon(rec["noise_multiplier"], rec["dp_sample_rate"], rec["dp_total_steps"], rec["delta"])
        clients.append({
            "client": cid,
            "rounds_trained": sum(1 for r in fit_records if r["client_id"] == cid),
            "sample_rate": rec["dp_sample_rate"],
            "total_steps": rec["dp_total_steps"],
            "epsilon": rec["epsilon"],
            "epsilon_recomputed": recomputed,
        })
    return {
        "guarantee": (
            "Record-level (example-level) (epsilon, delta)-DP for each institution's training records, "
            "add/remove-one-record adjacency, Opacus DP-SGD with Poisson sampling, RDP accountant, "
            "composed over every local step in every round the institution participated in. "
            "Not client-level DP. Does not cover preprocessing statistics or reported sample counts."
        ),
        "noise_multiplier": cfg.privacy.noise_multiplier,
        "max_grad_norm": cfg.privacy.max_grad_norm,
        "delta": cfg.privacy.delta,
        "accountant": "rdp",
        "secure_rng": cfg.privacy.secure_mode,
        "per_client": clients,
        "max_epsilon": max((c["epsilon"] for c in clients), default=None),
    }
