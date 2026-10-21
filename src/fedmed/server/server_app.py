"""Coordinator: a Flower ServerApp running FedAvg, optionally with SecAgg+."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from fedmed.clients.flower_client import IDENTITY_KEY
from fedmed.config import ExperimentCfg
from fedmed.evaluation.metrics import clean_for_json, evaluate_model
from fedmed.models.mlp import MLP, build_model, get_weights, save_checkpoint, set_weights
from fedmed.server.strategy import SimulatedFedAvg, availability_schedule, failure_schedule


def register_clients(grid: Grid, n_expected: int, timeout: float = 120.0) -> dict[int, int]:
    """Registration handshake: ask each connected node for its institution index."""
    deadline = time.time() + timeout
    node_ids: list[int] = []
    while time.time() < deadline:
        node_ids = list(grid.get_node_ids())
        if len(node_ids) >= n_expected:
            break
        time.sleep(0.5)
    if len(node_ids) < n_expected:
        raise RuntimeError(f"Only {len(node_ids)} of {n_expected} clients connected")
    query = [
        Message(RecordDict({IDENTITY_KEY: ConfigRecord({"request": "client_id"})}),
                dst_node_id=nid, message_type=MessageType.QUERY)
        for nid in node_ids
    ]
    mapping: dict[int, int] = {}
    for reply in grid.send_and_receive(query, timeout=timeout):
        if reply.has_error():
            raise RuntimeError(f"Registration failed: {reply.error}")
        mapping[reply.metadata.src_node_id] = int(reply.content.config_records[IDENTITY_KEY]["client_id"])
    if sorted(mapping.values()) != list(range(n_expected)):
        raise RuntimeError(f"Unexpected client ids {sorted(mapping.values())}")
    return mapping


def build_server_app(cfg: ExperimentCfg, run_dir: Path, eval_data: dict, n_features: int) -> ServerApp:
    """`eval_data` holds the coordinator's evaluation sets (simulation oracle):
    global val/test plus each client's test shard. The coordinator never
    receives any client's training data."""
    if cfg.secagg.enabled and cfg.federated.min_fit_clients < 3:
        # With two participants each one can subtract its own update from the
        # revealed sum and recover the other's, defeating secure aggregation.
        raise ValueError("SecAgg+ requires federated.min_fit_clients >= 3")
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        n_clients = len(eval_data["client_test"])
        node_to_client = register_clients(grid, n_clients)
        init_model = build_model(n_features, cfg.model.hidden, seed=cfg.seed)
        history: list[dict] = []
        final: dict = {}
        eval_model = MLP(n_features, cfg.model.hidden)

        def evaluate_fn(server_round, ndarrays, _config):
            t0 = time.perf_counter()
            set_weights(eval_model, ndarrays)
            thr = cfg.train.threshold
            entry = {
                "round": server_round,
                "val": evaluate_model(eval_model, *eval_data["val"], thr),
                "test": evaluate_model(eval_model, *eval_data["test"], thr),
                "client_test": [
                    {"client": k, **evaluate_model(eval_model, X, y, thr)}
                    for k, (X, y) in enumerate(eval_data["client_test"])
                ],
            }
            entry["eval_time_s"] = time.perf_counter() - t0
            history.append(entry)
            final["weights"] = [w.copy() for w in ndarrays]
            return entry["val"]["log_loss"], {"val_accuracy": entry["val"]["accuracy"]}

        fed = cfg.federated
        strategy = SimulatedFedAvg(
            node_to_client=node_to_client,
            available=availability_schedule(fed.rounds, n_clients, fed.dropout_rate, cfg.seed),
            fails=failure_schedule(fed.rounds, n_clients, fed.midround_failure_rate, cfg.seed),
            fraction_fit=fed.fraction_fit,
            min_fit_clients=fed.min_fit_clients,
            seed=cfg.seed,
            initial_parameters=ndarrays_to_parameters(get_weights(init_model)),
            evaluate_fn=evaluate_fn,
            secagg=cfg.secagg.enabled,
        )
        legacy = LegacyContext(
            context=context,
            config=ServerConfig(num_rounds=fed.rounds, round_timeout=fed.round_timeout),
            strategy=strategy,
        )
        if cfg.secagg.enabled:
            sa = cfg.secagg
            workflow = DefaultWorkflow(
                fit_workflow=SecAggPlusWorkflow(
                    num_shares=sa.num_shares,
                    reconstruction_threshold=sa.reconstruction_threshold,
                    max_weight=sa.max_weight,
                    clipping_range=sa.clipping_range,
                    quantization_range=sa.quantization_range,
                    timeout=sa.timeout,
                )
            )
        else:
            workflow = DefaultWorkflow()
        t0 = time.perf_counter()
        workflow(grid, legacy)
        wall = time.perf_counter() - t0

        if "weights" in final:
            model = MLP(n_features, cfg.model.hidden)
            set_weights(model, final["weights"])
            if cfg.output.save_checkpoints:
                save_checkpoint(model, run_dir / "checkpoints" / "global_final.pt",
                                {"rounds": fed.rounds, "mode": "federated"})
        out = {
            "node_to_client": {str(k): v for k, v in node_to_client.items()},
            "rounds": [strategy.rounds[r] for r in sorted(strategy.rounds)],
            "eval_history": history,
            "fl_wall_time_s": wall,
        }
        (run_dir / "server_log.json").write_text(json.dumps(clean_for_json(out)))

    return app
