"""Simulated institution: a Flower ClientApp.

Each client loads ONLY its own partition file (standing in for data kept on the
institution's premises), starts every round from the global parameters sent by
the server, trains locally (optionally with DP-SGD), and returns its updated
parameters and sample count.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from fedmed.config import ExperimentCfg
from fedmed.clients.trainer import train
from fedmed.evaluation.communication import append_jsonl, make_comm_meter_mod
from fedmed.models.mlp import MLP, get_weights, set_weights
from fedmed.privacy.dp import compute_epsilon

DP_STATE_KEY = "fedmed_dp_state"
IDENTITY_KEY = "fedmed_identity"


class SimulatedFailure(RuntimeError):
    """Raised to emulate an institution that crashes after being selected."""


def client_seed(base: int, server_round: int, client_id: int) -> int:
    return (base * 1_000_003 + server_round * 10_007 + client_id * 101) % (2**31 - 1)


def load_client_data(data_dir: Path, client_id: int) -> dict[str, np.ndarray]:
    with np.load(data_dir / f"client_{client_id}.npz") as f:
        return {k: f[k] for k in f.files}


class InstitutionClient(NumPyClient):
    def __init__(self, cfg: ExperimentCfg, client_id: int, data: dict[str, np.ndarray],
                 context: Context, log_path: Path):
        self.cfg = cfg
        self.client_id = client_id
        self.X, self.y = data["X_train"], data["y_train"]
        self.n_features = self.X.shape[1]
        self.context = context
        self.log_path = log_path

    def _fresh_model(self, parameters) -> MLP:
        # A new model object each round: no state can leak from previous rounds.
        model = MLP(self.n_features, self.cfg.model.hidden)
        set_weights(model, parameters)
        return model

    def fit(self, parameters, config):
        server_round = int(config["server_round"])
        if bool(config.get("simulate_failure", False)):
            append_jsonl(self.log_path, {"client_id": self.client_id, "round": server_round,
                                         "event": "simulated_failure", "t": time.time()})
            raise SimulatedFailure(f"client {self.client_id} dropped in round {server_round}")
        model = self._fresh_model(parameters)
        start_digest = float(sum(np.abs(p).sum() for p in parameters))
        privacy = self.cfg.privacy
        res = train(
            model, self.X, self.y, self.cfg.train, epochs=self.cfg.train.local_epochs,
            seed=client_seed(self.cfg.seed, server_round, self.client_id),
            privacy=privacy if privacy.enabled else None,
        )
        record = {
            "client_id": self.client_id, "round": server_round, "event": "fit",
            "n_train": res.n_samples, "steps": res.steps, "train_loss": res.mean_loss,
            "train_time_s": res.train_time_s, "start_param_abs_sum": start_digest,
        }
        metrics: dict = {"client_id": self.client_id, "train_time_s": res.train_time_s}
        if privacy.enabled:
            # Cumulative accounting across all rounds this client has trained in.
            state = self.context.state.config_records.get(DP_STATE_KEY, ConfigRecord({"steps": 0}))
            total_steps = int(state["steps"]) + res.steps
            self.context.state.config_records[DP_STATE_KEY] = ConfigRecord(
                {"steps": total_steps, "sample_rate": float(res.sample_rate)}
            )
            eps = compute_epsilon(privacy.noise_multiplier, res.sample_rate, total_steps, privacy.delta)
            record.update(dp_sample_rate=res.sample_rate, dp_total_steps=total_steps, epsilon=eps,
                          delta=privacy.delta, noise_multiplier=privacy.noise_multiplier,
                          max_grad_norm=privacy.max_grad_norm)
            metrics["epsilon"] = eps
        if not self.cfg.secagg.enabled:
            # Under SecAgg+ these plaintext metrics would still be visible to the
            # server, so we only send them when SecAgg is off (they are logged locally).
            metrics["train_loss"] = res.mean_loss
        append_jsonl(self.log_path, record)
        return get_weights(model), res.n_samples, metrics

    # No `evaluate`: the global model is evaluated by the coordinator on the
    # held-out global test set and per-client test shards (fraction_evaluate=0).


def identity_mod(msg: Message, ctx: Context, call_next) -> Message:
    """Answer the coordinator's registration query with this institution's index."""
    if msg.metadata.message_type == MessageType.QUERY and IDENTITY_KEY in msg.content.config_records:
        reply = RecordDict({IDENTITY_KEY: ConfigRecord({"client_id": int(ctx.node_config["partition-id"])})})
        return Message(reply, reply_to=msg)
    return call_next(msg, ctx)


def build_client_app(cfg: ExperimentCfg, run_dir: Path) -> ClientApp:
    data_dir = run_dir / "client_data"
    log_path = run_dir / "client_log.jsonl"

    def client_fn(context: Context):
        torch.set_num_threads(1)
        client_id = int(context.node_config["partition-id"])
        data = load_client_data(data_dir, client_id)
        return InstitutionClient(cfg, client_id, data, context, log_path).to_client()

    mods = [make_comm_meter_mod(str(run_dir / "comm_log.jsonl")), identity_mod]
    if cfg.secagg.enabled:
        mods.append(secaggplus_mod)
    return ClientApp(client_fn=client_fn, mods=mods)
