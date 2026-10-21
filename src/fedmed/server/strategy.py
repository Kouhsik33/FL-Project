"""Coordinator-side FedAvg strategy with simulated client availability.

Built on Flower's ``flwr.server.strategy.FedAvg`` (the strategy interface used
by Flower's ``DefaultWorkflow`` and ``SecAggPlusWorkflow``). Additions:

* availability: a seeded (rounds x clients) Bernoulli schedule decides which
  institutions are reachable at the start of each round; unreachable ones are
  never contacted (so the server never waits for them);
* mid-round failure: a seeded schedule tells some selected clients to crash
  during local training, exercising Flower's failure path and SecAgg+ recovery;
* minimum participation: a round with fewer than ``min_fit_clients`` available
  clients, or fewer valid results, leaves the global model unchanged;
* aggregation uses ``fedmed.server.aggregation.fedavg`` (unit-tested formula);
* per-round records of participation, failures, payload sizes and timing.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from fedmed.evaluation.communication import parameters_nbytes
from fedmed.server.aggregation import fedavg


def availability_schedule(rounds: int, n_clients: int, dropout_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7_919)
    return rng.random((rounds + 1, n_clients)) >= dropout_rate  # row 0 unused


def failure_schedule(rounds: int, n_clients: int, failure_rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 104_729)
    return rng.random((rounds + 1, n_clients)) < failure_rate


class SimulatedFedAvg(FedAvg):
    def __init__(
        self,
        *,
        node_to_client: dict[int, int],
        available: np.ndarray,
        fails: np.ndarray,
        fraction_fit: float,
        min_fit_clients: int,
        seed: int,
        initial_parameters: Parameters,
        evaluate_fn: Callable,
        secagg: bool,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,  # no federated evaluation; the coordinator evaluates
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=1,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            accept_failures=True,
        )
        self.node_to_client = node_to_client
        self.available = available
        self.fails = fails
        self.select_rng = np.random.default_rng(seed + 15_485_863)
        self.secagg = secagg
        self.rounds: dict[int, dict] = {}
        self._round_start: dict[int, float] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start[server_round] = time.perf_counter()
        proxies = sorted(client_manager.all().values(), key=lambda p: self.node_to_client[p.node_id])
        avail = [p for p in proxies if self.available[server_round, self.node_to_client[p.node_id]]]
        n_select = max(math.ceil(self.fraction_fit * len(avail)), 0)
        record = {
            "round": server_round,
            "available": [self.node_to_client[p.node_id] for p in avail],
            "unavailable": [self.node_to_client[p.node_id] for p in proxies if p not in avail],
        }
        self.rounds[server_round] = record
        if len(avail) < self.min_fit_clients or n_select < self.min_fit_clients:
            record.update(selected=[], status="skipped_insufficient_available", aggregated=False)
            return []
        idx = sorted(self.select_rng.choice(len(avail), size=n_select, replace=False))
        selected = [avail[i] for i in idx]
        payload = parameters_nbytes(parameters)
        instructions = []
        for p in selected:
            cid = self.node_to_client[p.node_id]
            cfg = {"server_round": server_round, "simulate_failure": bool(self.fails[server_round, cid])}
            instructions.append((p, FitIns(parameters, cfg)))
        record.update(
            selected=[self.node_to_client[p.node_id] for p, _ in instructions],
            instructed_to_fail=[self.node_to_client[p.node_id] for p, ins in instructions
                                if ins.config["simulate_failure"]],
            model_payload_bytes=payload,
            downlink_model_bytes=payload * len(instructions),
        )
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        t0 = time.perf_counter()
        record = self.rounds.setdefault(server_round, {"round": server_round})
        participants = [self.node_to_client[proxy.node_id] for proxy, _ in results]
        record.update(
            participants=sorted(participants),
            n_failures=len(failures),
            sample_counts={self.node_to_client[p.node_id]: r.num_examples for p, r in results},
        )
        if self.secagg:
            # SecAgg+ hands every FitRes the same already-aggregated parameters; the
            # coordinator never sees individual updates (see docs/architecture.md).
            record["uplink_model_bytes"] = None
        else:
            record["uplink_model_bytes"] = int(sum(parameters_nbytes(r.parameters) for _, r in results))
        if len(results) < self.min_fit_clients:
            record.update(status="skipped_insufficient_results", aggregated=False)
            return None, {}
        weights = [(parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in results]
        new = fedavg(weights)
        metrics: dict = {}
        eps = [r.metrics["epsilon"] for _, r in results if "epsilon" in r.metrics]
        if eps:
            metrics["max_epsilon_reported"] = max(eps)
        record.update(
            status="aggregated", aggregated=True,
            aggregation_time_s=time.perf_counter() - t0,
            round_wall_time_s=time.perf_counter() - self._round_start.get(server_round, t0),
            client_train_time_s={self.node_to_client[p.node_id]: r.metrics.get("train_time_s")
                                 for p, r in results},
            client_reported_epsilon={self.node_to_client[p.node_id]: r.metrics.get("epsilon")
                                     for p, r in results},
        )
        return ndarrays_to_parameters(new), metrics
