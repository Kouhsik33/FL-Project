"""End-to-end runs: baselines (fast) and Flower simulations (slow, uses Ray)."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_wdbc
from fedmed.config import load_config
from fedmed.experiments.run import run
from fedmed.models.mlp import get_weights, load_checkpoint


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


@requires_wdbc
@pytest.mark.parametrize("mode", ["centralized", "local"])
def test_baseline_end_to_end(mode, tmp_runs):
    d = run(load_config(None, [f"mode={mode}", f"output.runs_dir={tmp_runs}", "name=t"]))
    s = _summary(d)
    assert s["final"]["test"]["accuracy"] > 0.85
    assert s["dataset"]["split_sizes"]["test"] == 114
    assert len(s["final"]["client_test"]) == 5
    assert sum(c["n"] for c in s["final"]["client_test"]) == 114  # shards cover the global test set
    assert any((d / "checkpoints").iterdir())


@requires_wdbc
def test_dp_centralized_reports_epsilon(tmp_runs):
    d = run(load_config(None, ["mode=centralized", "privacy.enabled=true", f"output.runs_dir={tmp_runs}"]))
    p = _summary(d)["privacy"]
    assert p["epsilon"] > 0 and p["delta"] == 1e-5 and p["steps"] > 0


def _fl(tmp_runs, *extra):
    return run(load_config(None, [f"output.runs_dir={tmp_runs}", "partition.n_clients=4",
                                  "federated.rounds=3", "name=e2e", *extra]))


@pytest.mark.slow
@requires_wdbc
def test_federated_secagg_matches_plain_fedavg_with_dropout_and_dp(tmp_runs):
    """Same seed, same protocol: SecAgg+ must reproduce plaintext FedAvg up to
    quantisation error, including a round where a client crashes mid-round
    (after SecAgg+ key sharing) and DP-SGD noise at every client."""
    common = ["federated.midround_failure_rate=0.25", "federated.min_fit_clients=3",
              "privacy.enabled=true", "seed=1"]
    plain = _fl(tmp_runs, *common)
    sec = _fl(tmp_runs, *common, "secagg.enabled=true")
    sp, ss = _summary(plain), _summary(sec)

    # Identical participation (deterministic schedules) and at least one crash exercised.
    assert [r.get("participants") for r in sp["rounds"]] == [r.get("participants") for r in ss["rounds"]]
    assert sp["participation"]["simulated_failures"] >= 1
    status = lambda s: [(r["round"], r.get("status")) for r in s["rounds"]]  # noqa: E731
    print("plain:", status(sp), "\nsecagg:", status(ss))
    assert status(sp) == status(ss)
    assert any(r.get("aggregated") and r.get("n_failures", 0) > 0 for r in ss["rounds"]), \
        "expected a SecAgg+ round that recovered from a mid-round crash"

    wp = get_weights(load_checkpoint(plain / "checkpoints" / "global_final.pt")[0])
    ws = get_weights(load_checkpoint(sec / "checkpoints" / "global_final.pt")[0])
    max_diff = max(float(np.abs(a - b).max()) for a, b in zip(wp, ws))
    assert max_diff < 1e-3, max_diff

    # SecAgg+ adds key-exchange traffic; plaintext run has none.
    stages = ss["communication"]["by_stage"]
    assert {"train:secagg_setup", "train:secagg_share_keys", "train:secagg_unmask"} <= set(stages)
    assert ss["communication"]["training_total_bytes"] > sp["communication"]["training_total_bytes"]
    # Coordinator records no per-client uplink model bytes under SecAgg+.
    assert all(r.get("uplink_model_bytes") is None for r in ss["rounds"])

    # DP accounting accumulates over the rounds each client actually trained in.
    for c in sp["privacy"]["per_client"]:
        steps_per_round = int(round(1 / c["sample_rate"]))
        assert c["total_steps"] == c["rounds_trained"] * steps_per_round
        assert c["epsilon"] == pytest.approx(c["epsilon_recomputed"])


@pytest.mark.slow
@requires_wdbc
def test_federated_dropout_skips_and_records(tmp_runs):
    d = _fl(tmp_runs, "federated.dropout_rate=0.6", "federated.rounds=6", "federated.min_fit_clients=2")
    s = _summary(d)
    rounds = s["rounds"]
    for r in rounds:
        assert set(r.get("participants", [])) <= set(r["available"])
        if len(r["available"]) < 2:
            assert not r["aggregated"]
    assert s["participation"]["rounds_aggregated"] + s["participation"]["rounds_skipped"] == 6
    assert len(s["history"]) == 7  # round 0 (initial model) + 6 rounds
    # Every fit started from the same global parameters within a round.
    log = [json.loads(x) for x in (d / "client_log.jsonl").read_text().splitlines()]
    by_round = {}
    for rec in log:
        if rec["event"] == "fit":
            by_round.setdefault(rec["round"], set()).add(round(rec["start_param_abs_sum"], 4))
    assert all(len(v) == 1 for v in by_round.values())
