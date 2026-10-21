"""FedAvg formula, strategy behaviour (dropout, skipping), metrics, communication, config."""

from types import SimpleNamespace

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fedmed.config import load_config
from fedmed.evaluation.communication import parameters_nbytes
from fedmed.evaluation.metrics import compute_metrics, disparity
from fedmed.server.aggregation import fedavg
from fedmed.server.strategy import SimulatedFedAvg, availability_schedule, failure_schedule


# ---------------------------------------------------------------- FedAvg
def test_fedavg_hand_calculated():
    # Client A: n=10, w=[1, 2], b=[0]; client B: n=30, w=[5, 6], b=[4]
    # global = 10/40*A + 30/40*B = [0.25*1+0.75*5, 0.25*2+0.75*6] = [4, 5]; b = 0.75*4 = 3
    a = [np.array([1.0, 2.0]), np.array([0.0])]
    b = [np.array([5.0, 6.0]), np.array([4.0])]
    out = fedavg([(a, 10), (b, 30)])
    np.testing.assert_allclose(out[0], [4.0, 5.0])
    np.testing.assert_allclose(out[1], [3.0])


def test_fedavg_equal_weights_is_mean_and_single_client_identity():
    ws = [[np.full((2, 2), float(k))] for k in range(4)]
    np.testing.assert_allclose(fedavg([(w, 5) for w in ws])[0], np.full((2, 2), 1.5))
    np.testing.assert_allclose(fedavg([(ws[3], 7)])[0], ws[3][0])
    with pytest.raises(ValueError):
        fedavg([])
    with pytest.raises(ValueError):
        fedavg([(ws[0], 1), ([np.zeros(2), np.zeros(1)], 1)])


# ---------------------------------------------------------------- strategy
class _Proxy(SimpleNamespace):
    pass


class _CM:
    def __init__(self, ids):
        self.proxies = {str(i): _Proxy(node_id=i, cid=str(i)) for i in ids}

    def all(self):
        return self.proxies


def _strategy(n=5, rounds=4, dropout=0.0, fail=0.0, min_fit=2, seed=0):
    node_to_client = {100 + k: k for k in range(n)}
    return SimulatedFedAvg(
        node_to_client=node_to_client,
        available=availability_schedule(rounds, n, dropout, seed),
        fails=failure_schedule(rounds, n, fail, seed),
        fraction_fit=1.0, min_fit_clients=min_fit, seed=seed,
        initial_parameters=ndarrays_to_parameters([np.zeros(3, dtype=np.float32)]),
        evaluate_fn=lambda *a: None, secagg=False,
    ), _CM(node_to_client)


def test_availability_schedule_reproducible_and_rate():
    a = availability_schedule(2000, 5, 0.3, seed=1)
    assert np.array_equal(a, availability_schedule(2000, 5, 0.3, seed=1))
    assert a[1:].mean() == pytest.approx(0.7, abs=0.02)
    assert availability_schedule(10, 5, 0.0, 0)[1:].all()


def test_configure_fit_only_contacts_available_clients():
    strat, cm = _strategy(dropout=0.5, rounds=30, seed=3)
    params = ndarrays_to_parameters([np.ones(3, dtype=np.float32)])
    for r in range(1, 31):
        ins = strat.configure_fit(r, params, cm)
        rec = strat.rounds[r]
        expected = [k for k in range(5) if strat.available[r, k]]
        assert rec["available"] == expected
        if len(expected) < 2:
            assert ins == [] and rec["status"] == "skipped_insufficient_available"
        else:
            assert sorted(strat.node_to_client[p.node_id] for p, _ in ins) == expected
            assert all(fi.parameters is params for _, fi in ins)  # everyone starts from the global model
    assert any(not strat.available[r].all() for r in range(1, 31))


def _fitres(w, n, cid):
    return FitRes(Status(Code.OK, ""), ndarrays_to_parameters(w), n, {"client_id": cid, "train_time_s": 0.1})


def test_aggregate_fit_weighted_and_handles_failures():
    strat, cm = _strategy()
    strat.configure_fit(1, strat.initial_parameters, cm)
    proxies = list(cm.all().values())
    results = [(proxies[0], _fitres([np.array([1.0, 1.0, 1.0], np.float32)], 10, 0)),
               (proxies[1], _fitres([np.array([4.0, 4.0, 4.0], np.float32)], 20, 1))]
    params, _ = strat.aggregate_fit(1, results, [Exception("client crashed")])
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], [3.0, 3.0, 3.0])
    assert strat.rounds[1]["participants"] == [0, 1] and strat.rounds[1]["n_failures"] == 1


def test_aggregate_fit_skips_when_too_few_valid_results():
    strat, cm = _strategy(min_fit=3)
    strat.configure_fit(1, strat.initial_parameters, cm)
    p = list(cm.all().values())
    results = [(p[0], _fitres([np.ones(3, np.float32)], 10, 0)), (p[1], _fitres([np.ones(3, np.float32)], 10, 1))]
    params, _ = strat.aggregate_fit(1, results, [Exception("x"), Exception("y")])
    assert params is None and strat.rounds[1]["status"] == "skipped_insufficient_results"


def test_failure_schedule_marks_instructions():
    strat, cm = _strategy(fail=1.0)
    ins = strat.configure_fit(1, strat.initial_parameters, cm)
    assert all(fi.config["simulate_failure"] for _, fi in ins)
    assert strat.rounds[1]["instructed_to_fail"] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- metrics
def test_metrics_hand_calculated():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.4])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(12 / 15)  # 12 of 15 positive/negative pairs ranked correctly


def test_f1_is_zero_not_undefined_when_no_positives_predicted():
    # Positives exist but the model predicts none: precision undefined, recall 0, F1 = 0.
    m = compute_metrics(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert m["precision"] is None and m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_metrics_undefined_values_are_none():
    m = compute_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert m["recall"] is None and m["roc_auc"] is None and m["precision"] is None
    assert m["specificity"] == 1.0 and m["accuracy"] == 1.0
    assert compute_metrics(np.array([]), np.array([]))["accuracy"] is None


def test_disparity_gap():
    pcs = [{"client": 0, "f1": 0.9}, {"client": 1, "f1": 0.6}, {"client": 2, "f1": None}]
    d = disparity(pcs, "f1")
    assert d["gap"] == pytest.approx(0.3) and d["worst_client"] == 1 and d["n_undefined"] == 1


# ---------------------------------------------------------------- communication & config
def test_parameters_nbytes_counts_serialised_tensors():
    arrs = [np.zeros((30, 32), np.float32), np.zeros(32, np.float32)]
    p = ndarrays_to_parameters(arrs)
    n = parameters_nbytes(p)
    raw = sum(a.nbytes for a in arrs)
    assert raw < n < raw + 2 * 256  # .npy header per tensor


def test_message_size_measurement():
    from flwr.app import ArrayRecord, RecordDict

    from fedmed.evaluation.communication import recorddict_nbytes

    small = RecordDict({"a": ArrayRecord([np.zeros(10, np.float32)])})
    big = RecordDict({"a": ArrayRecord([np.zeros(1000, np.float32)])})
    assert recorddict_nbytes(big) - recorddict_nbytes(small) == pytest.approx(990 * 4, abs=16)


def test_config_overrides_and_validation(tmp_path):
    cfg = load_config(None, ["federated.rounds=7", "privacy.enabled=true", "model.hidden=[8,4]"])
    assert cfg.federated.rounds == 7 and cfg.privacy.enabled and cfg.model.hidden == [8, 4]
    with pytest.raises(ValueError):
        load_config(None, ["federated.roundz=7"])
    with pytest.raises(ValueError):
        load_config(None, ["noequals"])
