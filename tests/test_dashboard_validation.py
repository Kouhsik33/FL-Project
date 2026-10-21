"""Validate dashboard figures against the saved result files in results/runs.

Values are recomputed independently from summary.json (plain json + pandas),
not through the dashboard/report code paths. Skipped when no runs exist.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from fedmed.paths import RUNS_DIR

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))
from data import aggregated_by_group, history_frame, load_runs, participation_frame, per_client_frame  # noqa: E402

SUMMARIES = sorted(RUNS_DIR.glob("*/summary.json"))
pytestmark = pytest.mark.skipif(not SUMMARIES, reason="no saved runs in results/runs")
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy", "roc_auc"]


@pytest.fixture(scope="module")
def raw():
    return {s["run_id"]: s for s in (json.loads(p.read_text()) for p in SUMMARIES)}


@pytest.fixture(scope="module")
def store():
    return load_runs(RUNS_DIR)


def same(a, b, tol=1e-12):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return b is not None and abs(float(a) - float(b)) <= tol


def test_every_run_loaded_without_errors(raw, store):
    assert not store.errors
    assert set(store.runs) == set(raw)


def test_run_table_matches_summary_files(raw, store):
    table = store.table.set_index("run_id")
    for rid, s in raw.items():
        row = table.loc[rid]
        for m in METRICS:
            assert same(row[f"test_{m}"], s["final"]["test"].get(m)), (rid, m)
        if s["mode"] == "federated":
            c = s["communication"]
            assert row["comm_training_bytes"] == c["training_total_bytes"]
            assert row["model_payload_bytes"] == c["model_payload_bytes"]
            assert row["rounds_aggregated"] == s["participation"]["rounds_aggregated"]
            assert same(row["fl_wall_time_s"], s["timing"]["fl_wall_time_s"])
        if s.get("privacy"):
            p = s["privacy"]
            assert same(row["epsilon"], p.get("max_epsilon", p.get("epsilon"))), rid


def test_grouped_means_match_independent_recomputation(raw, store):
    agg = aggregated_by_group(store).set_index(["group", "name"])
    rows = [{"group": s["tag"], "name": s["name"], **{m: s["final"]["test"].get(m) for m in METRICS}}
            for s in raw.values() if s.get("tag")]
    df = pd.DataFrame(rows)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m])
    for (g, n), sub in df.groupby(["group", "name"]):
        for m in METRICS:
            assert same(agg.loc[(g, n), f"test_{m}_mean"], sub[m].mean(), 1e-9), (g, n, m)
            if len(sub) > 1:
                assert same(agg.loc[(g, n), f"test_{m}_std"], sub[m].std(ddof=1), 1e-9), (g, n, m)
        assert agg.loc[(g, n), "n_seeds"] == sub.shape[0]


def test_per_round_and_per_client_frames_match(raw):
    for rid, s in raw.items():
        pc = per_client_frame(s).to_dict("records")
        assert len(pc) == len(s["final"]["client_test"])
        for shown, saved in zip(pc, s["final"]["client_test"]):
            for k in ("client", "n", "n_pos", *METRICS):
                assert same(shown[k], saved[k]), (rid, k)
        if s["mode"] != "federated":
            continue
        h = history_frame(s, "test")
        assert h["round"].tolist() == [e["round"] for e in s["history"]]
        assert all(same(a, e["test"]["f1"]) for a, e in zip(h["f1"], s["history"]))
        pf = participation_frame(s)
        counts = pf[pf["state"] == "participated"].groupby("client").size().to_dict()
        for k, v in s["participation"]["participation_count"].items():
            assert counts.get(int(k), 0) == v, (rid, k)
        assert len(pf) == len(s["rounds"]) * len(s["partition"]["clients"])


def test_rendered_metric_widgets_match_files(raw):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120)
    at.run()
    shown = {m.label: m.value for m in at.metric}
    assert shown["Saved runs"] == str(len(raw))
    assert shown["Federated runs"] == str(sum(s["mode"] == "federated" for s in raw.values()))
    assert shown["Datasets"] == str(len({s["config"]["data"]["dataset"] for s in raw.values()}))

    at.sidebar.radio[0].set_value("Communication & time").run()
    assert not at.exception
    rid = at.selectbox[0].value  # default selection = last federated run
    c = raw[rid]["communication"]
    shown = {m.label: m.value for m in at.metric}
    assert shown["Model payload (bytes)"] == f"{c['model_payload_bytes']:,}"
    assert shown["Training traffic, total (bytes)"] == f"{c['training_total_bytes']:,}"
    assert shown["Per aggregated round (bytes)"] == f"{c['training_total_bytes_per_aggregated_round'] or 0:,.0f}"

    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_dp_page_points_are_single_configurations(raw, store):
    """Each plotted DP point is one group-C configuration (regression: points used to pool
    different experiments that shared a sigma)."""
    from data import dp_sweep

    sweep = dp_sweep(store.table, "f1")
    for _, row in sweep.iterrows():
        runs = [s for s in raw.values() if s.get("tag") == "C" and s["name"] == row["name"]]
        assert len({json.dumps(s["config"]["federated"]) + s["mode"] for s in runs}) == 1
        f1 = pd.Series([s["final"]["test"]["f1"] for s in runs], dtype=float)
        eps = pd.Series([s["privacy"].get("max_epsilon", s["privacy"].get("epsilon")) for s in runs], dtype=float)
        assert same(row["mean"], f1.mean(), 1e-9) and same(row["epsilon_mean"], eps.mean(), 1e-9)
        assert row["n_seeds"] == len(runs)
    # One point per (mode, sigma); vacuous when no group-C runs exist (e.g. a fresh clone).
    assert sweep.empty or sweep.groupby(["mode", "sigma"]).size().max() == 1
