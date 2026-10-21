"""Dashboard data loading and a headless render of every page."""

import sys
from pathlib import Path

import pytest

from conftest import requires_wdbc
from fedmed.config import load_config
from fedmed.experiments.run import run

DASH = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASH))
from data import history_frame, load_runs, participation_frame  # noqa: E402


def test_load_runs_reports_missing_and_corrupt(tmp_path):
    store = load_runs(tmp_path / "nope")
    assert store.errors and not store.runs
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "summary.json").write_text("{not json")
    store = load_runs(tmp_path)
    assert len(store.errors) == 1 and not store.runs


def test_frames_from_minimal_summary():
    s = {
        "partition": {"clients": [{}, {}, {}]},
        "rounds": [{"round": 1, "available": [0, 1], "participants": [0], "instructed_to_fail": [1],
                    "status": "aggregated"}],
        "history": [{"round": 0, "test": {"accuracy": 0.5}}, {"round": 1, "test": {"accuracy": 0.7}}],
    }
    pf = participation_frame(s)
    assert dict(zip(pf["client"], pf["state"])) == {0: "participated", 1: "crashed mid-round", 2: "unavailable"}
    assert history_frame(s)["accuracy"].tolist() == [0.5, 0.7]


@requires_wdbc
def test_dashboard_loads_real_run_and_renders_all_pages(tmp_runs):
    run(load_config(None, ["mode=centralized", f"output.runs_dir={tmp_runs}", "name=dash"]))
    store = load_runs(Path(tmp_runs))
    assert len(store.runs) == 1 and not store.errors
    row = store.table.iloc[0]
    assert row["mode"] == "centralized" and 0 < row["test_accuracy"] <= 1

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(tmp_runs).run()
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
