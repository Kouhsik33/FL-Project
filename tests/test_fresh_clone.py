"""Regression: CLIs must work in a fresh clone where results/ does not exist yet."""

import json

from conftest import requires_wdbc
from fedmed.experiments import tune


@requires_wdbc
def test_tune_creates_missing_results_dir(tmp_path, monkeypatch):
    missing = tmp_path / "fresh" / "results"
    monkeypatch.setattr(tune, "RESULTS_DIR", missing)
    monkeypatch.setattr(tune, "LRS", [0.005])
    monkeypatch.setattr(tune, "EPOCHS", [2])
    tune.main(["--dataset", "wdbc", "--seeds", "1"])
    out = json.loads((missing / "tuning_wdbc.json").read_text())
    assert out["best"]["lr"] == 0.005 and len(out["grid"]) == 1
