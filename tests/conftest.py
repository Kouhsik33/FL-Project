import pytest

from fedmed.paths import RAW_DIR


def _have(name: str, fname: str) -> bool:
    return (RAW_DIR / name / fname).exists()


requires_wdbc = pytest.mark.skipif(
    not _have("wdbc", "wdbc.data"), reason="WDBC not downloaded (python scripts/download_data.py)"
)
requires_heart = pytest.mark.skipif(
    not _have("heart_disease", "processed.va.data"), reason="Heart Disease not downloaded"
)


@pytest.fixture
def tmp_runs(tmp_path):
    return str(tmp_path / "runs")
