# Federated Learning for Privacy-Preserving Medical Diagnosis

A locally runnable research prototype. Several healthcare institutions are **simulated** on one computer. They collaboratively train a diagnostic classifier with **Federated Averaging (FedAvg)**, without pooling their records. Two privacy-enhancing mechanisms are added and evaluated: **record-level Differential Privacy** (Opacus DP-SGD) and **Secure Aggregation** (Flower SecAgg+).

> **Scope and honesty statement.** Every institution here is simulated, and the data are public, de-identified research datasets (UCI). No real hospital took part, no clinical validation was performed, and the models must **not** be used for diagnosis. Federated learning alone is not a privacy guarantee. See [docs/architecture.md §4](docs/architecture.md) for exactly what each mechanism does and does not protect.

## Quick start

Two ways to use this repository. Both are tested in a fresh clone. Commands assume macOS or Linux with Python 3.11; the environment was tested on an Intel Mac.

### A. View the results (dashboard only, about 5 minutes)

The dashboard reads the saved results in `results/runs/` (199 runs, committed with the code). It does **not** need the datasets, the model checkpoints, or the 1.5-hour experiment suite.

```bash
git clone https://github.com/Kouhsik33/FL-Project.git
cd FL-Project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock      # exact pinned versions
pip install -e . --no-deps             # installs the fedmed package from src/
streamlit run dashboard/app.py         # open http://localhost:8501
```

Expected on the Overview page: **199 saved runs**, 124 federated, 2 datasets, 5 seeds. No loader errors appear in the sidebar. Nine pages are listed in the sidebar; each renders from the saved `summary.json` files.

Check the dashboard against the result files (no data download needed):

```bash
pytest tests/test_dashboard.py tests/test_dashboard_validation.py -q
```

### B. Reproduce the experiments (needs the datasets, about 2 hours)

Run the steps in [Setup](#setup) and [Run](#run) below. The dataset download is needed only for this path, and only because the code reads the raw files. Do not run `python -m fedmed.experiments.suite` unless you intend to overwrite `results/suite_index.jsonl` and `results/runs/`.

### What is committed, and what is not

| Path | Committed? | Used by |
|---|---|---|
| `results/runs/<run_id>/summary.json`, `config.yaml`, `partition.json`, `client_log.jsonl`, `comm_log.jsonl`, `server_log.json` | yes (37 MB) | dashboard, report, tests |
| `results/runs/<run_id>/checkpoints/*.pt` | yes (3 MB) | checkpoint re-evaluation |
| `results/report/`, `results/suite_index.jsonl`, `results/tuning_*.json`, `results/archive/` | yes | report, provenance |
| `data/raw/` (datasets) | **no**, re-downloaded by `scripts/download_data.py` | experiments, data tests |
| `.venv/` | no | local environment |

### Notes for AI agents and reviewers

- Start with the dashboard path (A). It is the fastest way to confirm the results.
- Results are immutable inputs. Do not edit `results/runs/`, `results/suite_index.jsonl`, or `results/archive/`. Each run is a self-contained folder.
- `results/archive/summaries_before_f1_fix.tar.gz` holds the summaries from before the F1 correction. Current summaries are tagged `f1_recomputed_from_confusion_counts_2026-10-20` in their `corrections` field.
- Test selection: `pytest -m "not slow"` is the fast suite. Plain `pytest` runs everything, including Flower/Ray simulations (about 1.5 minutes).
- Read `PROJECT_STATUS.md` for the current status, the known issues, and the pending decisions. Read `docs/final_report.md` for the methodology and results.

## What it does (mapped to the submitted abstract)

| Abstract objective | Implementation |
|---|---|
| Institutions train collaboratively without sharing patient data | each Flower `ClientApp` loads only its own partition; the coordinator never receives training records |
| Records stay local; each institution trains the common model locally | `fedmed.clients.flower_client.InstitutionClient.fit` |
| Coordinator aggregates with FedAvg and redistributes | `fedmed.server.strategy.SimulatedFedAvg` + `fedmed.server.aggregation.fedavg` (Σ n_k/N · w_k) |
| Metrics: accuracy, precision, recall, F1, training time, communication | `summary.json` per run; measured protobuf message sizes; `results/report/tables.md` |
| Effect of data distribution | IID, Dirichlet label skew (α = 1.0, 0.3, 0.1), quantity skew, real-site partition (Heart Disease) |
| Differential Privacy and Secure Aggregation | Opacus DP-SGD with cumulative RDP accounting; Flower `SecAggPlusWorkflow` + `secaggplus_mod` |

## Setup

Tested on macOS 26.7 (Intel x86_64), Python 3.11.15, CPU only.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock      # exact versions used for the reported results
pip install -e . --no-deps            # the fedmed package (src/fedmed)
python scripts/download_data.py       # downloads + SHA-256-verifies both UCI datasets into data/raw/
```

The dependency pins are deliberate (see [docs/technical_research.md §2](docs/technical_research.md)):
- torch 2.2.2 is the last release for Intel Macs.
- Opacus 1.5.3 is the newest release that imports with that torch.
- Ray 2.49.2 is installed explicitly, because `flwr[simulation]` pins a Ray version with no Intel-Mac wheel.

## Run

```bash
# one experiment (any config key can be overridden with section.key=value)
python -m fedmed.experiments.run --config configs/default.yaml mode=centralized
python -m fedmed.experiments.run --config configs/default.yaml mode=local
python -m fedmed.experiments.run --config configs/default.yaml                         # FedAvg, 5 IID clients, 20 rounds
python -m fedmed.experiments.run privacy.enabled=true privacy.noise_multiplier=1.5     # + DP-SGD
python -m fedmed.experiments.run secagg.enabled=true federated.min_fit_clients=3       # + SecAgg+
python -m fedmed.experiments.run partition.strategy=dirichlet partition.alpha=0.3 federated.dropout_rate=0.2
python -m fedmed.experiments.run data.dataset=heart_disease partition.strategy=natural  # 4 real sites

# hyper-parameter selection (validation split only)
python -m fedmed.experiments.tune --dataset wdbc

# the full controlled experiment suite (resumable, ~1.5 h on the test machine) and its tables
python -m fedmed.experiments.suite
python -m fedmed.experiments.report        # -> results/report/tables.md, aggregated.csv, runs.csv

# dashboard (reads results/runs/*/summary.json)
streamlit run dashboard/app.py

# tests
pytest -m "not slow"     # fast unit + baseline end-to-end tests
pytest                   # everything, including Flower/Ray simulations (~1.5 min measured)
```

Each run writes `results/runs/<run_id>/` containing:
- `config.yaml`, `partition.json`, `summary.json` (metrics, history, participation, communication, timing, privacy, environment);
- `client_log.jsonl`, `comm_log.jsonl`, `server_log.json`;
- `checkpoints/`.

## Repository layout

```
configs/default.yaml        documented sample configuration
scripts/download_data.py    reproducible, checksum-verified dataset download
src/fedmed/
  data/          download, parsing, splits, partitioning, leakage-safe preprocessing
  models/        MLP + checkpoints
  clients/       local trainer (SGD / DP-SGD), Flower ClientApp
  server/        FedAvg formula, strategy (availability, failures, min participants), ServerApp (+SecAgg+)
  privacy/       Opacus wrapper and RDP accounting
  evaluation/    metrics, per-client disparity, communication measurement, summaries
  experiments/   run / tune / suite / report CLIs
dashboard/       Streamlit app (data.py loader + app.py)
tests/           unit, integration and end-to-end tests
docs/            research, dataset assessment, architecture, plan, references, final report
results/         the 199 saved runs (committed), suite index, report tables, tuning files, archive
```

## Results (199 runs; WDBC test set, n = 114; mean ± std over seeds)

| Setting | F1 | ROC-AUC |
|---|---|---|
| Centralised | 0.952 ± 0.000 | 0.991 ± 0.001 |
| Local-only, 5 institutions (IID) | 0.901 ± 0.018 | 0.973 ± 0.004 |
| FedAvg, 5 institutions, 20 rounds (pre-registered budget) | 0.513 ± 0.364 | 0.951 ± 0.008 |
| FedAvg, 5 institutions, 100 rounds (seeds 0–2) | 0.921 ± 0.029 | 0.983 ± 0.002 |
| FedAvg + DP-SGD σ = 1.0, 50 rounds (ε = 26.7, δ = 1e-5) | 0.880 ± 0.031 | 0.968 ± 0.004 |

- At the pre-registered 20 rounds, FedAvg was under-trained: validation loss was still falling. With 100 rounds it came within about 0.03 F1 of centralised training.
- SecAgg+ reproduced plain FedAvg (relative weight difference 0.04%). It cost 2.0× (5 clients) and 2.65× (10 clients) the bytes per round, and 4× the messages.
- The analysis, including negative and inconclusive results and the fixes made during validation, is in **[docs/final_report.md](docs/final_report.md)**. The full tables are in `results/report/tables.md`. Both are generated from the saved runs.

Validation: `pytest` gives 58 passed, both in this environment and in a fresh environment built from these instructions.

## Limitations

- Simulated institutions on one machine; public, small, historical datasets. Partitions are artificial, except the four Heart Disease sites.
- DP is **record-level**, not client-level. The reported ε does not cover the preprocessing statistics or the exact sample counts. Opacus' secure RNG was off.
- Flower's SecAgg+ still reveals `num_examples`, fit metrics and participation to the server. In-process simulation has no real trust boundary and no TLS.
- The coordinator evaluates on held-out data as an oracle; federated evaluation is not implemented.
- Communication is measured as serialised message content. Network time is not measured.

## Licence and attribution

The datasets are © their creators, licensed CC BY 4.0. WDBC: Wolberg, Mangasarian, Street & Street, https://doi.org/10.24432/C5DW2B. Heart Disease: Janosi, Steinbrunn, Pfisterer & Detrano, https://doi.org/10.24432/C52P4X.
