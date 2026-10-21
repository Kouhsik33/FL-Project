# Implementation plan and status

The milestones follow the project brief. Each is marked done only when its checks pass. The final validation results are in the README and `docs/final_report.md`.

| # | Milestone | Deliverables | Checks | Status |
|---|---|---|---|---|
| 1 | Research & setup | environment inspection; compatibility matrix; research documents; verified dataset download; venv | packages install and import; SHA-256 matches; parsing tests | done |
| 2 | Baselines | preprocessing, splits, partitions, MLP, centralised + local-only training, metrics, checkpoints | `tests/test_data.py`, `tests/test_model_training.py`, baseline end-to-end tests | done |
| 3 | FedAvg | Flower ClientApp/ServerApp, registration handshake, weighted FedAvg, per-round logging, communication meter | hand-calculated FedAvg test; strategy unit tests; multi-round simulation | done |
| 4 | Privacy & robustness | Opacus DP-SGD with cumulative RDP accounting; non-IID partitions; availability and mid-round failure; SecAgg+ | DP accounting equals Opacus; clipping and noise tests; SecAgg+ ≈ plaintext FedAvg under DP and crashes; dropout end-to-end test | done |
| 5 | Dashboard | Streamlit app over saved runs | headless `AppTest` render of all pages; loader tests | done |
| 6 | Final validation | full test suite; experiment suite (A–H, 199 runs); report; checkpoint re-evaluation; dashboard validation; fresh-environment reproduction | 58 tests pass (main + fresh env); 369 checkpoints re-evaluated with 0 mismatches; 4 issues found and fixed (docs/final_report.md §5) | done |

## Engineering defaults chosen without asking

These defaults were all low-risk and reversible:
- Python 3.11 virtual environment in `.venv`.
- `src/fedmed` package layout, installed editable. A top-level package named `src` is not importable by Ray workers.
- Plotly for charts. YAML configs with dotted command-line overrides.
- Five seeds for the main groups, three for the SecAgg and round-count groups.

## Decisions that would need approval (none were taken)

- Downloading data that needs registration. Not needed: both datasets are CC BY 4.0 with direct download.
- Architecture changes for Secure Aggregation. Not needed: Flower's SecAgg+ fits the chosen workflow API.
- Client-level DP, or replacing the coordinator's evaluation oracle with federated evaluation. Listed as future work, because either would change the experiment design.
