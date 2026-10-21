# Project status guide

*Written 2026-10-21. Complements `README.md` (setup and reproduction) and `docs/final_report.md` (full analysis). Numbers below are copied from those two sources and from `results/report/tables.md`; nothing here is new data.*

> Simulation only. All institutions are simulated on one computer using public UCI datasets. No real hospital participated and no clinical validation was performed. Nothing here supports clinical use.

---

## A. Project overview

**Problem.** Hospitals hold sensitive patient records that cannot be pooled, yet a model trained on more data is usually better. This project studies **federated learning (FL)** as a way to train one diagnostic classifier across several institutions without moving their records, and measures what that costs in accuracy, time, communication and privacy.

**How the simulation works** (details: `docs/architecture.md`):

- **Simulated institutions.** The dataset's training records are split into disjoint partitions, one per simulated institution. Each institution (a Flower `ClientApp`) loads only its own partition.
- **Coordinator.** A Flower `ServerApp` holds the global model. Each round it picks the available institutions, sends them the *current* global parameters, collects their updates, and combines them with **FedAvg**: the sample-count-weighted mean, Σ (nₖ/N) · wₖ. It then evaluates the new global model on held-out data.
- **Local training.** Each institution builds a fresh small MLP from the received parameters and trains it on its own records (SGD, or DP-SGD).
- **Differential privacy (DP).** Optional. Opacus DP-SGD clips each per-record gradient and adds Gaussian noise during local training. An RDP accountant tracks ε across all steps the institution has taken. The guarantee is **record-level**, not institution-level.
- **Secure aggregation (SecAgg+).** Optional. Flower's `SecAggPlusWorkflow` masks each institution's update so the coordinator can recover only their sum. It is separate from DP: SecAgg hides *individual updates from the server*; DP bounds *what the model reveals about any one record*.

**Technologies.** Python 3.11, PyTorch 2.2.2, Flower 1.40.0 (+ Ray 2.49.2 simulation backend), Opacus 1.5.3, scikit-learn, pandas, Streamlit, Plotly, pytest. The pins are forced by this Intel Mac (torch 2.2.2 is the last Intel-Mac release); see `docs/technical_research.md` §2.

**Datasets** (both UCI, CC BY 4.0, SHA-256-verified downloads): WDBC breast cancer (569 records, primary) and Heart Disease (920 records from 4 real collection sites, secondary). Details: `docs/dataset_assessment.md`.

**Repository map**

```
README.md  PROJECT_STATUS.md  requirements.lock  pyproject.toml
configs/default.yaml        documented sample configuration
scripts/                    download_data.py, recompute_f1.py (F1 correction, already applied)
src/fedmed/
  data/        download, parsing, splits, partitioning, leakage-safe preprocessing
  models/      MLP + checkpoints
  clients/     local trainer, Flower ClientApp
  server/      FedAvg formula, strategy (availability/failures), ServerApp (+SecAgg+)
  privacy/     Opacus wrapper + RDP accounting
  evaluation/  metrics, disparity, communication measurement, summaries
  experiments/ run / tune / suite / report CLIs
dashboard/     app.py (Streamlit) + data.py (loader)
tests/         58 tests
docs/          research, dataset assessment, architecture, plan, references, final_report
results/       runs/ (199 runs), report/, archive/, suite_index.jsonl, tuning_*.json, logs
```

## B. Current implementation (all items below are in code and exercised by tests or recorded runs)

| Capability | Status | Evidence |
|---|---|---|
| Centralised, local-only, federated training | Implemented, run | groups A–H; `tests/test_end_to_end.py` |
| Partition strategies | `iid`, `dirichlet` (label skew, α), `quantity_skew` (α), `natural` (Heart Disease sites). Seeded; disjoint cover verified | `tests/test_data.py` |
| Leakage control | Global 60/20/20 split fixed by `split_seed`; preprocessing statistics from client *training* partitions only; hyper-parameters chosen on validation only | `tests/test_data.py` |
| Record-level DP | Opacus DP-SGD, Poisson sampling, RDP accountant, cumulative across rounds per institution; reported ε = max over institutions. Our ε helper equals Opacus' own accountant for the same run | `tests/test_model_training.py` |
| Secure aggregation | Flower SecAgg+ integrated. Reproduces plain FedAvg (≤1e-3 per parameter) under DP and mid-round crashes. Minimum 3 participants enforced | `tests/test_end_to_end.py` |
| Client dropout | (i) unavailable at round start (never contacted); (ii) crash after selection (error reply, FedAvg uses valid results only); (iii) rounds below the minimum participants skipped | `tests/test_federated_units.py`, group D |
| Metrics | Accuracy, precision, recall, specificity, F1, balanced accuracy, ROC-AUC, log loss, confusion counts; global + per-client; best/worst/gap | `evaluation/metrics.py` |
| Experiment tracking | Per-run `summary.json`, `config.yaml`, `partition.json`, client/comm/server logs, checkpoints, environment versions | `results/runs/<run_id>/` |
| Communication | Measured protobuf-serialised Flower message content per stage. Excludes gRPC/TLS/TCP framing | `evaluation/communication.py` |
| Dashboard | 9 pages reading `summary.json` only | §C |
| Tests and checks | 58 tests; 369 checkpoints re-evaluated against stored metrics (0 mismatches); dashboard figures recomputed independently (6 checks) | `docs/final_report.md` §4 |

**Not implemented:** client-level DP, DP-protected preprocessing statistics, federated (client-side) evaluation, deployment-mode Flower with TLS/auth, robust aggregation against malicious clients, any network-time measurement.

## C. Dashboard guide

**Launch (from the project root):**

```bash
source .venv/bin/activate
streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501** (equivalently http://localhost:8501). Binding to `127.0.0.1` keeps it off the network; plain `streamlit run dashboard/app.py` also listens on your LAN address. Stop it with `pkill -f "streamlit run dashboard/app.py"`.

**Data source.** `results/runs/*/summary.json`, read live. The sidebar "Runs directory" box can point to another directory. Unreadable files are listed as sidebar warnings. The cache refreshes when the newest `summary.json` changes.

| Page | Shows |
|---|---|
| Overview | Scope warning, objectives, run/dataset/seed counts, legend (measured / configured / computed bound), environment |
| Dataset & partitions | Dataset facts, split sizes, per-institution class counts for a chosen run |
| Baseline comparison | Mean ± std bar chart + table for an experiment group (A–H) and metric |
| Training progress | Per-round global metrics for up to 8 federated runs (round 0 = initial model) |
| Differential privacy | Utility vs ε for the controlled group-C sweep only; other DP runs in a separate table; per-client accounting |
| Participation & dropout | Round × institution state map (participated / unavailable / crashed / available but not aggregated); group-D effect |
| Communication & time | Model payload, bytes by protocol stage, timing breakdown, bytes per round across runs |
| Per-client performance | Per-institution metric and best–worst gap for a chosen run |
| Experiment history | All 199 runs, CSV download, config and environment per run |

**Caveats.**
- The baseline-comparison group menu has friendly labels for A–G; group H appears under the bare letter "H".
- Streamlit logs deprecation warnings for `use_container_width` (its notice says removal after 2025-12-31; Streamlit 1.66 still accepts it). It works today but may break on a future Streamlit upgrade. Replace it with `width="stretch"` if you upgrade. Not changed, because it is not a failure.
- Startup logs a uvicorn/websockets warning. It concerns reverse proxies forwarding non-ASCII headers and does not apply to local use.
- Single-seed rows have no std; undefined precision shows as blank.

## D. Results snapshot (validated; WDBC, test set n = 114; mean ± std over seeds)

| Setting | Seeds | F1 | ROC-AUC |
|---|---|---|---|
| Centralised | 5 | 0.952 ± 0.000 | 0.991 ± 0.001 |
| Local-only, 5 institutions (IID, mean of local models) | 5 | 0.901 ± 0.018 | 0.973 ± 0.004 |
| FedAvg, 5 institutions, **20 rounds** (pre-registered) | 5 | 0.513 ± 0.364 | 0.951 ± 0.008 |
| FedAvg, 3 institutions, 20 rounds | 5 | 0.890 ± 0.031 | 0.969 ± 0.004 |
| FedAvg, 10 institutions, 20 rounds | 5 | 0.270 ± 0.306 | 0.902 ± 0.035 |

**20 vs 100 rounds (Group H; seed-matched, seeds 0–2 only; 5 institutions, IID):**

| Rounds | F1 | ROC-AUC | Total bytes |
|---|---|---|---|
| 20 | 0.376 ± 0.407 | 0.950 ± 0.011 | 1,445,700 |
| 50 | 0.911 ± 0.028 | 0.970 ± 0.005 | 3,614,250 |
| 100 | 0.921 ± 0.029 | 0.983 ± 0.002 | 7,228,685 |
| *Centralised, same seeds* | 0.952 ± 0.000 | 0.991 ± 0.000 | — |

Note the 20-round row here (0.376, 3 seeds) differs from the table above (0.513, 5 seeds) because of the seed subset; compare only within a table. With 3 seeds the intervals are wide. Group H was pre-registered before any of its results existed.

**Differential privacy (δ = 1e-5, record-level):**
- Converged model (centralised): F1 falls from 0.952 (no DP) to 0.943 / 0.941 / 0.936 / 0.929 / 0.923 at ε = 23.1 / 7.0 / 3.5 / 2.3 / 1.4.
- Federated, Group H (50 rounds, σ = 1.0, 3 seeds): F1 **0.880 ± 0.031** at **ε = 26.70**, versus 0.911 ± 0.028 without DP at the same rounds and seeds.
- Federated at 20 rounds: no usable trend (see §E).

**Secure aggregation overhead (3 seeds):** bytes per aggregated round **2.03×** (5 institutions: 146,732 vs 72,285) and **2.65×** (10 institutions: 382,859 vs 144,560); 4× the messages; FL wall time 16.5 s vs 6.0 s (5 institutions). Utility matched plain FedAvg.

**Heart Disease (4 real sites, 5 seeds):** F1 centralised 0.862 ± 0.012; local-only 0.770 ± 0.006; FedAvg natural sites 0.708 ± 0.208 (20 rounds).

These numbers describe a small public dataset in a simulation. They say nothing about clinical effectiveness. Full tables: `results/report/tables.md`.

## E. Known issues and methodological caveats

**Resolved**

| Issue | Resolution |
|---|---|
| F1 recorded as undefined whenever a model predicted no positives, silently dropping the worst runs from means (e.g. Dirichlet α=0.3, 20 rounds: 0.856 shown vs 0.571 correct) | Metric fixed (F1 = 2TP/(2TP+FP+FN)); 94 summaries recomputed exactly from stored confusion counts by `scripts/recompute_f1.py`; originals archived in `results/archive/summaries_before_f1_fix.tar.gz`; regression test added. No model retrained. |
| Dashboard DP page pooled different experiments sharing a σ (showed F1 0.592 / ε 17.98 instead of 0.409 / 15.97) | Page now plots the group-C sweep only; regression test added |
| Dashboard cache never refreshed (underscore-prefixed cache argument) | Renamed; fixed |
| `tune`/`suite` crashed in a fresh clone without `results/` | Fixed; regression test added |

**Outstanding caveats**

- **Legacy F1 in raw logs.** Only `server_log.json` (per-round `eval_history`) carries the old F1 definition: 671 metric dicts across the federated runs still hold `f1 = null` where the correct value is 0. `client_log.jsonl` and `comm_log.jsonl` contain no F1 values at all (training loss, steps, timing and message sizes only), so they are unaffected. Everything derived from `summary.json` is corrected, including the `history` the dashboard's "Training progress" page reads (6,708 history metric dicts checked against their confusion counts, 0 inconsistent). Do not use `server_log.json` for F1 without recomputing from the confusion counts.
- **Under-training at 20 FedAvg rounds.** Validation loss was still ≈0.63 at round 20 (centralised 0.08); the global weights moved only ~6% from initialisation; each institution takes 3–8 SGD steps per round with momentum reset every round. The pre-registered budget therefore understates FedAvg.
- **Inconclusive at 20 rounds:** the IID-vs-non-IID ranking (group B) and the federated DP sweep (group C: F1 flat across ε = 3.3–42.8). Only Group H (3 seeds) used adequately trained models, and it covers a few configurations, not the full sweeps.
- **Privacy scope.** ε is record-level (add/remove one record), not client-level. It excludes the preprocessing statistics and exact sample counts. Opacus `secure_mode` was off. SecAgg+ still reveals sample counts, fit metrics and participation to the server, and runs in-process with no real trust boundary or TLS. DP + SecAgg do not remove every attack (poisoning, inference from the released model).
- **Federated ε is large** (3–43) because ~68-record institutions force a high sampling rate (q ≈ 0.2).
- **Scale limits.** Simulated institutions; 569 and 920 records; per-institution test shards of 11–61 records, so per-client metrics are noisy; disparity analysis is not demographic or clinical fairness. 3 seeds for groups F and H, no significance tests.
- **Evaluation oracle.** The coordinator evaluates on held-out data; federated evaluation is not implemented. Network time is not measured.

## F. Pending work and recommended next steps

*Unfinished = needed to close out the current deliverable. Optional = potential improvement; nothing optional is required.*

| # | Item | Status | Why it matters | Next action |
|---|---|---|---|---|
| 1 | Manually inspect the dashboard | **Unfinished (yours)** | Automated checks cover loading and rendering, not visual layout, label collisions or chart readability | Open http://127.0.0.1:8501, walk the 9 pages (checklist below); report anything off |
| 2 | Decide whether to run a separately versioned FedAvg tuning experiment | **Decision pending** | The 20-round FedAvg results are under-trained; Group H shows more rounds help but isn't a tuned configuration | Approve or decline. If approved, first resolve the versioning prerequisite in §G |
| 3 | Tune learning rate, local epochs, server momentum on validation only | Optional; **needs approval** | Would give FedAvg a fair, pre-registered configuration | Pre-register the grid and selection rule, then run, only after item 2 |
| 4 | Re-run main comparisons with the selected configuration | Optional; depends on 3 | Makes the centralised vs local vs FedAvg comparison fair | New versioned run set; keep the baseline untouched |
| 5 | Re-evaluate heterogeneity (B) and federated-DP (C) with adequately trained models | Optional; depends on 3–4 | These two comparisons are currently inconclusive | Same versioned set; more seeds if affordable |
| 6 | Legacy per-round F1 in `server_log.json` | Open, low effort | That log disagrees with corrected summaries | Either document as legacy (already done here and in the final report) or write a *new* derived file; do **not** edit the originals |
| 7 | Presentation, report and demo preparation | **Unfinished** | The deliverable the project is actually graded on | Build slides and a live demo script from `docs/final_report.md` and the dashboard |
| 8 | Replace deprecated `use_container_width` | Optional, low risk | Avoids breakage on a future Streamlit upgrade | Only if you upgrade Streamlit |

**Dashboard inspection checklist (item 1):** Overview shows 199 runs / 124 federated / 2 datasets / 5 seeds. Baseline comparison: group A should show centralised F1 0.952 and FedAvg bars with large error bars. Differential privacy: the sweep chart (log-ε axis) and a separate "Other DP runs" table. Participation: pick a `drop0.5` run and confirm unavailable rounds appear as grey cells. Communication: payload 6,916 bytes for WDBC. Look for overlapping labels and unreadable legends.

## G. Reproduction and integrity

**Reproduce the environment and tests** (full steps in `README.md`):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install -e . --no-deps
python scripts/download_data.py
pytest                      # 58 passed (measured ~1.5 min here and in a fresh environment)
```

**Completed baseline (do not overwrite):**

| Item | Location |
|---|---|
| 199 runs (184 main + 15 Group H), all successful | `results/runs/<run_id>/` |
| Run index (199 lines; first 184 = main suite) | `results/suite_index.jsonl` |
| Archived index and log of the main suite | `results/archive/suite_index_main_184runs.jsonl`, `suite_main.log` |
| Pre-F1-fix summaries + report | `results/archive/summaries_before_f1_fix.tar.gz` (SHA-256 `057af454…`) |
| Suite logs | `results/suite.log`, `results/suite_H.log` |
| Validation-only tuning results | `results/tuning_wdbc.json`, `results/tuning_heart_disease.json` |
| Generated tables (regenerable from runs) | `results/report/` |

Status: 199/199 runs complete, 58/58 tests pass (main environment and a fresh environment built from the README), 369/369 checkpoints re-evaluated with 0 mismatches.

**Versioning future experiments — prerequisite, not yet done.**
- `fedmed.experiments.suite` and `fedmed.experiments.report` are hardcoded to `results/suite_index.jsonl` and `results/runs/`. Running a new or modified suite as-is would **append to the baseline index** and mix new runs into the baseline tables.
- Before any new suite: add a configurable index path and runs directory (e.g. `--version v2` writing to `results/runs_v2/` and `results/suite_index_v2.jsonl`), with a test that the baseline index is not touched.
- Single ad-hoc runs are already safe if you pass `output.runs_dir=results/runs_v2`. Run directories are timestamped and created with `exist_ok=False`, so a run cannot overwrite another run.
- The dashboard can point at another directory through its sidebar box.
- Keep `split_seed = 0` so new runs share the baseline test set; record any change to the pre-registered settings *before* looking at results.
- Never edit: `results/runs/*`, `results/suite_index.jsonl`, `results/archive/*`, raw logs, `results/tuning_*.json`. Never rerun `scripts/recompute_f1.py` on new runs unless they carry the old bug (it skips already-corrected summaries).
