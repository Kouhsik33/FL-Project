# Architecture

## 1. Components

```
                    ┌────────────────────────── one machine, one Python process tree ─────────────────────────┐
                    │                                                                                          │
  data/raw (UCI) ──▶│ prepare(): parse → global split (fixed seed) → partition → fit preprocessing on client   │
                    │            train partitions only → per-client arrays                                      │
                    │                                                                                          │
                    │  ┌──────────── Coordinator (Flower ServerApp, main thread) ─────────────┐                │
                    │  │ register_clients(): QUERY handshake → node_id ↔ institution index      │                │
                    │  │ SimulatedFedAvg (flwr.server.strategy.FedAvg subclass)                  │                │
                    │  │   configure_fit: availability schedule → select → FitIns(global params) │                │
                    │  │   aggregate_fit: fedavg(Σ n_k/N · w_k), min-participant rule, logging   │                │
                    │  │   evaluate_fn : global val/test + per-client test shards (oracle)       │                │
                    │  │ DefaultWorkflow(fit_workflow = default | SecAggPlusWorkflow)            │                │
                    │  └───────────────▲─────────────────────────────────────┬──────────────────┘                │
                    │                  │ Flower Messages (in-memory grid)     │                                   │
                    │  ┌───────────────┴──────────┐  ...  ┌─────────────────▼────────┐  Ray actors (1 CPU each)  │
                    │  │ ClientApp "institution 0"│       │ ClientApp "institution K" │                          │
                    │  │ mods: comm_meter →       │       │                           │                          │
                    │  │       identity → secagg+ │       │                           │                          │
                    │  │ NumPyClient.fit:         │       │                           │                          │
                    │  │  load own partition only │       │                           │                          │
                    │  │  fresh MLP ← global w    │       │                           │                          │
                    │  │  local SGD / DP-SGD      │       │                           │                          │
                    │  │  ε accountant (Context)  │       │                           │                          │
                    │  └──────────────────────────┘       └───────────────────────────┘                          │
                    └──────────────────────────────────────────────────────────────────────────────────────────┘
                                         │ summary.json, logs, checkpoints → results/runs/<run_id>/ → report, dashboard
```

| Package (`src/fedmed/`) | Responsibility |
|---|---|
| `data/` | download + SHA-256 verification, parsing, splits, partitioning, leakage-safe preprocessing |
| `models/` | MLP, weight (de)serialisation, checkpoints |
| `clients/` | shared training loop (`trainer.py`), Flower ClientApp (`flower_client.py`) |
| `server/` | FedAvg formula (`aggregation.py`), strategy with availability/failure simulation, ServerApp |
| `privacy/` | Opacus DP-SGD wrapper and RDP accounting helpers |
| `evaluation/` | metrics, disparity, communication measurement, log summaries |
| `experiments/` | single-run CLI, validation-only tuning, experiment suite, report generator |
| `dashboard/` | Streamlit app reading saved results only |

## 2. One federated round

1. **Selection.** For round *r*, the strategy reads row *r* of a seeded availability matrix: Bernoulli(1 − `dropout_rate`) per institution. Unavailable institutions are never contacted, so the server cannot wait for them. If fewer than `min_fit_clients` institutions are available, the round is **skipped** and the global model is kept.
2. **Distribution.** Each selected client receives the *current* global parameters in its `FitIns`, together with `server_round` and a `simulate_failure` flag taken from a seeded failure matrix.
3. **Local training.** The client builds a *new* MLP, loads the received weights, and trains for `local_epochs` on its own partition with a seed derived from (run seed, round, client). With DP enabled it uses Opacus DP-SGD and updates its cumulative step count in `Context.state`. A client flagged to fail raises an exception, which emulates a crash after selection; under SecAgg+ the crash happens after key sharing.
4. **Upload.** Without SecAgg, the client sends its parameters and `num_examples`. With SecAgg+, the four-stage protocol sends a masked, quantised vector. The server reconstructs only the sum, removing the masks of dropped clients via Shamir shares.
5. **Aggregation.** `fedavg([(w_k, n_k)])` = Σ (n_k / Σ n) · w_k over **valid results only**. If fewer than `min_fit_clients` valid results remain, the round is skipped.
6. **Evaluation.** The coordinator evaluates the new global model on the global validation and test sets, and on every client's test shard.
7. **Logging.** Participation, failures, sample counts, payload sizes, client train time, ε, and round wall time are recorded.

## 3. Data placement in the simulation

| Data | Who holds it |
|---|---|
| Training records of institution *k* | only client *k*, from `client_data/client_k.npz`, which is written before the run and **deleted afterwards** |
| Global validation / test sets, per-client test shards | the coordinator, *as an evaluation oracle*. This is a simulation convenience: in a deployment, evaluation would be federated. |
| Preprocessing statistics | computed from per-client sufficient statistics (counts, sums, sums of squares). This exchange happens **before** training and **outside** Flower, DP and SecAgg. |

## 4. What is and is not protected

| Mechanism | Protects | Does **not** protect |
|---|---|---|
| Federated learning alone | Raw records never leave a client's partition. | Model updates can leak information about training records (gradient inversion, membership inference [R15, R16]). FL alone is **not** a privacy guarantee. |
| DP-SGD (record-level) | Bounds, by (ε, δ), how much the sequence of a client's released updates (and so the global model) can depend on any one of its records. | Client-level properties; exact sample counts; preprocessing statistics; side channels; the guarantee assumes the RNG is not attacked (`secure_mode=False` here). |
| SecAgg+ | The coordinator cannot see an individual client's update, assuming an honest-but-curious server and fewer than `threshold` colluding parties. | The aggregate and final model; `num_examples` and fit metrics (sent in plaintext by Flower); participation metadata; poisoning by malicious clients; anything when ≤ 2 clients participate (we enforce ≥ 3). |
| DP + SecAgg together | The server sees only the sum of DP-noised updates. | They do **not** eliminate all attacks: model inversion against the released model is bounded only by ε; malicious-server variants, Sybil attacks and poisoning are outside the threat model. |

Transport encryption (TLS) is a separate property and is not used in the in-process simulation. It must not be confused with Secure Aggregation.

## 5. Client availability semantics

- *Unavailable at round start* (`dropout_rate`): the institution is never contacted, and the round proceeds with the others.
- *Crash after selection* (`midround_failure_rate`): the client's reply is an error. Flower converts the `ClientApp` exception into an error message, so the server does not hang. FedAvg uses only the valid results. Under SecAgg+, the protocol reconstructs the dropped client's masks, provided at least `threshold` of its neighbours remain.
- `round_timeout` and the SecAgg+ `timeout` bound waiting time in deployments. In simulation, errors return immediately.

## 6. Measurement definitions

- **Model payload bytes**: total serialised tensor bytes in a Flower `Parameters` object: 6,916 bytes for the 1,537-parameter WDBC MLP.
- **Message bytes**: the Protocol Buffers `ByteSize()` of the `RecordDict` content of every message a client receives or sends, recorded by `comm_meter_mod`. It is broken down by stage (`train`, `train:secagg_setup`, …, `query`). gRPC, TLS and TCP framing are excluded.
- **Times**: `train_time_s` is the client's local training loop. `aggregation_time_s` is the server's FedAvg computation. `round_wall_time_s` and `fl_wall_time_s` are wall-clock times that include Ray scheduling and in-memory message passing. **Network communication time is not measured.**
