# Technical research

Project: **Federated Learning for Privacy-Preserving Medical Diagnosis** (simulated research prototype).
Research date: 2026-10-20. Numbered citations refer to [references.md](references.md).

Each statement is tagged:

- **[Fact]**: established by official documentation, the installed source code, or a primary paper.
- **[Finding]**: a published research result.
- **[Decision]**: an engineering choice made for this project, with its reason.
- **[Hypothesis]**: something our experiments test. Nothing tagged this way is assumed true.

---

## 1. Local environment (inspected 2026-10-20)

| Item | Value |
|---|---|
| OS | macOS 26.7.1 (Darwin 25.6.0), **x86_64 (Intel)** |
| CPU / RAM | 12 logical cores, 16 GB |
| GPU | none used (CPU-only design) |
| Python | 3.11.15 (also 3.14 present; not used) |

**[Fact]** PyTorch stopped publishing macOS x86_64 binaries after the 2.2 series. PyTorch 2.2.x is the last release that supports Intel Macs [P1, P2]. On this machine `pip index versions torch` lists 2.2.2 as the newest installable version.

## 2. Dependency compatibility (verified by installing and importing)

| Package | Version | Why this version |
|---|---|---|
| torch | 2.2.2 | Last release with macOS x86_64 wheels [P1]. |
| numpy | 1.26.x (< 2) | torch 2.2 wheels are built against NumPy 1.x. |
| flwr | 1.40.0 | Latest release at research time. |
| ray | 2.49.2 | `flwr[simulation]` 1.24–1.40 pins `ray==2.51.1` or `2.55.1`. **Neither has a macOS x86_64 wheel**; 2.49.2 is the newest that does. Flower's core package does not depend on Ray, so `flwr==1.40.0` + `ray==2.49.2` resolves cleanly, and the simulation runtime works with it (verified by the end-to-end tests). |
| opacus | 1.5.3 | 1.6.0 requires a newer torch. 1.5.4 declares `torch>=2.0` but fails to import on torch 2.2 (`AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`; `nn.RMSNorm` first appeared in torch 2.4). 1.5.3 imports, and its DP-SGD + RDP accounting was verified by our tests. |
| pyarrow | 21.0.0 | Streamlit pulls pyarrow; pyarrow ≥ 22 requires NumPy ≥ 2, which breaks the scikit-learn import under NumPy 1.26. |
| scikit-learn, pandas, streamlit, plotly, pyyaml, pytest | see `requirements.lock` | No conflicts. |

**[Decision]** These pins are recorded in `pyproject.toml`, and the exact versions are frozen in `requirements.lock`. On Apple-silicon or Linux machines newer versions would be installable. They are **not** required, and the pins keep results reproducible across machines.

## 3. Flower (federated learning framework)

**[Fact]** Flower 1.40 offers two server-side programming models:

1. The **Message API strategies** in `flwr.serverapp.strategy` (e.g. `FedAvg` operating on `ArrayRecord`s).
2. The **legacy-compatible workflow API**: a `ServerApp` with `@app.main()` that builds a `LegacyContext` holding a `flwr.server.strategy` strategy, then executes `DefaultWorkflow`.

**[Fact]** Flower's Secure Aggregation implementation, `SecAggPlusWorkflow`, is a *fit workflow* for `DefaultWorkflow`. It requires a `LegacyContext` (it raises `TypeError` otherwise) and calls `context.strategy.configure_fit` / `aggregate_fit` of a legacy strategy. The client side is `flwr.client.mod.secaggplus_mod`. This was confirmed in the installed source (`flwr/server/workflow/secure_aggregation/secaggplus_workflow.py`) and in Flower's official `flower-secure-aggregation` example [F3].

**[Decision]** We use the legacy-compatible workflow API for *all* federated runs. Plain FedAvg and SecAgg+ FedAvg then share one strategy class, so the only difference between them is the fit workflow. This makes their comparison controlled.

**[Fact]** `flwr.simulation.run_simulation` is marked deprecated in 1.40 in favour of the `flwr run` CLI, but is still supported. We use it because the experiment suite launches more than 100 configured runs programmatically. Each run executes in its own process (`fedmed.experiments.suite`) to isolate Ray state.

**[Fact]** In the simulation runtime, an exception raised by a `ClientApp` is converted into an error reply. The server therefore receives a failure rather than waiting indefinitely. Node `Context.state` persists across rounds for the same node (`vce_api.py`). We rely on this to keep each client's cumulative DP step count.

**[Fact]** Flower also ships DP wrappers: central DP with server-side or client-side clipping (`DifferentialPrivacyServerSide*`, `DifferentialPrivacyClientSide*`) and `LocalDpMod`. Its DP guide notes that instance-level privacy during local training is provided by privacy engines such as Opacus [F2].

## 4. PyTorch model

**[Decision]** We use a small MLP (30 → 32 → 16 → 1 with ReLU; 1,537 parameters for WDBC), `BCEWithLogitsLoss`, and SGD with momentum 0.9. It has no batch normalisation, because BatchNorm mixes samples and so is incompatible with per-sample gradients in DP-SGD (Opacus rejects it). It has no dropout regularisation, so the term "dropout" in this project only ever means client unavailability.

**[Decision]** Hyper-parameters (learning rate 0.005, 20 epochs, batch size 16) were selected by grid search on the **validation split only**, minimising mean validation log loss over 3 seeds (`python -m fedmed.experiments.tune`; `results/tuning_*.json`). The test set was not used.

## 5. Differential privacy with Opacus

**[Fact]** DP-SGD [R2] clips each per-example gradient to L2 norm ≤ C, adds Gaussian noise N(0, σ²C²) to the sum, and averages. Combined with Poisson subsampling at rate q, the privacy loss of T steps is accounted with Rényi DP for the subsampled Gaussian mechanism [R3, R4]. Opacus implements this through `PrivacyEngine.make_private(..., poisson_sampling=True)` and the `RDPAccountant` [O1, R12].

**[Fact]** Opacus' default adjacency for Poisson-sampled DP-SGD is add/remove one record. Its default RNG is not cryptographically secure unless `secure_mode=True`, and Opacus warns about this. Floating-point implementation attacks on DP are documented in the literature.

**[Decision]** Each simulated institution runs DP-SGD locally on its own records. The client stores its cumulative number of noisy steps in `Context.state`. After every round it computes ε via the RDP accountant, using the history `(σ, q, total_steps)`; q is constant per client because the dataset size and batch size are fixed. A test verifies that this helper reproduces Opacus' own `PrivacyEngine.get_epsilon` for the same run.

**[Fact]** Composition across rounds is valid because each round's global model is a post-processing of earlier releases and of *other* institutions' data, which is disjoint from this client's records. Adaptive composition of DP-SGD steps therefore gives the cumulative guarantee. Different clients hold disjoint records, so a given record is only ever touched by one client's mechanism. We report the **maximum per-client ε**.

**Scope of the guarantee (must accompany any reported ε):**

- Record-level (example-level), **not** client-level DP. It does not hide whether an institution took part, or properties of its data distribution. Client-level DP would require clipping and noising whole client updates [R13], which Flower's central-DP wrappers provide. With 3–10 clients, the noise needed for client-level DP is large; we note it as future work.
- It does **not** cover the preprocessing statistics (per-feature counts, sums and sums of squares from each client, used for standardisation) or the exact sample counts that FedAvg uses as weights. Under add/remove adjacency, an exact count is itself not DP.
- The guarantee holds under the assumptions of the accountant: Poisson sampling, the data-independent σ and C shown, and the chosen δ = 1e-5, which is below 1/n for every client.
- `secure_mode=False` in our experiments. This is fine for a research simulation, but a deployable guarantee requires `secure_mode=True`.

**[Fact]** Example of planning numbers, computed with the accountant rather than guessed (5 clients, ~68 records each, batch 16, so q = 0.2 and 5 steps per round; δ = 1e-5; 20 rounds): σ = 1.0 gives ε ≈ 16.0, σ = 2.0 gives ε ≈ 5.5, σ = 3.0 gives ε ≈ 3.3. Small local datasets force a high sampling rate, which weakens the privacy amplification from subsampling. This is an inherent cost of DP-SGD on small cohorts.

## 6. Secure Aggregation

**[Finding]** Bonawitz et al. [R5] introduced practical secure aggregation for FL. Clients mask their updates with pairwise masks (derived via Diffie–Hellman key agreement) plus a self-mask, and secret-share their keys with Shamir's scheme so the server can remove the masks of dropped clients. The server learns only the sum of the updates of the surviving clients, provided fewer than a threshold of parties collude. Bell et al. [R6] (SecAgg+) replace the complete communication graph with a sparse random one, reducing cost to polylogarithmic in the number of clients.

**[Fact]** Flower's `SecAggPlusWorkflow` + `secaggplus_mod` implement SecAgg+ in four stages: setup (public keys), share keys (encrypted Shamir shares), collect masked vectors, and unmask. Parameters are clipped to `[-clipping_range, clipping_range]`, scaled by `num_examples / max_weight`, quantised to `quantization_range` levels, and masked modulo `modulus_range`.

**[Fact]: what the coordinator observes under Flower SecAgg+ (from the installed source):**

- **Hidden:** each client's individual model parameters. The plaintext `ArrayRecord` is cleared, and only the masked vector is sent.
- **Visible in plaintext:** each client's `num_examples` (the `fitres.num_examples` metric record) and the client's fit-metrics dictionary, which the mod passes through unchanged. Also visible: which clients participated or dropped in each stage, public keys, encrypted shares, message timing and sizes. And of course the aggregate.
- **[Decision]** Under SecAgg we therefore stop sending per-client training loss to the server. Each client's ε is still reported, because it is a public accountant output that depends only on σ, q and the step count.

**[Fact]: assumptions and failure conditions:**

- Honest-but-curious server, with fewer than `reconstruction_threshold` colluding parties. Flower's documentation calls this a simulation-grade implementation. Its README recommends TLS and SuperNode authentication for deployments [F3]; our simulation runs in one process, so the trust boundary is notional.
- If, at any stage, some sampled client has fewer than `threshold` active neighbours, the round fails ("Insufficient available nodes").
- With only **two** participants, each one can subtract its own update from the revealed sum and recover the other's. We therefore enforce `min_fit_clients ≥ 3` whenever SecAgg is enabled (a `ValueError` otherwise).
- Quantisation introduces a small error. We measured that SecAgg+ reproduces plaintext FedAvg's global model to within 1e-3 per parameter (`tests/test_end_to_end.py`).
- SecAgg does not protect the final model, does not stop a malicious client from poisoning the model, and does not replace DP.

**[Decision]** SecAgg+ was integrated because Flower provides an auditable, tested implementation that slots into our workflow without any architecture change.

## 7. Federated learning in medicine and non-IID data

- **[Finding]** FedAvg [R1] performs local SGD epochs and averages the resulting models weighted by sample count. It can converge with fewer communication rounds than synchronous SGD, but it degrades under non-IID data [R7, R8].
- **[Finding]** Hsu et al. [R9] introduced Dirichlet(α) label partitioning to control heterogeneity: smaller α means more skew. This is the method our `dirichlet` strategy uses.
- **[Finding]** Reviews of FL in healthcare [R10, R11, R14] stress that FL reduces data movement but is not by itself a privacy guarantee. Model updates can leak training data through gradient inversion and membership inference [R15, R16], which motivates DP and SecAgg.
- **[Finding]** FLamby [D3] is a benchmark with natural cross-silo splits. One of its datasets, Fed-Heart-Disease, uses the four UCI Heart Disease sites. We use the same raw source in our secondary experiment, with our own preprocessing (see dataset_assessment.md); our results are not comparable to FLamby's numbers.

## 8. Hypotheses tested (outcomes in [final_report.md](final_report.md) §2)

- **H1.** Under IID partitions, FedAvg reaches test performance close to centralised training and better than the average local-only model. We do *not* assume it matches or exceeds centralised training.
- **H2.** Stronger label skew (smaller Dirichlet α) reduces FedAvg performance and increases the disparity between clients.
- **H3.** Lower ε (more DP noise) reduces utility, and the cost is larger in federated settings with small client datasets.
- **H4.** Moderate client unavailability (20%) has little effect on final performance. High unavailability (50%) slows convergence and causes skipped rounds.
- **H5.** SecAgg+ leaves utility unchanged (up to quantisation error) and increases the number and volume of messages.

## 9. Novelty

**[Decision]** We make **no novelty claim.** FedAvg, DP-SGD, SecAgg+ and their combination in FL are all established. The contribution of this project is a careful, reproducible, tested implementation, plus an honest empirical analysis on public tabular medical data.
