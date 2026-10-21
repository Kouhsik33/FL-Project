# Dataset assessment

Both datasets come from the UCI Machine Learning Repository and are licensed **CC BY 4.0**. They need no registration, credentials or special terms. Attribution is given below and in [references.md](references.md). They are public, de-identified research datasets: using them here does **not** demonstrate protection of real private hospital records.

Download and verification are handled by `python scripts/download_data.py`. It fetches each archive, checks a SHA-256 digest recorded on 2026-10-20, rejects unsafe archive paths, extracts the files into `data/raw/<name>/`, and writes a `MANIFEST.json`. Raw files are never modified. Derived arrays exist only in memory, or temporarily per run, and are git-ignored.

## 1. Primary: Breast Cancer Wisconsin (Diagnostic), "WDBC"

| Property | Verified value |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| DOI / citation | 10.24432/C5DW2B. Wolberg, Mangasarian, Street & Street (1993) [D1]; feature method in Street et al. (1993) [D4] |
| Licence | CC BY 4.0 |
| Download | `https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip` (SHA-256 `bc154869…05d3af`) |
| File used | `wdbc.data` (SHA-256 `d606af41…c3695f4a`), 569 rows, no header |
| Columns | ID, diagnosis (M/B), then 30 real-valued features: the mean, standard error and "worst" value of 10 nuclear measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) computed from a digitised image of a fine-needle aspirate |
| Labels | 212 malignant (M), 357 benign (B). Verified by count. |
| Missing values | none (verified) |
| Task | binary classification. **Positive class = malignant (1).** |

**Processing.** The ID column is dropped (it is an identifier and must never be used as a feature). The 30 features are standardised with mean and standard deviation fitted on training data only.

**Limitations.**
- Small: 569 records, so each simulated institution holds only about 34–114 training records. Per-client test shards contain about 11–38 records, so per-client metrics are noisy.
- Single source: one clinical group in Wisconsin, early 1990s. It is not representative of current populations, imaging equipment or clinical workflows.
- The features are engineered image descriptors, not raw clinical data. The task is known to be easy (published accuracies are often above 95%), which leaves limited headroom to separate methods.
- There are no demographic attributes, so no demographic fairness analysis is possible, and none is attempted.
- Splitting one dataset into "institutions" is artificial. It does not reproduce the real differences between hospitals.

## 2. Secondary validation: Heart Disease (UCI), four collection sites

### Why this dataset rather than Pima Indians Diabetes

| Criterion | Pima Indians Diabetes | Heart Disease (UCI id 45) |
|---|---|---|
| Current authoritative source | Not listed on the current UCI repository (searched 2026-10-20). It circulates via mirrors (Kaggle, R packages) with unclear provenance and licence. | UCI repository, DOI 10.24432/C52P4X, CC BY 4.0 |
| Data quality | Zeros encode missing values for glucose, blood pressure, skin thickness, insulin and BMI, despite older documentation claiming "no missing values" [D5] | Missing values are explicitly marked `?`. Some are encoded as zeros (cholesterol = 0). |
| Relevance to FL | Single population | **Four real collection sites**: Cleveland Clinic, Hungarian Institute of Cardiology (Budapest), University Hospitals Zurich and Basel (Switzerland), VA Long Beach. This gives a *natural* cross-silo split, as used by the FLamby benchmark [D3]. |

**[Decision]** Heart Disease is the recommended secondary dataset. It has a verifiable source and licence, and its natural partition by real site lets us test FL under genuine (not synthetic) heterogeneity. Pima was not used, so as not to depend on a source whose licence could not be verified.

| Property | Verified value |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/45/heart+disease |
| DOI / citation | 10.24432/C52P4X. Janosi, Steinbrunn, Pfisterer & Detrano (1989) [D2]; Detrano et al. (1989) [D6] |
| Licence | CC BY 4.0 |
| Download | `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` (SHA-256 `b17cd273…51563adc`) |
| Files used | `processed.cleveland.data` (303), `processed.hungarian.data` (294), `processed.switzerland.data` (123), `processed.va.data` (200), 920 rows in total |
| Label | `num` (0–4). Binary target: **positive = disease present (num > 0)**. 509 positive (Cleveland 139, Hungary 106, Switzerland 115, VA 149). |

**Missingness measured per site (percent of rows):**

| Feature | Cleveland | Hungary | Switzerland | VA |
|---|---|---|---|---|
| slope | 0 | 65 | 14 | 51 |
| ca | 1 | 99 | 96 | 99 |
| thal | 1 | 90 | 42 | 83 |
| chol = 0 (encodes missing) | 0 | 0 | 100 | 25 |

**Processing.**
- Drop `slope`, `ca` and `thal`. They are mostly missing outside Cleveland, and imputing them would mostly encode site identity.
- Treat `chol == 0` as missing (0 mg/dl is physiologically impossible).
- One-hot encode the nominal codes `cp` and `restecg`, using the fixed category sets from `heart-disease.names`, which does not depend on the data.
- Mean-impute the remaining missing values and standardise, with statistics fitted on training data only.
- Result: 16 features.

**Limitations.**
- Data collected in the 1980s. The site cohorts differ sharply: Switzerland is 93% positive, and its cholesterol is entirely unrecorded. Site-level effects are therefore confounded with label prevalence and with missingness.
- The `processed.*` files keep 14 of the original 76 attributes. Patient names and social-security numbers were replaced with dummy values by the donors.
- The angiographic disease label is not a modern clinical endpoint.
- Small per-site test shards (16–61 records).
- Results are not comparable with FLamby's reported numbers, because our preprocessing and splits differ.

## 3. Splits and leakage controls (both datasets)

- One stratified **60 / 20 / 20 train / validation / test split** per dataset, fixed by `data.split_seed = 0`. It is stratified by label, and for Heart Disease by site × label. **Every** run of a dataset is evaluated on the same test set: 114 records for WDBC, 184 for Heart Disease.
- Simulated institutions receive disjoint shares of the training pool. The same per-class proportions are used to divide the validation and test pools into per-client shards, so the union of the shards is exactly the global test set.
- Imputation and standardisation statistics come from the clients' training partitions only. They are computed from summed sufficient statistics, which are mathematically identical to fitting on the pooled training split. Tests check this equivalence and check that test data is never used.
- Hyper-parameters were chosen on validation data. DP parameters are experimental variables, reported for every setting rather than tuned on test data.
