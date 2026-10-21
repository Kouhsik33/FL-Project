"""Streamlit dashboard for saved experiment outputs.

Run:  streamlit run dashboard/app.py
Every number shown is read from results/runs/*/summary.json; nothing is hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (  # noqa: E402
    aggregated_by_group, dp_sweep, history_frame, load_runs, other_dp_runs, participation_frame,
    per_client_frame,
)

from fedmed.paths import RUNS_DIR  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATE_COLORS = {  # participation states: fixed mapping, never reordered by filters
    "participated": "#2a78d6", "available, not aggregated": "#b4d0f0",
    "crashed mid-round": "#e34948", "unavailable": "#d9d8d4",
}
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]

MEASURED = "🟢 **Measured** — computed from a saved run"
CONFIGURED = "🔵 **Configured** — a simulation parameter set before the run"
ESTIMATED = "🟠 **Computed bound** — derived by the privacy accountant, not observed"


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(colorway=SERIES, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="closest", legend=dict(orientation="h", y=-0.2))
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2), marker=dict(size=8))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


st.set_page_config(page_title="FL Medical Diagnosis", page_icon="🩺", layout="wide")


@st.cache_data(show_spinner=False)
def _load(runs_dir: str, stamp: float):  # stamp (newest mtime) is part of the cache key
    return load_runs(Path(runs_dir))


runs_dir = st.sidebar.text_input("Runs directory", str(RUNS_DIR))
stamp = max((p.stat().st_mtime for p in Path(runs_dir).glob("*/summary.json")), default=0.0)
store = _load(runs_dir, stamp)
for err in store.errors:
    st.sidebar.warning(err)

PAGES = ["Overview", "Dataset & partitions", "Baseline comparison", "Training progress",
         "Differential privacy", "Participation & dropout", "Communication & time",
         "Per-client performance", "Experiment history"]
page = st.sidebar.radio("Page", PAGES)
st.sidebar.caption(f"{len(store.runs)} saved runs loaded")

if not store.runs:
    st.title("Federated Learning for Privacy-Preserving Medical Diagnosis")
    st.info("No experiment results found. Run e.g. "
            "`python -m fedmed.experiments.run --config configs/default.yaml` or the suite "
            "`python -m fedmed.experiments.suite`, then refresh.")
    st.stop()

table = store.table
agg = aggregated_by_group(store)


def pick_run(label: str, ids: list[str], key: str) -> dict:
    if not ids:
        st.warning("No matching runs.")
        st.stop()
    rid = st.selectbox(label, ids, index=len(ids) - 1, key=key)
    return store.runs[rid]


def mean_std_bar(sub: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=sub["name"], y=sub[f"{metric}_mean"],
        error_y=dict(type="data", array=sub[f"{metric}_std"].fillna(0)),
        marker_color=SERIES[0], hovertemplate="%{x}<br>mean %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, yaxis_title=f"{metric} (mean ± std over seeds)")
    return style(fig)


# ============================================================== pages
if page == "Overview":
    st.title("Federated Learning for Privacy-Preserving Medical Diagnosis")
    st.warning("**Research prototype — simulation only.** Multiple institutions are *simulated* on one "
               "computer using public datasets. No real hospital participated, no clinical validation was "
               "performed, and the models must not be used for diagnosis.")
    st.markdown("""
**Objectives (from the submitted abstract)**
1. Institutions collaboratively train a model without sharing patient records — records stay in each client's partition.
2. A coordinator aggregates updates with sample-weighted **FedAvg** and redistributes the global model.
3. Evaluate accuracy, precision, recall, F1, training time and communication overhead.
4. Analyse how data distributions across institutions affect performance.
5. Investigate **Differential Privacy** (Opacus DP-SGD, record-level) and **Secure Aggregation** (Flower SecAgg+).
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved runs", len(table))
    c2.metric("Federated runs", int((table["mode"] == "federated").sum()))
    c3.metric("Datasets", table["dataset"].nunique())
    c4.metric("Seeds", table["seed"].nunique())
    st.markdown("**Legend used throughout**")
    st.markdown(f"- {MEASURED}\n- {CONFIGURED}\n- {ESTIMATED}")
    env = next(iter(store.runs.values()))["environment"]
    st.caption("Environment of the first run: " + ", ".join(f"{k}={v}" for k, v in env.items()))

elif page == "Dataset & partitions":
    st.header("Dataset and simulated institutions")
    s = pick_run("Run", sorted(store.runs), "ds_run")
    d = s["dataset"]
    st.markdown(f"**{d['name']}** — {d['n_records']} records, {d['n_features']} features. "
                f"Positive class: *{d['positive_label']}* ({d['n_positive']} records).")
    for note in d.get("notes", []):
        st.caption("• " + note)
    st.markdown(f"{MEASURED}: split sizes {d['split_sizes']}, positives per split {d['split_positive']}")
    p = s["partition"]
    st.markdown(f"{CONFIGURED}: strategy **{p['strategy']}**, α = {p['alpha']}, seed {p['seed']}, "
                f"{len(p['clients'])} clients. *Artificial partitions do not represent real hospital populations.*")
    cl = pd.DataFrame(p["clients"])
    long = cl.melt(id_vars=["client", "name"], value_vars=["n_train_neg", "n_train_pos"],
                   var_name="class", value_name="records")
    long["class"] = long["class"].map({"n_train_neg": d["negative_label"], "n_train_pos": d["positive_label"]})
    fig = px.bar(long, x="name", y="records", color="class", barmode="stack",
                 title="Training records per simulated institution", color_discrete_sequence=SERIES)
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(cl, use_container_width=True)

elif page == "Baseline comparison":
    st.header("Centralised vs local-only vs FedAvg")
    st.markdown(f"{MEASURED} on the held-out global test set; mean ± std over seeds. "
                "Local-only = unweighted mean of each institution's isolated model on the global test set.")
    if agg.empty:
        st.info("Run the experiment suite to populate grouped comparisons.")
        st.stop()
    group = st.selectbox("Experiment group", sorted(agg["group"].unique()),
                         format_func=lambda g: {"A": "A: baselines (IID)", "B": "B: heterogeneity",
                                                "C": "C: differential privacy", "D": "D: dropout",
                                                "F": "F: secure aggregation", "G": "G: Heart Disease (natural sites)"}.get(g, g))
    metric = st.selectbox("Metric", METRICS, index=4)
    sub = agg[agg["group"] == group]
    st.plotly_chart(mean_std_bar(sub, f"test_{metric}", f"Test {metric} by run"), use_container_width=True)
    show = ["name", "mode", "n_seeds"] + [c for c in sub.columns if c.startswith("test_") and c.endswith(("_mean", "_std"))]
    st.dataframe(sub[show].round(4), use_container_width=True)

elif page == "Training progress":
    st.header("Convergence across communication rounds")
    ids = store.federated_ids()
    chosen = st.multiselect("Federated runs (overlay up to 8)", ids, default=ids[-1:] if ids else [], max_selections=8)
    split = st.radio("Evaluation set", ["test", "val"], horizontal=True)
    metric = st.selectbox("Metric", METRICS + ["log_loss"], index=4)
    fig = go.Figure()
    for i, rid in enumerate(chosen):
        h = history_frame(store.runs[rid], split)
        fig.add_trace(go.Scatter(x=h["round"], y=h[metric], mode="lines+markers", name=store.runs[rid]["name"]
                                 + f" (s{store.runs[rid]['seed']})", line=dict(color=SERIES[i % 8])))
    fig.update_layout(title=f"Global model {metric} on {split} set (round 0 = initial model)",
                      xaxis_title="round", yaxis_title=metric)
    st.plotly_chart(style(fig, 420), use_container_width=True)
    st.caption(f"{MEASURED}. Rounds skipped for insufficient participation keep the previous global model.")

elif page == "Differential privacy":
    st.header("Differential privacy: utility vs privacy budget")
    st.markdown(f"{ESTIMATED}: ε is the RDP-accountant bound for **record-level** DP at each institution "
                "(add/remove one record, δ shown), composed over all local steps in all rounds it trained. "
                "It is *not* client-level DP and does not cover preprocessing statistics or sample counts.")
    dp = table[table["dp"] == True]  # noqa: E712
    if dp.empty:
        st.info("No DP runs saved yet.")
        st.stop()
    metric = st.selectbox("Utility metric", METRICS, index=4)
    g = dp_sweep(table, metric)
    st.caption("Plotted: the controlled DP sweep (group C: WDBC, 5 IID clients or centralised, 20 rounds/epochs). "
               "Other DP runs are different experiments and are listed separately below.")
    fig = go.Figure()
    for i, (mode, sub) in enumerate(g.groupby("mode")):
        sub = sub.sort_values("epsilon_mean")
        fig.add_trace(go.Scatter(x=sub["epsilon_mean"], y=sub["mean"], error_y=dict(array=sub["std"].fillna(0)),
                                 mode="lines+markers", name=mode, line=dict(color=SERIES[i]),
                                 text=[f"σ={s}" for s in sub["sigma"]],
                                 hovertemplate="%{text}<br>ε=%{x:.2f}<br>%{y:.3f}<extra></extra>"))
    nondp = table[(table["dp"] == False) & (table["group"].isin(["A"])) & (table["name"].isin(["central", "fedavg_iid_k5"]))]  # noqa: E712
    for i, (name, sub) in enumerate(nondp.groupby("name")):
        fig.add_hline(y=sub[f"test_{metric}"].mean(), line_dash="dot", line_color="#898781",
                      annotation_text=f"{name} without DP", annotation_position="bottom right")
    fig.update_layout(title=f"Test {metric} vs ε (mean ± std over seeds)", xaxis_title="ε (δ = 1e-5), log scale",
                      yaxis_title=metric, xaxis_type="log")
    st.plotly_chart(style(fig, 420), use_container_width=True)
    st.dataframe(g.round(4), use_container_width=True)
    st.markdown("**Other DP runs (not part of the sweep)**")
    st.dataframe(other_dp_runs(table, metric).round(4), use_container_width=True)
    ids = [k for k in store.federated_ids() if store.runs[k].get("privacy")]
    if ids:
        s = pick_run("Per-client accounting for run", ids, "dp_run")
        st.dataframe(pd.DataFrame(s["privacy"]["per_client"]), use_container_width=True)
        st.caption(s["privacy"]["guarantee"])

elif page == "Participation & dropout":
    st.header("Client availability and participation")
    st.markdown("*Client dropout* = an institution unavailable or crashing during a round. "
                "This is unrelated to dropout regularisation (the MLP uses none).")
    s = pick_run("Federated run", store.federated_ids(), "part_run")
    f = s["config"]["federated"]
    st.markdown(f"{CONFIGURED}: P(unavailable) = {f['dropout_rate']}, P(mid-round crash) = "
                f"{f['midround_failure_rate']}, minimum participants = {f['min_fit_clients']}.")
    pf = participation_frame(s)
    fig = px.scatter(pf, x="round", y="client", color="state", color_discrete_map=STATE_COLORS,
                     symbol="state", title="Participation per round (measured)", hover_data=["round_status"])
    fig.update_traces(marker=dict(size=14, line=dict(width=2, color="white")))
    fig.update_yaxes(dtick=1)
    st.plotly_chart(style(fig, 360), use_container_width=True)
    st.json(s["participation"])
    if not agg.empty and "D" in set(agg["group"]):
        st.subheader("Effect of dropout (group D, plus no-dropout baselines)")
        base = agg[(agg["group"] == "A") & agg["name"].isin(["fedavg_iid_k5", "fedavg_iid_k10"])]
        sub = pd.concat([base, agg[agg["group"] == "D"]])
        st.plotly_chart(mean_std_bar(sub, "test_f1", "Test F1 under client unavailability"), use_container_width=True)
        st.dataframe(sub[["name", "n_seeds", "test_f1_mean", "test_f1_std", "rounds_aggregated_mean"]].round(4),
                     use_container_width=True)

elif page == "Communication & time":
    st.header("Communication volume and training time")
    st.markdown(f"{MEASURED}: protobuf-serialised Flower message content per client message "
                "(excludes gRPC/TLS/TCP framing). Clients and server run on one machine, so **network "
                "latency and transfer time were not measured**. Wall-clock times include in-process "
                "simulation overhead (Ray scheduling).")
    s = pick_run("Federated run", store.federated_ids(), "comm_run")
    comm = s["communication"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model payload (bytes)", f"{comm['model_payload_bytes']:,}")
    c2.metric("Training traffic, total (bytes)", f"{comm['training_total_bytes']:,}")
    c3.metric("Per aggregated round (bytes)", f"{comm['training_total_bytes_per_aggregated_round'] or 0:,.0f}")
    stages = pd.DataFrame([{"stage": k, **v} for k, v in comm["by_stage"].items()])
    long = stages.melt(id_vars=["stage", "messages"], value_vars=["downlink_bytes", "uplink_bytes"],
                       var_name="direction", value_name="bytes")
    fig = px.bar(long, x="stage", y="bytes", color="direction", barmode="group",
                 title="Bytes by protocol stage", color_discrete_sequence=SERIES)
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(stages, use_container_width=True)
    st.json(s["timing"])
    fed = table[table["mode"] == "federated"]
    if not fed.empty:
        st.subheader("Bytes per aggregated round across runs")
        g = fed.groupby(["name"]).agg(bytes_per_round=("comm_bytes_per_round", "mean"),
                                      clients=("n_clients", "first"), secagg=("secagg", "first")).reset_index()
        fig = px.bar(g.sort_values("bytes_per_round"), x="name", y="bytes_per_round", color="secagg",
                     color_discrete_sequence=SERIES, title="Measured training traffic per aggregated round")
        st.plotly_chart(style(fig), use_container_width=True)

elif page == "Per-client performance":
    st.header("Per-client performance disparity")
    st.markdown("Disparity across **simulated** institutions on their own test shards. This is not evidence "
                "about demographic or clinical fairness. Shards are small, so undefined metrics (e.g. recall "
                "with no positive cases) are shown as blank.")
    s = pick_run("Run", sorted(store.runs), "pc_run")
    pc = per_client_frame(s)
    metric = st.selectbox("Metric", METRICS, index=2)
    fig = px.bar(pc, x="client", y=metric, hover_data=["n", "n_pos"], title=f"Per-client test {metric}",
                 color_discrete_sequence=SERIES)
    st.plotly_chart(style(fig), use_container_width=True)
    st.dataframe(pc, use_container_width=True)
    st.json(s["final"]["disparity"].get(metric, {}))

elif page == "Experiment history":
    st.header("All saved runs")
    st.dataframe(table, use_container_width=True)
    st.download_button("Download runs table (CSV)", table.to_csv(index=False), "runs.csv", "text/csv")
    s = pick_run("Inspect run", sorted(store.runs), "hist_run")
    st.markdown(f"Started {s['started_utc']} — finished {s['finished_utc']}")
    st.subheader("Configuration"); st.json(s["config"], expanded=False)  # noqa: E702
    st.subheader("Environment"); st.json(s["environment"], expanded=False)  # noqa: E702
