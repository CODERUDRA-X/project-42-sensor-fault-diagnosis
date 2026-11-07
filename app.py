from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Make the repository's src/ package importable when Streamlit is launched
# from the repository root with: streamlit run app.py
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project42.inference import (  # noqa: E402
    CLASSES,
    FEATURES,
    FEATURE_LABELS,
    OBSERVED_RANGES,
    SensorDiagnosisEngine,
    SensorReadings,
)

st.set_page_config(
    page_title="Project 42 | Sensor Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown(
    """
    <style>
      .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.28);
             border-radius: 18px; background: linear-gradient(115deg, rgba(60,90,130,.15), rgba(35,150,130,.08));}
      .eyebrow {font-size: .78rem; letter-spacing: .13em; text-transform: uppercase;
                opacity: .72; font-weight: 700;}
      .hero h1 {margin: .25rem 0 .35rem 0; font-size: 2.15rem;}
      .muted {opacity: .72;}
      .small-note {font-size: .84rem; opacity: .78;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem;
                                     border-radius: 12px;}
      .state-normal {color: #23966b; font-weight: 800;}
      .state-warning {color: #d69a25; font-weight: 800;}
      .state-critical {color: #d65b55; font-weight: 800;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the saved Gaussian Naive Bayes model...")
def load_engine() -> SensorDiagnosisEngine:
    return SensorDiagnosisEngine(model_path=ROOT / "model" / "sensor_fault_gnb_pipeline.joblib")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    path = ROOT / "data" / "machinery_vibration_telemetry.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = [*FEATURES, "operational_state"]
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()
    return frame


try:
    engine = load_engine()
except Exception as exc:  # Keep the failure visible and actionable.
    st.error(f"Could not load the saved model: {exc}")
    st.info("Check that model/sensor_fault_gnb_pipeline.joblib exists and dependencies are installed.")
    st.stop()


def state_class(state: str) -> str:
    return {"Normal": "state-normal", "Warning": "state-warning", "Critical": "state-critical"}.get(state, "")


def run_diagnosis(values: dict[str, float]) -> dict:
    return engine.diagnose(values)


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Project 42 · Research prototype</div>
      <h1>Machine Sensor Fault Diagnosis</h1>
      <div class="muted">Explore how four sensor readings are analysed by a saved Gaussian Naive Bayes model.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")
st.warning(
    "Demonstration only: this model was trained on synthetic telemetry. It is not a certified industrial "
    "monitoring, protection, or shutdown system. Its displayed probabilities are uncalibrated."
)

# ---------- Sidebar: input controls ----------
st.sidebar.header("Sensor input")
st.sidebar.caption("Enter readings or load a labelled example from the synthetic dataset.")

sample_df = load_dataset()
if not sample_df.empty:
    sample_classes = [c for c in CLASSES if c in set(sample_df["operational_state"].astype(str))]
    sample_choice = st.sidebar.selectbox(
        "Load a dataset example",
        ["Manual input"] + sample_classes,
        help="Examples are real rows from this project's synthetic dataset. The model may predict a different class from the dataset label.",
    )
else:
    sample_choice = "Manual input"

if "sensor_values" not in st.session_state:
    st.session_state.sensor_values = {
        "vibration_amplitude_g": 1.0,
        "acoustic_decibels": 68.0,
        "bearing_temperature_c": 65.0,
        "motor_amperage": 22.0,
    }

if sample_choice != "Manual input" and not sample_df.empty:
    if st.sidebar.button("Load selected dataset example", use_container_width=True):
        matching = sample_df[sample_df["operational_state"].astype(str) == sample_choice]
        if not matching.empty:
            row = matching.iloc[0]
            st.session_state.sensor_values = {feature: float(row[feature]) for feature in FEATURES}
            for feature in FEATURES:
                st.session_state[f"input_{feature}"] = float(row[feature])
            st.session_state.loaded_sample_id = str(row.get("machine_id", "Dataset example"))

labels = {
    "vibration_amplitude_g": ("Vibration amplitude", "g", "How strongly the machine vibrates."),
    "acoustic_decibels": ("Acoustic level", "dB", "Sound level measured near the machine."),
    "bearing_temperature_c": ("Bearing temperature", "°C", "Temperature around the bearing."),
    "motor_amperage": ("Motor amperage", "A", "Electrical current drawn by the motor."),
}

values: dict[str, float] = {}
for feature in FEATURES:
    low, high = OBSERVED_RANGES[feature]
    label, unit, help_text = labels[feature]
    current = float(st.session_state.sensor_values[feature])
    values[feature] = st.sidebar.number_input(
        f"{label} ({unit})",
        min_value=0.0,
        max_value=float(max(high * 1.5, current + 1.0)),
        value=current,
        step=0.01 if unit in ("g",) else 0.1,
        help=f"{help_text} Observed synthetic-data range: {low:g}–{high:g} {unit}. This is not a safety limit.",
        key=f"input_{feature}",
    )

if st.sidebar.button("Reset sensor readings", use_container_width=True):
    st.session_state.sensor_values = {
        "vibration_amplitude_g": 1.0,
        "acoustic_decibels": 68.0,
        "bearing_temperature_c": 65.0,
        "motor_amperage": 22.0,
    }
    for feature in FEATURES:
        st.session_state.pop(f"input_{feature}", None)
    st.rerun()

# Use values from the actual widgets, not preset labels, for every prediction.
try:
    result = run_diagnosis(values)
except Exception as exc:
    st.error(f"Diagnosis could not be computed: {exc}")
    st.stop()

# ---------- Main dashboard ----------
tab_live, tab_explain, tab_performance, tab_about = st.tabs(
    ["🔎 Live diagnosis", "🧠 How it works", "📊 Model evaluation", "ℹ️ Dataset & limitations"]
)

with tab_live:
    st.subheader("Diagnosis result")
    st.caption("Every result below is generated at runtime by the saved model in model/.")

    predicted = result["predicted_state"]
    col_state, col_standard, col_risk = st.columns([1.25, 1, 1])
    with col_state:
        st.markdown("**Model prediction**")
        st.markdown(f'<div class="{state_class(predicted)}" style="font-size:2rem">{predicted}</div>', unsafe_allow_html=True)
    with col_standard:
        st.metric("Standard decision", result["standard_decision"])
    with col_risk:
        st.metric("Cost-sensitive decision", result["cost_sensitive_decision"])

    if result["range_warnings"]:
        for warning in result["range_warnings"]:
            st.warning(warning)
    else:
        st.success("All four readings fall within the observed ranges of the synthetic dataset. This does not mean they are safe operating values.")

    st.markdown("#### Model probability output")
    # Keep the same Normal → Warning → Critical order in both the chart and table.
    probability_df = pd.DataFrame(
        {"Raw probability (%)": [result["probabilities"][class_name] * 100 for class_name in CLASSES]},
        index=pd.Index(CLASSES, name="Class"),
    )
    # Horizontal bars keep class labels readable and make the class order explicit.
    st.bar_chart(probability_df, y="Raw probability (%)", horizontal=True, height=220)
    st.dataframe(
        probability_df.style.format({"Raw probability (%)": "{:.6f}%"}),
        use_container_width=True,
    )
    st.caption("These are raw GaussianNB outputs from an uncalibrated pipeline. They are not calibrated real-world confidence scores.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Sensor readings used")
        reading_rows = []
        for feature in FEATURES:
            label, unit, _ = labels[feature]
            reading_rows.append({"Sensor": label, "Reading": values[feature], "Unit": unit})
        st.dataframe(pd.DataFrame(reading_rows), hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Expected illustrative risk")
        risk_df = pd.DataFrame(
            {"Chosen action": list(result["expected_risk_illustrative_units"].keys()),
             "Expected risk (illustrative units)": list(result["expected_risk_illustrative_units"].values())}
        ).set_index("Chosen action")
        st.dataframe(risk_df.style.format("{:.4f}"), use_container_width=True)
        st.caption("Lower is preferred by the notebook's illustrative cost matrix. These are not monetary values or measured plant losses.")

    st.markdown("#### What sensor evidence is contributing?")
    evidence = result["critical_vs_normal_evidence"]["sensor_contributions"]
    evidence_df = pd.DataFrame(
        [{"Sensor": item["sensor"], "Evidence direction": item["direction"],
          "Log-likelihood contribution": item["critical_vs_normal_log_likelihood_ratio"]} for item in evidence]
    )
    st.dataframe(evidence_df.style.format({"Log-likelihood contribution": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.caption("Positive contribution supports Critical relative to Normal; negative supports Normal. This is model evidence, not proof of physical cause.")

    with st.expander("View the cost matrix and decision rule"):
        st.write("Rows are the actual class; columns are the chosen action. The model selects the action with the lowest expected illustrative cost.")
        cost_df = pd.DataFrame(result["cost_matrix"], index=CLASSES, columns=CLASSES)
        cost_df.index.name = "Actual ↓ / Action →"
        st.dataframe(cost_df, use_container_width=True)

with tab_explain:
    st.subheader("How the diagnosis works")
    st.markdown(
        """
        Think of a machine as reporting four clues. The model compares the combination of clues
        with patterns it learned from labelled synthetic examples.

        1. **Read the sensors** — vibration, sound, bearing temperature, and motor current.
        2. **Scale the inputs** — the saved `StandardScaler` transforms the four measurements.
        3. **Estimate class probabilities** — Gaussian Naive Bayes estimates support for Normal, Warning, and Critical.
        4. **Choose an action** — standard prediction selects the largest raw probability; the cost-sensitive rule selects the action with the lowest expected cost under the supplied matrix.
        """
    )
    st.info("Naive Bayes assumes the features are conditionally independent given the class. Real sensor signals can be correlated, so this assumption is a modelling simplification—not a statement about machine physics.")
    st.markdown("#### The four input signals")
    for feature in FEATURES:
        label, unit, help_text = labels[feature]
        lo, hi = OBSERVED_RANGES[feature]
        st.markdown(f"**{label} ({unit})** — {help_text}  ")
        st.caption(f"Observed synthetic dataset range: {lo:g}–{hi:g} {unit}; not an engineering threshold.")

with tab_performance:
    st.subheader("What was measured during model evaluation")
    st.markdown(
        "The following values are from the project's executed notebook. They describe separate evaluation protocols; do not treat them as the live prediction's confidence."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("5-fold CV accuracy", "86.40%", help="Mean across five cross-validation folds.")
    c2.metric("5-fold CV macro F1", "0.7928")
    c3.metric("5-fold CV ROC-AUC", "0.9387")
    c4, c5, c6 = st.columns(3)
    c4.metric("Holdout accuracy", "87.27%", help="Separate 25% holdout evaluation.")
    c5.metric("Holdout Critical recall", "77.93%")
    c6.metric("Holdout log loss", "0.4220")
    st.markdown("#### Calibration experiment (holdout)")
    calibration = pd.DataFrame(
        [
            {"Method": "Raw GaussianNB", "Brier score": 0.2074, "ECE": 0.0687},
            {"Method": "Platt scaling", "Brier score": 0.2012, "ECE": 0.0459},
            {"Method": "Isotonic calibration", "Brier score": 0.1895, "ECE": 0.0039},
        ]
    )
    st.dataframe(calibration.style.format({"Brier score": "{:.4f}", "ECE": "{:.4f}"}), hide_index=True, use_container_width=True)
    st.caption("Calibration results are retrospective notebook experiments. The saved live model remains the uncalibrated StandardScaler → GaussianNB pipeline.")
    st.markdown("#### Cost-sensitive evaluation (out-of-fold predictions)")
    cost_eval = pd.DataFrame(
        [
            {"Decision policy": "Standard argmax", "Accuracy": "86.40%", "Critical recall": "82.56%", "Total illustrative cost": "6,827"},
            {"Decision policy": "Minimum expected cost", "Accuracy": "79.52%", "Critical recall": "90.67%", "Total illustrative cost": "4,895"},
        ]
    )
    st.dataframe(cost_eval, hide_index=True, use_container_width=True)
    st.caption("This is an out-of-fold evaluation, not the holdout test. The cost policy used raw cross-validated probabilities; interpret the comparison as illustrative research evidence.")

with tab_about:
    st.subheader("Dataset, scope, and limitations")
    st.markdown(
        """
        - **Dataset:** 6,000 rows of physics-inspired synthetic machinery telemetry.
        - **Inputs:** four sensor features; target labels are Normal, Warning, and Critical.
        - **Saved model:** `StandardScaler → GaussianNB`.
        - **Live probability output:** uncalibrated.
        - **Sensor ranges:** observed minimum and maximum in the synthetic dataset, not safe operating limits.
        - **Cost matrix:** illustrative units, not currency or measured industrial loss.
        """
    )
    st.error("Do not connect this prototype to real machinery controls or use it to make safety-critical decisions. Real deployment would require representative plant data, engineering validation, monitoring, calibration, fail-safe design, and formal safety review.")
    if not sample_df.empty:
        st.markdown("#### Dataset preview")
        st.dataframe(sample_df.head(10), hide_index=True, use_container_width=True)

st.divider()
st.caption("Project 42 · Synthetic Sensor Fault Diagnosis · Research and educational demonstration")
