"""
app.py
------
Streamlit application: "Maritime Predictive Maintenance Assistant"

Run with:
    streamlit run app.py

Lets a maintenance engineer either:
  1. Upload a CSV of recent sensor cycles for one or more engines, or
  2. Pick a simulated unit from the demo dataset,
and get a failure-risk prediction, probability trend, and the sensor
readings most responsible for the risk score (feature importances from
the trained Random Forest model).
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from features import build_feature_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="Maritime PdM Assistant", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(ROOT / "models" / "best_model.joblib")
    return bundle["model"], bundle["feature_cols"], bundle["name"]


@st.cache_data
def load_demo_data():
    return pd.read_csv(ROOT / "data" / "maritime_sensor_data.csv")


def main():
    st.title("Maritime Predictive Maintenance Assistant")
    st.caption(
        "Machine-learning decision support for engine/propulsion condition "
        "monitoring. Predicts the probability that an engine will require "
        "maintenance within the next 30 operating cycles."
    )

    model, feature_cols, model_name = load_model()
    demo = load_demo_data()

    with st.sidebar:
        st.header("Data source")
        source = st.radio("Choose input", ["Demo fleet", "Upload CSV"])
        if source == "Upload CSV":
            uploaded = st.file_uploader("Sensor CSV (same schema as demo)", type="csv")
            data = pd.read_csv(uploaded) if uploaded else None
        else:
            unit_ids = sorted(demo["unit_id"].unique())
            unit = st.selectbox("Engine unit", unit_ids)
            data = demo[demo["unit_id"] == unit].copy()

        st.markdown(f"**Active model:** `{model_name}`")

    if data is None:
        st.info("Upload a CSV or select a demo unit from the sidebar to begin.")
        return

    X, _, _, _ = build_feature_matrix(data)
    X = X[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    data = data.reset_index(drop=True)
    data["risk_score"] = proba

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.line(
            data, x="cycle", y="risk_score",
            title="Predicted failure risk over time",
            labels={"risk_score": "P(maintenance needed within 30 cycles)"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                       annotation_text="Alert threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        latest_risk = data["risk_score"].iloc[-1]
        st.metric("Current risk score", f"{latest_risk:.2%}")
        if latest_risk >= 0.5:
            st.error("⚠️ Maintenance recommended soon.")
        else:
            st.success("✅ Engine within normal operating envelope.")

    st.subheader("Recent sensor readings")
    st.dataframe(
        data[["cycle", "exhaust_temp_C", "coolant_temp_C", "lube_oil_pressure_bar",
              "vibration_rms_mm_s", "oil_particle_count", "risk_score"]].tail(20),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
