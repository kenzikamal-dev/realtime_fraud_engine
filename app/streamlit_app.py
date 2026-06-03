import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FICO Fraud Engine",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    background-color: #00C897;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧠 Real-Time Fraud Detection Dashboard")
st.markdown("### AI-Powered Financial Transaction Risk Engine")

# ---------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------
st.sidebar.header("💳 Transaction Input")

def get_user_input():
    time = st.sidebar.number_input("Transaction Time", 0.0)
    amount = st.sidebar.number_input("Transaction Amount", 0.0)

    v_features = []
    for i in range(1, 29):
        v = st.sidebar.number_input(f"V{i}", value=0.0)
        v_features.append(v)

    data = [time] + v_features + [amount]
    return np.array(data).reshape(1, -1), amount

input_data, amount = get_user_input()

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("🚀 Predict Fraud Risk"):

    prob = model.predict_proba(input_data)[0][1]

    # Risk Classification
    if prob < 0.2:
        risk = "LOW"
        color = "green"

    elif prob < 0.5:
        risk = "MEDIUM"
        color = "orange"

    else:
        risk = "HIGH"
        color = "red"

    # ---------------------------------------------------
    # TOP METRICS
    # ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Fraud Probability", f"{prob:.4f}")

    with col2:
        st.metric("Risk Level", risk)

    with col3:
        st.metric("Transaction Amount", f"${amount:.2f}")

    st.markdown("---")

    # ---------------------------------------------------
    # GAUGE CHART
    # ---------------------------------------------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Fraud Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 20], 'color': "green"},
                {'range': [20, 50], 'color': "orange"},
                {'range': [50, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    # ---------------------------------------------------
    # ANALYTICS CHART
    # ---------------------------------------------------
    chart_data = pd.DataFrame({
        "Metric": ["Safe Score", "Fraud Score"],
        "Value": [1 - prob, prob]
    })

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Value",
        color="Metric",
        title="Fraud Analytics",
        color_discrete_map={
            "Safe Score": "green",
            "Fraud Score": "red"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # ALERTS
    # ---------------------------------------------------
    if risk == "HIGH":
        st.error("⚠️ High Fraud Risk Transaction Detected")

    elif risk == "MEDIUM":
        st.warning("⚠️ Medium Risk Transaction")

    else:
        st.success("✅ Low Risk Transaction")

    # ---------------------------------------------------
    # FOOTER
    # ---------------------------------------------------
    st.markdown("---")
    st.caption("Powered by XGBoost + Streamlit + Plotly")