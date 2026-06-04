import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Real-Time Fraud Analytics",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL (PRODUCTION SAFE)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/xgb_fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# SESSION STATE (SIMULATE REAL BEHAVIOR)
# ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# UI HEADER
# ---------------------------------------------------
st.title("🧠 Real-Time Fraud Analytics Dashboard")
st.markdown("AI-powered financial risk monitoring system")

# ---------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------
st.sidebar.header("Transaction Input")

time = st.sidebar.number_input("Transaction Time", 0.0, 100000.0, 1000.0)
amount = st.sidebar.number_input("Transaction Amount", 0.0, 100000.0, 500.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Mode (Auto Feature Generator)")
auto_mode = st.sidebar.checkbox("Auto-generate V1–V28 (Recommended)", value=True)

# ---------------------------------------------------
# FEATURE ENGINE (IMPORTANT FIX)
# ---------------------------------------------------
def generate_features(amount):

    # realistic fraud signal injection
    base = np.random.normal(0, 1, 28)

    # high amount increases anomaly chance
    fraud_signal = min(amount / 10000, 5)

    # inject fraud behavior pattern
    if amount > 10000:
        base += np.random.normal(fraud_signal, 1.5, 28)

    return base

def get_input():

    if auto_mode:
        v_features = generate_features(amount)
    else:
        v_features = np.array([
            st.sidebar.number_input(f"V{i}", value=0.0)
            for i in range(1, 29)
        ])

    data = np.hstack([[time], v_features, [amount]])
    return data.reshape(1, -1)

input_data = get_input()

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("🚀 Predict Fraud Risk"):

    prob = model.predict_proba(input_data)[0][1]

    # stronger risk logic
    if prob < 0.3:
        risk = "LOW"
        color = "green"
    elif prob < 0.7:
        risk = "MEDIUM"
        color = "orange"
    else:
        risk = "HIGH"
        color = "red"

    # store history
    st.session_state.history.append(prob)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud Probability", f"{prob:.4f}")
    col2.metric("Risk Level", risk)
    col3.metric("Transaction Amount", f"${amount:,.2f}")

    st.markdown("---")

    # ---------------------------------------------------
    # GAUGE CHART
    # ---------------------------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Fraud Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # RISK BREAKDOWN CHART
    # ---------------------------------------------------
    chart = pd.DataFrame({
        "Type": ["Safe Score", "Fraud Score"],
        "Value": [1 - prob, prob]
    })

    fig2 = px.bar(
        chart,
        x="Type",
        y="Value",
        color="Type",
        color_discrete_map={
            "Safe Score": "green",
            "Fraud Score": "red"
        },
        title="Risk Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------
    # ALERTS
    # ---------------------------------------------------
    if risk == "HIGH":
        st.error("⚠️ HIGH FRAUD RISK DETECTED")
    elif risk == "MEDIUM":
        st.warning("⚠️ Medium Risk Transaction")
    else:
        st.success("✅ Transaction is Safe")

# ---------------------------------------------------
# ANALYTICS SECTION
# ---------------------------------------------------
st.markdown("---")
st.subheader("📊 Live Fraud Analytics")

if len(st.session_state.history) > 1:

    hist_df = pd.DataFrame({
        "Transaction": list(range(len(st.session_state.history))),
        "Fraud Probability": st.session_state.history
    })

    fig3 = px.line(hist_df, x="Transaction", y="Fraud Probability", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Run multiple predictions to see analytics")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Production Fraud Detection System | XGBoost + Streamlit + Feature Engineering")