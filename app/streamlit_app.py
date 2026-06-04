import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random
from datetime import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="FICO Fraud Analytics",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL (SAFE PATH)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# FRAUD RULE ENGINE (NEW - FIX)
# ---------------------------------------------------
def fraud_rule_engine(amount, ml_prob, transaction_time):
    """
    Enhances ML prediction with banking rules
    """

    risk_boost = 0

    # Rule 1: High amount risk
    if amount > 5000:
        risk_boost += 0.25
    if amount > 20000:
        risk_boost += 0.35

    # Rule 2: Midnight risk (0-5 AM)
    if 0 <= transaction_time <= 5:
        risk_boost += 0.20

    # Rule 3: Very large ML + rules synergy
    final_score = ml_prob + risk_boost

    return min(final_score, 1.0)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main { background-color: #0E1117; color: white; }

.stButton>button {
    background-color: #00C897;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧠 Real-Time Fraud Analytics Dashboard")
st.markdown("### AI-Powered Financial Risk Monitoring System (Production Mode)")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Fraud Prediction", "Fraud Analytics", "Transaction Monitoring"]
)

# ---------------------------------------------------
# PAGE 1 — FRAUD PREDICTION
# ---------------------------------------------------
if page == "Fraud Prediction":

    st.header("💳 Transaction Risk Prediction")

    # Time now default (IMPORTANT FIX)
    current_hour = datetime.now().hour

    def get_user_input():

        time = st.number_input("Transaction Hour (0–23)", 0, 23, current_hour)
        amount = st.number_input("Transaction Amount ($)", 0.0)

        v_features = []
        for i in range(1, 29):
            v = st.number_input(f"V{i}", value=0.0)
            v_features.append(v)

        data = [time] + v_features + [amount]

        return np.array(data).reshape(1, -1), amount, time

    input_data, amount, time = get_user_input()

    if st.button("🚀 Predict Fraud Risk"):

        # ML prediction
        ml_prob = model.predict_proba(input_data)[0][1]

        # APPLY RULE ENGINE (FIX)
        final_prob = fraud_rule_engine(amount, ml_prob, time)

        # Risk classification
        if final_prob < 0.2:
            risk = "LOW"
            color = "green"

        elif final_prob < 0.5:
            risk = "MEDIUM"
            color = "orange"

        else:
            risk = "HIGH"
            color = "red"

        # KPIs
        col1, col2, col3 = st.columns(3)

        col1.metric("ML Probability", f"{ml_prob:.4f}")
        col2.metric("Final Fraud Score", f"{final_prob:.4f}")
        col3.metric("Risk Level", risk)

        st.markdown("---")

        # GAUGE
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_prob * 100,
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

        # ALERTS
        if risk == "HIGH":
            st.error("⚠️ HIGH RISK TRANSACTION DETECTED")

        elif risk == "MEDIUM":
            st.warning("⚠️ Medium Risk Transaction")

        else:
            st.success("✅ Low Risk Transaction")

# ---------------------------------------------------
# PAGE 2 — ANALYTICS
# ---------------------------------------------------
elif page == "Fraud Analytics":

    st.header("📊 Fraud Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions", "12,847")
    col2.metric("Fraud Alerts", "312")
    col3.metric("Fraud Rate", "2.4%")
    col4.metric("Revenue Protected", "$84K")

    trend_data = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Fraud Cases": [12, 19, 8, 15, 22, 10, 17]
    })

    fig = px.line(trend_data, x="Day", y="Fraud Cases", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PAGE 3 — TRANSACTION MONITORING
# ---------------------------------------------------
elif page == "Transaction Monitoring":

    st.header("🛰️ Live Transactions")

    transactions = []

    for i in range(15):

        amount = round(random.uniform(10, 50000), 2)
        risk_score = round(random.uniform(0, 1), 2)

        transactions.append({
            "Transaction ID": f"TXN-{1000+i}",
            "Amount": amount,
            "Risk Score": risk_score,
            "Status": "FRAUD" if amount > 20000 else "SAFE"
        })

    df = pd.DataFrame(transactions)

    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Production Fraud Detection System | XGBoost + Rule Engine + Streamlit")