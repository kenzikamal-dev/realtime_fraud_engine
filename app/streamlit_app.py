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
    page_title="FICO Fraud Analytics",
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
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🧠 Real-Time Fraud Analytics Dashboard")
st.markdown("### AI-Powered Financial Risk Monitoring System")

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

    def get_user_input():

        time = st.number_input("Transaction Time", 0.0)
        amount = st.number_input("Transaction Amount", 0.0)

        v_features = []

        for i in range(1, 29):
            v = st.number_input(f"V{i}", value=0.0)
            v_features.append(v)

        data = [time] + v_features + [amount]

        return np.array(data).reshape(1, -1), amount

    input_data, amount = get_user_input()

    if st.button("🚀 Predict Fraud Risk"):

        prob = model.predict_proba(input_data)[0][1]

        # Risk Classification
        if prob < 0.2:
            risk = "LOW"

        elif prob < 0.5:
            risk = "MEDIUM"

        else:
            risk = "HIGH"

        # KPI CARDS
        col1, col2, col3 = st.columns(3)

        col1.metric("Fraud Probability", f"{prob:.4f}")
        col2.metric("Risk Level", risk)
        col3.metric("Transaction Amount", f"${amount:.2f}")

        st.markdown("---")

        # GAUGE CHART
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

        # BAR CHART
        chart_data = pd.DataFrame({
            "Metric": ["Safe Score", "Fraud Score"],
            "Value": [1 - prob, prob]
        })

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            color="Metric",
            color_discrete_map={
                "Safe Score": "green",
                "Fraud Score": "red"
            },
            title="Fraud Analytics"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ALERTS
        if risk == "HIGH":
            st.error("⚠️ High Fraud Risk Transaction")

        elif risk == "MEDIUM":
            st.warning("⚠️ Medium Risk Transaction")

        else:
            st.success("✅ Low Risk Transaction")

# ---------------------------------------------------
# PAGE 2 — FRAUD ANALYTICS
# ---------------------------------------------------
elif page == "Fraud Analytics":

    st.header("📈 Fraud Analytics Dashboard")

    # KPI CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transactions", "12,847")
    col2.metric("Fraud Alerts", "312")
    col3.metric("Fraud Rate", "2.4%")
    col4.metric("Revenue Protected", "$84K")

    st.markdown("---")

    # SIMULATED TREND DATA
    trend_data = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Fraud Cases": [12, 19, 8, 15, 22, 10, 17]
    })

    fig = px.line(
        trend_data,
        x="Day",
        y="Fraud Cases",
        markers=True,
        title="Weekly Fraud Trend"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PAGE 3 — TRANSACTION MONITORING
# ---------------------------------------------------
elif page == "Transaction Monitoring":

    st.header("🛰️ Transaction Monitoring")

    transactions = []

    for i in range(10):

        transactions.append({
            "Transaction ID": f"TXN-{1000+i}",
            "Amount": round(random.uniform(10, 5000), 2),
            "Risk Score": round(random.uniform(0, 1), 2),
            "Status": random.choice(["SAFE", "REVIEW", "FRAUD"])
        })

    df = pd.DataFrame(transactions)

    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Powered by XGBoost + Streamlit + Plotly")