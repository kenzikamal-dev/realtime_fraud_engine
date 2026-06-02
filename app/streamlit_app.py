import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# LOAD MODEL (FIXED FOR STREAMLIT CLOUD)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_fraud_model.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="FICO Fraud Engine", layout="wide")

st.title("🧠 Real-Time Fraud Detection System (FICO-Style)")
st.markdown("Enter transaction details to calculate fraud risk score (0–1).")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Transaction Input")

def get_user_input():
    time = st.sidebar.number_input("Time", 0.0)
    amount = st.sidebar.number_input("Amount", 0.0)

    v_features = []
    for i in range(1, 29):
        v = st.sidebar.number_input(f"V{i}", value=0.0)
        v_features.append(v)

    data = [time] + v_features + [amount]
    return np.array(data).reshape(1, -1)

input_data = get_user_input()

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Fraud Risk"):
    prob = model.predict_proba(input_data)[0][1]

    # Risk classification
    if prob < 0.2:
        risk = "LOW"
        color = "green"
    elif prob < 0.5:
        risk = "MEDIUM"
        color = "orange"
    else:
        risk = "HIGH"
        color = "red"

    # -----------------------------
    # OUTPUT UI
    # -----------------------------
    st.subheader("Prediction Result")

    st.metric(label="Fraud Probability", value=f"{prob:.4f}")

    st.markdown(f"### Risk Level: :{color}[{risk}]")

    if risk == "HIGH":
        st.error("⚠️ High Fraud Risk Detected")
    elif risk == "MEDIUM":
        st.warning("⚠️ Medium Risk Transaction")
    else:
        st.success("✅ Low Risk Transaction")

    st.markdown("---")
    st.write("Model Type: XGBoost Fraud Detection Engine")