# app.py
import streamlit as st
import pandas as pd
import requests

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>💳 Credit Card Fraud Detection</h1>
        <p>This app predicts whether a credit card transaction is <b>fraudulent</b> or <b>legitimate</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Enter Transaction Features</h3>",
            unsafe_allow_html=True)

# ---------------------------
# Input features
# ---------------------------
cols = st.columns(2)
with cols[0]:
    Time = st.number_input(
        "⏱ Time (seconds since first transaction)", min_value=0, value=0)
with cols[1]:
    Amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=0.0)

# V1-V28 input
feature_values = {}
num_features = 28
cols_per_row = 5
for i in range(1, num_features + 1, cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j <= num_features:
            feature_values[f"V{i+j}"] = col.number_input(f"V{i+j}", value=0.0)

feature_values['Time'] = Time
feature_values['Amount'] = Amount

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Fraud"):
    # Call FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=feature_values
    )

    if response.status_code == 200:
        result = response.json()
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 20px;'>
                <h2>📊 Prediction Results</h2>
                <h3>Fraud Probability: <b>{result['fraud_probability']:.3f}</b></h3>
                <h3>Prediction: <b>{result['prediction']}</b></h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Error: Could not get prediction from API")
