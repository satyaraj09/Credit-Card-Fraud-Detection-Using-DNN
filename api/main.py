# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from keras.models import load_model

# ---------------------------
# Load Model and Scaler
# ---------------------------
model = load_model('model/credit_card_detection.h5')
scaler = joblib.load('model/scaler.pkl')

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

# ---------------------------
# Pydantic Model for input validation
# ---------------------------


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# ---------------------------
# Root endpoint
# ---------------------------


@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running."}

# ---------------------------
# Prediction endpoint
# ---------------------------


@app.post("/predict")
def predict(transaction: Transaction):
    # Convert input to DataFrame
    input_df = pd.DataFrame([transaction.model_dump()])

    # Scale 'Time' and 'Amount'
    input_df[['Amount', 'Time']] = scaler.transform(
        input_df[['Amount', 'Time']]
    )

    # Predict
    prob = model.predict(input_df)[0][0]
    prediction = int(prob > 0.5)

    return {
        "fraud_probability": round(float(prob), 4),
        "prediction": "Fraud" if prediction == 1 else "Legitimate"
    }
