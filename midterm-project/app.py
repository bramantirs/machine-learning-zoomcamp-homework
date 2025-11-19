# app_fastapi.py
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from catboost import Pool

# =========================
# Load models & preprocessing
# =========================
with open("models/dv_lr.pkl", "rb") as f:
    dv_lr = pickle.load(f)

with open("models/scaler_lr.pkl", "rb") as f:
    scaler_lr = pickle.load(f)

with open("models/model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)

with open("models/model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

with open("models/model_cb.pkl", "rb") as f:
    model_cb = pickle.load(f)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Loan Prediction API")

@app.get("/")
def read_root():
    return {"message": "API is running"}

# =========================
# Define input schema
# =========================
class LoanInput(BaseModel):
    age: float
    years_employed: float
    annual_income: float
    credit_score: float
    credit_history_years: float
    savings_assets: float
    current_debt: float
    delinquencies_last_2yrs: float
    derogatory_marks: float
    loan_amount: float
    interest_rate: float
    debt_to_income_ratio: float
    loan_to_income_ratio: float
    payment_to_income_ratio: float
    occupation_status: str
    defaults_on_file: str
    product_type: str
    loan_intent: str

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict_loan_endpoint(input_data: LoanInput):
    df_input = pd.DataFrame([input_data.dict()])

    # --- Logistic Regression ---
    X_lr = dv_lr.transform(df_input.to_dict(orient='records'))
    X_lr_scaled = scaler_lr.transform(X_lr)
    proba_lr = model_lr.predict_proba(X_lr_scaled)[:, 1]
    label_lr = (proba_lr >= 0.5).astype(int)

    # --- Random Forest ---
    X_rf = dv_lr.transform(df_input.to_dict(orient='records'))
    proba_rf = model_rf.predict_proba(X_rf)[:, 1]
    label_rf = (proba_rf >= 0.5).astype(int)

    # --- CatBoost ---
    X_cb = df_input.copy()
    cat_features = ['occupation_status', 'defaults_on_file', 'product_type', 'loan_intent']
    for c in cat_features:
        X_cb[c] = X_cb[c].astype(str)
    pool_cb = Pool(data=X_cb, cat_features=cat_features)
    proba_cb = model_cb.predict_proba(pool_cb)[:, 1]
    label_cb = (proba_cb >= 0.5).astype(int)

    return {
        "LR_label": int(label_lr[0]), "LR_proba": float(proba_lr[0]),
        "RF_label": int(label_rf[0]), "RF_proba": float(proba_rf[0]),
        "CB_label": int(label_cb[0]), "CB_proba": float(proba_cb[0])
    }
