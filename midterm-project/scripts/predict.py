import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from catboost import Pool


# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'models')


# Model paths
model_lr_path = os.path.join(models_dir, "model_lr.pkl")
model_rf_path = os.path.join(models_dir, "model_rf.pkl")
model_cb_path = os.path.join(models_dir, "model_cb.pkl")

dv_path = os.path.join(models_dir, "dv_lr.pkl")
scaler_path = os.path.join(models_dir, "scaler_lr.pkl")


# Load models
with open(model_lr_path, "rb") as f:
    model_lr = pickle.load(f)

with open(model_rf_path, "rb") as f:
    model_rf = pickle.load(f)

with open(model_cb_path, "rb") as f:
    model_cb = pickle.load(f)


# Load preprocessing objects
with open(dv_path, "rb") as f:
    dv_lr = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler_lr = pickle.load(f)


# Prediction function
def predict_loan(df_input):
    numerical = [
        'age', 'years_employed', 'annual_income', 'credit_score', 'credit_history_years',
        'savings_assets', 'current_debt', 'delinquencies_last_2yrs', 'derogatory_marks',
        'loan_amount', 'interest_rate', 'debt_to_income_ratio', 'loan_to_income_ratio',
        'payment_to_income_ratio'
    ]
    categorical = ['occupation_status', 'defaults_on_file', 'product_type', 'loan_intent']

    # Ensure all columns exist
    for col in numerical + categorical:
        if col not in df_input.columns:
            # Assign default values
            if col in numerical:
                df_input[col] = 0.0
            else:
                df_input[col] = 'unknown'

    # Convert types
    for col in numerical:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)
    for col in categorical:
        df_input[col] = df_input[col].astype(str).fillna('unknown')


    # Logistic Regression
    X_lr = dv_lr.transform(df_input.to_dict(orient='records'))
    X_lr_scaled = scaler_lr.transform(X_lr)
    proba_lr = model_lr.predict_proba(X_lr_scaled)[:, 1]
    label_lr = (proba_lr >= 0.5).astype(int)


    # Random Forest
    X_rf = dv_lr.transform(df_input.to_dict(orient='records'))
    proba_rf = model_rf.predict_proba(X_rf)[:, 1]
    label_rf = (proba_rf >= 0.5).astype(int)


    # CatBoost
    X_cb = df_input.copy()
    pool_cb = Pool(data=X_cb, cat_features=categorical)
    proba_cb = model_cb.predict_proba(pool_cb)[:, 1]
    label_cb = (proba_cb >= 0.5).astype(int)


    # Compile results
    results = pd.DataFrame({
        'LR_label': label_lr, 'LR_proba': proba_lr,
        'RF_label': label_rf, 'RF_proba': proba_rf,
        'CB_label': label_cb, 'CB_proba': proba_cb
    })

    return results


# Example usage
if __name__ == "__main__":
    df_new = pd.DataFrame([{
        'age': 30,
        'years_employed': 10,
        'annual_income': 300000,
        'credit_score': 800,
        'credit_history_years': 7,
        'savings_assets': 5000000,
        'current_debt': 5000,
        'delinquencies_last_2yrs': 0,
        'derogatory_marks': 0,
        'loan_amount': 20000,
        'interest_rate': 16.36,
        'debt_to_income_ratio': 0.01,
        'loan_to_income_ratio': 0.01,
        'payment_to_income_ratio': 0.01,
        'occupation_status': 'Employed',
        'defaults_on_file': 'no',
        'product_type': 'Credit Card',
        'loan_intent': 'Business'
    }])

    pred = predict_loan(df_new)
    print(pred)
