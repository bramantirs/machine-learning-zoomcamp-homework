# Import and load dataset
import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve
from catboost import CatBoostClassifier


# Config / Seed
RANDOM_STATE = 11


# Path dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'Loan_approval_data_2025.csv')

models_dir = os.path.join(BASE_DIR, 'models')

df = pd.read_csv(data_path)


# Define features
numerical = [
    'age', 
    'years_employed', 
    'annual_income', 
    'credit_score', 
    'credit_history_years', 
    'savings_assets',
    'current_debt',
    'delinquencies_last_2yrs',
    'derogatory_marks',
    'loan_amount',
    'interest_rate',
    'debt_to_income_ratio',
    'loan_to_income_ratio',
    'payment_to_income_ratio'
]

categorical = [
    'occupation_status',
    'defaults_on_file',
    'product_type',
    'loan_intent'
]


# Train / Val / Test Split
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=RANDOM_STATE, stratify=df.loan_status)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=RANDOM_STATE, stratify=df_full_train.loan_status)

y_train = df_train.loan_status.values
y_val = df_val.loan_status.values
y_test = df_test.loan_status.values

X_train = df_train.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)
X_val = df_val.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)
X_test = df_test.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)


# Logistic Regression model
dv_lr = DictVectorizer(sparse=False)
scaler_lr = StandardScaler()

X_train_lr = dv_lr.fit_transform(X_train[categorical + numerical].to_dict(orient='records'))
X_val_lr = dv_lr.transform(X_val[categorical + numerical].to_dict(orient='records'))
X_test_lr = dv_lr.transform(X_test[categorical + numerical].to_dict(orient='records'))

X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr)
X_val_lr_scaled = scaler_lr.transform(X_val_lr)
X_test_lr_scaled = scaler_lr.transform(X_test_lr)


# Save DictVectorizer & Scaler
with open(os.path.join(models_dir, "dv_lr.pkl"), "wb") as f:
    pickle.dump(dv_lr, f)
with open(os.path.join(models_dir, "scaler_lr.pkl"), "wb") as f:
    pickle.dump(scaler_lr, f)


model_lr = LogisticRegression(
    C=1,
    solver='liblinear',
    max_iter=500,
    random_state=RANDOM_STATE
)
model_lr.fit(X_train_lr_scaled, y_train)


# Save Logistic Regression model
model_lr_path = os.path.join(models_dir, "model_lr.pkl")
with open(model_lr_path, "wb") as f:
    pickle.dump(model_lr, f)


# Random Forest model
X_train_enc = dv_lr.transform(X_train[categorical + numerical].to_dict(orient='records'))
X_val_enc = dv_lr.transform(X_val[categorical + numerical].to_dict(orient='records'))

model_rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model_rf.fit(X_train_enc, y_train)


# Save Random Forest model
model_rf_path = os.path.join(models_dir, "model_rf.pkl")
with open(model_rf_path, "wb") as f:
    pickle.dump(model_rf, f)


# CatBoost model
X_train_cb = df_train.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)
X_val_cb = df_val.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)
X_test_cb = df_test.drop(columns=['customer_id', 'loan_status']).reset_index(drop=True)

cat_features = ['occupation_status', 'defaults_on_file', 'product_type', 'loan_intent']

for c in cat_features:
    X_train_cb[c] = X_train_cb[c].astype(str)
    X_val_cb[c] = X_val_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

model_cb = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    verbose=0
)
model_cb.fit(X_train_cb, y_train, cat_features=cat_features, eval_set=(X_val_cb, y_val))


# Save CatBoost model
model_cb_path = os.path.join(models_dir, "model_cb.pkl")
with open(model_cb_path, "wb") as f:
    pickle.dump(model_cb, f)


# Evaluation
def evaluate_model(model, X_input, y_true):
    y_pred_proba = model.predict_proba(X_input)[:,1]
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    print("AUC:", roc_auc_score(y_true, y_pred_proba))
    print("Accuracy:", accuracy_score(y_true, y_pred_label))
    print("F1 Score:", f1_score(y_true, y_pred_label))
    print("Classification Report:\n", classification_report(y_true, y_pred_label))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_label))
    print("\n")

print("=== Logistic Regression ===")
evaluate_model(model_lr, X_test_lr_scaled, y_test)

print("=== Random Forest ===")
evaluate_model(model_rf, X_val_enc, y_val)

print("=== CatBoost ===")
evaluate_model(model_cb, X_test_cb, y_test)