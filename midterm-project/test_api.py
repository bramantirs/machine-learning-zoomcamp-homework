import requests

url = "http://127.0.0.1:8000/predict"  # alamat endpoint FastAPI
data = {
    "age": 30,
    "years_employed": 5,
    "annual_income": 50000,
    "credit_score": 700,
    "credit_history_years": 5,
    "savings_assets": 10000,
    "current_debt": 5000,
    "delinquencies_last_2yrs": 0,
    "derogatory_marks": 0,
    "loan_amount": 20000,
    "interest_rate": 5,
    "debt_to_income_ratio": 20,
    "loan_to_income_ratio": 40,
    "payment_to_income_ratio": 15,
    "occupation_status": "Employed",
    "defaults_on_file": "no",
    "product_type": "Personal Loan",
    "loan_intent": "Education"
}

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())