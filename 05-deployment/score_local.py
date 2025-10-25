# score_local.py
import pickle
import json

with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# If pipeline expects list of dicts
score = pipeline.predict_proba([record])[:,1][0] if hasattr(pipeline, "predict_proba") else pipeline.predict([record])[0]

print("probability:", round(float(score), 3))
