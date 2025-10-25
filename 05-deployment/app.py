import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI()

with open("pipeline_v1.bin","rb") as f_in:
    pipeline = pickle.load(f_in)

@app.get("/")
def root():
    return {"status":"ok"}

@app.post("/predict")
def predict(lead: Lead):
    rec = lead.dict()
    proba = pipeline.predict_proba([rec])[:,1][0] if hasattr(pipeline, "predict_proba") else pipeline.predict([rec])[0]
    return {"probability": float(proba)}
