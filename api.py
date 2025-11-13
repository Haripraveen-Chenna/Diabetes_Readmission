from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI(title="Diabetes Readmission ML API")

# Load models (ensure ensemble_xgb.pkl exists in this folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ensemble_xgb.pkl")
CALIB_PATH = os.path.join(os.path.dirname(__file__), "calibrator.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("ensemble_xgb.pkl not found in ml_service/ - add your saved models")

models = pickle.load(open(MODEL_PATH, "rb"))

# optional calibrator
if os.path.exists(CALIB_PATH):
    calibrator = pickle.load(open(CALIB_PATH, "rb"))
    HAS_CALIB = True
else:
    calibrator = None
    HAS_CALIB = False

class Patient(BaseModel):
    # include the fields your model expects. Minimal example:
    age_num: float
    time_in_hospital: float
    num_medications: float
    number_diagnoses: float

def ensemble_predict(df: pd.DataFrame) -> float:
    # df should have same preprocessed columns as model expects.
    preds = [m.predict_proba(df)[0][1] for m in models]
    p = float(np.mean(preds))
    if HAS_CALIB and calibrator is not None:
        p = float(calibrator.predict_proba([[p]])[0][1])
    return p

@app.get("/")
def root():
    return {"status": "ok", "message": "Diabetes ML API running"}

@app.post("/predict")
def predict(patient: Patient):
    # convert to dataframe with expected column names
    data = {k: [v] for k, v in patient.dict().items()}
    df = pd.DataFrame(data)
    prob = ensemble_predict(df)
    return {"probability": prob, "prediction": int(prob >= 0.5)}
