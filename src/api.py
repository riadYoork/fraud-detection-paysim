import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import PRE_TX_FEATURES, build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
model = joblib.load(MODEL_DIR / "model.joblib")
threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]

app = FastAPI(
    title="Fraud detection API",
    description="Scores CASH_OUT and TRANSFER transactions (PaySim simulation).",
)


class Transaction(BaseModel):
    step: int = Field(ge=1, description="Hour of the simulation")
    type: Literal["CASH_OUT", "TRANSFER"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    row = build_features(pd.DataFrame([tx.model_dump()]))
    score = float(model.predict_proba(row[PRE_TX_FEATURES])[0, 1])
    return {
           "fraud_score": round(score, 6),
           "flagged": score >= threshold,
           "threshold": round(threshold, 6),
    }