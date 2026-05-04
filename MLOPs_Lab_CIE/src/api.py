import os
import json
import joblib
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

MODEL_PATH = "models/best_model.pkl"
LOGS_PATH = "logs/predictions.jsonl"

model = None


# ✅ Correct input schema with STRICT validation
class InputFeatures(BaseModel):
    qubit_count: int = Field(..., ge=5, le=100)
    gate_depth: int = Field(..., ge=10, le=500)
    error_rate_pct: float = Field(..., ge=0.1, le=5)
    is_error_corrected: int = Field(..., ge=0, le=1)


# ✅ Load model correctly (joblib)
@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found at models/best_model.pkl")

    model = joblib.load(MODEL_PATH)


# ✅ Exact required response
@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "QuantumBench circuit_exec_ms API"}


# ✅ Inference endpoint
@app.post("/infer")
def infer(data: InputFeatures):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_dict = data.dict()

    df = pd.DataFrame([input_dict])
    prediction = float(model.predict(df)[0])

    # ✅ Logging (Task 4 requirement)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_dict,
        "prediction": prediction,
    }

    os.makedirs("logs", exist_ok=True)
    with open(LOGS_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"prediction": prediction}
