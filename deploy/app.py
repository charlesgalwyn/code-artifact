import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

from deploy.preprocess import preprocess_input

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "saved_models/gru/12122025-190051-gru-e8.h5"
SEQ_LEN = 49
N_NUMERIC = 4  # must match training

with open("deploy/labels.json", "r") as f:
    LABELS = json.load(f)

# ------------------------
# LOAD MODEL
# ------------------------
try:
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ------------------------
# FASTAPI APP
# ------------------------
app = FastAPI(
    title="Learning State Prediction API",
    description="Predicts student learning phase from interaction sequences",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for academic demo
    allow_credentials=True,
    allow_methods=["*"],   # POST, OPTIONS
    allow_headers=["*"],
)

# ------------------------
# REQUEST SCHEMAS
# ------------------------
class Timestep(BaseModel):
    action: int
    reg: int
    self: int
    numeric: list[float]

class SequenceInput(BaseModel):
    sequence: list[Timestep]

# ------------------------
# HEALTH CHECK
# ------------------------
@app.get("/")
def root():
    return {"status": "API is running"}

# ------------------------
# PREDICTION ENDPOINT
# ------------------------
@app.post("/predict")
def predict_learning_state(data: SequenceInput):
    try:
        if len(data.sequence) == 0:
            raise HTTPException(status_code=400, detail="Sequence cannot be empty")

        # Validate numeric length early
        for step in data.sequence:
            if len(step.numeric) != N_NUMERIC:
                raise HTTPException(
                    status_code=400,
                    detail=f"Each numeric vector must have length {N_NUMERIC}"
                )

        # Preprocess (padding + truncation handled here)
        X = preprocess_input(
            sequence=[step.dict() for step in data.sequence],
            seq_len=SEQ_LEN,
            n_numeric=N_NUMERIC
        )

        # Predict
        probs = model.predict(X, verbose=0)
        pred_class = int(np.argmax(probs, axis=1)[0])
        confidence = float(np.max(probs))

        return {
            "predicted_class": pred_class,
            "predicted_phase": LABELS[str(pred_class)],
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise

    except Exception as e:
        # Any unexpected error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
