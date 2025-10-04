from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import pandas as pd
import os
from typing import List
from pydantic import BaseModel

# relative imports
from .utils import FEATURE_COLS
from .schemas import SingleFeatures, BulkFeatures
from .model import AnomalyModel

app = FastAPI(title="Network Anomaly Detection API")

# ✅ Add CORS (development: allow everything; restrict for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # during dev allow all (replace with specific origins for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Define model paths
MODELS = {
    "autoencoder": {
        "model": os.path.join(MODEL_DIR, "autoencoder", "autoencoder.h5"),
        "scaler": os.path.join(MODEL_DIR, "autoencoder", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "autoencoder", "features.joblib"),
    },
    "isolation_forest": {
        "model": os.path.join(MODEL_DIR, "isolation", "isolation_forest.joblib"),
        "scaler": os.path.join(MODEL_DIR, "isolation", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "isolation", "features.joblib"),
    }
}


# Cache loaded models
loaded_models = {}
for model_type, paths in MODELS.items():
    try:
        loaded_models[model_type] = AnomalyModel(
            paths["model"],
            paths["scaler"],
            paths["features"],
            model_type=model_type
        )
        print(f"✅ Loaded {model_type} model")
    except Exception as e:
        print(f"⚠️ Could not load {model_type}: {e}")


class InputRecord(BaseModel):
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    count: float = 0
    srv_count: float = 0
    wrong_fragment: float = 0


@app.get("/health")
async def health():
    return {"status": "ok", "available_models": list(loaded_models.keys())}


@app.post("/predict")
def predict_json(
    items: List[InputRecord],
    model: str = Query("autoencoder", description="Choose model: autoencoder or isolation_forest")
):
    """Predict on JSON array of multiple records using selected model"""
    if model not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")

    df = pd.DataFrame([item.dict() for item in items])
    out = loaded_models[model].predict_on_df(df)
    return {"model": model, "results": out.to_dict(orient="records")}


@app.post("/predict_single")
async def predict_single(
    payload: SingleFeatures,
    model: str = Query("autoencoder", description="Choose model: autoencoder or isolation_forest")
):
    """Predict on a single JSON record"""
    if model not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")

    df = pd.DataFrame([payload.dict()])
    out = loaded_models[model].predict_on_df(df)
    return {"model": model, "results": out.to_dict(orient="records")[0]}


@app.post("/predict_file")
async def predict_file(
    file: UploadFile = File(...),
    model: str = Query("autoencoder", description="Choose model: autoencoder or isolation_forest")
):
    """Predict on uploaded CSV file"""
    if model not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    df = pd.read_csv(file.file)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    out = loaded_models[model].predict_on_df(df)
    return {"model": model, "results": out.to_dict(orient="records")}
