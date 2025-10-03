from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from typing import List
from pydantic import BaseModel

# relative imports inside package
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

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.joblib")

# Load model once at startup
anomaly_model = AnomalyModel(MODEL_PATH, SCALER_PATH, FEATURES_PATH)


# Schema for manual JSON input
class InputRecord(BaseModel):
    duration: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    count: float = 0
    srv_count: float = 0
    wrong_fragment: float = 0




@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_json(items: List[InputRecord]):
    print("🔍 predict_json called with anomaly_model:", anomaly_model)
    print("🔍 anomaly_model class:", anomaly_model.__class__)
    print("🔍 anomaly_model module:", anomaly_model.__module__)

    """Predict on JSON array of multiple records"""
    df = pd.DataFrame([item.dict() for item in items])
    out = anomaly_model.predict_on_df(df)
    return {"results": out.to_dict(orient="records")}


@app.post("/predict_single")
async def predict_single(payload: SingleFeatures):
    """Predict on a single JSON record"""
    df = pd.DataFrame([payload.dict()])
    out = anomaly_model.predict_on_df(df)
    return {"results": out.to_dict(orient="records")[0]}


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """Predict on uploaded CSV file"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported in this demo")
    
    df = pd.read_csv(file.file)

    # Ensure all required features exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    
    out = anomaly_model.predict_on_df(df)
    return {"results": out.to_dict(orient="records")}
