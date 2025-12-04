# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import os, joblib, pandas as pd
from typing import List
from .schemas import SingleFeatures
from .model import AnomalyModel

app = FastAPI(title="Network Anomaly Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODELS = {
    "autoencoder": {
        "model": os.path.join(MODEL_DIR, "autoencoder", "autoencoder.h5"),
        "scaler": os.path.join(MODEL_DIR, "autoencoder", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "autoencoder", "features.joblib"),
        "metadata": os.path.join(MODEL_DIR, "autoencoder", "metadata.joblib"),
        "type": "autoencoder"
    },
    "isolation": {
        "model": os.path.join(MODEL_DIR, "isolation", "isolation_forest.joblib"),
        "scaler": os.path.join(MODEL_DIR, "isolation", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "isolation", "features.joblib"),
        "metadata": os.path.join(MODEL_DIR, "isolation", "metadata.joblib"),
        "type": "isolation_forest"
    },
    "ocsvm": {
        "model": os.path.join(MODEL_DIR, "ocsvm", "ocsvm.joblib"),
        "scaler": os.path.join(MODEL_DIR, "ocsvm", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "ocsvm", "features.joblib"),
        "metadata": os.path.join(MODEL_DIR, "ocsvm", "metadata.joblib"),
        "type": "ocsvm"
    },
    "rf": {
        "model": os.path.join(MODEL_DIR, "rf", "random_forest.joblib"),
        "scaler": os.path.join(MODEL_DIR, "rf", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "rf", "features.joblib"),
        "metadata": os.path.join(MODEL_DIR, "rf", "metadata.joblib"),
        "type": "random_forest"
    },
    "xgboost": {
        "model": os.path.join(MODEL_DIR, "xgboost", "xgboost.joblib"),
        "scaler": os.path.join(MODEL_DIR, "xgboost", "scaler.joblib"),
        "features": os.path.join(MODEL_DIR, "xgboost", "features.joblib"),
        "metadata": os.path.join(MODEL_DIR, "xgboost", "metadata.joblib"),
        "type": "xgboost"
    },
}

loaded = {}
for name, cfg in MODELS.items():
    try:
        loaded[name] = AnomalyModel(cfg["model"], cfg["scaler"], cfg["features"], model_type=cfg["type"], metadata_path=cfg["metadata"])
        print("✅ Loaded", name)
    except Exception as e:
        print("⚠️ Could not load", name, e)

@app.get("/health")
async def health():
    return {"status": "ok", "available_models": list(loaded.keys())}

@app.post("/predict_single")
async def predict_single(payload: SingleFeatures):
    df = pd.DataFrame([payload.dict()])
    results = {}
    for name, model in loaded.items():
        try:
            out = model.predict_on_df(df)
            pred = out["prediction"].iloc[0]
            score = float(out["score"].iloc[0])
            results[name] = {"prediction": pred, "score": score}
        except Exception as e:
            results[name] = {"error": str(e)}
    # summary
    anomaly_count = sum(1 for r in results.values() if r.get("prediction") == "anomaly")
    insight = "Normal behavior" if anomaly_count == 0 else f"{anomaly_count} models flagged anomaly"
    return {"results": results, "summary": {"anomaly_models": int(anomaly_count), "insight": insight}}
