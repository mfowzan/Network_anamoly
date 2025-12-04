# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import os, joblib, pandas as pd
from typing import List
from .schemas import SingleFeatures
from .model import AnomalyModel

from fastapi import UploadFile, File
from .pcap_parser import extract_flows_from_pcap

import tempfile

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
    "ensemble": {
    "model": os.path.join(MODEL_DIR, "ensemble", "ensemble_weights.joblib"),
    "type": "ensemble"
},
}

loaded = {}

# First load all base models
base_models = {}

for name, cfg in MODELS.items():
    if name == "ensemble":
        continue  # skip ensemble for now

    try:
        model = AnomalyModel(
            cfg["model"], 
            cfg.get("scaler"), 
            cfg.get("features"), 
            model_type=cfg["type"], 
            metadata_path=cfg.get("metadata")
        )
        base_models[name] = model
        loaded[name] = model
        print("✅ Loaded", name)
    except Exception as e:
        print("⚠️ Could not load", name, e)

# Now load ensemble
from .model import EnsembleModel
try:
    ensemble_path = MODELS["ensemble"]["model"]
    loaded["ensemble"] = EnsembleModel(base_models, ensemble_path)
    print("✅ Loaded ensemble model")
except Exception as e:
    print("⚠️ Could not load ensemble:", e)


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
    if "ensemble" in loaded:
        ens = loaded["ensemble"].predict(df)
        results["ensemble"] = ens
    # summary
    anomaly_count = sum(1 for r in results.values() if r.get("prediction") == "anomaly")
    insight = "Normal behavior" if anomaly_count == 0 else f"{anomaly_count} models flagged anomaly"
    return {"results": results, "summary": {"anomaly_models": int(anomaly_count), "insight": insight}}




@app.post("/upload_pcap")
async def upload_pcap(file: UploadFile = File(...)):
    try:
        # Use OS-safe temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)

        # Save file
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        print("Saved PCAP to:", temp_path)

        # Extract flows from pcap
        df = extract_flows_from_pcap(temp_path)

        if df.empty:
            return {"error": "No valid traffic flows were found in the PCAP file."}

        results_list = []

        for idx, row in df.iterrows():
            row_df = pd.DataFrame([row])

            result = {}
            for name, model in loaded.items():

            # Ensemble model has a different method
                if hasattr(model, "predict_on_df"):
                    out = model.predict_on_df(row_df)       # base models
                    pred = out["prediction"].iloc[0]
                    score = float(out["score"].iloc[0])
                else:
                    out = model.predict(row_df)             # ensemble model
                    pred = out["prediction"]
                    score = float(out["score"])

                result[name] = {
                    "prediction": pred,
                    "score": score
                }

            results_list.append({
                "flow_id": idx,
                "features": row.to_dict(),
                "predictions": result
            })


        return {
            "flow_count": len(df),
            "analysis": results_list
        }

    except Exception as e:
        return {"error": str(e)}

