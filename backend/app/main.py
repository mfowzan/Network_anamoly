# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import joblib
import pandas as pd
import tempfile
from typing import Dict, Any

from .schemas import SingleFeatures
from .model import AnomalyModel, EnsembleModel  # EnsembleModel exists in model.py
from .pcap_parser import extract_flows_from_pcap


from threading import Thread, Event
import time
from scapy.all import sniff

realtime_running = False
realtime_thread = None
realtime_stop_event = Event()
realtime_results = []   # store most recent predictions

last_single_input = None



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

# storage for loaded models
loaded: Dict[str, Any] = {}

# First load all base models
base_models: Dict[str, AnomalyModel] = {}

for name, cfg in MODELS.items():
    if name == "ensemble":
        continue  # skip ensemble for now

    try:
        model = AnomalyModel(
            cfg["model"],
            cfg.get("scaler"),
            cfg.get("features"),
            model_type=cfg["type"],
            metadata_path=cfg.get("metadata"),
        )
        base_models[name] = model
        loaded[name] = model
        print("✅ Loaded", name)
    except Exception as e:
        print("⚠️ Could not load", name, e)

# Now load ensemble (if weights exist)
try:
    ensemble_cfg = MODELS.get("ensemble")
    ensemble_path = ensemble_cfg["model"]
    if os.path.exists(ensemble_path):
        loaded["ensemble"] = EnsembleModel(base_models, ensemble_path)
        print("✅ Loaded ensemble model")
    else:
        print("⚠️ Ensemble weights file not found:", ensemble_path)
except Exception as e:
    print("⚠️ Could not load ensemble:", e)


@app.get("/health")
async def health():
    return {"status": "ok", "available_models": list(loaded.keys())}


def process_live_packet(packet):
    global realtime_results
    
    try:
        # Convert single packet → 1-flow-like features
        # SIMPLE VERSION: treat every packet as its own flow
        if not packet.haslayer("IP"):
            return
        
        size = len(packet)
        
        row = {
            "duration": 0,
            "src_bytes": size,
            "dst_bytes": 0,
            "count": 1,
            "srv_count": 0,
            "wrong_fragment": 0,
            "serror_rate": 0,
            "srv_serror_rate": 0,
            "rerror_rate": 0,
            "srv_rerror_rate": 0,
            "same_srv_rate": 0.5,
            "diff_srv_rate": 0.1,
            "dst_host_count": 1,
            "dst_host_srv_count": 1,
            "dst_host_same_srv_rate": 0.5,
            "dst_host_diff_srv_rate": 0.1,
        }

        df = pd.DataFrame([row])
        result = {}

        # Run predictions
        for name, model in loaded.items():
            if hasattr(model, "predict_on_df"):
                out = model.predict_on_df(df)
                pred = out["prediction"].iloc[0]
                score = float(out["score"].iloc[0])
            else:
                out = model.predict(df)
                pred = out["prediction"]
                score = float(out["score"])

            result[name] = {"prediction": pred, "score": score}

        # Save last 20 results
        realtime_results.append(result)
        if len(realtime_results) > 20:
            realtime_results.pop(0)

    except Exception as e:
        print("Realtime error:", e)

def realtime_capture():
    global realtime_running

    try:
        sniff(prn=process_live_packet, stop_filter=lambda x: realtime_stop_event.is_set())
    except Exception as e:
        print("Realtime sniff error:", e)

    realtime_running = False



def _predict_with_model(model_obj: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Unified wrapper: call base models (predict_on_df) or ensemble (.predict)
    and return a normalized dict: {"prediction": str, "score": float, "details": optional}
    """
    try:
        if hasattr(model_obj, "predict_on_df"):
            out = model_obj.predict_on_df(df)
            pred = out["prediction"].iloc[0]
            score = float(out["score"].iloc[0])
            return {"prediction": pred, "score": score}
        elif hasattr(model_obj, "predict"):
            out = model_obj.predict(df)
            # expected out: {"prediction": ..., "score": ..., "details": ...}
            # but be defensive
            pred = out.get("prediction", "unknown")
            score = float(out.get("score", 0.0))
            details = out.get("details", None)
            result = {"prediction": pred, "score": score}
            if details is not None:
                result["details"] = details
            return result
        else:
            return {"prediction": "unknown", "score": 0.0}
    except Exception as e:
        return {"error": str(e)}



@app.post("/realtime/start")
async def start_realtime():
    global realtime_running, realtime_thread, realtime_stop_event

    if realtime_running:
        return {"status": "already_running"}

    realtime_stop_event.clear()
    realtime_running = True
    realtime_thread = Thread(target=realtime_capture, daemon=True)
    realtime_thread.start()

    return {"status": "started"}


@app.post("/realtime/stop")
async def stop_realtime():
    global realtime_running, realtime_stop_event

    if not realtime_running:
        return {"status": "not_running"}

    realtime_stop_event.set()
    realtime_running = False

    return {"status": "stopped"}


@app.get("/realtime/latest")
async def get_latest():
    return {
        "running": realtime_running,
        "results": realtime_results[-10:],   # last 10 predictions
    }


@app.post("/predict_single")
async def predict_single(payload: SingleFeatures):
    global last_single_input
    last_single_input = payload.dict()
    df = pd.DataFrame([payload.dict()])
    results: Dict[str, Any] = {}

    for name, model in loaded.items():
        res = _predict_with_model(model, df)
        results[name] = res

    # Build summary (count models that explicitly flagged anomaly)
    anomaly_count = 0
    for r in results.values():
        if isinstance(r, dict) and r.get("prediction") == "anomaly":
            anomaly_count += 1

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

        if df is None or df.empty:
            return {"error": "No valid traffic flows were found in the PCAP file."}

        results_list = []

        for idx, row in df.iterrows():
            row_df = pd.DataFrame([row])
            per_model_result = {}

            for name, model in loaded.items():
                res = _predict_with_model(model, row_df)
                per_model_result[name] = res

            results_list.append({
                "flow_id": int(idx),
                "features": row.to_dict(),
                "predictions": per_model_result
            })

        return {
            "flow_count": len(df),
            "analysis": results_list
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/explain_single")
async def explain_single(payload: SingleFeatures):
    """
    Returns XAI explanations for a single flow using SHAP + deviation scores.
    Safe against all SHAP array formats.
    """

    import numpy as np
    import shap

    try:
        df = pd.DataFrame([payload.dict()])
        explanation = {}

        shap_features = df.values

        # ---------------------------
        # 1) RANDOM FOREST SHAP
        # ---------------------------
        if "rf" in loaded:
            rf_model = loaded["rf"].model
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(shap_features)

            # SAFE EXTRACTION
            if isinstance(shap_values, list):
                sv = np.array(shap_values[-1][0])   # last class
            else:
                sv = np.array(shap_values[0])

            sv = sv.flatten().tolist()
            feature_importance = dict(zip(df.columns, sv))

            explanation["rf"] = {
                "feature_importance": feature_importance,
                "top_features": sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            }

        # ---------------------------
        # 2) XGBOOST SHAP
        # ---------------------------
        if "xgboost" in loaded:
            xgb_model = loaded["xgboost"].model
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(shap_features)

            if isinstance(shap_values, list):
                sv = np.array(shap_values[0][0])
            else:
                sv = np.array(shap_values[0])

            sv = sv.flatten().tolist()
            feature_importance = dict(zip(df.columns, sv))

            explanation["xgboost"] = {
                "feature_importance": feature_importance,
                "top_features": sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            }

        # ---------------------------
        # 3) AUTOENCODER DEVIATION
        # ---------------------------
        if "autoencoder" in loaded:
            model = loaded["autoencoder"]

            scaled = model.scaler.transform(df[model.features])
            reconstructed = model.model.predict(scaled)

            # convert keras outputs safely
            reconstructed = np.array(reconstructed).reshape(scaled.shape)

            diff = np.abs(scaled - reconstructed)[0]
            diff = diff.tolist()

            feature_diffs = dict(zip(model.features, diff))

            explanation["autoencoder"] = {
                "deviation": feature_diffs,
                "top_features": sorted(feature_diffs.items(), key=lambda x: x[1], reverse=True)[:5]
            }

        # ---------------------------
        # 4) ISOLATION FOREST
        # ---------------------------
        if "isolation" in loaded:
            model = loaded["isolation"]
            score = float(model.model.decision_function(df[model.features])[0])

            explanation["isolation"] = {
                "score": score,
                "meaning": "Lower score = more anomalous"
            }

        # ---------------------------
        # 5) OCSVM
        # ---------------------------
        if "ocsvm" in loaded:
            model = loaded["ocsvm"]
            score = float(model.model.decision_function(df[model.features])[0])

            explanation["ocsvm"] = {
                "score": score,
                "meaning": "More negative = anomaly"
            }

        # ---------------------------
        # 6) ENSEMBLE
        # ---------------------------
        if "ensemble" in loaded:
            explanation["ensemble"] = loaded["ensemble"].predict(df)

        return {"explanation": explanation}

    except Exception as e:
        return {"error": str(e)}
    

@app.get("/explain_last")
async def explain_last():
    from fastapi import HTTPException
    if last_single_input is None:
        raise HTTPException(status_code=400, detail="No previous prediction available")

    return await explain_single(SingleFeatures(**last_single_input))

