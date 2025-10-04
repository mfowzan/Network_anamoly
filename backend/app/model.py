# backend/app/model.py
print("🚀 model.py LOADED!", flush=True)

import os
import joblib
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model

print("🚀 USING UPDATED model.py")

try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class AnomalyModel:
    def __init__(self, model_path: str, scaler_path: str, features_path: str, model_type="isolation_forest"):
        self.model_type = model_type
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)

        if model_type == "isolation_forest":
            self.model = joblib.load(model_path)
        elif model_type == "autoencoder":
            self.model = load_model(model_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._shap_explainer = None

    def predict_on_df(self, df: pd.DataFrame):
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.features].copy()

        Xs = self.scaler.transform(df)

        if self.model_type == "isolation_forest":
            preds = self.model.predict(Xs)   # 1 = normal, -1 = anomaly
            scores = self.model.decision_function(Xs)

        elif self.model_type == "autoencoder":
            reconstructions = self.model.predict(Xs)
            mse = np.mean(np.power(Xs - reconstructions, 2), axis=1)
            threshold = np.percentile(mse, 95)   # 🔹 dynamic threshold
            preds = (mse < threshold).astype(int)  # 1 = normal, 0 = anomaly
            scores = -mse   # lower mse = more normal

        results = []
        for i in range(len(df)):
            top_features = [[self.features[j], float(abs(Xs[i][j]))] for j in range(len(self.features))]
            top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:3]

            results.append({
                "prediction": int(preds[i]),
                "score": float(scores[i]),
                "top_features": top_features,
                "explanation_method": "autoencoder" if self.model_type == "autoencoder" else "isolation_forest"
            })

        return pd.DataFrame(results)
