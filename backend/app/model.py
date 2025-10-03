# model.py
print("🚀 model.py LOADED!", flush=True)

import os
import joblib
import numpy as np
import pandas as pd
import logging

print("🚀 USING UPDATED model.py")

# optional: try shap
try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    _SHAP_AVAILABLE = False

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")

logger = logging.getLogger(__name__)


class AnomalyModel:
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)

        # lazy SHAP explainer
        self._shap_explainer = None
        if _SHAP_AVAILABLE:
            logger.info("✅ SHAP available — will use SHAP explanations when possible.")

    def _init_shap_explainer(self, X):
        """Create SHAP explainer lazily (tree or generic)."""
        if not _SHAP_AVAILABLE:
            return None
        if self._shap_explainer is not None:
            return self._shap_explainer
        try:
            # For tree-based models like IsolationForest
            self._shap_explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                self._shap_explainer = shap.Explainer(self.model, X)
            except Exception as e:
                logger.warning(f"⚠️ Failed to init SHAP explainer: {e}")
                self._shap_explainer = None
        return self._shap_explainer

    def predict_on_df(self, df: pd.DataFrame):
        print("DEBUG >>> predict_on_df CALLED", flush=True)

        # Ensure all required features
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.features].copy()

        # Scale features
        Xs = self.scaler.transform(df)

        # Model outputs
        preds = self.model.predict(Xs)          # 1 = normal, -1 = anomaly
        scores = self.model.decision_function(Xs)

        # --- Explanations ---
        shap_used = False
        explanations = None
        if _SHAP_AVAILABLE:
            try:
                explainer = self._init_shap_explainer(Xs)
                if explainer is not None:
                    try:
                        exp = explainer(Xs)
                        shap_vals = exp.values
                    except Exception:
                        shap_vals = explainer.shap_values(Xs)

                    shap_arr = np.array(shap_vals)
                    if shap_arr.ndim == 3:   # (n_classes, n_samples, n_features)
                        shap_arr = np.abs(shap_arr).sum(axis=0)
                    elif shap_arr.ndim == 2:
                        shap_arr = np.abs(shap_arr)
                    else:
                        shap_arr = np.abs(shap_arr).reshape(len(Xs), -1)

                    explanations = shap_arr
                    shap_used = True
            except Exception as e:
                logger.warning(f"⚠️ SHAP explanation failed: {e}")

        # fallback: z-score heuristic
        if explanations is None:
            mu = Xs.mean(axis=0)
            sigma = Xs.std(axis=0) + 1e-6
            explanations = np.abs((Xs - mu) / sigma)

        # --- Build per-sample results ---
        results = []
        for i in range(len(df)):
            impact_arr = np.asarray(explanations[i])
            if impact_arr.shape[0] != len(self.features):
                impact_arr = impact_arr.flatten()
                if impact_arr.shape[0] < len(self.features):
                    impact_arr = np.pad(impact_arr, (0, len(self.features) - impact_arr.shape[0]))
                else:
                    impact_arr = impact_arr[:len(self.features)]

            feat_imp_pairs = [
                [self.features[j], float(impact_arr[j])]
                for j in range(len(self.features))
            ]
            top_features = sorted(feat_imp_pairs, key=lambda x: x[1], reverse=True)[:3]

            results.append({
                "prediction": int(1 if preds[i] == 1 else 0),
                "score": float(scores[i]),
                "top_features": top_features,
                "explanation_method": "shap" if shap_used else "zscore"
            })

        print("Returning keys:", results[0].keys(), flush=True)
        return pd.DataFrame(results)
