import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")

class AnomalyModel:
    def __init__(self, model_path: str, scaler_path: str, features_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)

    def predict_on_df(self, df: pd.DataFrame):
        # Ensure all required features exist
        for col in self.features:
            if col not in df.columns:
                df[col] = 0  # fill missing with 0
        df = df[self.features]  # reorder

        # Scale + predict
        Xs = self.scaler.transform(df)
        preds = self.model.predict(Xs)
        scores = self.model.decision_function(Xs)

        # Feature importance via simple z-scores
        z_scores = np.abs((Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-6))

        results = []
        for i in range(len(df)):
            feature_importances = {
                self.features[j]: float(z_scores[i][j])
                for j in range(len(self.features))
            }
            top_features = sorted(
                feature_importances.items(), key=lambda x: x[1], reverse=True
            )[:3]

            results.append({
                "prediction": int(1 if preds[i] == 1 else 0),
                "score": float(scores[i]),
                "top_features": top_features
            })

        return pd.DataFrame(results)
