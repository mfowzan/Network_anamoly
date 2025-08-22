import os
import joblib
import numpy as np
from typing import List, Dict
from .utils import df_to_features




import joblib
import pandas as pd
import numpy as np
import os

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
        return pd.DataFrame({"prediction": preds, "score": scores})

