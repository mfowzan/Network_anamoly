import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

class AnomalyModel:
    def __init__(self, model_path, scaler_path, features_path, model_type=None, metadata_path=None):
        """
        Load a trained anomaly detection model
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            features_path: Path to features file
            model_type: One of: 'autoencoder', 'isolation_forest', 'ocsvm', 'rf', 'xgboost'
            metadata_path: Optional path to metadata
        """
        self.model_type = model_type
        self.model_path = model_path
        
        # Load model based on type
        if model_type == "autoencoder":
            self.model = load_model(model_path)
            print(f"✅ Loaded Keras model: {model_path}")
        else:
            self.model = joblib.load(model_path)
            print(f"✅ Loaded sklearn/xgboost model: {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler: {scaler_path}")
        
        # Load features
        self.features = joblib.load(features_path)
        print(f"✅ Loaded features: {len(self.features)} features")
        print(f"   Features: {self.features}")
        
        # Load metadata
        self.metadata = None
        if metadata_path and os.path.exists(metadata_path):
            try:
                self.metadata = joblib.load(metadata_path)
                print(f"✅ Loaded metadata")
            except Exception as e:
                print(f"⚠️  Could not load metadata: {e}")
                self.metadata = None

    def _ensure_df(self, df):
        """
        Ensure DataFrame has all required features in correct order
        """
        df_out = pd.DataFrame()
        
        for feat in self.features:
            if feat in df.columns:
                df_out[feat] = df[feat]
            else:
                # Feature missing - use default value
                if self.metadata and "expected_ranges" in self.metadata and feat in self.metadata["expected_ranges"]:
                    default_val = self.metadata["expected_ranges"][feat]["mean"]
                    print(f"⚠️  Feature '{feat}' missing, using mean={default_val}")
                else:
                    default_val = 0
                    print(f"⚠️  Feature '{feat}' missing, using 0")
                df_out[feat] = default_val
        
        # Convert to float and handle inf/nan
        df_out = df_out.astype(float)
        df_out = df_out.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df_out

    def _scale(self, df):
        """
        Scale features using loaded scaler
        """
        try:
            arr = self.scaler.transform(df)
            return arr
        except Exception as e:
            print(f"❌ Scaling error: {e}")
            # Fallback: return raw array
            return df.values

    def predict_on_df(self, df: pd.DataFrame):
        """
        Make predictions on a DataFrame
        
        Args:
            df: DataFrame with network traffic features
            
        Returns:
            DataFrame with columns: 'prediction' and 'score'
            - prediction: 'normal' or 'anomaly'
            - score: confidence score (0-1, higher = more normal)
        """
        print(f"\n🔍 Making predictions for {len(df)} samples...")
        
        # Ensure correct features
        df_in = self._ensure_df(df.copy())
        print(f"   Features prepared: {df_in.shape}")
        
        # Scale features
        X_scaled = self._scale(df_in)
        print(f"   Features scaled: {X_scaled.shape}")
        
        # Make predictions based on model type
        if self.model_type == "autoencoder":
            # Autoencoder: high reconstruction error = anomaly
            recon = self.model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
            
            # Use 95th percentile as threshold
            threshold = np.percentile(mse, 95)
            preds = np.where(mse > threshold, "anomaly", "normal")
            
            # Score: lower MSE = more normal (higher score)
            # Normalize using sigmoid-like function
            score = 1.0 / (1.0 + mse)
            score = np.clip(score, 0.0, 1.0)
            
        elif self.model_type in ("isolation", "isolation_forest"):
            # IsolationForest: -1 = anomaly, 1 = normal
            raw_pred = self.model.predict(X_scaled)
            preds = np.where(raw_pred == -1, "anomaly", "normal")
            
            # Decision function: higher = more normal
            decision = self.model.decision_function(X_scaled)
            score = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid normalization
            
        elif self.model_type == "ocsvm":
            # OneClassSVM: -1 = anomaly, 1 = normal
            raw_pred = self.model.predict(X_scaled)
            preds = np.where(raw_pred == -1, "anomaly", "normal")
            
            # Decision function: higher = more normal
            decision = self.model.decision_function(X_scaled)
            score = 1.0 / (1.0 + np.exp(-decision))
            
        elif self.model_type in ("rf", "random_forest"):
            # RandomForest: 0 = normal, 1 = attack/anomaly
            raw_pred = self.model.predict(X_scaled)
            preds = np.where(raw_pred == 0, "normal", "anomaly")
            
            # Use probability if available
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_scaled)
                # Score = probability of being normal (class 0)
                score = proba[:, 0] if proba.shape[1] >= 1 else np.ones(len(X_scaled))
            else:
                score = np.where(raw_pred == 0, 1.0, 0.0)
                
        elif self.model_type == "xgboost":
            # XGBoost: 0 = normal, 1 = attack/anomaly
            raw_pred = self.model.predict(X_scaled)
            preds = np.where(raw_pred == 0, "normal", "anomaly")
            
            # Use probability
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_scaled)
                score = proba[:, 0] if proba.shape[1] >= 1 else np.ones(len(X_scaled))
            else:
                score = np.where(raw_pred == 0, 1.0, 0.0)
                
        else:
            print(f"⚠️  Unknown model type: {self.model_type}, defaulting to 'normal'")
            preds = np.array(["normal"] * len(X_scaled))
            score = np.array([1.0] * len(X_scaled))

        # Convert to native Python types (avoid numpy types in JSON)
        preds_list = [str(p) for p in preds]
        scores_list = [float(s) for s in score]
        
        print(f"✅ Predictions complete:")
        print(f"   Normal: {preds_list.count('normal')}")
        print(f"   Anomaly: {preds_list.count('anomaly')}")
        print(f"   Avg score: {np.mean(scores_list):.4f}")

        # Return DataFrame
        result = pd.DataFrame({
            "prediction": preds_list,
            "score": scores_list
        })
        
        return result


def load_model_from_dir(model_dir: str, model_type: str):
    """
    Helper function to load model from directory
    
    Args:
        model_dir: Path to model directory
        model_type: One of 'autoencoder', 'isolation_forest', 'ocsvm', 'rf', 'xgboost'
    
    Returns:
        AnomalyModel instance
    """
    # Map model types to filenames
    model_files = {
        "autoencoder": "autoencoder.h5",
        "isolation_forest": "isolation_forest.joblib",
        "ocsvm": "ocsvm.joblib",
        "rf": "random_forest.joblib",
        "xgboost": "xgboost.joblib"
    }
    
    if model_type not in model_files:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model_path = os.path.join(model_dir, model_files[model_type])
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    features_path = os.path.join(model_dir, "features.joblib")
    metadata_path = os.path.join(model_dir, "metadata.joblib")
    
    # Validate files exist
    for path, name in [(model_path, "model"), (scaler_path, "scaler"), (features_path, "features")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Normalize model_type
    type_mapping = {
        "isolation_forest": "isolation",
        "random_forest": "rf"
    }
    normalized_type = type_mapping.get(model_type, model_type)
    
    return AnomalyModel(
        model_path=model_path,
        scaler_path=scaler_path,
        features_path=features_path,
        model_type=normalized_type,
        metadata_path=metadata_path if os.path.exists(metadata_path) else None
    )


class EnsembleModel:
    """Hybrid Ensemble with (50% static accuracy weight + 50% dynamic confidence score)"""

    def __init__(self, model_dict, weight_path):
        """
        Args:
            model_dict: dict of {name: AnomalyModel instance}
            weight_path: joblib file containing accuracies + weights
        """
        self.models = model_dict
        self.weights = joblib.load(weight_path)
        self.model_weights = self.weights["weights"]
        print(f"✅ Loaded EnsembleModel with weights: {self.model_weights}")

    def predict(self, df):
        """
        Returns weighted anomaly score + label
        """
        total_weight = 0
        total_score = 0

        for name, model in self.models.items():

            out = model.predict_on_df(df)
            pred = out["prediction"].iloc[0]      # "normal" or "anomaly"
            score = out["score"].iloc[0]          # 0–1 (normality score)

            static_weight = self.model_weights.get(name, 0)
            dynamic_weight = score                 # higher score → more normal

            hybrid_weight = (static_weight + dynamic_weight) / 2

            # Convert pred to anomaly probability (1 = anomaly)
            anomaly_prob = 1 - score

            total_weight += hybrid_weight
            total_score += hybrid_weight * anomaly_prob

        final_score = total_score / total_weight
        final_pred = "anomaly" if final_score > 0.5 else "normal"

        return {
            "prediction": final_pred,
            "score": float(1 - final_score),  # return normality score
            "details": {
                "hybrid_scores": self.weights
            }
        }
