import pandas as pd
import joblib
import numpy as np
import argparse
from tensorflow.keras.models import load_model

def test_autoencoder(data_path, model_dir, threshold=0.01):
    # Load data
    df = pd.read_csv(data_path)

    # Load artifacts
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    features = joblib.load(f"{model_dir}/features.joblib")
    model = load_model(f"{model_dir}/autoencoder.h5")

    # Select features & scale
    X = df[features]
    X_scaled = scaler.transform(X)

    # Reconstruct
    X_pred = model.predict(X_scaled, verbose=0)

    # Reconstruction error
    errors = np.mean(np.square(X_scaled - X_pred), axis=1)

    # Predict anomaly if error > threshold
    preds = (errors > threshold).astype(int)

    anomaly_count = preds.sum()
    normal_count = len(preds) - anomaly_count

    print(f"✅ Autoencoder Results")
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="Reconstruction error threshold")
    args = parser.parse_args()

    test_autoencoder(args.data, args.model_dir, args.threshold)
