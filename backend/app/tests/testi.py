import pandas as pd
import joblib
import argparse

def test_isolation(data_path, model_dir):
    # Load data
    df = pd.read_csv(data_path)

    # Load artifacts
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    features = joblib.load(f"{model_dir}/features.joblib")
    model = joblib.load(f"{model_dir}/isolation_forest.joblib")

    # Select features & scale
    X = df[features]
    X_scaled = scaler.transform(X)

    # Predictions (-1 = anomaly, 1 = normal)
    preds = model.predict(X_scaled)

    # Print summary
    anomaly_count = (preds == -1).sum()
    normal_count = (preds == 1).sum()
    print(f"✅ Isolation Forest Results")
    print(f"Normal: {normal_count}, Anomalies: {anomaly_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    args = parser.parse_args()

    test_isolation(args.data, args.model_dir)
