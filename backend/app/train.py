"""
Train and save an IsolationForest model and scaler.

Run as module from project root:
  python -m backend.app.train --data data/sample_flows.csv --out-dir models --contamination 0.05
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import os

def load_and_preprocess(data_path: str):
    # Load raw dataset (no headers in KDD dataset)
    col_names = [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes",
        "land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count",
        "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
        "dst_host_srv_rerror_rate","label"
    ]

    df = pd.read_csv(data_path, names=col_names)

    # Drop categorical cols for now (simple version)
    df = df.drop(columns=["protocol_type","service","flag","label"])

    # Fill missing with 0
    df = df.fillna(0)

    return df

def train_model(data_path: str, out_dir: str, contamination: float = 0.05):
    # Load dataset
    df = load_and_preprocess(data_path)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)

    # Save model + scaler
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    feature_names = df.columns.tolist()
    joblib.dump(feature_names, os.path.join(out_dir, "features.joblib"))
    print(f"✅ Features saved to: {os.path.join(out_dir, 'features.joblib')}")

    print(f"✅ Trained model saved to: {os.path.join(out_dir, 'isolation_forest.joblib')}")
    print(f"✅ Scaler saved to: {os.path.join(out_dir, 'scaler.joblib')}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    train_model(args.data, args.out_dir, args.contamination)
