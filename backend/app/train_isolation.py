"""
Train and save an IsolationForest model and scaler using KDD dataset.

Usage:
  python train_isolation.py --data data/kddcup.csv --out-dir backend/app/models/isolation --contamination 0.05
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import argparse
import os

def load_and_preprocess(data_path: str):
    # Column names for KDD Cup dataset
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

    # Load dataset
    df = pd.read_csv(data_path, names=col_names)

    # Drop categorical + label columns
    df = df.drop(columns=["protocol_type", "service", "flag", "label"], errors="ignore")

    # Replace NaN with 0
    df = df.fillna(0)

    return df

def train_isolation_forest(data_path: str, out_dir: str, contamination: float = 0.05):
    print(f"📂 Loading KDD dataset from: {data_path}")
    df = load_and_preprocess(data_path)

    feature_names = df.columns.tolist()
    print(f"✅ Preprocessed dataset with {len(df)} rows and {len(feature_names)} numeric features")

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)

    # Save model artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(out_dir, "features.joblib"))

    print(f"✅ Trained IsolationForest saved to {out_dir}")
    print(f"   - isolation_forest.joblib")
    print(f"   - scaler.joblib")
    print(f"   - features.joblib")
    print(f"Total samples trained on: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to KDD dataset CSV (e.g., data/kddcup.csv)")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save model files")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination rate (outlier proportion)")
    args = parser.parse_args()

    train_isolation_forest(args.data, args.out_dir, args.contamination)
