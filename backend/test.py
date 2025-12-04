import joblib, os, pandas as pd

# Paths to all model folders (edit if your structure differs)
base = "app/models"
folders = ["autoencoder", "isolation", "ocsvm", "rf"]

for f in folders:
    print("\n==========", f.upper(), "==========")
    try:
        features = joblib.load(os.path.join(base, f, "features.joblib"))
        print("Feature count:", len(features))
        print("First few features:", features[:10])
    except Exception as e:
        print("Could not load features.joblib:", e)
    try:
        scaler = joblib.load(os.path.join(base, f, "scaler.joblib"))
        print("Scaler type:", type(scaler))
    except Exception as e:
        print("Could not load scaler:", e)
