import os
import joblib
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

autoencoder = tf.keras.models.load_model(os.path.join(MODEL_DIR, "autoencoder.h5"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_autoencoder.joblib"))
features = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))

def preprocess_input(data: dict):
    values = []
    for f in features:
        values.append(float(data.get(f, 0)))
    X = np.array(values).reshape(1, -1)
    return scaler.transform(X)

def anomaly_score(data: dict):
    X_scaled = preprocess_input(data)
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - reconstructed))
    
    # Convert to confidence-like score for frontend
    confidence = 100 - (mse * 100 / (mse + 1))
    return {
        "confidence": confidence,
        "anomaly": confidence < 50,  # threshold can be tuned
        "error": mse
    }
