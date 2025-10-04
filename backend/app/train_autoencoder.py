import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

def load_kdd(data_path):
    # Load KDD dataset
    df = pd.read_csv(data_path)

    # Drop non-numeric or categorical columns if present
    df = df.select_dtypes(include=[np.number])

    # Features only
    X = df.values
    return X

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)
    encoded = layers.Dense(8, activation="relu")(encoded)

    decoded = layers.Dense(16, activation="relu")(encoded)
    decoded = layers.Dense(32, activation="relu")(decoded)
    output_layer = layers.Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to KDD dataset (CSV)")
    parser.add_argument("--out-dir", default="app/models/autoencoder", help="Where to save model + scaler")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("📥 Loading dataset...")
    X = load_kdd(args.data)

    print("📏 Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🏗 Building autoencoder...")
    autoencoder = build_autoencoder(X_scaled.shape[1])

    print("⚡ Training autoencoder...")
    autoencoder.fit(X_scaled, X_scaled, 
                    epochs=20, 
                    batch_size=64, 
                    shuffle=True, 
                    validation_split=0.2)

    print("💾 Saving model + scaler...")
    autoencoder.save(os.path.join(args.out_dir, "autoencoder.h5"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    joblib.dump(list(range(X.shape[1])), os.path.join(args.out_dir, "features.joblib"))

    print(f"✅ Autoencoder saved in {args.out_dir}")

if __name__ == "__main__":
    main()
