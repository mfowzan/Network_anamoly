import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ✅ CRITICAL: Feature order must be consistent across all files
FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", "srv_count", "wrong_fragment",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate"
]

CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_kdd_subset.csv")

def load_data():
    """Load preprocessed dataset"""
    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"❌ {CLEAN_DATA_PATH} not found. Run pre_process.py first!")
    
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape}")
    
    # Ensure all features exist
    for f in FEATURES:
        if f not in df.columns:
            print(f"⚠️  Adding missing feature: {f}")
            df[f] = 0
    
    # Ensure label column exists
    if 'attack' not in df.columns:
        raise ValueError("❌ 'attack' column not found in dataset!")
    
    return df

def prepare_data(df):
    """Prepare features and labels"""
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = shuffle(df, random_state=42).reset_index(drop=True)
    
    X = df[FEATURES].astype(float).values
    y = df['attack'].astype(int).values
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(FEATURES)}")
    print(f"   Normal (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   Attack (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    return X, y

def save_model(model_dir, model, scaler, features, metadata, model_file):
    """Save model and associated files"""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, model_file))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(features, os.path.join(model_dir, "features.joblib"))
    joblib.dump(metadata, os.path.join(model_dir, "metadata.joblib"))

def evaluate_model(y_true, y_pred, model_name):
    """Quick evaluation"""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n   Accuracy: {acc:.4f}")
    print(f"   Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    return acc

# ===================== MAIN TRAINING =====================
print("="*60)
print("🚀 TRAINING ALL ANOMALY DETECTION MODELS")
print("="*60)

# Load data
df = load_data()
X, y = prepare_data(df)

# Create metadata
metadata = {
    "features": FEATURES,
    "expected_ranges": {
        f: {
            "min": float(df[f].min()),
            "max": float(df[f].max()),
            "mean": float(df[f].mean()),
            "std": float(df[f].std())
        } for f in FEATURES
    },
    "label_distribution": {
        "normal": int((y==0).sum()),
        "attack": int((y==1).sum())
    }
}

# Split data for validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n✂️  Train: {len(X_train)}, Validation: {len(X_val)}")

# ==================== 1. AUTOENCODER ====================
print("\n" + "="*60)
print("🔹 1/5 TRAINING AUTOENCODER")
print("="*60)

ae_scaler = StandardScaler()
X_train_scaled = ae_scaler.fit_transform(X_train)
X_val_scaled = ae_scaler.transform(X_val)

# Train only on normal data
X_train_normal = X_train_scaled[y_train == 0]
print(f"   Using {len(X_train_normal)} normal samples for training")

if len(X_train_normal) < 100:
    print("⚠️  WARNING: Very few normal samples!")

input_dim = X_train_normal.shape[1]
autoencoder = Sequential([
    Dense(32, activation="relu", input_shape=(input_dim,)),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(input_dim, activation="linear")
], name="Autoencoder")
autoencoder.compile(optimizer=Adam(0.001), loss="mse")

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)
print(f"   Final loss: {history.history['loss'][-1]:.6f}")

# Validate
recon = autoencoder.predict(X_val_scaled, verbose=0)
mse = np.mean(np.power(X_val_scaled - recon, 2), axis=1)
threshold = np.percentile(mse, 95)
y_pred = (mse > threshold).astype(int)
acc = evaluate_model(y_val, y_pred, "Autoencoder")

ae_out = os.path.join(OUT_DIR, "autoencoder")
autoencoder.save(os.path.join(ae_out, "autoencoder.h5"))
joblib.dump(ae_scaler, os.path.join(ae_out, "scaler.joblib"))
joblib.dump(FEATURES, os.path.join(ae_out, "features.joblib"))
joblib.dump(metadata, os.path.join(ae_out, "metadata.joblib"))
print(f"✅ Saved to {ae_out}")

# ==================== 2. ISOLATION FOREST ====================
print("\n" + "="*60)
print("🔹 2/5 TRAINING ISOLATION FOREST")
print("="*60)

iso_scaler = StandardScaler()
X_train_iso = iso_scaler.fit_transform(X_train)
X_val_iso = iso_scaler.transform(X_val)

iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
iso.fit(X_train_iso)

y_pred = (iso.predict(X_val_iso) == -1).astype(int)
acc = evaluate_model(y_val, y_pred, "IsolationForest")

iso_out = os.path.join(OUT_DIR, "isolation")
save_model(iso_out, iso, iso_scaler, FEATURES, metadata, "isolation_forest.joblib")
print(f"✅ Saved to {iso_out}")

# ==================== 3. ONE-CLASS SVM ====================
print("\n" + "="*60)
print("🔹 3/5 TRAINING ONE-CLASS SVM")
print("="*60)

ocsvm_scaler = StandardScaler()
X_train_normal_idx = (y_train == 0)
X_train_ocsvm = ocsvm_scaler.fit_transform(X_train[X_train_normal_idx])
X_val_ocsvm = ocsvm_scaler.transform(X_val)

print(f"   Using {len(X_train_ocsvm)} normal samples")

# Use smaller subset if dataset is large (OCSVM is slow)
if len(X_train_ocsvm) > 5000:
    print(f"   ⚠️  Subsampling to 5000 for speed")
    X_train_ocsvm = X_train_ocsvm[:5000]

ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
ocsvm.fit(X_train_ocsvm)

y_pred = (ocsvm.predict(X_val_ocsvm) == -1).astype(int)
acc = evaluate_model(y_val, y_pred, "OCSVM")

ocsvm_out = os.path.join(OUT_DIR, "ocsvm")
save_model(ocsvm_out, ocsvm, ocsvm_scaler, FEATURES, metadata, "ocsvm.joblib")
print(f"✅ Saved to {ocsvm_out}")

# ==================== 4. RANDOM FOREST ====================
print("\n" + "="*60)
print("🔹 4/5 TRAINING RANDOM FOREST")
print("="*60)

rf_scaler = StandardScaler()
X_train_rf = rf_scaler.fit_transform(X_train)
X_val_rf = rf_scaler.transform(X_val)

rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train)

y_pred = rf.predict(X_val_rf)
acc = evaluate_model(y_val, y_pred, "RandomForest")

rf_out = os.path.join(OUT_DIR, "rf")
save_model(rf_out, rf, rf_scaler, FEATURES, metadata, "random_forest.joblib")
print(f"✅ Saved to {rf_out}")

# ==================== 5. XGBOOST ====================
print("\n" + "="*60)
print("🔹 5/5 TRAINING XGBOOST")
print("="*60)

xgb_scaler = StandardScaler()
X_train_xgb = xgb_scaler.fit_transform(X_train)
X_val_xgb = xgb_scaler.transform(X_val)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train_xgb, y_train)

y_pred = xgb.predict(X_val_xgb)
acc = evaluate_model(y_val, y_pred, "XGBoost")

xgb_out = os.path.join(OUT_DIR, "xgboost")
save_model(xgb_out, xgb, xgb_scaler, FEATURES, metadata, "xgboost.joblib")
print(f"✅ Saved to {xgb_out}")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("🎯 ALL MODELS TRAINED & SAVED SUCCESSFULLY")
print("="*60)
print(f"Models saved in: {OUT_DIR}")
print("\n📁 Directory structure:")
for model_name in ["autoencoder", "isolation", "ocsvm", "rf", "xgboost"]:
    model_path = os.path.join(OUT_DIR, model_name)
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"   {model_name}/")
        for f in files:
            print(f"      - {f}")