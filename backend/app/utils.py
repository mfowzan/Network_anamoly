import pandas as pd

# ✅ CRITICAL: Feature order MUST match training order exactly
FEATURE_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "wrong_fragment",  # ⚠️ MOVED to match train_all.py
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate"
]

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV and ensure all required features exist"""
    df = pd.read_csv(path)
    
    # Add missing features with default value 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"⚠️  Warning: Missing feature '{col}', filling with 0")
            df[col] = 0
    
    # Remove rows with NaN in feature columns
    df = df.dropna(subset=FEATURE_COLS)
    return df

def df_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature columns in correct order"""
    return df[FEATURE_COLS].astype(float)