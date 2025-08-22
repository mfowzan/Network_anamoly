import pandas as pd
from typing import List

FEATURE_COLS = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "wrong_fragment"]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only numeric feature cols (drop rows with NaNs in those cols)
    df = df.dropna(subset=FEATURE_COLS)
    return df

def df_to_features(df: pd.DataFrame) -> pd.DataFrame:
    # Return DataFrame containing only feature columns (float)
    return df[FEATURE_COLS].astype(float)
