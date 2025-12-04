import pandas as pd
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
OUT_FILE = os.path.join(DATA_DIR, "clean_kdd_subset.csv")

# ✅ NSL-KDD column names
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'
]

# ✅ CRITICAL: Feature order MUST match utils.py and train_all.py
selected_features = [
    'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 'wrong_fragment',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate'
]

print("📂 Loading NSL-KDD datasets...")

try:
    df_train = pd.read_csv(TRAIN_FILE, names=columns)
    print(f"✅ Loaded training: {df_train.shape[0]} rows")
except FileNotFoundError:
    print(f"❌ Training file not found: {TRAIN_FILE}")
    exit(1)

try:
    df_test = pd.read_csv(TEST_FILE, names=columns)
    print(f"✅ Loaded testing: {df_test.shape[0]} rows")
except FileNotFoundError:
    print(f"❌ Test file not found: {TEST_FILE}")
    exit(1)

# Combine datasets
df = pd.concat([df_train, df_test], ignore_index=True)
print(f"📊 Total dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ✅ Convert categorical features to numeric
for col in ['protocol_type', 'service', 'flag']:
    if col in df.columns:
        df[col] = df[col].astype('category').cat.codes

# ✅ Clean and standardize attack labels
def clean_attack_label(x):
    """Convert attack labels to binary: 0=normal, 1=attack"""
    if pd.isna(x):
        return 1  # Assume unknown is attack for safety
    s = str(x).strip().lower()
    return 0 if s == 'normal' else 1

df['attack'] = df['attack'].apply(clean_attack_label)

print("\n📈 Label distribution:")
print(df['attack'].value_counts())
print(f"Normal: {(df['attack']==0).sum()} ({(df['attack']==0).sum()/len(df)*100:.1f}%)")
print(f"Attack: {(df['attack']==1).sum()} ({(df['attack']==1).sum()/len(df)*100:.1f}%)")

# ✅ Select features + label
df = df[selected_features + ['attack']]

# ✅ Handle infinities and NaNs
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# ✅ Data validation
print("\n🔍 Data validation:")
for col in selected_features:
    if col in df.columns:
        print(f"  {col:30s} | min={df[col].min():10.2f} | max={df[col].max():10.2f} | mean={df[col].mean():10.2f}")
    else:
        print(f"  ⚠️  Missing column: {col}")

# ✅ Save cleaned dataset
df.to_csv(OUT_FILE, index=False)
print(f"\n✅ Cleaned dataset saved to: {OUT_FILE}")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print("\nFirst 3 rows:")
print(df.head(3))