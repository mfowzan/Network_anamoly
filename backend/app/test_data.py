"""
Test models with manual input (simulates frontend)
"""
import os
import pandas as pd
from model import load_model_from_dir

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Sample network traffic data (you can modify these values)
SAMPLE_TRAFFIC = {
    "duration": 0,
    "src_bytes": 181,
    "dst_bytes": 5450,
    "count": 8,
    "srv_count": 8,
    "wrong_fragment": 0,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0
}

# Example of attack traffic
ATTACK_TRAFFIC = {
    "duration": 0,
    "src_bytes": 0,
    "dst_bytes": 0,
    "count": 123,
    "srv_count": 6,
    "wrong_fragment": 0,
    "serror_rate": 1.0,
    "srv_serror_rate": 1.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 0.05,
    "diff_srv_rate": 0.06,
    "dst_host_count": 255,
    "dst_host_srv_count": 6,
    "dst_host_same_srv_rate": 0.04,
    "dst_host_diff_srv_rate": 0.06
}

def test_traffic(traffic_data, description=""):
    """Test a single traffic sample with all models"""
    print("\n" + "="*70)
    print(f"🔍 Testing: {description}")
    print("="*70)
    
    # Convert to DataFrame
    df = pd.DataFrame([traffic_data])
    
    print("\n📊 Input features:")
    for key, value in traffic_data.items():
        print(f"   {key:30s} = {value}")
    
    # Test with each model
    models = [
        ("Autoencoder", "autoencoder", "autoencoder"),
        ("Isolation Forest", "isolation_forest", "isolation"),
        ("One-Class SVM", "ocsvm", "ocsvm"),
        ("Random Forest", "rf", "rf"),
        ("XGBoost", "xgboost", "xgboost")
    ]
    
    print("\n" + "-"*70)
    print("PREDICTIONS FROM ALL MODELS:")
    print("-"*70)
    
    for model_name, model_type, model_dir_name in models:
        model_dir = os.path.join(MODELS_DIR, model_dir_name)
        
        if not os.path.exists(model_dir):
            print(f"❌ {model_name:20s} - Model not found")
            continue
        
        try:
            model = load_model_from_dir(model_dir, model_type)
            result = model.predict_on_df(df)
            
            prediction = result['prediction'].iloc[0]
            score = result['score'].iloc[0]
            
            emoji = "✅" if prediction == "normal" else "🚨"
            print(f"{emoji} {model_name:20s} → {prediction.upper():8s} (score: {score:.4f})")
            
        except Exception as e:
            print(f"❌ {model_name:20s} - Error: {e}")

def main():
    print("="*70)
    print("🧪 MANUAL INPUT TESTING (Frontend Simulation)")
    print("="*70)
    
    # Test normal traffic
    test_traffic(SAMPLE_TRAFFIC, "Normal Network Traffic")
    
    # Test attack traffic
    test_traffic(ATTACK_TRAFFIC, "Suspicious Network Traffic (Potential Attack)")
    
    print("\n" + "="*70)
    print("💡 TIP: Modify SAMPLE_TRAFFIC and ATTACK_TRAFFIC in this script")
    print("         to test with your own network traffic data")
    print("="*70)

if __name__ == "__main__":
    main()