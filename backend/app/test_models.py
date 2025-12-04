"""
Test script to verify all models work correctly
Run this after training to validate predictions
"""
import os
import pandas as pd
import numpy as np
from model import load_model_from_dir

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DATA = os.path.join(DATA_DIR, "clean_kdd_subset.csv")

def test_model(model_type, model_dir, test_df):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()}")
    print('='*60)
    
    try:
        # Load model
        model = load_model_from_dir(model_dir, model_type)
        
        # Make predictions
        results = model.predict_on_df(test_df)
        
        # Display results
        print(f"\n📊 Results Summary:")
        print(f"   Total samples: {len(results)}")
        print(f"   Normal: {(results['prediction']=='normal').sum()}")
        print(f"   Anomaly: {(results['prediction']=='anomaly').sum()}")
        print(f"   Avg score: {results['score'].mean():.4f}")
        print(f"   Min score: {results['score'].min():.4f}")
        print(f"   Max score: {results['score'].max():.4f}")
        
        # Show sample predictions
        print(f"\n🔍 Sample predictions (first 5):")
        print(results.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🧪 MODEL TESTING SCRIPT")
    print("="*60)
    
    # Load test data
    if not os.path.exists(CLEAN_DATA):
        print(f"❌ Test data not found: {CLEAN_DATA}")
        print("   Run pre_process.py first!")
        return
    
    df = pd.read_csv(CLEAN_DATA)
    print(f"\n✅ Loaded test data: {df.shape}")
    
    # Sample some data for testing (mix of normal and attacks)
    if 'attack' in df.columns:
        normal_samples = df[df['attack'] == 0].sample(min(50, len(df[df['attack']==0])), random_state=42)
        attack_samples = df[df['attack'] == 1].sample(min(50, len(df[df['attack']==1])), random_state=42)
        test_df = pd.concat([normal_samples, attack_samples]).sample(frac=1, random_state=42)
    else:
        test_df = df.sample(min(100, len(df)), random_state=42)
    
    print(f"   Using {len(test_df)} samples for testing")
    if 'attack' in test_df.columns:
        print(f"   Normal: {(test_df['attack']==0).sum()}, Attack: {(test_df['attack']==1).sum()}")
    
    # Test each model
    models_to_test = [
        ("autoencoder", "autoencoder"),
        ("isolation_forest", "isolation"),
        ("ocsvm", "ocsvm"),
        ("rf", "rf"),
        ("xgboost", "xgboost")
    ]
    
    results = {}
    for model_type, model_dir_name in models_to_test:
        model_dir = os.path.join(MODELS_DIR, model_dir_name)
        if os.path.exists(model_dir):
            success = test_model(model_type, model_dir, test_df)
            results[model_type] = success
        else:
            print(f"\n⚠️  Model directory not found: {model_dir}")
            results[model_type] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 TESTING SUMMARY")
    print("="*60)
    for model_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_type:20s} {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All models passed testing!")
    else:
        print("\n⚠️  Some models failed. Check errors above.")

if __name__ == "__main__":
    main()