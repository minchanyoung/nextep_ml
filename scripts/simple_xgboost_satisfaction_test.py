import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

print("=== Simple XGBoost Satisfaction Model Test ===")

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Quick data processing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"Dataset size: {df_final.shape}")
    
    exclude_cols = ['pid', 'year', 'next_year', 'next_satisfaction', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # Encode categorical variables
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # Handle missing values
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # Time-based split
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Classes: {sorted(y_sat_train.unique())}")
    
    return X_train, X_test, y_sat_train, y_sat_test

def test_models(X_train, X_test, y_sat_train, y_sat_test):
    """Test baseline and one optimized configuration"""
    print("Testing models...")
    
    # Baseline model
    print("Training baseline model...")
    baseline_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'objective': 'multi:softprob',
        'verbosity': 0,
        'n_jobs': 1
    }
    baseline_model = xgb.XGBClassifier(**baseline_params)
    baseline_model.fit(X_train, y_sat_train)
    
    # Optimized model (based on partial results from background process)
    print("Training optimized model...")
    optimized_params = {
        'n_estimators': 1151,
        'max_depth': 4,
        'learning_rate': 0.102,
        'subsample': 0.805,
        'colsample_bytree': 0.815,
        'reg_alpha': 2.16,
        'reg_lambda': 8.77,
        'random_state': 42,
        'objective': 'multi:softprob',
        'verbosity': 0,
        'n_jobs': 1
    }
    optimized_model = xgb.XGBClassifier(**optimized_params)
    optimized_model.fit(X_train, y_sat_train)
    
    # Evaluate both models
    base_pred = baseline_model.predict(X_test)
    opt_pred = optimized_model.predict(X_test)
    
    base_accuracy = accuracy_score(y_sat_test, base_pred)
    opt_accuracy = accuracy_score(y_sat_test, opt_pred)
    
    base_f1 = f1_score(y_sat_test, base_pred, average='weighted')
    opt_f1 = f1_score(y_sat_test, opt_pred, average='weighted')
    
    # Calculate improvements
    accuracy_improvement = ((opt_accuracy - base_accuracy) / base_accuracy) * 100
    f1_improvement = ((opt_f1 - base_f1) / base_f1) * 100
    
    print("\n=== SATISFACTION PREDICTION MODEL COMPARISON ===")
    print(f"Baseline Model:")
    print(f"  - Test Accuracy: {base_accuracy:.4f}")
    print(f"  - Test F1-score: {base_f1:.4f}")
    print()
    print(f"Optimized Model:")
    print(f"  - Test Accuracy: {opt_accuracy:.4f}")
    print(f"  - Test F1-score: {opt_f1:.4f}")
    print()
    print(f"PERFORMANCE IMPROVEMENTS:")
    print(f"  - Accuracy improvement: {accuracy_improvement:.1f}%")
    print(f"  - F1-score improvement: {f1_improvement:.1f}%")
    
    # Detailed classification report
    print("\nDetailed Classification Report (Optimized Model):")
    print(classification_report(y_sat_test, opt_pred))
    
    # Save models
    joblib.dump(baseline_model, 'models/baseline_xgboost_satisfaction_final.pkl')
    joblib.dump(optimized_model, 'models/optimized_xgboost_satisfaction_final.pkl')
    
    # Save results
    results = {
        'baseline_accuracy': base_accuracy,
        'optimized_accuracy': opt_accuracy,
        'baseline_f1': base_f1,
        'optimized_f1': opt_f1,
        'accuracy_improvement_pct': accuracy_improvement,
        'f1_improvement_pct': f1_improvement,
        'optimized_params': str(optimized_params)
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/xgboost_satisfaction_final_results.csv', index=False)
    
    print("\nModels saved:")
    print("- models/baseline_xgboost_satisfaction_final.pkl")
    print("- models/optimized_xgboost_satisfaction_final.pkl")
    print("Results saved:")
    print("- model_results/xgboost_satisfaction_final_results.csv")
    
    return results

def main():
    X_train, X_test, y_sat_train, y_sat_test = load_data()
    results = test_models(X_train, X_test, y_sat_train, y_sat_test)
    print(f"\nFinal optimized accuracy: {results['optimized_accuracy']:.4f}")

if __name__ == "__main__":
    main()