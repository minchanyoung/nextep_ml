import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

print("=== Quick Class Balance Test ===")

def quick_test():
    """Quick class balance improvement test"""
    
    # Load minimal data for quick test
    print("Loading data...")
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Quick processing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    # Sample for speed (take first 50k rows)
    df_sample = df_sorted.head(50000)
    
    valid_mask = (~df_sample['next_satisfaction'].isna()) & (df_sample['next_satisfaction'] > 0)
    df_final = df_sample[valid_mask].copy()
    
    print(f"Sample dataset size: {df_final.shape}")
    
    # Simple feature selection (top features from SHAP)
    feature_cols = ['wave', 'p4311', 'p_age', 'p_sex', 'p_edu', 'occupation_code']
    
    # Quick encoding
    for col in feature_cols:
        if df_final[col].dtype == 'object':
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # Fill missing values
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # Split data
    train_mask = df_final['year'] <= 2018
    test_mask = df_final['year'] > 2018
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Class distribution
    print("\nClass distribution:")
    for cls in sorted(y_train.unique()):
        count = (y_train == cls).sum()
        print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Test models
    models = {}
    
    # 1. Baseline
    print("\nTesting baseline model...")
    baseline = xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    baseline.fit(X_train, y_train)
    pred_baseline = baseline.predict(X_test)
    acc_baseline = accuracy_score(y_test, pred_baseline)
    f1_baseline = f1_score(y_test, pred_baseline, average='weighted')
    models['Baseline'] = {'accuracy': acc_baseline, 'f1': f1_baseline}
    
    # 2. Class weights (balanced)
    print("Testing balanced class weights...")
    weighted = xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    
    # Simple balanced weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[classes == y][0] for y in y_train])
    
    weighted.fit(X_train, y_train, sample_weight=sample_weights)
    pred_weighted = weighted.predict(X_test)
    acc_weighted = accuracy_score(y_test, pred_weighted)
    f1_weighted = f1_score(y_test, pred_weighted, average='weighted')
    models['Weighted'] = {'accuracy': acc_weighted, 'f1': f1_weighted}
    
    # 3. Custom extreme weights for minority classes
    print("Testing custom weights...")
    custom_weights = {0: 15.0, 1: 1.0, 2: 0.8, 3: 10.0, 4: 25.0}
    sample_weights_custom = np.array([custom_weights.get(y, 1.0) for y in y_train])
    
    custom_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, verbosity=0)
    custom_model.fit(X_train, y_train, sample_weight=sample_weights_custom)
    pred_custom = custom_model.predict(X_test)
    acc_custom = accuracy_score(y_test, pred_custom)
    f1_custom = f1_score(y_test, pred_custom, average='weighted')
    models['Custom Weighted'] = {'accuracy': acc_custom, 'f1': f1_custom}
    
    # Results
    print("\n=== Results ===")
    for name, metrics in models.items():
        print(f"{name:15s}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Best model
    best_model = max(models.keys(), key=lambda k: models[k]['f1'])
    print(f"\nBest model (by F1): {best_model}")
    print(f"Best F1 score: {models[best_model]['f1']:.4f}")
    
    # Save quick results
    results_df = pd.DataFrame(models).T
    results_df.to_csv('model_results/quick_class_balance_results.csv')
    print("\nResults saved to: model_results/quick_class_balance_results.csv")
    
    return models, best_model

if __name__ == "__main__":
    models, best_model = quick_test()
    print(f"\nQuick class balance test complete. Best method: {best_model}")