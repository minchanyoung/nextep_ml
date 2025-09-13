import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import joblib
import time

print("=== Quick XGBoost Satisfaction Optimization ===")

def load_data():
    """Load and preprocess data"""
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
    return X_train, X_test, y_sat_train, y_sat_test

def quick_optimize(X_train, X_test, y_sat_train, y_sat_test, n_trials=20):
    """Quick optimization with fewer trials"""
    print(f"Quick optimization with {n_trials} trials...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'n_jobs': 1
        }
        
        # Simple 2-fold CV for speed
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_sat_train.iloc[train_idx]
            y_fold_val = y_sat_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            accuracy = accuracy_score(y_fold_val, y_pred)
            scores.append(accuracy)
        
        return -np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Train best model
    best_params = study.best_params
    best_params.update({
        'random_state': 42,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'n_jobs': 1
    })
    
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_sat_train)
    
    # Baseline model
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
    
    # Evaluate both models
    opt_pred = best_model.predict(X_test)
    base_pred = baseline_model.predict(X_test)
    
    opt_accuracy = accuracy_score(y_sat_test, opt_pred)
    base_accuracy = accuracy_score(y_sat_test, base_pred)
    
    opt_f1 = f1_score(y_sat_test, opt_pred, average='weighted')
    base_f1 = f1_score(y_sat_test, base_pred, average='weighted')
    
    improvement = ((opt_accuracy - base_accuracy) / base_accuracy) * 100
    
    print(f"\nResults:")
    print(f"Optimized Accuracy: {opt_accuracy:.4f}")
    print(f"Baseline Accuracy: {base_accuracy:.4f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Best params: {best_params}")
    
    # Save results
    joblib.dump(best_model, 'models/quick_optimized_xgboost_satisfaction.pkl')
    
    results = {
        'optimized_accuracy': opt_accuracy,
        'baseline_accuracy': base_accuracy,
        'optimized_f1': opt_f1,
        'baseline_f1': base_f1,
        'improvement_pct': improvement,
        'best_params': str(best_params)
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/quick_xgboost_satisfaction_results.csv', index=False)
    
    return results

def main():
    start_time = time.time()
    
    X_train, X_test, y_sat_train, y_sat_test = load_data()
    results = quick_optimize(X_train, X_test, y_sat_train, y_sat_test, n_trials=20)
    
    print(f"\nQuick optimization completed in {(time.time() - start_time)/60:.1f} minutes")
    print("Files saved:")
    print("- models/quick_optimized_xgboost_satisfaction.pkl")
    print("- model_results/quick_xgboost_satisfaction_results.csv")

if __name__ == "__main__":
    main()