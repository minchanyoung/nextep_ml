import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import joblib
import time

print("=== XGBoost Satisfaction Model Optimization ===")

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Filter predictable cases
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    
    # Create next year targets
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    # Filter consecutive years only
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    # Select cases with valid targets
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    
    # Filter valid satisfaction values (> 0) and convert to 0-based classes
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"Final dataset size: {df_final.shape}")
    
    # Feature selection
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
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
    
    # Convert satisfaction to 0-based classes (1-5 -> 0-4)
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"Features: {len(feature_cols)} features")
    print(f"Satisfaction classes: {sorted(y_sat_train.unique())}")
    
    return X_train, X_test, y_sat_train, y_sat_test, feature_cols

def optimize_xgboost_satisfaction(X_train, X_test, y_sat_train, y_sat_test, n_trials=50):
    """Optimize XGBoost satisfaction prediction model"""
    print(f"\nXGBoost Satisfaction Model Optimization (trials: {n_trials})")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'n_jobs': 1  # Single thread for stability
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
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
        
        return -np.mean(scores)  # Negative for minimization
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    # Train best model
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['objective'] = 'multi:softprob'
    best_params['eval_metric'] = 'mlogloss'
    best_params['verbosity'] = 0
    best_params['n_jobs'] = 1
    
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_sat_train)
    
    # Performance evaluation
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_accuracy = accuracy_score(y_sat_train, y_pred_train)
    test_accuracy = accuracy_score(y_sat_test, y_pred_test)
    train_f1 = f1_score(y_sat_train, y_pred_train, average='weighted')
    test_f1 = f1_score(y_sat_test, y_pred_test, average='weighted')
    
    print(f"\nXGBoost Satisfaction Optimization Complete")
    print(f"Best validation accuracy: {-study.best_value:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Best parameters: {best_params}")
    
    return {
        'model': best_model,
        'study': study,
        'best_params': best_params,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'best_cv_score': -study.best_value
    }

def compare_with_baseline(X_train, X_test, y_sat_train, y_sat_test, optimized_results):
    """Compare optimized model with baseline"""
    print("\nComparing with baseline model...")
    
    # Baseline parameters
    baseline_params = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'n_jobs': 1
    }
    
    baseline_model = xgb.XGBClassifier(**baseline_params)
    baseline_model.fit(X_train, y_sat_train)
    
    base_pred_test = baseline_model.predict(X_test)
    base_accuracy = accuracy_score(y_sat_test, base_pred_test)
    base_f1 = f1_score(y_sat_test, base_pred_test, average='weighted')
    
    # Results comparison
    opt_accuracy = optimized_results['test_accuracy']
    opt_f1 = optimized_results['test_f1']
    
    accuracy_improvement = ((opt_accuracy - base_accuracy) / base_accuracy) * 100
    f1_improvement = ((opt_f1 - base_f1) / base_f1) * 100
    
    print("\n=== SATISFACTION PREDICTION MODEL COMPARISON ===")
    print(f"Optimized Model:")
    print(f"  - Test Accuracy: {opt_accuracy:.4f}")
    print(f"  - Test F1-score: {opt_f1:.4f}")
    print()
    print(f"Baseline Model:")
    print(f"  - Test Accuracy: {base_accuracy:.4f}")
    print(f"  - Test F1-score: {base_f1:.4f}")
    print()
    print(f"PERFORMANCE IMPROVEMENTS:")
    print(f"  - Accuracy improvement: {accuracy_improvement:.1f}%")
    print(f"  - F1-score improvement: {f1_improvement:.1f}%")
    
    return {
        'baseline_accuracy': base_accuracy,
        'baseline_f1': base_f1,
        'optimized_accuracy': opt_accuracy,
        'optimized_f1': opt_f1,
        'accuracy_improvement': accuracy_improvement,
        'f1_improvement': f1_improvement,
        'baseline_model': baseline_model
    }

def save_results(optimized_results, comparison_results):
    """Save models and results"""
    print("\nSaving models and results...")
    
    # Save optimized model
    joblib.dump(optimized_results['model'], 'models/final_optimized_xgboost_satisfaction.pkl')
    joblib.dump(comparison_results['baseline_model'], 'models/baseline_xgboost_satisfaction.pkl')
    
    # Save results
    results = {
        'optimized_accuracy': optimized_results['test_accuracy'],
        'optimized_f1': optimized_results['test_f1'],
        'baseline_accuracy': comparison_results['baseline_accuracy'],
        'baseline_f1': comparison_results['baseline_f1'],
        'accuracy_improvement_pct': comparison_results['accuracy_improvement'],
        'f1_improvement_pct': comparison_results['f1_improvement'],
        'best_cv_score': optimized_results['best_cv_score'],
        'best_params': str(optimized_results['best_params'])
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/xgboost_satisfaction_optimization_results.csv', index=False)
    
    print("Models saved:")
    print("  - models/final_optimized_xgboost_satisfaction.pkl")
    print("  - models/baseline_xgboost_satisfaction.pkl")
    print("Results saved:")
    print("  - model_results/xgboost_satisfaction_optimization_results.csv")
    
    return results

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_sat_train, y_sat_test, feature_cols = load_data()
    
    # Optimize XGBoost satisfaction model
    optimized_results = optimize_xgboost_satisfaction(
        X_train, X_test, y_sat_train, y_sat_test, n_trials=50
    )
    
    # Compare with baseline
    comparison_results = compare_with_baseline(
        X_train, X_test, y_sat_train, y_sat_test, optimized_results
    )
    
    # Save results
    final_results = save_results(optimized_results, comparison_results)
    
    total_time = time.time() - start_time
    print(f"\n=== XGBoost Satisfaction Optimization Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final test accuracy: {final_results['optimized_accuracy']:.4f}")
    print(f"Accuracy improvement: {final_results['accuracy_improvement_pct']:.1f}%")

if __name__ == "__main__":
    main()