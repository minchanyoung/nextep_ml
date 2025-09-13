import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import catboost as cb
import joblib

print("=== Hyperparameter Optimization Results Analysis ===")

def load_data():
    """Load and preprocess data"""
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
    
    # Filter valid satisfaction values (> 0)
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    # Feature selection
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
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
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    
    return X_train, X_test, y_wage_train, y_wage_test

def test_models():
    """Test optimized parameters"""
    X_train, X_test, y_wage_train, y_wage_test = load_data()
    
    # Optimized parameters from Trial 15
    optimized_params = {
        'iterations': 860,
        'learning_rate': 0.010426694594004318,
        'depth': 8,
        'l2_leaf_reg': 8.653255320409505,
        'border_count': 216,
        'random_seed': 42,
        'verbose': False
    }
    
    # Baseline parameters
    baseline_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False
    }
    
    print("Training models...")
    
    # Optimized model
    optimized_model = cb.CatBoostRegressor(**optimized_params)
    optimized_model.fit(X_train, y_wage_train)
    
    # Baseline model  
    baseline_model = cb.CatBoostRegressor(**baseline_params)
    baseline_model.fit(X_train, y_wage_train)
    
    # Predictions
    opt_pred_test = optimized_model.predict(X_test)
    base_pred_test = baseline_model.predict(X_test)
    
    # Performance calculation
    opt_rmse = np.sqrt(mean_squared_error(y_wage_test, opt_pred_test))
    opt_mae = mean_absolute_error(y_wage_test, opt_pred_test)
    opt_r2 = r2_score(y_wage_test, opt_pred_test)
    
    base_rmse = np.sqrt(mean_squared_error(y_wage_test, base_pred_test))
    base_mae = mean_absolute_error(y_wage_test, base_pred_test)
    base_r2 = r2_score(y_wage_test, base_pred_test)
    
    # Results
    print("\n=== WAGE PREDICTION MODEL COMPARISON ===")
    print(f"Optimized Model:")
    print(f"  - Test RMSE: {opt_rmse:.2f}")
    print(f"  - Test MAE: {opt_mae:.2f}")
    print(f"  - Test R2: {opt_r2:.4f}")
    print()
    print(f"Baseline Model:")
    print(f"  - Test RMSE: {base_rmse:.2f}")
    print(f"  - Test MAE: {base_mae:.2f}")
    print(f"  - Test R2: {base_r2:.4f}")
    print()
    
    # Improvements
    rmse_improvement = ((base_rmse - opt_rmse) / base_rmse) * 100
    mae_improvement = ((base_mae - opt_mae) / base_mae) * 100
    r2_improvement = ((opt_r2 - base_r2) / abs(base_r2)) * 100 if base_r2 != 0 else 0
    
    print(f"PERFORMANCE IMPROVEMENTS:")
    print(f"  - RMSE improvement: {rmse_improvement:.1f}%")
    print(f"  - MAE improvement: {mae_improvement:.1f}%") 
    print(f"  - R2 improvement: {r2_improvement:.1f}%")
    
    # Save models
    joblib.dump(optimized_model, 'models/final_optimized_catboost_wage.pkl')
    print(f"\nOptimized model saved to: models/final_optimized_catboost_wage.pkl")
    
    # Save results
    results = {
        'optimized_rmse': opt_rmse,
        'optimized_mae': opt_mae,
        'optimized_r2': opt_r2,
        'baseline_rmse': base_rmse,
        'baseline_mae': base_mae,
        'baseline_r2': base_r2,
        'rmse_improvement_pct': rmse_improvement,
        'mae_improvement_pct': mae_improvement,
        'r2_improvement_pct': r2_improvement
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/final_optimization_results.csv', index=False)
    print(f"Results saved to: model_results/final_optimization_results.csv")
    
    return results

if __name__ == "__main__":
    results = test_models()