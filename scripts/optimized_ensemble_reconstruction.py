import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import joblib
import time

print("=== Optimized Ensemble Reconstruction ===")

def load_data():
    """Load and preprocess data"""
    print("Loading data...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Quick data processing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"Dataset size: {df_final.shape}")
    
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
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test

def create_optimized_wage_ensemble(X_train, X_test, y_wage_train, y_wage_test):
    """Create optimized wage prediction ensemble"""
    print("\n=== Creating Optimized Wage Ensemble ===")
    
    # Optimized CatBoost (from previous optimization)
    optimized_catboost = cb.CatBoostRegressor(
        iterations=860,
        learning_rate=0.010426694594004318,
        depth=8,
        l2_leaf_reg=8.653255320409505,
        border_count=216,
        random_seed=42,
        verbose=False
    )
    
    # Optimized XGBoost (manual tuning for regression)
    optimized_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        verbosity=0
    )
    
    # Optimized LightGBM
    optimized_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        verbosity=-1
    )
    
    print("Training individual models...")
    
    # Train individual models
    optimized_catboost.fit(X_train, y_wage_train)
    optimized_xgb.fit(X_train, y_wage_train)
    optimized_lgb.fit(X_train, y_wage_train)
    
    # Individual model predictions
    catboost_pred = optimized_catboost.predict(X_test)
    xgb_pred = optimized_xgb.predict(X_test)
    lgb_pred = optimized_lgb.predict(X_test)
    
    # Individual model performance
    individual_results = {}
    for name, pred in [('CatBoost', catboost_pred), ('XGBoost', xgb_pred), ('LightGBM', lgb_pred)]:
        rmse = np.sqrt(mean_squared_error(y_wage_test, pred))
        mae = mean_absolute_error(y_wage_test, pred)
        r2 = r2_score(y_wage_test, pred)
        individual_results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    # Create voting ensemble
    print("\nCreating voting ensemble...")
    voting_ensemble = VotingRegressor([
        ('catboost', optimized_catboost),
        ('xgb', optimized_xgb),
        ('lgb', optimized_lgb)
    ])
    
    voting_ensemble.fit(X_train, y_wage_train)
    voting_pred = voting_ensemble.predict(X_test)
    
    # Ensemble performance
    ensemble_rmse = np.sqrt(mean_squared_error(y_wage_test, voting_pred))
    ensemble_mae = mean_absolute_error(y_wage_test, voting_pred)
    ensemble_r2 = r2_score(y_wage_test, voting_pred)
    
    print(f"Voting Ensemble - RMSE: {ensemble_rmse:.2f}, MAE: {ensemble_mae:.2f}, R²: {ensemble_r2:.4f}")
    
    # Save ensemble
    joblib.dump(voting_ensemble, 'models/optimized_wage_voting_ensemble.pkl')
    
    return {
        'individual_results': individual_results,
        'ensemble_rmse': ensemble_rmse,
        'ensemble_mae': ensemble_mae,
        'ensemble_r2': ensemble_r2,
        'ensemble_model': voting_ensemble
    }

def create_optimized_satisfaction_ensemble(X_train, X_test, y_sat_train, y_sat_test):
    """Create optimized satisfaction prediction ensemble"""
    print("\n=== Creating Optimized Satisfaction Ensemble ===")
    
    # Optimized XGBoost (from previous optimization)
    optimized_xgb = xgb.XGBClassifier(
        n_estimators=1151,
        max_depth=4,
        learning_rate=0.102,
        subsample=0.805,
        colsample_bytree=0.815,
        reg_alpha=2.16,
        reg_lambda=8.77,
        random_state=42,
        objective='multi:softprob',
        verbosity=0
    )
    
    # Optimized CatBoost for classification
    optimized_catboost = cb.CatBoostClassifier(
        iterations=800,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False
    )
    
    # Optimized LightGBM for classification
    optimized_lgb = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=5.0,
        random_state=42,
        verbosity=-1
    )
    
    print("Training individual models...")
    
    # Train individual models
    optimized_xgb.fit(X_train, y_sat_train)
    optimized_catboost.fit(X_train, y_sat_train)
    optimized_lgb.fit(X_train, y_sat_train)
    
    # Individual model predictions
    xgb_pred = optimized_xgb.predict(X_test)
    catboost_pred = optimized_catboost.predict(X_test)
    lgb_pred = optimized_lgb.predict(X_test)
    
    # Individual model performance
    individual_results = {}
    for name, pred in [('XGBoost', xgb_pred), ('CatBoost', catboost_pred), ('LightGBM', lgb_pred)]:
        accuracy = accuracy_score(y_sat_test, pred)
        f1 = f1_score(y_sat_test, pred, average='weighted')
        individual_results[name] = {'accuracy': accuracy, 'f1': f1}
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Create voting ensemble
    print("\nCreating voting ensemble...")
    voting_ensemble = VotingClassifier([
        ('xgb', optimized_xgb),
        ('catboost', optimized_catboost),
        ('lgb', optimized_lgb)
    ], voting='soft')
    
    voting_ensemble.fit(X_train, y_sat_train)
    voting_pred = voting_ensemble.predict(X_test)
    
    # Ensemble performance
    ensemble_accuracy = accuracy_score(y_sat_test, voting_pred)
    ensemble_f1 = f1_score(y_sat_test, voting_pred, average='weighted')
    
    print(f"Voting Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
    
    # Save ensemble
    joblib.dump(voting_ensemble, 'models/optimized_satisfaction_voting_ensemble.pkl')
    
    return {
        'individual_results': individual_results,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'ensemble_model': voting_ensemble
    }

def compare_with_baseline(wage_results, satisfaction_results):
    """Compare with previous baseline results"""
    print("\n=== Comparison with Baseline ===")
    
    # Previous baseline results (from documentation)
    baseline_wage_rmse = 118.89  # Previous voting ensemble
    baseline_satisfaction_accuracy = 0.6716  # Previous voting ensemble
    
    # Current optimized results
    current_wage_rmse = wage_results['ensemble_rmse']
    current_satisfaction_accuracy = satisfaction_results['ensemble_accuracy']
    
    # Calculate improvements
    wage_improvement = ((baseline_wage_rmse - current_wage_rmse) / baseline_wage_rmse) * 100
    satisfaction_improvement = ((current_satisfaction_accuracy - baseline_satisfaction_accuracy) / baseline_satisfaction_accuracy) * 100
    
    print(f"Wage Prediction:")
    print(f"  Previous Ensemble RMSE: {baseline_wage_rmse:.2f}")
    print(f"  Optimized Ensemble RMSE: {current_wage_rmse:.2f}")
    print(f"  Improvement: {wage_improvement:.1f}%")
    print()
    print(f"Satisfaction Prediction:")
    print(f"  Previous Ensemble Accuracy: {baseline_satisfaction_accuracy:.4f}")
    print(f"  Optimized Ensemble Accuracy: {current_satisfaction_accuracy:.4f}")
    print(f"  Improvement: {satisfaction_improvement:.1f}%")
    
    return {
        'wage_improvement': wage_improvement,
        'satisfaction_improvement': satisfaction_improvement,
        'current_wage_rmse': current_wage_rmse,
        'current_satisfaction_accuracy': current_satisfaction_accuracy
    }

def save_ensemble_results(wage_results, satisfaction_results, comparison_results):
    """Save all ensemble results"""
    print("\nSaving ensemble results...")
    
    # Combine all results
    final_results = {
        'wage_ensemble_rmse': wage_results['ensemble_rmse'],
        'wage_ensemble_mae': wage_results['ensemble_mae'],
        'wage_ensemble_r2': wage_results['ensemble_r2'],
        'satisfaction_ensemble_accuracy': satisfaction_results['ensemble_accuracy'],
        'satisfaction_ensemble_f1': satisfaction_results['ensemble_f1'],
        'wage_improvement_vs_baseline': comparison_results['wage_improvement'],
        'satisfaction_improvement_vs_baseline': comparison_results['satisfaction_improvement']
    }
    
    # Save to CSV
    results_df = pd.DataFrame([final_results])
    results_df.to_csv('model_results/optimized_ensemble_results.csv', index=False)
    
    print("Results saved to: model_results/optimized_ensemble_results.csv")
    print("Models saved:")
    print("- models/optimized_wage_voting_ensemble.pkl")
    print("- models/optimized_satisfaction_voting_ensemble.pkl")
    
    return final_results

def main():
    """Main execution function"""
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test = load_data()
    
    # Create optimized ensembles
    wage_results = create_optimized_wage_ensemble(X_train, X_test, y_wage_train, y_wage_test)
    satisfaction_results = create_optimized_satisfaction_ensemble(X_train, X_test, y_sat_train, y_sat_test)
    
    # Compare with baseline
    comparison_results = compare_with_baseline(wage_results, satisfaction_results)
    
    # Save results
    final_results = save_ensemble_results(wage_results, satisfaction_results, comparison_results)
    
    total_time = time.time() - start_time
    print(f"\n=== Optimized Ensemble Reconstruction Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final wage RMSE: {final_results['wage_ensemble_rmse']:.2f}")
    print(f"Final satisfaction accuracy: {final_results['satisfaction_ensemble_accuracy']:.4f}")

if __name__ == "__main__":
    main()