import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Advanced Performance Improvement Analysis ===")

def load_and_engineer_features():
    """Load data and create advanced engineered features"""
    print("Loading data and engineering advanced features...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Basic preprocessing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"Base dataset size: {df_final.shape}")
    
    # ADVANCED FEATURE ENGINEERING
    print("Creating advanced features...")
    
    # 1. Personal trajectory features (individual growth patterns)
    df_final = df_final.sort_values(['pid', 'year'])
    
    # Wage growth trends
    df_final['wage_growth_1yr'] = df_final.groupby('pid')['p_wage'].pct_change()
    df_final['wage_growth_2yr'] = df_final.groupby('pid')['p_wage'].pct_change(periods=2)
    df_final['wage_volatility'] = df_final.groupby('pid')['p_wage'].rolling(window=3, min_periods=2).std().reset_index(drop=True)
    
    # Career progression indicators
    df_final['job_changes'] = df_final.groupby('pid')['occupation_code'].apply(lambda x: (x != x.shift()).cumsum()).reset_index(drop=True)
    df_final['years_in_current_job'] = df_final.groupby(['pid', 'occupation_code']).cumcount() + 1
    df_final['career_advancement'] = df_final.groupby('pid')['p_edu'].apply(lambda x: (x > x.shift()).cumsum()).reset_index(drop=True)
    
    # Satisfaction trends
    df_final['satisfaction_trend'] = df_final.groupby('pid')['p4321'].apply(lambda x: x.diff()).reset_index(drop=True)
    df_final['satisfaction_volatility'] = df_final.groupby('pid')['p4321'].rolling(window=3, min_periods=2).std().reset_index(drop=True)
    
    # 2. Interaction features based on SHAP importance
    # Top important features from SHAP analysis
    df_final['wage_age_interaction'] = df_final['wage_vs_occupation_avg'] * df_final['p_age']
    df_final['edu_occupation_interaction'] = df_final['p_edu'] * df_final['occupation_avg_wage']
    df_final['sex_wage_gap'] = df_final['p_sex'] * df_final['wage_quartile_in_occupation']
    df_final['age_experience_interaction'] = df_final['p_age'] * df_final['years_in_current_job']
    
    # 3. Time-based features
    df_final['economic_cycle'] = np.sin(2 * np.pi * (df_final['year'] - 2000) / 10)  # 10-year economic cycle
    df_final['career_stage'] = pd.cut(df_final['p_age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3])
    df_final['career_stage'] = df_final['career_stage'].astype(float)
    
    # 4. Relative performance features
    # Individual vs cohort comparisons
    cohort_groups = df_final.groupby(['year', 'p_edu', 'p_sex'])
    df_final['wage_vs_cohort'] = df_final['p_wage'] / cohort_groups['p_wage'].transform('mean')
    df_final['satisfaction_vs_cohort'] = df_final['p4321'] / cohort_groups['p4321'].transform('mean')
    
    # 5. Lag features (using past values as predictors)
    for lag in [1, 2]:
        df_final[f'wage_lag_{lag}'] = df_final.groupby('pid')['p_wage'].shift(lag)
        df_final[f'satisfaction_lag_{lag}'] = df_final.groupby('pid')['p4321'].shift(lag)
    
    print(f"Enhanced dataset size: {df_final.shape}")
    
    # Encode categorical variables
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols and not col.startswith('Unnamed')]
    
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # Handle missing values (advanced imputation)
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # Remove infinite values
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    df_final = df_final.fillna(df_final.median(numeric_only=True))
    
    # Time-based split
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Final training features: {X_train.shape[1]}")
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_cols

def implement_stacking_ensemble(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test):
    """Implement Stacking ensemble for better performance"""
    print("\n=== Implementing Stacking Ensemble ===")
    
    # Define base models for wage prediction
    wage_base_models = [
        ('catboost', cb.CatBoostRegressor(
            iterations=860, learning_rate=0.01, depth=8, l2_leaf_reg=8.65,
            random_seed=42, verbose=False
        )),
        ('xgb', xgb.XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.08,
            random_state=42, verbosity=0
        )),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.08,
            random_state=42, verbosity=-1
        ))
    ]
    
    # Define base models for satisfaction prediction
    satisfaction_base_models = [
        ('xgb', xgb.XGBClassifier(
            n_estimators=1151, max_depth=4, learning_rate=0.102,
            random_state=42, objective='multi:softprob', verbosity=0
        )),
        ('catboost', cb.CatBoostClassifier(
            iterations=800, learning_rate=0.08, depth=6,
            random_seed=42, verbose=False
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=800, max_depth=6, learning_rate=0.08,
            random_state=42, verbosity=-1
        ))
    ]
    
    # Stacking for wage prediction
    print("Training Stacking ensemble for wage prediction...")
    wage_stacking = StackingRegressor(
        estimators=wage_base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1
    )
    
    wage_stacking.fit(X_train, y_wage_train)
    wage_pred_stacking = wage_stacking.predict(X_test)
    
    wage_rmse_stacking = np.sqrt(mean_squared_error(y_wage_test, wage_pred_stacking))
    wage_r2_stacking = r2_score(y_wage_test, wage_pred_stacking)
    
    print(f"Wage Stacking - RMSE: {wage_rmse_stacking:.2f}, R²: {wage_r2_stacking:.4f}")
    
    # Stacking for satisfaction prediction
    print("Training Stacking ensemble for satisfaction prediction...")
    satisfaction_stacking = StackingClassifier(
        estimators=satisfaction_base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1
    )
    
    satisfaction_stacking.fit(X_train, y_sat_train)
    satisfaction_pred_stacking = satisfaction_stacking.predict(X_test)
    
    satisfaction_acc_stacking = accuracy_score(y_sat_test, satisfaction_pred_stacking)
    satisfaction_f1_stacking = f1_score(y_sat_test, satisfaction_pred_stacking, average='weighted')
    
    print(f"Satisfaction Stacking - Accuracy: {satisfaction_acc_stacking:.4f}, F1: {satisfaction_f1_stacking:.4f}")
    
    return {
        'wage_model': wage_stacking,
        'satisfaction_model': satisfaction_stacking,
        'wage_rmse': wage_rmse_stacking,
        'wage_r2': wage_r2_stacking,
        'satisfaction_accuracy': satisfaction_acc_stacking,
        'satisfaction_f1': satisfaction_f1_stacking
    }

def feature_selection_optimization(X_train, X_test, y_wage_train, y_wage_test):
    """Optimize feature selection for better performance"""
    print("\n=== Feature Selection Optimization ===")
    
    # Train a model to get feature importance
    model = cb.CatBoostRegressor(iterations=500, random_seed=42, verbose=False)
    model.fit(X_train, y_wage_train)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Test different numbers of top features
    top_features_counts = [20, 30, 40, 50, len(feature_names)]
    best_performance = float('inf')
    best_n_features = len(feature_names)
    
    print("Testing different numbers of features:")
    for n_features in top_features_counts:
        if n_features >= len(feature_names):
            selected_features = feature_names
        else:
            top_indices = np.argsort(feature_importance)[-n_features:]
            selected_features = feature_names[top_indices]
        
        # Train model with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        test_model = cb.CatBoostRegressor(iterations=500, random_seed=42, verbose=False)
        test_model.fit(X_train_selected, y_wage_train)
        pred = test_model.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_wage_test, pred))
        
        print(f"  {n_features:2d} features: RMSE = {rmse:.2f}")
        
        if rmse < best_performance:
            best_performance = rmse
            best_n_features = n_features
    
    print(f"Best performance with {best_n_features} features: RMSE = {best_performance:.2f}")
    
    return best_n_features

def compare_all_approaches(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test):
    """Compare all improvement approaches"""
    print("\n=== Comprehensive Performance Comparison ===")
    
    results = {}
    
    # 1. Baseline (current best)
    print("1. Testing baseline model...")
    baseline_wage = cb.CatBoostRegressor(
        iterations=860, learning_rate=0.01, depth=8, l2_leaf_reg=8.65,
        random_seed=42, verbose=False
    )
    baseline_wage.fit(X_train, y_wage_train)
    baseline_wage_pred = baseline_wage.predict(X_test)
    baseline_wage_rmse = np.sqrt(mean_squared_error(y_wage_test, baseline_wage_pred))
    
    baseline_sat = xgb.XGBClassifier(
        n_estimators=1151, max_depth=4, learning_rate=0.102,
        random_state=42, objective='multi:softprob', verbosity=0
    )
    baseline_sat.fit(X_train, y_sat_train)
    baseline_sat_pred = baseline_sat.predict(X_test)
    baseline_sat_acc = accuracy_score(y_sat_test, baseline_sat_pred)
    
    results['Baseline'] = {
        'wage_rmse': baseline_wage_rmse,
        'satisfaction_accuracy': baseline_sat_acc
    }
    
    # 2. Stacking ensemble
    print("2. Testing Stacking ensemble...")
    stacking_results = implement_stacking_ensemble(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test)
    results['Stacking'] = {
        'wage_rmse': stacking_results['wage_rmse'],
        'satisfaction_accuracy': stacking_results['satisfaction_accuracy']
    }
    
    # 3. Extended hyperparameter optimization
    print("3. Testing extended hyperparameter optimization...")
    extended_wage = cb.CatBoostRegressor(
        iterations=1500,  # More iterations
        learning_rate=0.005,  # Lower learning rate
        depth=10,  # Deeper trees
        l2_leaf_reg=10.0,  # Stronger regularization
        random_seed=42,
        verbose=False
    )
    extended_wage.fit(X_train, y_wage_train)
    extended_wage_pred = extended_wage.predict(X_test)
    extended_wage_rmse = np.sqrt(mean_squared_error(y_wage_test, extended_wage_pred))
    
    results['Extended_Hyperparams'] = {
        'wage_rmse': extended_wage_rmse,
        'satisfaction_accuracy': baseline_sat_acc  # Use same for comparison
    }
    
    # Display results
    print("\n=== Results Summary ===")
    for method, metrics in results.items():
        wage_improvement = ((results['Baseline']['wage_rmse'] - metrics['wage_rmse']) / results['Baseline']['wage_rmse']) * 100
        sat_improvement = ((metrics['satisfaction_accuracy'] - results['Baseline']['satisfaction_accuracy']) / results['Baseline']['satisfaction_accuracy']) * 100
        
        print(f"{method:20s}: Wage RMSE = {metrics['wage_rmse']:.2f} ({wage_improvement:+.1f}%), "
              f"Satisfaction Acc = {metrics['satisfaction_accuracy']:.4f} ({sat_improvement:+.1f}%)")
    
    # Find best method
    best_wage_method = min(results.keys(), key=lambda k: results[k]['wage_rmse'])
    best_sat_method = max(results.keys(), key=lambda k: results[k]['satisfaction_accuracy'])
    
    print(f"\nBest wage prediction: {best_wage_method}")
    print(f"Best satisfaction prediction: {best_sat_method}")
    
    return results, stacking_results

def save_improvement_results(results, stacking_models):
    """Save improvement results and models"""
    print("\nSaving improvement results...")
    
    # Save best stacking models
    joblib.dump(stacking_models['wage_model'], 'models/advanced_stacking_wage.pkl')
    joblib.dump(stacking_models['satisfaction_model'], 'models/advanced_stacking_satisfaction.pkl')
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results/advanced_improvement_results.csv')
    
    print("Advanced models and results saved:")
    print("- models/advanced_stacking_wage.pkl")
    print("- models/advanced_stacking_satisfaction.pkl")
    print("- model_results/advanced_improvement_results.csv")
    
    return results_df

def main():
    """Main execution function"""
    import time
    start_time = time.time()
    
    # Load data with advanced feature engineering
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_cols = load_and_engineer_features()
    
    # Optimize feature selection
    best_n_features = feature_selection_optimization(X_train, X_test, y_wage_train, y_wage_test)
    
    # Compare all approaches
    results, stacking_models = compare_all_approaches(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test)
    
    # Save results
    results_df = save_improvement_results(results, stacking_models)
    
    total_time = time.time() - start_time
    print(f"\n=== Advanced Performance Improvement Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best wage RMSE: {min(r['wage_rmse'] for r in results.values()):.2f}")
    print(f"Best satisfaction accuracy: {max(r['satisfaction_accuracy'] for r in results.values()):.4f}")

if __name__ == "__main__":
    main()