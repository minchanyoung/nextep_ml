import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== SHAP Analysis for Optimized Models ===")

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
    print(f"Features: {len(feature_cols)}")
    
    return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_cols

def train_optimized_models(X_train, y_wage_train, y_sat_train):
    """Train optimized models for SHAP analysis"""
    print("\nTraining optimized models...")
    
    # Optimized CatBoost for wage prediction
    wage_model = cb.CatBoostRegressor(
        iterations=860,
        learning_rate=0.010426694594004318,
        depth=8,
        l2_leaf_reg=8.653255320409505,
        border_count=216,
        random_seed=42,
        verbose=False
    )
    wage_model.fit(X_train, y_wage_train)
    
    # Optimized XGBoost for satisfaction prediction
    satisfaction_model = xgb.XGBClassifier(
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
    satisfaction_model.fit(X_train, y_sat_train)
    
    print("Models trained successfully!")
    return wage_model, satisfaction_model

def perform_shap_analysis_wage(wage_model, X_test, feature_names, sample_size=1000):
    """Perform SHAP analysis for wage prediction model"""
    print(f"\nPerforming SHAP analysis for wage prediction (sample size: {sample_size})...")
    
    # Sample for analysis (for speed)
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
    else:
        X_sample = X_test
    
    # Create SHAP explainer
    explainer = shap.Explainer(wage_model)
    shap_values = explainer(X_sample)
    
    # Get feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-15:][::-1]
    
    print("Top 15 features for wage prediction:")
    for i, idx in enumerate(top_features_idx):
        feature_name = feature_names[idx]
        importance = feature_importance[idx]
        print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
    
    return shap_values, feature_importance, top_features_idx

def perform_shap_analysis_satisfaction(satisfaction_model, X_test, feature_names, sample_size=1000):
    """Perform SHAP analysis for satisfaction prediction model"""
    print(f"\nPerforming SHAP analysis for satisfaction prediction (sample size: {sample_size})...")
    
    # Sample for analysis (for speed)
    if len(X_test) > sample_size:
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
    else:
        X_sample = X_test
    
    # Create SHAP explainer
    explainer = shap.Explainer(satisfaction_model)
    shap_values = explainer(X_sample)
    
    # For multi-class, get average importance across all classes
    if len(shap_values.values.shape) == 3:
        feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        feature_importance = np.abs(shap_values.values).mean(0)
    
    top_features_idx = np.argsort(feature_importance)[-15:][::-1]
    
    print("Top 15 features for satisfaction prediction:")
    for i, idx in enumerate(top_features_idx):
        feature_name = feature_names[idx]
        importance = feature_importance[idx]
        print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
    
    return shap_values, feature_importance, top_features_idx

def create_shap_visualizations(wage_shap_values, satisfaction_shap_values, feature_names, 
                              wage_top_features, satisfaction_top_features):
    """Create SHAP visualizations"""
    print("\nCreating SHAP visualizations...")
    
    plt.style.use('default')
    
    try:
        # Wage prediction SHAP plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Summary plot for wage
        plt.sca(axes[0, 0])
        shap.summary_plot(wage_shap_values, max_display=10, show=False)
        axes[0, 0].set_title('Wage Prediction - SHAP Summary Plot', fontsize=14, fontweight='bold')
        
        # Bar plot for wage
        plt.sca(axes[0, 1])
        shap.summary_plot(wage_shap_values, plot_type="bar", max_display=10, show=False)
        axes[0, 1].set_title('Wage Prediction - Feature Importance', fontsize=14, fontweight='bold')
        
        # Waterfall plot for wage (first sample)
        plt.sca(axes[1, 0])
        try:
            shap.waterfall_plot(wage_shap_values[0], show=False)
            axes[1, 0].set_title('Wage Prediction - Waterfall Plot (Sample 1)', fontsize=14, fontweight='bold')
        except:
            axes[1, 0].text(0.5, 0.5, 'Waterfall plot not available', ha='center', va='center')
            axes[1, 0].set_title('Wage Prediction - Waterfall Plot', fontsize=14, fontweight='bold')
        
        # Force plot placeholder
        axes[1, 1].text(0.5, 0.5, 'Force plots available\nin Jupyter notebook\nenvironment', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Force Plot Info', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/shap_analysis_wage_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Satisfaction prediction SHAP plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Summary plot for satisfaction
        plt.sca(axes[0, 0])
        shap.summary_plot(satisfaction_shap_values, max_display=10, show=False)
        axes[0, 0].set_title('Satisfaction Prediction - SHAP Summary Plot', fontsize=14, fontweight='bold')
        
        # Bar plot for satisfaction
        plt.sca(axes[0, 1])
        shap.summary_plot(satisfaction_shap_values, plot_type="bar", max_display=10, show=False)
        axes[0, 1].set_title('Satisfaction Prediction - Feature Importance', fontsize=14, fontweight='bold')
        
        # Waterfall plot for satisfaction (first sample)
        plt.sca(axes[1, 0])
        try:
            shap.waterfall_plot(satisfaction_shap_values[0], show=False)
            axes[1, 0].set_title('Satisfaction Prediction - Waterfall Plot (Sample 1)', fontsize=14, fontweight='bold')
        except:
            axes[1, 0].text(0.5, 0.5, 'Waterfall plot not available', ha='center', va='center')
            axes[1, 0].set_title('Satisfaction Prediction - Waterfall Plot', fontsize=14, fontweight='bold')
        
        # Multi-class info
        axes[1, 1].text(0.5, 0.5, 'Multi-class SHAP values\nanalyzed across all classes\n\nDetailed analysis available\nin saved results', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 1].set_title('Multi-class SHAP Info', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/shap_analysis_satisfaction_optimized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("SHAP visualizations saved:")
        print("- visualizations/shap_analysis_wage_optimized.png")
        print("- visualizations/shap_analysis_satisfaction_optimized.png")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        print("SHAP analysis completed, but some visualizations may not be available")

def save_shap_results(wage_feature_importance, satisfaction_feature_importance, 
                     feature_names, wage_top_features, satisfaction_top_features):
    """Save SHAP analysis results"""
    print("\nSaving SHAP analysis results...")
    
    # Wage feature importance
    wage_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': wage_feature_importance
    }).sort_values('importance', ascending=False)
    
    wage_importance_df.to_csv('model_results/shap_wage_feature_importance.csv', index=False)
    
    # Satisfaction feature importance
    satisfaction_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': satisfaction_feature_importance
    }).sort_values('importance', ascending=False)
    
    satisfaction_importance_df.to_csv('model_results/shap_satisfaction_feature_importance.csv', index=False)
    
    # Top features summary
    top_features_summary = {
        'wage_top_10': [feature_names[i] for i in wage_top_features[:10]],
        'satisfaction_top_10': [feature_names[i] for i in satisfaction_top_features[:10]]
    }
    
    # Create summary DataFrame
    max_len = max(len(top_features_summary['wage_top_10']), len(top_features_summary['satisfaction_top_10']))
    
    summary_df = pd.DataFrame({
        'rank': list(range(1, max_len + 1)),
        'wage_prediction_top_features': top_features_summary['wage_top_10'] + [''] * (max_len - len(top_features_summary['wage_top_10'])),
        'satisfaction_prediction_top_features': top_features_summary['satisfaction_top_10'] + [''] * (max_len - len(top_features_summary['satisfaction_top_10']))
    })
    
    summary_df.to_csv('model_results/shap_top_features_summary.csv', index=False)
    
    print("SHAP results saved:")
    print("- model_results/shap_wage_feature_importance.csv")
    print("- model_results/shap_satisfaction_feature_importance.csv")
    print("- model_results/shap_top_features_summary.csv")
    
    return top_features_summary

def main():
    """Main execution function"""
    import time
    start_time = time.time()
    
    # Load data
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_names = load_data()
    
    # Train optimized models
    wage_model, satisfaction_model = train_optimized_models(X_train, y_wage_train, y_sat_train)
    
    # Perform SHAP analysis
    wage_shap_values, wage_importance, wage_top_features = perform_shap_analysis_wage(
        wage_model, X_test, feature_names, sample_size=500
    )
    
    satisfaction_shap_values, satisfaction_importance, satisfaction_top_features = perform_shap_analysis_satisfaction(
        satisfaction_model, X_test, feature_names, sample_size=500
    )
    
    # Create visualizations
    create_shap_visualizations(
        wage_shap_values, satisfaction_shap_values, feature_names,
        wage_top_features, satisfaction_top_features
    )
    
    # Save results
    top_features_summary = save_shap_results(
        wage_importance, satisfaction_importance, feature_names,
        wage_top_features, satisfaction_top_features
    )
    
    total_time = time.time() - start_time
    print(f"\n=== SHAP Analysis Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Top wage prediction feature: {feature_names[wage_top_features[0]]}")
    print(f"Top satisfaction prediction feature: {feature_names[satisfaction_top_features[0]]}")

if __name__ == "__main__":
    main()