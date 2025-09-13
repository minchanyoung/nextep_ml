import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import xgboost as xgb

print("=== Quick Performance Improvement Analysis ===")

def analyze_current_performance():
    """Analyze current model performance and identify improvement opportunities"""
    
    # Read SHAP results to understand feature importance
    try:
        wage_importance = pd.read_csv('model_results/shap_wage_feature_importance.csv')
        satisfaction_importance = pd.read_csv('model_results/shap_satisfaction_feature_importance.csv')
        
        print("Top 5 wage prediction features:")
        for i, row in wage_importance.head().iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        print("\nTop 5 satisfaction prediction features:")
        for i, row in satisfaction_importance.head().iterrows():
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
    except FileNotFoundError:
        print("SHAP results not found, proceeding with general analysis...")

def quick_feature_engineering():
    """Quick feature engineering for immediate improvements"""
    print("\nTesting quick feature engineering improvements...")
    
    # Load data
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # Basic preprocessing
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    # Filter valid cases
    valid_mask = (~df_sorted['next_wage'].isna()) & (~df_sorted['next_satisfaction'].isna()) & (df_sorted['next_satisfaction'] > 0)
    df_final = df_sorted[valid_mask].copy()
    
    print(f"Working with {df_final.shape[0]} samples")
    
    # QUICK FEATURE ENGINEERING IDEAS
    
    # 1. Wage momentum features
    df_final['wage_change_1yr'] = df_final.groupby('pid')['p_wage'].pct_change()
    df_final['wage_trend_2yr'] = df_final.groupby('pid')['p_wage'].pct_change(periods=2)
    
    # 2. Career stability features  
    df_final['job_tenure'] = df_final.groupby(['pid', 'occupation_code']).cumcount() + 1
    df_final['total_job_changes'] = df_final.groupby('pid')['occupation_code'].apply(lambda x: (x != x.shift()).cumsum()).reset_index(drop=True)
    
    # 3. Relative performance features
    df_final['wage_vs_age_peer'] = df_final['p_wage'] / df_final.groupby(['year', 'p_age'])['p_wage'].transform('mean')
    df_final['satisfaction_vs_age_peer'] = df_final['p4321'] / df_final.groupby(['year', 'p_age'])['p4321'].transform('mean')
    
    # 4. Interaction features (based on SHAP top features)
    df_final['wage_age_interaction'] = df_final['wage_vs_occupation_avg'] * df_final['p_age']
    df_final['education_wage_interaction'] = df_final['p_edu'] * df_final['occupation_avg_wage']
    
    # 5. Time-based features
    df_final['career_phase'] = pd.cut(df_final['p_age'], bins=[0, 30, 45, 60, 100], labels=[1, 2, 3, 4]).astype(float)
    df_final['years_since_2000'] = df_final['year'] - 2000
    
    # Select features
    exclude_cols = ['pid', 'year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols and not col.startswith('Unnamed')]
    
    # Handle categorical variables
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # Fill missing values
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # Split data
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"Enhanced features: {len(feature_cols)} (added {len(feature_cols) - 40} new features)")
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test

def test_improved_models(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test):
    """Test models with improved features and parameters"""
    print("\n=== Testing Improved Models ===")
    
    results = {}
    
    # 1. Baseline (current best)
    print("1. Baseline model...")
    baseline_wage = cb.CatBoostRegressor(
        iterations=860, learning_rate=0.01, depth=8, l2_leaf_reg=8.65, 
        random_seed=42, verbose=False
    )
    baseline_wage.fit(X_train, y_wage_train)
    baseline_pred = baseline_wage.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_wage_test, baseline_pred))
    baseline_r2 = r2_score(y_wage_test, baseline_pred)
    
    results['Baseline'] = {'rmse': baseline_rmse, 'r2': baseline_r2}
    print(f"   Baseline - RMSE: {baseline_rmse:.2f}, R²: {baseline_r2:.4f}")
    
    # 2. Enhanced parameters model
    print("2. Enhanced parameters model...")
    enhanced_wage = cb.CatBoostRegressor(
        iterations=1200,      # More iterations
        learning_rate=0.008,  # Lower learning rate for fine-tuning
        depth=10,             # Deeper trees
        l2_leaf_reg=12.0,     # Stronger regularization
        border_count=256,     # More border count
        random_seed=42,
        verbose=False
    )
    enhanced_wage.fit(X_train, y_wage_train)
    enhanced_pred = enhanced_wage.predict(X_test)
    enhanced_rmse = np.sqrt(mean_squared_error(y_wage_test, enhanced_pred))
    enhanced_r2 = r2_score(y_wage_test, enhanced_pred)
    
    results['Enhanced_Params'] = {'rmse': enhanced_rmse, 'r2': enhanced_r2}
    print(f"   Enhanced - RMSE: {enhanced_rmse:.2f}, R²: {enhanced_r2:.4f}")
    
    # 3. Feature selection model (top important features only)
    print("3. Feature selection model...")
    # Train a quick model to get feature importance
    temp_model = cb.CatBoostRegressor(iterations=200, random_seed=42, verbose=False)
    temp_model.fit(X_train, y_wage_train)
    
    # Select top 30 features
    feature_importance = temp_model.feature_importances_
    top_30_indices = np.argsort(feature_importance)[-30:]
    selected_features = X_train.columns[top_30_indices]
    
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    selected_wage = cb.CatBoostRegressor(
        iterations=1000, learning_rate=0.01, depth=8, l2_leaf_reg=8.65,
        random_seed=42, verbose=False
    )
    selected_wage.fit(X_train_selected, y_wage_train)
    selected_pred = selected_wage.predict(X_test_selected)
    selected_rmse = np.sqrt(mean_squared_error(y_wage_test, selected_pred))
    selected_r2 = r2_score(y_wage_test, selected_pred)
    
    results['Feature_Selection'] = {'rmse': selected_rmse, 'r2': selected_r2}
    print(f"   Selected - RMSE: {selected_rmse:.2f}, R²: {selected_r2:.4f}")
    
    # 4. Simple ensemble (average of top 2 models)
    print("4. Simple ensemble...")
    ensemble_pred = (enhanced_pred + selected_pred) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_wage_test, ensemble_pred))
    ensemble_r2 = r2_score(y_wage_test, ensemble_pred)
    
    results['Simple_Ensemble'] = {'rmse': ensemble_rmse, 'r2': ensemble_r2}
    print(f"   Ensemble - RMSE: {ensemble_rmse:.2f}, R²: {ensemble_r2:.4f}")
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
    best_rmse = results[best_model]['rmse']
    
    print(f"\nBest model: {best_model}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Improvement over baseline: {((baseline_rmse - best_rmse) / baseline_rmse * 100):+.1f}%")
    
    return results, best_model

def test_satisfaction_improvements(X_train, X_test, y_sat_train, y_sat_test):
    """Test satisfaction prediction improvements"""
    print("\n=== Testing Satisfaction Prediction Improvements ===")
    
    results = {}
    
    # 1. Baseline
    baseline_sat = xgb.XGBClassifier(
        n_estimators=1151, max_depth=4, learning_rate=0.102,
        random_state=42, objective='multi:softprob', verbosity=0
    )
    baseline_sat.fit(X_train, y_sat_train)
    baseline_pred = baseline_sat.predict(X_test)
    baseline_acc = accuracy_score(y_sat_test, baseline_pred)
    baseline_f1 = f1_score(y_sat_test, baseline_pred, average='weighted')
    
    results['Baseline'] = {'accuracy': baseline_acc, 'f1': baseline_f1}
    print(f"Baseline - Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}")
    
    # 2. Enhanced XGBoost
    enhanced_sat = xgb.XGBClassifier(
        n_estimators=1500, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=3.0, reg_lambda=10.0,
        random_state=42, objective='multi:softprob', verbosity=0
    )
    enhanced_sat.fit(X_train, y_sat_train)
    enhanced_pred = enhanced_sat.predict(X_test)
    enhanced_acc = accuracy_score(y_sat_test, enhanced_pred)
    enhanced_f1 = f1_score(y_sat_test, enhanced_pred, average='weighted')
    
    results['Enhanced_XGBoost'] = {'accuracy': enhanced_acc, 'f1': enhanced_f1}
    print(f"Enhanced - Accuracy: {enhanced_acc:.4f}, F1: {enhanced_f1:.4f}")
    
    # 3. CatBoost for satisfaction
    catboost_sat = cb.CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=8,
        random_seed=42, verbose=False
    )
    catboost_sat.fit(X_train, y_sat_train)
    catboost_pred = catboost_sat.predict(X_test)
    catboost_acc = accuracy_score(y_sat_test, catboost_pred)
    catboost_f1 = f1_score(y_sat_test, catboost_pred, average='weighted')
    
    results['CatBoost'] = {'accuracy': catboost_acc, 'f1': catboost_f1}
    print(f"CatBoost - Accuracy: {catboost_acc:.4f}, F1: {catboost_f1:.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_acc = results[best_model]['accuracy']
    
    print(f"\nBest satisfaction model: {best_model}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Improvement over baseline: {((best_acc - baseline_acc) / baseline_acc * 100):+.1f}%")
    
    return results, best_model

def generate_improvement_recommendations():
    """Generate actionable improvement recommendations"""
    print("\n=== PERFORMANCE IMPROVEMENT RECOMMENDATIONS ===")
    
    recommendations = {
        "즉시 적용 가능한 개선사항": [
            "1. 고급 특성 엔지니어링: 개인별 임금 증가율, 직업 안정성 지표 추가",
            "2. 향상된 하이퍼파라미터: 더 깊은 트리, 낮은 학습률, 강한 정규화",
            "3. 특성 선택 최적화: 상위 30개 중요 특성만 사용하여 과적합 방지",
            "4. 간단한 앙상블: 최고 성능 모델들의 평균으로 안정성 향상"
        ],
        
        "중기 개선 방안 (1-2주)": [
            "1. Stacking 앙상블: Ridge/LogisticRegression 메타모델로 개별 모델 조합",
            "2. 시계열 특화 특성: 개인별 경력 궤적, 산업 트렌드 반영 특성",
            "3. 고차원 상호작용: 중요 특성들 간의 2차, 3차 상호작용 특성",
            "4. 앙상블 다양성 증대: Neural Network, Random Forest 등 추가 모델"
        ],
        
        "장기 개선 방안 (1개월+)": [
            "1. 딥러닝 모델: TabNet, Neural ODEs 등 테이블 데이터 특화 신경망",
            "2. 외부 데이터 통합: 경제 지표, 산업별 트렌드 데이터 추가",
            "3. 개인화 모델: 개인별 특성을 반영한 맞춤형 예측 모델",
            "4. 시계열 전문 모델: LSTM, Transformer 기반 시퀀스 모델"
        ],
        
        "예상 성능 개선": [
            "• 임금 예측 RMSE: 현재 116.42 → 목표 110-114만원 (2-5% 개선)",
            "• 만족도 예측 정확도: 현재 67.53% → 목표 70-72% (3-6% 개선)",
            "• 전체적으로 5-10% 성능 향상 가능성 높음"
        ]
    }
    
    for category, items in recommendations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    return recommendations

def main():
    """Main function"""
    import time
    start_time = time.time()
    
    # Analyze current performance
    analyze_current_performance()
    
    # Quick feature engineering
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test = quick_feature_engineering()
    
    # Test improved models
    wage_results, best_wage_model = test_improved_models(X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test)
    satisfaction_results, best_sat_model = test_satisfaction_improvements(X_train, X_test, y_sat_train, y_sat_test)
    
    # Generate recommendations
    recommendations = generate_improvement_recommendations()
    
    # Save results
    all_results = {
        'wage_models': wage_results,
        'satisfaction_models': satisfaction_results,
        'best_wage_model': best_wage_model,
        'best_satisfaction_model': best_sat_model
    }
    
    results_df = pd.DataFrame(wage_results).T
    results_df.to_csv('model_results/quick_improvement_analysis.csv')
    
    total_time = time.time() - start_time
    print(f"\n=== Quick Improvement Analysis Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("Results saved to: model_results/quick_improvement_analysis.csv")

if __name__ == "__main__":
    main()