import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

print("=== 빠른 Stacking 테스트 ===")

def quick_stacking_test():
    # 데이터 로드 및 극도로 간소화
    print("데이터 로드 및 10,000개 샘플링...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
    df_final = df_consecutive[valid_mask].copy()
    
    # 매우 작은 샘플
    df_final = df_final.sample(n=10000, random_state=42)
    
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"훈련: {len(X_train):,}개, 테스트: {len(X_test):,}개")
    
    # 매우 간단한 Stacking (2-fold, 적은 iterations)
    print("\n간단한 Stacking 앙상블 생성...")
    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    
    # 임금 예측 Stacking
    print("  임금 예측...")
    wage_estimators = [
        ('cb', cb.CatBoostRegressor(iterations=100, learning_rate=0.15, depth=4, verbose=False, random_seed=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=4, verbosity=0, random_state=42))
    ]
    
    wage_stacking = StackingRegressor(
        estimators=wage_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=cv,
        n_jobs=1
    )
    
    start = time.time()
    wage_stacking.fit(X_train, y_wage_train)
    wage_time = time.time() - start
    print(f"    완료 ({wage_time:.1f}초)")
    
    # 만족도 예측 Stacking  
    print("  만족도 예측...")
    sat_estimators = [
        ('cb', cb.CatBoostClassifier(iterations=100, learning_rate=0.15, depth=4, verbose=False, random_seed=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.15, max_depth=4, verbosity=0, random_state=42))
    ]
    
    sat_stacking = StackingClassifier(
        estimators=sat_estimators,
        final_estimator=LogisticRegression(max_iter=300, random_state=42),
        cv=cv,
        n_jobs=1
    )
    
    start = time.time()
    sat_stacking.fit(X_train, y_sat_train)
    sat_time = time.time() - start
    print(f"    완료 ({sat_time:.1f}초)")
    
    # 성능 평가
    print("\n성능 평가...")
    wage_pred = wage_stacking.predict(X_test)
    wage_rmse = np.sqrt(mean_squared_error(y_wage_test, wage_pred))
    wage_mae = mean_absolute_error(y_wage_test, wage_pred)
    wage_r2 = r2_score(y_wage_test, wage_pred)
    
    sat_pred = sat_stacking.predict(X_test)
    sat_acc = accuracy_score(y_sat_test, sat_pred)
    
    print(f"\n=== 빠른 Stacking 결과 ===")
    print(f"샘플: 10,000개 (테스트용)")
    print(f"임금 - RMSE: {wage_rmse:.2f}, MAE: {wage_mae:.2f}, R2: {wage_r2:.4f}")
    print(f"만족도 - 정확도: {sat_acc:.4f}")
    
    # 비교
    current_wage_rmse = 118.89
    current_sat_acc = 0.6716
    
    print(f"\n=== 현재 성능 대비 ===")
    print(f"임금 RMSE: {current_wage_rmse:.2f} -> {wage_rmse:.2f} ({current_wage_rmse-wage_rmse:+.2f})")
    print(f"만족도 정확도: {current_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-current_sat_acc:+.4f})")
    
    wage_improved = wage_rmse < current_wage_rmse
    sat_improved = sat_acc > current_sat_acc
    
    if wage_improved or sat_improved:
        print("\n결론: Stacking 앙상블이 성능 개선을 보여줌!")
        print("-> 전체 데이터셋 적용 권장")
    else:
        print("\n결론: 현재 샘플에서는 개선 미미")
        print("-> 추가 파라미터 튜닝 또는 특성 엔지니어링 필요")
    
    # 간단한 저장
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(wage_stacking, 'models/quick_stacking_wage.pkl')
    joblib.dump(sat_stacking, 'models/quick_stacking_satisfaction.pkl')
    
    results = {
        'sample_size': 10000,
        'wage_rmse': wage_rmse,
        'wage_mae': wage_mae,
        'wage_r2': wage_r2,
        'sat_accuracy': sat_acc,
        'training_time': wage_time + sat_time
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/quick_stacking_results.csv', index=False)
    
    print("빠른 테스트 모델 저장 완료!")

if __name__ == "__main__":
    start_time = time.time()
    quick_stacking_test()
    elapsed = (time.time() - start_time) / 60
    print(f"\n총 소요시간: {elapsed:.1f}분")