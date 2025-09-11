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

print("=== 중형 Stacking 테스트 (30,000개 샘플) ===")

def medium_stacking_test():
    print("데이터 로드 및 30,000개 샘플링...")
    
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
    df_final = df_consecutive[valid_mask].copy()
    
    # 30k 샘플
    df_final = df_final.sample(n=30000, random_state=42)
    
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
    
    # 3-fold, 더 많은 iterations
    print("\n중형 Stacking 앙상블 생성...")
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # 임금 예측 Stacking (3개 모델)
    print("  임금 예측...")
    wage_estimators = [
        ('cb', cb.CatBoostRegressor(iterations=200, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
    ]
    
    wage_stacking = StackingRegressor(
        estimators=wage_estimators,
        final_estimator=Ridge(alpha=0.5, random_state=42),
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
        ('cb', cb.CatBoostClassifier(iterations=200, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
    ]
    
    sat_stacking = StackingClassifier(
        estimators=sat_estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
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
    
    print(f"\n=== 중형 Stacking 결과 (30K 샘플) ===")
    print(f"임금 - RMSE: {wage_rmse:.2f}, MAE: {wage_mae:.2f}, R2: {wage_r2:.4f}")
    print(f"만족도 - 정확도: {sat_acc:.4f}")
    
    # 비교
    baseline_wage_rmse = 115.92
    baseline_sat_acc = 0.694
    current_wage_rmse = 118.89
    current_sat_acc = 0.6716
    quick_wage_rmse = 97.13
    quick_sat_acc = 0.6641
    
    print(f"\n=== 성능 비교 ===")
    print(f"베이스라인 대비:")
    print(f"  임금 RMSE: {baseline_wage_rmse:.2f} -> {wage_rmse:.2f} ({baseline_wage_rmse-wage_rmse:+.2f})")
    print(f"  만족도 정확도: {baseline_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-baseline_sat_acc:+.4f})")
    
    print(f"\n현재 Voting 앙상블 대비:")
    print(f"  임금 RMSE: {current_wage_rmse:.2f} -> {wage_rmse:.2f} ({current_wage_rmse-wage_rmse:+.2f})")
    print(f"  만족도 정확도: {current_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-current_sat_acc:+.4f})")
    
    print(f"\n빠른 Stacking(10K) 대비:")
    print(f"  임금 RMSE: {quick_wage_rmse:.2f} -> {wage_rmse:.2f} ({quick_wage_rmse-wage_rmse:+.2f})")
    print(f"  만족도 정확도: {quick_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-quick_sat_acc:+.4f})")
    
    # 성공 여부 종합 판단
    wage_vs_baseline = wage_rmse < baseline_wage_rmse
    wage_vs_current = wage_rmse < current_wage_rmse
    sat_vs_baseline = sat_acc > baseline_sat_acc
    sat_vs_current = sat_acc > current_sat_acc
    
    print(f"\n=== 최종 평가 ===")
    print(f"베이스라인 대비: 임금 {'개선' if wage_vs_baseline else '유지/하락'}, 만족도 {'개선' if sat_vs_baseline else '유지/하락'}")
    print(f"현재 모델 대비: 임금 {'개선' if wage_vs_current else '유지/하락'}, 만족도 {'개선' if sat_vs_current else '유지/하락'}")
    
    if (wage_vs_baseline or sat_vs_baseline) and (wage_vs_current or sat_vs_current):
        print("\n결론: Stacking 앙상블 성공! 전체 데이터셋 적용 권장")
        recommendation = "SUCCESS"
    elif wage_vs_current or sat_vs_current:
        print("\n결론: 부분 개선, 전체 적용 고려 가능")
        recommendation = "PARTIAL"
    else:
        print("\n결론: 추가 최적화 필요")
        recommendation = "NEEDS_WORK"
    
    # 저장
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(wage_stacking, 'models/medium_stacking_wage.pkl')
    joblib.dump(sat_stacking, 'models/medium_stacking_satisfaction.pkl')
    
    results = {
        'sample_size': 30000,
        'wage_rmse': wage_rmse,
        'wage_mae': wage_mae,
        'wage_r2': wage_r2,
        'sat_accuracy': sat_acc,
        'training_time': wage_time + sat_time,
        'recommendation': recommendation
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('model_results/medium_stacking_results.csv', index=False)
    
    print("중형 테스트 모델 저장 완료!")
    return results

if __name__ == "__main__":
    start_time = time.time()
    results = medium_stacking_test()
    elapsed = (time.time() - start_time) / 60
    print(f"\n총 소요시간: {elapsed:.1f}분")
    print(f"권장사항: {results['recommendation']}")