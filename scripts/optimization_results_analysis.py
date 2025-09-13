import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import catboost as cb
import matplotlib.pyplot as plt

print("=== 하이퍼파라미터 최적화 결과 분석 ===")

def load_data():
    """데이터 로드 및 전처리"""
    df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
    
    # 예측 가능한 케이스만 선별
    df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
    
    # 다음 연도 타겟 생성
    df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
    df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
    df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
    
    # 연속된 연도만 필터링
    consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
    df_consecutive = df_sorted[consecutive_mask].copy()
    
    # 타겟이 모두 있는 케이스만 선별
    valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna())
    df_final = df_consecutive[valid_mask].copy()
    
    # 만족도 타겟 변수의 유효한 값(0 초과)만 필터링
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    # 특성 선택
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # 범주형 변수 인코딩
    from sklearn.preprocessing import LabelEncoder
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # 결측값 처리
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # 시간 기반 분할
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    
    return X_train, X_test, y_wage_train, y_wage_test

def test_optimized_parameters():
    """최적화된 파라미터로 모델 테스트"""
    X_train, X_test, y_wage_train, y_wage_test = load_data()
    
    # 최적화된 파라미터 (Trial 15 결과)
    optimized_params = {
        'iterations': 860,
        'learning_rate': 0.010426694594004318,
        'depth': 8,
        'l2_leaf_reg': 8.653255320409505,
        'border_count': 216,
        'random_seed': 42,
        'verbose': False
    }
    
    # 기존 베이스라인 파라미터
    baseline_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'random_seed': 42,
        'verbose': False
    }
    
    print("모델 훈련 및 비교 중...")
    
    # 최적화된 모델
    optimized_model = cb.CatBoostRegressor(**optimized_params)
    optimized_model.fit(X_train, y_wage_train)
    
    # 베이스라인 모델
    baseline_model = cb.CatBoostRegressor(**baseline_params)
    baseline_model.fit(X_train, y_wage_train)
    
    # 예측
    opt_pred_train = optimized_model.predict(X_train)
    opt_pred_test = optimized_model.predict(X_test)
    
    base_pred_train = baseline_model.predict(X_train)
    base_pred_test = baseline_model.predict(X_test)
    
    # 성능 계산
    results = {
        'optimized': {
            'train_rmse': np.sqrt(mean_squared_error(y_wage_train, opt_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_wage_test, opt_pred_test)),
            'train_r2': r2_score(y_wage_train, opt_pred_train),
            'test_r2': r2_score(y_wage_test, opt_pred_test),
            'train_mae': mean_absolute_error(y_wage_train, opt_pred_train),
            'test_mae': mean_absolute_error(y_wage_test, opt_pred_test)
        },
        'baseline': {
            'train_rmse': np.sqrt(mean_squared_error(y_wage_train, base_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_wage_test, base_pred_test)),
            'train_r2': r2_score(y_wage_train, base_pred_train),
            'test_r2': r2_score(y_wage_test, base_pred_test),
            'train_mae': mean_absolute_error(y_wage_train, base_pred_train),
            'test_mae': mean_absolute_error(y_wage_test, base_pred_test)
        }
    }
    
    return results, optimized_model, baseline_model

def print_results(results):
    """결과 출력"""
    print("\n=== 하이퍼파라미터 최적화 성능 비교 ===\n")
    
    print("📊 **임금 예측 모델 성능 비교**")
    print("=" * 50)
    
    opt = results['optimized']
    base = results['baseline']
    
    print(f"🔹 **최적화된 모델**")
    print(f"   - 테스트 RMSE: {opt['test_rmse']:.2f}만원")
    print(f"   - 테스트 MAE: {opt['test_mae']:.2f}만원")
    print(f"   - 테스트 R²: {opt['test_r2']:.4f}")
    print()
    
    print(f"🔸 **기존 베이스라인**")
    print(f"   - 테스트 RMSE: {base['test_rmse']:.2f}만원")
    print(f"   - 테스트 MAE: {base['test_mae']:.2f}만원")
    print(f"   - 테스트 R²: {base['test_r2']:.4f}")
    print()
    
    # 개선율 계산
    rmse_improvement = ((base['test_rmse'] - opt['test_rmse']) / base['test_rmse']) * 100
    mae_improvement = ((base['test_mae'] - opt['test_mae']) / base['test_mae']) * 100
    r2_improvement = ((opt['test_r2'] - base['test_r2']) / base['test_r2']) * 100
    
    print(f"📈 **성능 개선**")
    print(f"   - RMSE 개선: {rmse_improvement:.1f}% (더 낮을수록 좋음)")
    print(f"   - MAE 개선: {mae_improvement:.1f}% (더 낮을수록 좋음)")
    print(f"   - R² 개선: {r2_improvement:.1f}% (더 높을수록 좋음)")
    
    return {
        'rmse_improvement': rmse_improvement,
        'mae_improvement': mae_improvement,
        'r2_improvement': r2_improvement
    }

def save_results(results, improvements):
    """결과 저장"""
    # CSV 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results/hyperparameter_optimization_comparison.csv')
    
    # 개선율 저장
    improvements_df = pd.DataFrame([improvements])
    improvements_df.to_csv('model_results/optimization_improvements.csv', index=False)
    
    print(f"\n💾 결과 저장 완료")
    print(f"   - model_results/hyperparameter_optimization_comparison.csv")
    print(f"   - model_results/optimization_improvements.csv")

def main():
    """메인 실행 함수"""
    print("하이퍼파라미터 최적화 결과 분석 시작...")
    
    # 모델 비교
    results, opt_model, base_model = test_optimized_parameters()
    
    # 결과 출력
    improvements = print_results(results)
    
    # 결과 저장
    save_results(results, improvements)
    
    # 모델 저장
    import joblib
    joblib.dump(opt_model, 'models/optimized_catboost_wage.pkl')
    joblib.dump(base_model, 'models/baseline_catboost_wage.pkl')
    
    print(f"\n🎯 **최종 결론**")
    if improvements['rmse_improvement'] > 0:
        print(f"   ✅ 하이퍼파라미터 최적화가 성공적으로 모델 성능을 개선했습니다!")
        print(f"   📈 RMSE {improvements['rmse_improvement']:.1f}% 개선으로 예측 정확도가 향상되었습니다.")
    else:
        print(f"   ⚠️ 최적화 결과가 베이스라인과 유사하거나 약간 낮습니다.")
        print(f"   🔍 추가적인 특성 엔지니어링이나 다른 접근이 필요할 수 있습니다.")

if __name__ == "__main__":
    main()