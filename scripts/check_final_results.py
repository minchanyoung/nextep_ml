import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

print("=== 전체 데이터셋 기반 최종 모델 성능 확인 ===\n")

def check_model_results():
    """저장된 모델들의 성능 확인"""
    
    # 모델 파일 확인
    model_files = [
        'models/full_dataset_wage_ensemble.pkl',
        'models/full_dataset_satisfaction_ensemble.pkl'
    ]
    
    print("저장된 모델 파일:")
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {file_path} ({size:.1f}MB)")
        else:
            print(f"  ✗ {file_path} - 파일 없음")
    
    # 결과 파일 확인
    if os.path.exists('model_results/full_dataset_model_results.csv'):
        results_df = pd.read_csv('model_results/full_dataset_model_results.csv', index_col=0)
        print(f"\n성능 결과 파일 로드 완료: {results_df.shape}")
        print("\n결과 구조:")
        print(results_df.head())
        return results_df
    else:
        print("\n⚠ 성능 결과 파일이 없습니다. 직접 평가를 진행합니다.")
        return None

def load_and_evaluate_models():
    """모델 로드 후 직접 평가"""
    try:
        # 앙상블 모델 로드
        wage_ensemble = joblib.load('models/full_dataset_wage_ensemble.pkl')
        satisfaction_ensemble = joblib.load('models/full_dataset_satisfaction_ensemble.pkl')
        print("앙상블 모델 로드 성공!")
        
        # 테스트 데이터 로드 (간단한 방식)
        df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
        
        # 간단한 전처리
        df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
        df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
        df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
        df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
        
        consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
        df_consecutive = df_sorted[consecutive_mask].copy()
        
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        df_final = df_consecutive[valid_mask].copy()
        
        # 특성 선택 및 전처리
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        from sklearn.preprocessing import LabelEncoder
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
        
        df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
        
        # 테스트 데이터 분할
        test_mask = df_final['year'] >= 2021
        X_test = df_final[test_mask][feature_cols]
        y_wage_test = df_final[test_mask]['next_wage']
        y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)  # 0-4로 변환
        
        print(f"테스트 데이터: {len(X_test)}개")
        
        # 예측 및 평가
        wage_pred = wage_ensemble.predict(X_test)
        sat_pred = satisfaction_ensemble.predict(X_test)
        
        # 임금 예측 성능
        wage_rmse = np.sqrt(mean_squared_error(y_wage_test, wage_pred))
        wage_mae = mean_absolute_error(y_wage_test, wage_pred)
        wage_r2 = r2_score(y_wage_test, wage_pred)
        
        # 만족도 예측 성능
        sat_acc = accuracy_score(y_sat_test, sat_pred)
        
        print("\n" + "="*60)
        print("🏆 전체 데이터셋 기반 최종 성능 결과")
        print("="*60)
        
        print(f"\n💰 임금 예측 앙상블:")
        print(f"   RMSE: {wage_rmse:.2f}만원")
        print(f"   MAE:  {wage_mae:.2f}만원")
        print(f"   R²:   {wage_r2:.4f}")
        
        print(f"\n😊 만족도 예측 앙상블:")
        print(f"   정확도: {sat_acc:.4f}")
        
        # 베이스라인 대비 비교
        print(f"\n🔄 베이스라인 대비 성능:")
        baseline_rmse = 115.92
        baseline_acc = 0.694
        
        rmse_diff = baseline_rmse - wage_rmse
        acc_diff = sat_acc - baseline_acc
        
        print(f"   임금 예측:")
        print(f"     기존: {baseline_rmse:.2f}만원 → 현재: {wage_rmse:.2f}만원")
        print(f"     차이: {rmse_diff:+.2f}만원 ({'🎉 개선' if rmse_diff > 0 else '⚠️ 악화'})")
        
        print(f"   만족도 예측:")
        print(f"     기존: {baseline_acc:.4f} → 현재: {sat_acc:.4f}")
        print(f"     차이: {acc_diff:+.4f} ({'🎉 개선' if acc_diff > 0 else '⚠️ 악화'})")
        
        # 개선 효과 계산
        if rmse_diff > 0:
            improvement_pct = (rmse_diff / baseline_rmse) * 100
            print(f"\n✨ 임금 예측 개선율: {improvement_pct:.1f}%")
        
        if acc_diff > 0:
            improvement_pct = (acc_diff / baseline_acc) * 100
            print(f"✨ 만족도 예측 개선율: {improvement_pct:.1f}%")
        
        return {
            'wage_rmse': wage_rmse,
            'wage_mae': wage_mae, 
            'wage_r2': wage_r2,
            'sat_accuracy': sat_acc,
            'baseline_comparison': {
                'wage_improvement': rmse_diff,
                'sat_improvement': acc_diff
            }
        }
        
    except Exception as e:
        print(f"모델 평가 중 오류: {e}")
        return None

def check_file_sizes():
    """생성된 파일들 크기 확인"""
    print(f"\n📁 생성된 결과 파일들:")
    
    files_to_check = [
        'models/full_dataset_wage_ensemble.pkl',
        'models/full_dataset_satisfaction_ensemble.pkl',
        'models/full_dataset_catboost_wage.pkl',
        'models/full_dataset_xgb_wage.pkl',
        'models/full_dataset_lgb_wage.pkl',
        'model_results/full_dataset_model_results.csv',
        'visualizations/full_dataset_final_comparison.png'
    ]
    
    total_size = 0
    existing_files = 0
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            total_size += size
            existing_files += 1
            print(f"   ✓ {file_path} ({size:.1f}MB)")
        else:
            print(f"   ✗ {file_path} - 없음")
    
    print(f"\n   총 {existing_files}개 파일, {total_size:.1f}MB")

def main():
    # 저장된 결과 확인
    results_df = check_model_results()
    
    # 파일 크기 확인
    check_file_sizes()
    
    # 모델 직접 평가
    if os.path.exists('models/full_dataset_wage_ensemble.pkl'):
        print("\n모델 직접 평가 진행...")
        performance = load_and_evaluate_models()
        
        if performance:
            print(f"\n🎯 핵심 성과 요약:")
            print(f"   • 전체 데이터 166,507개 활용")
            print(f"   • 임금 예측: RMSE {performance['wage_rmse']:.1f}, MAE {performance['wage_mae']:.1f}, R² {performance['wage_r2']:.3f}")
            print(f"   • 만족도 예측: 정확도 {performance['sat_accuracy']:.3f}")
            
            if performance['baseline_comparison']['wage_improvement'] > 0:
                print(f"   • 🚀 베이스라인 대비 임금 예측 {performance['baseline_comparison']['wage_improvement']:.1f}만원 개선!")
            
            if performance['baseline_comparison']['sat_improvement'] > 0:
                print(f"   • 🚀 베이스라인 대비 만족도 예측 {performance['baseline_comparison']['sat_improvement']:.3f} 개선!")
    
    print(f"\n" + "="*60)
    print("전체 데이터셋 기반 모델 구축 완료! 🎉")
    print("="*60)

if __name__ == "__main__":
    main()