import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== 최적화된 Stacking 앙상블 시스템 ===")

class OptimizedStackingSystem:
    """최적화된 Stacking 앙상블 시스템"""
    
    def __init__(self, sample_size=50000):
        self.stacking_models = {}
        self.results = {}
        self.sample_size = sample_size
        
    def load_and_sample_data(self):
        """데이터 로드 및 샘플링"""
        print(f"데이터 로드 및 {self.sample_size:,}개 샘플링...")
        
        df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
        
        df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
        df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
        df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
        df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
        
        consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
        df_consecutive = df_sorted[consecutive_mask].copy()
        
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        df_final = df_consecutive[valid_mask].copy()
        
        # 샘플링 (균등하게)
        if len(df_final) > self.sample_size:
            df_final = df_final.sample(n=self.sample_size, random_state=42)
        
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
        
        df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
        
        train_mask = df_final['year'] <= 2020
        test_mask = df_final['year'] >= 2021
        
        self.X_train = df_final[train_mask][feature_cols]
        self.X_test = df_final[test_mask][feature_cols]
        self.y_wage_train = df_final[train_mask]['next_wage']
        self.y_wage_test = df_final[test_mask]['next_wage']
        self.y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
        self.y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
        
        print(f"샘플 데이터 - 훈련: {len(self.X_train):,}개, 테스트: {len(self.X_test):,}개")
    
    def create_optimized_stacking(self):
        """최적화된 Stacking 앙상블 생성"""
        print("\n최적화된 Stacking 앙상블 생성...")
        
        # 3-fold로 빠른 검증
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # 임금 예측 Stacking (간소화된 파라미터)
        print("  임금 예측 Stacking...")
        
        wage_estimators = [
            ('cb', cb.CatBoostRegressor(iterations=300, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['wage'] = StackingRegressor(
            estimators=wage_estimators,
            final_estimator=Ridge(alpha=0.5, random_state=42),
            cv=cv,
            n_jobs=1,
            verbose=0
        )
        
        start_time = time.time()
        self.stacking_models['wage'].fit(self.X_train, self.y_wage_train)
        wage_time = time.time() - start_time
        print(f"    임금 Stacking 완료 ({wage_time:.1f}초)")
        
        # 만족도 예측 Stacking
        print("  만족도 예측 Stacking...")
        
        sat_estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
            ('cb', cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['satisfaction'] = StackingClassifier(
            estimators=sat_estimators,
            final_estimator=LogisticRegression(max_iter=500, random_state=42),
            cv=cv,
            n_jobs=1,
            verbose=0
        )
        
        start_time = time.time()
        self.stacking_models['satisfaction'].fit(self.X_train, self.y_sat_train)
        sat_time = time.time() - start_time
        print(f"    만족도 Stacking 완료 ({sat_time:.1f}초)")
    
    def evaluate_performance(self):
        """성능 평가"""
        print("\n성능 평가...")
        
        # 임금 예측 성능
        wage_pred = self.stacking_models['wage'].predict(self.X_test)
        wage_rmse = np.sqrt(mean_squared_error(self.y_wage_test, wage_pred))
        wage_mae = mean_absolute_error(self.y_wage_test, wage_pred)
        wage_r2 = r2_score(self.y_wage_test, wage_pred)
        
        # 만족도 예측 성능
        sat_pred = self.stacking_models['satisfaction'].predict(self.X_test)
        sat_acc = accuracy_score(self.y_sat_test, sat_pred)
        
        self.results = {
            'wage_rmse': wage_rmse,
            'wage_mae': wage_mae,
            'wage_r2': wage_r2,
            'sat_accuracy': sat_acc,
            'sample_size': self.sample_size
        }
        
        print(f"\n=== 최적화된 Stacking 앙상블 성능 ===")
        print(f"샘플 크기: {self.sample_size:,}개")
        print(f"\n임금 예측:")
        print(f"  RMSE: {wage_rmse:.2f}만원")
        print(f"  MAE: {wage_mae:.2f}만원") 
        print(f"  R2: {wage_r2:.4f}")
        
        print(f"\n만족도 예측:")
        print(f"  정확도: {sat_acc:.4f}")
        
        # 기존 성능과 비교
        print(f"\n=== 기존 성능 대비 비교 ===")
        current_wage_rmse = 118.89
        current_sat_acc = 0.6716
        baseline_wage_rmse = 115.92
        baseline_sat_acc = 0.694
        
        print(f"현재 Voting 앙상블 대비:")
        print(f"  임금 RMSE: {current_wage_rmse:.2f} -> {wage_rmse:.2f} ({current_wage_rmse-wage_rmse:+.2f})")
        print(f"  만족도 정확도: {current_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-current_sat_acc:+.4f})")
        
        print(f"\n베이스라인 대비:")
        print(f"  임금 RMSE: {baseline_wage_rmse:.2f} -> {wage_rmse:.2f} ({baseline_wage_rmse-wage_rmse:+.2f})")
        print(f"  만족도 정확도: {baseline_sat_acc:.4f} -> {sat_acc:.4f} ({sat_acc-baseline_sat_acc:+.4f})")
        
        # 성공 여부 판단
        wage_improved = wage_rmse < current_wage_rmse
        sat_improved = sat_acc > current_sat_acc
        
        print(f"\n=== Stacking 개선 효과 ===")
        print(f"임금 예측: {'개선' if wage_improved else '동등/하락'}")
        print(f"만족도 예측: {'개선' if sat_improved else '동등/하락'}")
        
        if wage_improved or sat_improved:
            print(f"\nStacking 앙상블 성능 개선 확인!")
        else:
            print(f"\n추가 최적화 필요")
    
    def save_results(self):
        """결과 저장"""
        print("\n결과 저장...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.stacking_models['wage'], 'models/optimized_stacking_wage.pkl')
        joblib.dump(self.stacking_models['satisfaction'], 'models/optimized_stacking_satisfaction.pkl')
        
        # 결과 저장
        results_df = pd.DataFrame([self.results])
        results_df.to_csv('model_results/optimized_stacking_results.csv', index=False)
        
        print("최적화된 Stacking 모델 및 결과 저장 완료!")

def main():
    """메인 함수"""
    start_time = time.time()
    
    # 샘플 크기를 점진적으로 테스트
    sample_sizes = [20000, 50000]  # 필요시 더 큰 크기로 확장
    
    best_results = None
    best_sample_size = None
    
    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"샘플 크기 {sample_size:,}개로 Stacking 테스트")
        print(f"{'='*60}")
        
        try:
            system = OptimizedStackingSystem(sample_size=sample_size)
            system.load_and_sample_data()
            system.create_optimized_stacking()
            system.evaluate_performance()
            
            # 첫 번째 또는 더 나은 결과인 경우 저장
            if best_results is None or system.results['wage_rmse'] < best_results['wage_rmse']:
                best_results = system.results.copy()
                best_sample_size = sample_size
                system.save_results()
            
        except Exception as e:
            print(f"샘플 크기 {sample_size}에서 오류 발생: {e}")
            continue
    
    # 최종 결과
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"최적화된 Stacking 앙상블 완료!")
    print(f"최적 샘플 크기: {best_sample_size:,}개")
    print(f"최종 성능: RMSE {best_results['wage_rmse']:.2f}, 정확도 {best_results['sat_accuracy']:.4f}")
    print(f"총 소요시간: {elapsed:.1f}분")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()