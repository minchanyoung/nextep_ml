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

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 수정된 Stacking 앙상블 시스템 ===")

class FixedStackingSystem:
    """수정된 Stacking 앙상블 시스템"""
    
    def __init__(self):
        self.stacking_models = {}
        self.results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("데이터 로드...")
        
        df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
        
        df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
        df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
        df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
        df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
        
        consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
        df_consecutive = df_sorted[consecutive_mask].copy()
        
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        df_final = df_consecutive[valid_mask].copy()
        
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
        
        print(f"훈련: {len(self.X_train):,}개, 테스트: {len(self.X_test):,}개")
    
    def create_quick_stacking(self):
        """빠른 Stacking 앙상블 생성"""
        print("\n빠른 Stacking 앙상블 생성...")
        
        # 일반 KFold 사용 (시계열이 아닌)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 임금 예측 Stacking
        print("  임금 예측 Stacking...")
        
        wage_estimators = [
            ('catboost', cb.CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=False, random_seed=42)),
            ('xgboost', xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, verbosity=0, random_state=42)),
            ('lightgbm', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['wage'] = StackingRegressor(
            estimators=wage_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=42),
            cv=cv,
            n_jobs=1  # 병렬 처리 비활성화로 안정성 확보
        )
        
        self.stacking_models['wage'].fit(self.X_train, self.y_wage_train)
        print("    임금 Stacking 완료")
        
        # 만족도 예측 Stacking
        print("  만족도 예측 Stacking...")
        
        sat_estimators = [
            ('xgboost', xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, verbosity=0, random_state=42)),
            ('catboost', cb.CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=False, random_seed=42)),
            ('lightgbm', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, max_depth=6, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['satisfaction'] = StackingClassifier(
            estimators=sat_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=cv,
            n_jobs=1
        )
        
        self.stacking_models['satisfaction'].fit(self.X_train, self.y_sat_train)
        print("    만족도 Stacking 완료")
    
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
            'sat_accuracy': sat_acc
        }
        
        print(f"\n=== Stacking 앙상블 성능 ===")
        print(f"임금 예측:")
        print(f"  RMSE: {wage_rmse:.2f}만원")
        print(f"  MAE: {wage_mae:.2f}만원") 
        print(f"  R²: {wage_r2:.4f}")
        
        print(f"\n만족도 예측:")
        print(f"  정확도: {sat_acc:.4f}")
        
        # 베이스라인 비교
        print(f"\n=== 베이스라인 비교 ===")
        baseline_wage_rmse = 115.92
        baseline_sat_acc = 0.694
        previous_wage_rmse = 118.89
        previous_sat_acc = 0.6716
        
        print(f"원본 베이스라인 대비:")
        print(f"  임금 RMSE: {baseline_wage_rmse:.2f} → {wage_rmse:.2f} ({baseline_wage_rmse-wage_rmse:+.2f})")
        print(f"  만족도 정확도: {baseline_sat_acc:.4f} → {sat_acc:.4f} ({sat_acc-baseline_sat_acc:+.4f})")
        
        print(f"\n이전 Voting 대비:")
        print(f"  임금 RMSE: {previous_wage_rmse:.2f} → {wage_rmse:.2f} ({previous_wage_rmse-wage_rmse:+.2f})")
        print(f"  만족도 정확도: {previous_sat_acc:.4f} → {sat_acc:.4f} ({sat_acc-previous_sat_acc:+.4f})")
        
        # 성공 여부
        wage_success = wage_rmse < baseline_wage_rmse
        sat_success = sat_acc > baseline_sat_acc
        
        print(f"\n=== 개선 성공 여부 ===")
        print(f"임금 예측: {'✅ 성공' if wage_success else '❌ 미달'}")
        print(f"만족도 예측: {'✅ 성공' if sat_success else '❌ 미달'}")
        
        if wage_success or sat_success:
            print(f"\n🎉 Stacking 앙상블 성능 개선 달성!")
        else:
            print(f"\n⚠️ 추가 개선 방안 필요")
    
    def save_results(self):
        """결과 저장"""
        print("\n결과 저장...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        joblib.dump(self.stacking_models['wage'], 'models/fixed_stacking_wage.pkl')
        joblib.dump(self.stacking_models['satisfaction'], 'models/fixed_stacking_satisfaction.pkl')
        
        results_df = pd.DataFrame([self.results])
        results_df.to_csv('model_results/fixed_stacking_results.csv', index=False)
        
        print("저장 완료!")

def main():
    """메인 함수"""
    start_time = time.time()
    
    system = FixedStackingSystem()
    system.load_data()
    system.create_quick_stacking()
    system.evaluate_performance()
    system.save_results()
    
    elapsed = (time.time() - start_time) / 60
    print(f"\n총 소요시간: {elapsed:.1f}분")

if __name__ == "__main__":
    main()