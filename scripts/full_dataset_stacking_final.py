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

print("=== 전체 데이터셋 Stacking 앙상블 시스템 ===")

class FullDatasetStackingSystem:
    """전체 데이터셋을 사용하여 Stacking 앙상블을 훈련하는 시스템"""
    
    def __init__(self):
        self.stacking_models = {}
        self.results = {}
        
    def load_full_data(self):
        """전체 데이터 로드"""
        print("전체 데이터셋 로드 중...")
        
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
        
        print(f"전체 데이터 - 훈련: {len(self.X_train):,}개, 테스트: {len(self.X_test):,}개")
    
    def create_stacking_model(self):
        """Stacking 앙상블 생성 및 훈련"""
        print("\nStacking 앙상블 훈련 시작...")
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # 임금 예측 Stacking
        print("  임금 예측 모델 훈련 중...")
        
        wage_estimators = [
            ('cb', cb.CatBoostRegressor(iterations=300, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['wage'] = StackingRegressor(
            estimators=wage_estimators,
            final_estimator=Ridge(alpha=0.5, random_state=42),
            cv=cv,
            n_jobs=-1, # 모든 CPU 코어 사용
            verbose=1
        )
        
        start_time = time.time()
        self.stacking_models['wage'].fit(self.X_train, self.y_wage_train)
        wage_time = time.time() - start_time
        print(f"    임금 모델 훈련 완료 ({wage_time:.1f}초)")
        
        # 만족도 예측 Stacking
        print("  만족도 예측 모델 훈련 중...")
        
        sat_estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, verbosity=0, random_state=42)),
            ('cb', cb.CatBoostClassifier(iterations=300, learning_rate=0.1, depth=5, verbose=False, random_seed=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, verbose=-1, random_state=42))
        ]
        
        self.stacking_models['satisfaction'] = StackingClassifier(
            estimators=sat_estimators,
            final_estimator=LogisticRegression(max_iter=500, random_state=42),
            cv=cv,
            n_jobs=-1, # 모든 CPU 코어 사용
            verbose=1
        )
        
        start_time = time.time()
        self.stacking_models['satisfaction'].fit(self.X_train, self.y_sat_train)
        sat_time = time.time() - start_time
        print(f"    만족도 모델 훈련 완료 ({sat_time:.1f}초)")
    
    def evaluate_and_save(self):
        """성능 평가 및 결과 저장"""
        print("\n성능 평가 및 결과 저장 중...")
        
        # 임금 예측 성능
        wage_pred = self.stacking_models['wage'].predict(self.X_test)
        wage_rmse = np.sqrt(mean_squared_error(self.y_wage_test, wage_pred))
        wage_mae = mean_absolute_error(self.y_wage_test, wage_pred)
        wage_r2 = r2_score(self.y_wage_test, wage_pred)
        
        # 만족도 예측 성능
        sat_pred = self.stacking_models['satisfaction'].predict(self.X_test)
        sat_acc = accuracy_score(self.y_sat_test, sat_pred)
        
        self.results = {
            'model_type': 'Full Dataset Stacking',
            'wage_rmse': wage_rmse,
            'wage_mae': wage_mae,
            'wage_r2': wage_r2,
            'sat_accuracy': sat_acc,
            'sample_size': len(self.X_train) + len(self.X_test)
        }
        
        print(f"\n=== 전체 데이터셋 Stacking 앙상블 성능 ===")
        print(f"훈련 데이터: {len(self.X_train):,}개, 테스트 데이터: {len(self.X_test):,}개")
        print(f"\n임금 예측:")
        print(f"  RMSE: {wage_rmse:.2f}만원")
        print(f"  MAE: {wage_mae:.2f}만원") 
        print(f"  R2: {wage_r2:.4f}")
        
        print(f"\n만족도 예측:")
        print(f"  정확도: {sat_acc:.4f}")

        # 결과 저장
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        joblib.dump(self.stacking_models['wage'], 'models/full_dataset_stacking_wage.pkl')
        joblib.dump(self.stacking_models['satisfaction'], 'models/full_dataset_stacking_satisfaction.pkl')
        
        # 기존 결과 파일에 추가
        try:
            results_df = pd.read_csv('model_results/stacking_results_comparison.csv')
        except FileNotFoundError:
            results_df = pd.DataFrame()
            
        new_results_df = pd.DataFrame([self.results])
        final_df = pd.concat([results_df, new_results_df], ignore_index=True)
        final_df.to_csv('model_results/stacking_results_comparison.csv', index=False)
        
        print("\n전체 데이터셋 Stacking 모델 및 결과 저장 완료!")

def main():
    """메인 함수"""
    total_start_time = time.time()
    
    system = FullDatasetStackingSystem()
    system.load_full_data()
    system.create_stacking_model()
    system.evaluate_and_save()
    
    elapsed = (time.time() - total_start_time) / 60
    print(f"\n총 소요시간: {elapsed:.1f}분")
    print("모든 작업이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()
