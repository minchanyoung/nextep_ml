import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import catboost as cb
import shap
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 빠른 모델 최적화 및 SHAP 분석 ===")

class QuickOptimizationSystem:
    """빠른 최적화 시스템 - 제한된 파라미터 그리드 사용"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.optimization_results = {}
        
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로드 중...")
        
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
        
        print(f"최종 데이터셋 크기: {df_final.shape}")
        
        # 특성 선택
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        le_dict = {}
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
            le_dict[col] = le
        
        # 결측값 처리
        df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
        
        # 시간 기반 분할
        train_mask = df_final['year'] <= 2020
        test_mask = df_final['year'] >= 2021
        
        self.X_train = df_final[train_mask][feature_cols]
        self.X_test = df_final[test_mask][feature_cols]
        self.y_wage_train = df_final[train_mask]['next_wage']
        self.y_wage_test = df_final[test_mask]['next_wage']
        self.y_sat_train = df_final[train_mask]['next_satisfaction'].astype(int)
        self.y_sat_test = df_final[test_mask]['next_satisfaction'].astype(int)
        
        self.feature_names = feature_cols
        
        print(f"훈련 데이터: {self.X_train.shape[0]}개")
        print(f"테스트 데이터: {self.X_test.shape[0]}개")
        print(f"특성 개수: {len(self.feature_names)}개")
    
    def optimize_catboost_quick(self):
        """CatBoost 빠른 최적화"""
        print("\nCatBoost 임금 예측 모델 빠른 최적화...")
        
        # 제한된 파라미터 그리드
        param_grid = {
            'iterations': [500, 1000],
            'learning_rate': [0.05, 0.1, 0.2],
            'depth': [6, 8],
            'l2_leaf_reg': [3, 5]
        }
        
        catboost_model = cb.CatBoostRegressor(
            random_seed=self.random_state,
            verbose=False
        )
        
        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            catboost_model, 
            param_grid, 
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_wage_train)
        
        self.best_models['catboost_wage'] = grid_search.best_estimator_
        
        # 성능 평가
        y_pred_train = self.best_models['catboost_wage'].predict(self.X_train)
        y_pred_test = self.best_models['catboost_wage'].predict(self.X_test)
        
        self.optimization_results['catboost_wage'] = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'train_rmse': np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test)),
            'train_r2': r2_score(self.y_wage_train, y_pred_train),
            'test_r2': r2_score(self.y_wage_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_wage_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_wage_test, y_pred_test)
        }
        
        result = self.optimization_results['catboost_wage']
        print(f"✓ CatBoost 최적화 완료")
        print(f"  최적 파라미터: {grid_search.best_params_}")
        print(f"  테스트 RMSE: {result['test_rmse']:.2f}")
        print(f"  테스트 R²: {result['test_r2']:.4f}")
    
    def optimize_xgboost_quick(self):
        """XGBoost 빠른 최적화"""
        print("\nXGBoost 만족도 예측 모델 빠른 최적화...")
        
        # 제한된 파라미터 그리드
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_sat_train)
        
        self.best_models['xgboost_satisfaction'] = grid_search.best_estimator_
        
        # 성능 평가
        y_pred_train = self.best_models['xgboost_satisfaction'].predict(self.X_train)
        y_pred_test = self.best_models['xgboost_satisfaction'].predict(self.X_test)
        
        self.optimization_results['xgboost_satisfaction'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'train_accuracy': accuracy_score(self.y_sat_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_sat_test, y_pred_test),
            'train_report': classification_report(self.y_sat_train, y_pred_train, output_dict=True),
            'test_report': classification_report(self.y_sat_test, y_pred_test, output_dict=True)
        }
        
        result = self.optimization_results['xgboost_satisfaction']
        print(f"✓ XGBoost 최적화 완료")
        print(f"  최적 파라미터: {grid_search.best_params_}")
        print(f"  테스트 정확도: {result['test_accuracy']:.4f}")
    
    def perform_shap_analysis(self):
        """빠른 SHAP 분석"""
        print("\nSHAP 분석 시작...")
        
        # 샘플 선택 (빠른 분석)
        sample_size = min(300, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        X_shap = self.X_test.iloc[sample_indices]
        
        # CatBoost SHAP
        if 'catboost_wage' in self.best_models:
            print("  CatBoost SHAP 분석...")
            explainer = shap.Explainer(self.best_models['catboost_wage'])
            shap_values = explainer(X_shap.iloc[:100])  # 더 작은 샘플
            
            # Top 10 특성 중요도
            feature_importance = np.abs(shap_values.values).mean(0)
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            
            print("    임금 예측 주요 특성 (상위 10개):")
            for i, idx in enumerate(top_features_idx):
                feature_name = self.feature_names[idx]
                importance = feature_importance[idx]
                print(f"      {i+1}. {feature_name}: {importance:.4f}")
        
        # XGBoost SHAP
        if 'xgboost_satisfaction' in self.best_models:
            print("  XGBoost SHAP 분석...")
            explainer = shap.Explainer(self.best_models['xgboost_satisfaction'])
            shap_values = explainer(X_shap.iloc[:100])
            
            # 다중 클래스의 경우 첫 번째 클래스의 중요도 사용
            if len(shap_values.values.shape) == 3:
                feature_importance = np.abs(shap_values.values[:, :, 0]).mean(0)
            else:
                feature_importance = np.abs(shap_values.values).mean(0)
                
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            
            print("    만족도 예측 주요 특성 (상위 10개):")
            for i, idx in enumerate(top_features_idx):
                feature_name = self.feature_names[idx]
                importance = feature_importance[idx]
                print(f"      {i+1}. {feature_name}: {importance:.4f}")
    
    def save_models_and_results(self):
        """모델 및 결과 저장"""
        print("\n모델 및 결과 저장...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        for model_name, model in self.best_models.items():
            joblib.dump(model, f'models/quick_{model_name}.pkl')
            print(f"  ✓ {model_name} 저장")
        
        results_df = pd.DataFrame(self.optimization_results).T
        results_df.to_csv('model_results/quick_optimization_results.csv')
        print(f"  ✓ 결과 저장")
    
    def generate_performance_report(self):
        """성능 보고서 생성"""
        print("\n" + "="*60)
        print("빠른 모델 최적화 결과 보고서")
        print("="*60)
        
        if 'catboost_wage' in self.optimization_results:
            result = self.optimization_results['catboost_wage']
            print(f"\n💰 CatBoost 임금 예측:")
            print(f"   테스트 RMSE: {result['test_rmse']:.2f}만원")
            print(f"   테스트 R²: {result['test_r2']:.4f}")
            print(f"   테스트 MAE: {result['test_mae']:.2f}만원")
            print(f"   기존 대비: RMSE {115.92:.2f} → {result['test_rmse']:.2f}")
        
        if 'xgboost_satisfaction' in self.optimization_results:
            result = self.optimization_results['xgboost_satisfaction']
            print(f"\n😊 XGBoost 만족도 예측:")
            print(f"   테스트 정확도: {result['test_accuracy']:.4f}")
            print(f"   기존 대비: 0.694 → {result['test_accuracy']:.4f}")
        
        print(f"\n📊 분석 완료 시각 정보:")
        print(f"   - 모델: models/ 폴더")
        print(f"   - 결과: model_results/quick_optimization_results.csv")

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    optimizer = QuickOptimizationSystem(random_state=42)
    
    # 데이터 로드
    optimizer.load_and_prepare_data()
    
    # 빠른 최적화
    optimizer.optimize_catboost_quick()
    optimizer.optimize_xgboost_quick()
    
    # SHAP 분석
    optimizer.perform_shap_analysis()
    
    # 결과 저장
    optimizer.save_models_and_results()
    
    # 보고서
    optimizer.generate_performance_report()
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 전체 최적화 완료! 소요시간: {total_time:.1f}분")

if __name__ == "__main__":
    main()