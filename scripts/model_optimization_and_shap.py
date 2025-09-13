import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import catboost as cb
import shap
import optuna
from optuna.samplers import TPESampler
import warnings
import joblib
import time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 모델 최적화 및 SHAP 분석 시스템 ===")

class ModelOptimizationSystem:
    """모델 하이퍼파라미터 최적화 및 해석 시스템"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.optimization_results = {}
        self.shap_explainers = {}
        self.shap_values = {}
        
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("데이터 로드 중...")
        
        # 최종 ML 데이터셋 로드
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
        
        print(f"최종 데이터셋 크기: {df_final.shape}")
        
        # 특성 선택 (타겟 변수 제외)
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 
                       'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        le_dict = {}
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
            le_dict[col] = le
        
        # 결측값 처리
        df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
        
        # 시간 기반 분할 (2000-2020: 훈련, 2021-2022: 테스트)
        train_mask = df_final['year'] <= 2020
        test_mask = df_final['year'] >= 2021
        
        self.X_train = df_final[train_mask][feature_cols]
        self.X_test = df_final[test_mask][feature_cols]
        self.y_wage_train = df_final[train_mask]['next_wage']
        self.y_wage_test = df_final[test_mask]['next_wage']
        self.y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
        self.y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
        
        self.feature_names = feature_cols
        
        print(f"훈련 데이터: {self.X_train.shape[0]}개")
        print(f"테스트 데이터: {self.X_test.shape[0]}개")
        print(f"특성 개수: {len(self.feature_names)}개")
        
        return df_final
    
    def optimize_catboost_wage(self, n_trials=100):
        """CatBoost 임금 예측 모델 최적화"""
        print(f"\nCatBoost 임금 예측 모델 최적화 시작 (시행 횟수: {n_trials})")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': self.random_state,
                'verbose': False
            }
            
            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_train = self.y_wage_train.iloc[train_idx]
                y_fold_val = self.y_wage_train.iloc[val_idx]
                
                model = cb.CatBoostRegressor(**params)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 모델 훈련
        best_params = study.best_params
        best_params['random_seed'] = self.random_state
        best_params['verbose'] = False
        
        self.best_models['catboost_wage'] = cb.CatBoostRegressor(**best_params)
        self.best_models['catboost_wage'].fit(self.X_train, self.y_wage_train)
        
        # 성능 평가
        y_pred_train = self.best_models['catboost_wage'].predict(self.X_train)
        y_pred_test = self.best_models['catboost_wage'].predict(self.X_test)
        
        self.optimization_results['catboost_wage'] = {
            'best_params': best_params,
            'best_score': study.best_value,
            'train_rmse': np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test)),
            'train_r2': r2_score(self.y_wage_train, y_pred_train),
            'test_r2': r2_score(self.y_wage_test, y_pred_test),
            'train_mae': mean_absolute_error(self.y_wage_train, y_pred_train),
            'test_mae': mean_absolute_error(self.y_wage_test, y_pred_test)
        }
        
        print(f"CatBoost 임금 예측 최적화 완료")
        print(f"최적 검증 RMSE: {study.best_value:.4f}")
        print(f"테스트 RMSE: {self.optimization_results['catboost_wage']['test_rmse']:.4f}")
        print(f"테스트 R²: {self.optimization_results['catboost_wage']['test_r2']:.4f}")
        
        return study
    
    def optimize_xgboost_satisfaction(self, n_trials=100):
        """XGBoost 만족도 예측 모델 최적화"""
        print(f"\nXGBoost 만족도 예측 모델 최적화 시작 (시행 횟수: {n_trials})")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            
            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_train = self.y_sat_train.iloc[train_idx]
                y_fold_val = self.y_sat_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_fold_train, y_fold_train)
                
                y_pred = model.predict(X_fold_val)
                accuracy = accuracy_score(y_fold_val, y_pred)
                scores.append(accuracy)
            
            return -np.mean(scores)  # Optuna는 최소화 방향이므로 음수 사용
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 모델 훈련
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        best_params['objective'] = 'multi:softprob'
        best_params['eval_metric'] = 'mlogloss'
        best_params['verbosity'] = 0
        
        self.best_models['xgboost_satisfaction'] = xgb.XGBClassifier(**best_params)
        self.best_models['xgboost_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        # 성능 평가
        y_pred_train = self.best_models['xgboost_satisfaction'].predict(self.X_train)
        y_pred_test = self.best_models['xgboost_satisfaction'].predict(self.X_test)
        
        self.optimization_results['xgboost_satisfaction'] = {
            'best_params': best_params,
            'best_score': -study.best_value,  # 다시 양수로 변환
            'train_accuracy': accuracy_score(self.y_sat_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_sat_test, y_pred_test),
            'train_report': classification_report(self.y_sat_train, y_pred_train, output_dict=True),
            'test_report': classification_report(self.y_sat_test, y_pred_test, output_dict=True)
        }
        
        print(f"XGBoost 만족도 예측 최적화 완료")
        print(f"최적 검증 정확도: {-study.best_value:.4f}")
        print(f"테스트 정확도: {self.optimization_results['xgboost_satisfaction']['test_accuracy']:.4f}")
        
        return study
    
    def perform_shap_analysis(self, n_samples=1000):
        """SHAP 분석 수행"""
        print(f"\nSHAP 분석 시작 (샘플 수: {n_samples})")
        
        # 샘플 선택 (분석 속도 향상)
        sample_indices = np.random.choice(len(self.X_test), 
                                        min(n_samples, len(self.X_test)), 
                                        replace=False)
        X_shap = self.X_test.iloc[sample_indices]
        
        # CatBoost 임금 예측 SHAP 분석
        if 'catboost_wage' in self.best_models:
            print("CatBoost 임금 예측 모델 SHAP 분석...")
            explainer_wage = shap.Explainer(self.best_models['catboost_wage'])
            shap_values_wage = explainer_wage(X_shap)
            
            self.shap_explainers['catboost_wage'] = explainer_wage
            self.shap_values['catboost_wage'] = shap_values_wage
        
        # XGBoost 만족도 예측 SHAP 분석
        if 'xgboost_satisfaction' in self.best_models:
            print("XGBoost 만족도 예측 모델 SHAP 분석...")
            explainer_sat = shap.Explainer(self.best_models['xgboost_satisfaction'])
            shap_values_sat = explainer_sat(X_shap)
            
            self.shap_explainers['xgboost_satisfaction'] = explainer_sat
            self.shap_values['xgboost_satisfaction'] = shap_values_sat
        
        print("SHAP 분석 완료")
    
    def create_shap_visualizations(self):
        """SHAP 시각화 생성"""
        print("SHAP 시각화 생성 중...")
        
        plt.style.use('default')
        
        # 임금 예측 SHAP 시각화
        if 'catboost_wage' in self.shap_values:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Summary plot
            plt.sca(axes[0, 0])
            shap.summary_plot(self.shap_values['catboost_wage'], 
                            max_display=15, show=False)
            axes[0, 0].set_title('CatBoost 임금 예측 - SHAP Summary Plot', fontsize=14, fontweight='bold')
            
            # Bar plot
            plt.sca(axes[0, 1])
            shap.summary_plot(self.shap_values['catboost_wage'], 
                            plot_type="bar", max_display=15, show=False)
            axes[0, 1].set_title('CatBoost 임금 예측 - Feature Importance', fontsize=14, fontweight='bold')
            
            # Waterfall plot (첫 번째 예측 사례)
            plt.sca(axes[1, 0])
            shap.waterfall_plot(self.shap_values['catboost_wage'][0], show=False)
            axes[1, 0].set_title('CatBoost 임금 예측 - Waterfall Plot (첫 번째 사례)', fontsize=14, fontweight='bold')
            
            # Force plot을 위한 공간
            axes[1, 1].text(0.5, 0.5, 'Interactive plots는\nJupyter notebook에서\n확인 가능합니다', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            axes[1, 1].set_title('Force Plot 안내', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('visualizations/shap_analysis_wage_catboost.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 만족도 예측 SHAP 시각화
        if 'xgboost_satisfaction' in self.shap_values:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Summary plot
            plt.sca(axes[0, 0])
            shap.summary_plot(self.shap_values['xgboost_satisfaction'], 
                            max_display=15, show=False)
            axes[0, 0].set_title('XGBoost 만족도 예측 - SHAP Summary Plot', fontsize=14, fontweight='bold')
            
            # Bar plot
            plt.sca(axes[0, 1])
            shap.summary_plot(self.shap_values['xgboost_satisfaction'], 
                            plot_type="bar", max_display=15, show=False)
            axes[0, 1].set_title('XGBoost 만족도 예측 - Feature Importance', fontsize=14, fontweight='bold')
            
            # Waterfall plot
            plt.sca(axes[1, 0])
            shap.waterfall_plot(self.shap_values['xgboost_satisfaction'][0], show=False)
            axes[1, 0].set_title('XGBoost 만족도 예측 - Waterfall Plot (첫 번째 사례)', fontsize=14, fontweight='bold')
            
            # 안내 메시지
            axes[1, 1].text(0.5, 0.5, 'Multi-class SHAP values는\n각 클래스별로 해석됩니다\n\n상세한 분석은 Jupyter\nnotebook에서 확인 가능', 
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[1, 1].set_title('Multi-class SHAP 안내', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('visualizations/shap_analysis_satisfaction_xgboost.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_models_and_results(self):
        """모델과 결과 저장"""
        print("\n모델 및 결과 저장 중...")
        
        # 모델 저장
        for model_name, model in self.best_models.items():
            joblib.dump(model, f'models/best_{model_name}.pkl')
            print(f"{model_name} 모델 저장 완료")
        
        # 결과 저장
        results_df = pd.DataFrame(self.optimization_results).T
        results_df.to_csv('model_results/optimization_results.csv')
        print("최적화 결과 저장 완료")
    
    def generate_performance_report(self):
        """종합 성능 보고서 생성"""
        print("\n=== 모델 최적화 및 성능 분석 보고서 ===\n")
        
        # CatBoost 임금 예측 결과
        if 'catboost_wage' in self.optimization_results:
            result = self.optimization_results['catboost_wage']
            print("🎯 CatBoost 임금 예측 모델")
            print("=" * 40)
            print(f"테스트 RMSE: {result['test_rmse']:.2f}만원")
            print(f"테스트 MAE: {result['test_mae']:.2f}만원")
            print(f"테스트 R²: {result['test_r2']:.4f}")
            print(f"최적 파라미터: {result['best_params']}")
            print()
        
        # XGBoost 만족도 예측 결과
        if 'xgboost_satisfaction' in self.optimization_results:
            result = self.optimization_results['xgboost_satisfaction']
            print("😊 XGBoost 만족도 예측 모델")
            print("=" * 40)
            print(f"테스트 정확도: {result['test_accuracy']:.4f}")
            print("클래스별 성능:")
            for class_name, metrics in result['test_report'].items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    print(f"  클래스 {class_name}: F1-score {metrics['f1-score']:.3f}")
            print(f"최적 파라미터: {result['best_params']}")
            print()
        
        # SHAP 분석 결과 요약
        print("🔍 SHAP 분석 결과")
        print("=" * 40)
        if 'catboost_wage' in self.shap_values:
            # 임금 예측 중요 특성 top 5
            feature_importance = np.abs(self.shap_values['catboost_wage'].values).mean(0)
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            
            print("임금 예측 주요 특성 (상위 5개):")
            for i, idx in enumerate(top_features_idx):
                feature_name = self.feature_names[idx]
                importance = feature_importance[idx]
                print(f"  {i+1}. {feature_name}: {importance:.4f}")
            print()
        
        if 'xgboost_satisfaction' in self.shap_values:
            # 만족도 예측 중요 특성 top 5 (첫 번째 클래스 기준)
            if len(self.shap_values['xgboost_satisfaction'].values.shape) == 3:
                feature_importance = np.abs(self.shap_values['xgboost_satisfaction'].values[:, :, 0]).mean(0)
            else:
                feature_importance = np.abs(self.shap_values['xgboost_satisfaction'].values).mean(0)
                
            top_features_idx = np.argsort(feature_importance)[-5:][::-1]
            
            print("만족도 예측 주요 특성 (상위 5개):")
            for i, idx in enumerate(top_features_idx):
                feature_name = self.feature_names[idx]
                importance = feature_importance[idx]
                print(f"  {i+1}. {feature_name}: {importance:.4f}")
            print()

def main():
    """메인 실행 함수"""
    # 결과 저장 폴더 생성
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 시스템 초기화
    optimizer = ModelOptimizationSystem(random_state=42)
    
    # 데이터 로드
    data = optimizer.load_and_prepare_data()
    
    # 모델 최적화 (시행 횟수를 적게 설정하여 빠른 테스트)
    print("하이퍼파라미터 최적화 시작...")
    start_time = time.time()
    
    # CatBoost 임금 예측 최적화
    optimizer.optimize_catboost_wage(n_trials=50)  # 실제 운영시에는 100-200으로 증가
    
    # XGBoost 만족도 예측 최적화
    optimizer.optimize_xgboost_satisfaction(n_trials=50)  # 실제 운영시에는 100-200으로 증가
    
    optimization_time = time.time() - start_time
    print(f"하이퍼파라미터 최적화 완료 (소요시간: {optimization_time/60:.1f}분)")
    
    # SHAP 분석
    optimizer.perform_shap_analysis(n_samples=500)  # 빠른 분석을 위해 샘플 수 조정
    
    # 시각화 생성
    optimizer.create_shap_visualizations()
    
    # 모델 및 결과 저장
    optimizer.save_models_and_results()
    
    # 성능 보고서 생성
    optimizer.generate_performance_report()
    
    print(f"\n전체 분석 완료! 총 소요시간: {(time.time() - start_time)/60:.1f}분")
    print("결과 파일 위치:")
    print("- 모델: models/ 폴더")
    print("- 시각화: visualizations/ 폴더")
    print("- 성능 결과: model_results/ 폴더")

if __name__ == "__main__":
    main()