import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
import xgboost as xgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 앙상블 모델 구축 시스템 ===")

class EnsembleModelSystem:
    """최적화된 모델들을 결합한 앙상블 시스템"""
    
    def __init__(self):
        self.wage_models = {}
        self.satisfaction_models = {}
        self.ensemble_wage = None
        self.ensemble_satisfaction = None
        self.results = {}
        
    def load_optimized_models(self):
        """최적화된 모델들 로드"""
        print("최적화된 모델 로드 중...")
        
        try:
            # 최적화된 CatBoost 임금 예측 모델
            self.wage_models['catboost_optimized'] = joblib.load('models/best_catboost_wage.pkl')
            print("✓ 최적화된 CatBoost 임금 모델 로드 완료")
        except:
            print("⚠ 최적화된 CatBoost 모델을 찾을 수 없습니다. 기본 파라미터로 훈련합니다.")
            
        try:
            # 최적화된 XGBoost 만족도 예측 모델
            self.satisfaction_models['xgboost_optimized'] = joblib.load('models/best_xgboost_satisfaction.pkl')
            print("✓ 최적화된 XGBoost 만족도 모델 로드 완료")
        except:
            print("⚠ 최적화된 XGBoost 모델을 찾을 수 없습니다. 기본 파라미터로 훈련합니다.")
    
    def load_data(self):
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
        
        print(f"최종 데이터셋 크기: {df_final.shape}")
        
        # 특성 선택
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 
                       'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        from sklearn.preprocessing import LabelEncoder
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
    
    def create_baseline_models(self):
        """기본 모델들 생성 (최적화된 모델이 없을 경우)"""
        
        if 'catboost_optimized' not in self.wage_models:
            print("기본 CatBoost 임금 모델 훈련 중...")
            catboost_wage = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            )
            catboost_wage.fit(self.X_train, self.y_wage_train)
            self.wage_models['catboost_baseline'] = catboost_wage
        
        if 'xgboost_optimized' not in self.satisfaction_models:
            print("기본 XGBoost 만족도 모델 훈련 중...")
            xgboost_sat = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
            xgboost_sat.fit(self.X_train, self.y_sat_train)
            self.satisfaction_models['xgboost_baseline'] = xgboost_sat
    
    def create_additional_models(self):
        """앙상블을 위한 추가 모델들 생성"""
        print("추가 모델들 훈련 중...")
        
        # 임금 예측용 추가 모델들
        print("임금 예측 추가 모델 훈련...")
        
        # LightGBM 임금 모델
        import lightgbm as lgb
        lgb_wage = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        lgb_wage.fit(self.X_train, self.y_wage_train)
        self.wage_models['lightgbm'] = lgb_wage
        
        # XGBoost 임금 모델 (회귀용)
        xgb_wage = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        xgb_wage.fit(self.X_train, self.y_wage_train)
        self.wage_models['xgboost'] = xgb_wage
        
        # 만족도 예측용 추가 모델들
        print("만족도 예측 추가 모델 훈련...")
        
        # CatBoost 만족도 모델
        cb_sat = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        )
        cb_sat.fit(self.X_train, self.y_sat_train)
        self.satisfaction_models['catboost'] = cb_sat
        
        # LightGBM 만족도 모델
        lgb_sat = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        lgb_sat.fit(self.X_train, self.y_sat_train)
        self.satisfaction_models['lightgbm'] = lgb_sat
    
    def create_ensemble_models(self):
        """앙상블 모델 생성"""
        print("\n앙상블 모델 생성 중...")
        
        # 임금 예측 앙상블 (회귀)
        wage_estimators = [(name, model) for name, model in self.wage_models.items()]
        
        # Soft voting for regression (평균)
        self.ensemble_wage = VotingRegressor(
            estimators=wage_estimators,
            n_jobs=-1
        )
        
        print("임금 예측 앙상블 훈련 중...")
        self.ensemble_wage.fit(self.X_train, self.y_wage_train)
        
        # 만족도 예측 앙상블 (분류)
        satisfaction_estimators = [(name, model) for name, model in self.satisfaction_models.items()]
        
        # Soft voting for classification
        self.ensemble_satisfaction = VotingClassifier(
            estimators=satisfaction_estimators,
            voting='soft',
            n_jobs=-1
        )
        
        print("만족도 예측 앙상블 훈련 중...")
        self.ensemble_satisfaction.fit(self.X_train, self.y_sat_train)
        
        print("앙상블 모델 생성 완료!")
    
    def evaluate_models(self):
        """개별 모델 및 앙상블 성능 평가"""
        print("\n=== 모델 성능 평가 ===")
        
        self.results = {
            'wage_models': {},
            'satisfaction_models': {},
            'ensemble': {}
        }
        
        # 임금 예측 모델 평가
        print("\n💰 임금 예측 모델 성능:")
        print("-" * 50)
        
        for name, model in self.wage_models.items():
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
            train_r2 = r2_score(self.y_wage_train, y_pred_train)
            test_r2 = r2_score(self.y_wage_test, y_pred_test)
            
            self.results['wage_models'][name] = {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_mae': train_mae, 'test_mae': test_mae,
                'train_r2': train_r2, 'test_r2': test_r2
            }
            
            print(f"{name:20} | RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f} | R²: {test_r2:.4f}")
        
        # 임금 앙상블 평가
        y_pred_train = self.ensemble_wage.predict(self.X_train)
        y_pred_test = self.ensemble_wage.predict(self.X_test)
        
        ensemble_train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
        ensemble_test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
        ensemble_train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
        ensemble_test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
        ensemble_train_r2 = r2_score(self.y_wage_train, y_pred_train)
        ensemble_test_r2 = r2_score(self.y_wage_test, y_pred_test)
        
        self.results['ensemble']['wage'] = {
            'train_rmse': ensemble_train_rmse, 'test_rmse': ensemble_test_rmse,
            'train_mae': ensemble_train_mae, 'test_mae': ensemble_test_mae,
            'train_r2': ensemble_train_r2, 'test_r2': ensemble_test_r2
        }
        
        print(f"{'🌟 ENSEMBLE':20} | RMSE: {ensemble_test_rmse:.2f} | MAE: {ensemble_test_mae:.2f} | R²: {ensemble_test_r2:.4f}")
        
        # 만족도 예측 모델 평가
        print("\n😊 만족도 예측 모델 성능:")
        print("-" * 50)
        
        for name, model in self.satisfaction_models.items():
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_acc = accuracy_score(self.y_sat_train, y_pred_train)
            test_acc = accuracy_score(self.y_sat_test, y_pred_test)
            
            self.results['satisfaction_models'][name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            
            print(f"{name:20} | 정확도: {test_acc:.4f}")
        
        # 만족도 앙상블 평가
        y_pred_train = self.ensemble_satisfaction.predict(self.X_train)
        y_pred_test = self.ensemble_satisfaction.predict(self.X_test)
        
        ensemble_train_acc = accuracy_score(self.y_sat_train, y_pred_train)
        ensemble_test_acc = accuracy_score(self.y_sat_test, y_pred_test)
        
        self.results['ensemble']['satisfaction'] = {
            'train_accuracy': ensemble_train_acc,
            'test_accuracy': ensemble_test_acc
        }
        
        print(f"{'🌟 ENSEMBLE':20} | 정확도: {ensemble_test_acc:.4f}")
    
    def create_performance_visualization(self):
        """성능 비교 시각화"""
        print("\n성능 비교 시각화 생성 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 임금 예측 RMSE 비교
        wage_models = list(self.results['wage_models'].keys()) + ['Ensemble']
        wage_rmse = [self.results['wage_models'][m]['test_rmse'] for m in self.results['wage_models'].keys()]
        wage_rmse.append(self.results['ensemble']['wage']['test_rmse'])
        
        axes[0, 0].bar(wage_models, wage_rmse, color=['lightblue' if 'Ensemble' not in m else 'orange' for m in wage_models])
        axes[0, 0].set_title('임금 예측 모델 RMSE 비교', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('RMSE (만원)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 임금 예측 R² 비교
        wage_r2 = [self.results['wage_models'][m]['test_r2'] for m in self.results['wage_models'].keys()]
        wage_r2.append(self.results['ensemble']['wage']['test_r2'])
        
        axes[0, 1].bar(wage_models, wage_r2, color=['lightblue' if 'Ensemble' not in m else 'orange' for m in wage_models])
        axes[0, 1].set_title('임금 예측 모델 R² 비교', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 만족도 예측 정확도 비교
        sat_models = list(self.results['satisfaction_models'].keys()) + ['Ensemble']
        sat_acc = [self.results['satisfaction_models'][m]['test_accuracy'] for m in self.results['satisfaction_models'].keys()]
        sat_acc.append(self.results['ensemble']['satisfaction']['test_accuracy'])
        
        axes[1, 0].bar(sat_models, sat_acc, color=['lightgreen' if 'Ensemble' not in m else 'orange' for m in sat_models])
        axes[1, 0].set_title('만족도 예측 모델 정확도 비교', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 종합 성능 레이더 차트 (상위 모델들)
        from math import pi
        
        # 최고 성능 모델들 선택
        best_wage_model = max(self.results['wage_models'].items(), key=lambda x: x[1]['test_r2'])
        best_sat_model = max(self.results['satisfaction_models'].items(), key=lambda x: x[1]['test_accuracy'])
        
        categories = ['임금 RMSE\n(낮을수록 좋음)', '임금 R²\n(높을수록 좋음)', '만족도 정확도\n(높을수록 좋음)']
        N = len(categories)
        
        # 정규화된 점수 계산 (0-1 스케일)
        max_rmse = max(wage_rmse)
        ensemble_scores = [
            1 - (self.results['ensemble']['wage']['test_rmse'] / max_rmse),  # RMSE는 반전
            self.results['ensemble']['wage']['test_r2'],
            self.results['ensemble']['satisfaction']['test_accuracy']
        ]
        
        best_individual_scores = [
            1 - (best_wage_model[1]['test_rmse'] / max_rmse),
            best_wage_model[1]['test_r2'],
            best_sat_model[1]['test_accuracy']
        ]
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # 완전한 원을 위해
        
        ensemble_scores += ensemble_scores[:1]
        best_individual_scores += best_individual_scores[:1]
        
        axes[1, 1].plot(angles, ensemble_scores, 'o-', linewidth=2, label='앙상블 모델', color='orange')
        axes[1, 1].fill(angles, ensemble_scores, alpha=0.25, color='orange')
        axes[1, 1].plot(angles, best_individual_scores, 'o-', linewidth=2, label='최고 개별 모델', color='blue')
        axes[1, 1].fill(angles, best_individual_scores, alpha=0.25, color='blue')
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('모델 성능 종합 비교\n(레이더 차트)', fontweight='bold', fontsize=12)
        axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/ensemble_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_ensemble_models(self):
        """앙상블 모델 저장"""
        print("\n앙상블 모델 저장 중...")
        
        joblib.dump(self.ensemble_wage, 'models/ensemble_wage_model.pkl')
        joblib.dump(self.ensemble_satisfaction, 'models/ensemble_satisfaction_model.pkl')
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('model_results/ensemble_performance_results.csv')
        
        print("앙상블 모델 저장 완료!")
        print("- 임금 예측 앙상블: models/ensemble_wage_model.pkl")
        print("- 만족도 예측 앙상블: models/ensemble_satisfaction_model.pkl")
        print("- 성능 결과: model_results/ensemble_performance_results.csv")
    
    def generate_final_report(self):
        """최종 성능 보고서"""
        print("\n" + "="*60)
        print("🏆 최종 앙상블 모델 성능 보고서")
        print("="*60)
        
        # 최고 개별 모델
        best_wage = max(self.results['wage_models'].items(), key=lambda x: x[1]['test_r2'])
        best_sat = max(self.results['satisfaction_models'].items(), key=lambda x: x[1]['test_accuracy'])
        
        print(f"\n💰 임금 예측 결과:")
        print(f"   최고 개별 모델: {best_wage[0]} (R² = {best_wage[1]['test_r2']:.4f})")
        print(f"   앙상블 모델: R² = {self.results['ensemble']['wage']['test_r2']:.4f}")
        print(f"   성능 향상: {(self.results['ensemble']['wage']['test_r2'] - best_wage[1]['test_r2']):.4f}")
        
        print(f"\n😊 만족도 예측 결과:")
        print(f"   최고 개별 모델: {best_sat[0]} (정확도 = {best_sat[1]['test_accuracy']:.4f})")
        print(f"   앙상블 모델: 정확도 = {self.results['ensemble']['satisfaction']['test_accuracy']:.4f}")
        print(f"   성능 향상: {(self.results['ensemble']['satisfaction']['test_accuracy'] - best_sat[1]['test_accuracy']):.4f}")
        
        print(f"\n📊 앙상블 모델 구성:")
        print(f"   임금 예측: {len(self.wage_models)}개 모델 결합")
        print(f"   만족도 예측: {len(self.satisfaction_models)}개 모델 결합")

def main():
    """메인 실행 함수"""
    # 필요 폴더 생성
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 앙상블 시스템 초기화
    ensemble_system = EnsembleModelSystem()
    
    # 데이터 로드
    ensemble_system.load_data()
    
    # 최적화된 모델 로드 시도
    ensemble_system.load_optimized_models()
    
    # 기본 모델 생성 (필요한 경우)
    ensemble_system.create_baseline_models()
    
    # 추가 모델들 생성
    ensemble_system.create_additional_models()
    
    # 앙상블 모델 생성
    ensemble_system.create_ensemble_models()
    
    # 성능 평가
    ensemble_system.evaluate_models()
    
    # 시각화 생성
    ensemble_system.create_performance_visualization()
    
    # 모델 저장
    ensemble_system.save_ensemble_models()
    
    # 최종 보고서
    ensemble_system.generate_final_report()
    
    print("\n🎉 앙상블 모델 구축 완료!")

if __name__ == "__main__":
    main()