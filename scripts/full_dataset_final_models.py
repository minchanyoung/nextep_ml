import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import VotingRegressor, VotingClassifier
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 전체 데이터셋 기반 최종 모델 구축 ===")

class FullDatasetModelSystem:
    """전체 데이터셋을 사용한 최종 모델 시스템"""
    
    def __init__(self):
        self.individual_models = {}
        self.ensemble_models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """전체 데이터셋 로드 및 전처리"""
        print("전체 데이터셋 로드 중...")
        
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
        
        # 타겟이 모두 있는 케이스만 선별 및 만족도 -1 제거
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        df_final = df_consecutive[valid_mask].copy()
        
        print(f"전체 유효 데이터: {df_final.shape[0]}개")
        
        # 특성 선택
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        print("범주형 변수 인코딩 중...")
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
        
        # 결측값 처리
        print("결측값 처리 중...")
        df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
        
        # 시간 기반 분할
        train_mask = df_final['year'] <= 2020
        test_mask = df_final['year'] >= 2021
        
        self.X_train = df_final[train_mask][feature_cols]
        self.X_test = df_final[test_mask][feature_cols]
        self.y_wage_train = df_final[train_mask]['next_wage']
        self.y_wage_test = df_final[test_mask]['next_wage']
        # 만족도를 0부터 시작하도록 변환 (1-5 → 0-4)
        self.y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
        self.y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
        
        self.feature_names = feature_cols
        
        print(f"훈련 데이터: {self.X_train.shape[0]:,}개")
        print(f"테스트 데이터: {self.X_test.shape[0]:,}개")
        print(f"특성 개수: {len(self.feature_names)}개")
    
    def train_individual_models(self):
        """개별 모델 훈련 - 전체 데이터셋 사용"""
        print("\n개별 모델 훈련 중... (전체 데이터셋 사용)")
        
        # 임금 예측 모델들
        print("  임금 예측 모델들...")
        
        # CatBoost (최고 성능 모델)
        print("    CatBoost 훈련 중...")
        self.individual_models['catboost_wage'] = cb.CatBoostRegressor(
            iterations=2000,  # 더 많은 반복
            learning_rate=0.05,  # 더 낮은 학습률
            depth=8,  # 더 깊은 모델
            l2_leaf_reg=3,
            random_seed=42,
            verbose=100  # 진행상황 표시
        )
        self.individual_models['catboost_wage'].fit(self.X_train, self.y_wage_train)
        
        # XGBoost
        print("    XGBoost 훈련 중...")
        self.individual_models['xgb_wage'] = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=1
        )
        self.individual_models['xgb_wage'].fit(self.X_train, self.y_wage_train)
        
        # LightGBM
        print("    LightGBM 훈련 중...")
        self.individual_models['lgb_wage'] = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=100
        )
        self.individual_models['lgb_wage'].fit(self.X_train, self.y_wage_train)
        
        # 만족도 예측 모델들
        print("  만족도 예측 모델들...")
        
        # XGBoost (최고 성능 모델)
        print("    XGBoost 분류 모델 훈련 중...")
        self.individual_models['xgb_satisfaction'] = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=1
        )
        self.individual_models['xgb_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        # CatBoost
        print("    CatBoost 분류 모델 훈련 중...")
        self.individual_models['catboost_satisfaction'] = cb.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            random_seed=42,
            verbose=100
        )
        self.individual_models['catboost_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        # LightGBM
        print("    LightGBM 분류 모델 훈련 중...")
        self.individual_models['lgb_satisfaction'] = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=100
        )
        self.individual_models['lgb_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        print("  개별 모델 훈련 완료!")
    
    def create_ensemble_models(self):
        """앙상블 모델 생성"""
        print("\n앙상블 모델 생성...")
        
        # 임금 예측 앙상블
        wage_estimators = [
            ('catboost', self.individual_models['catboost_wage']),
            ('xgboost', self.individual_models['xgb_wage']),
            ('lightgbm', self.individual_models['lgb_wage'])
        ]
        
        self.ensemble_models['wage_ensemble'] = VotingRegressor(
            estimators=wage_estimators,
            n_jobs=-1
        )
        print("  임금 앙상블 훈련 중...")
        self.ensemble_models['wage_ensemble'].fit(self.X_train, self.y_wage_train)
        
        # 만족도 예측 앙상블
        satisfaction_estimators = [
            ('xgboost', self.individual_models['xgb_satisfaction']),
            ('catboost', self.individual_models['catboost_satisfaction']),
            ('lightgbm', self.individual_models['lgb_satisfaction'])
        ]
        
        self.ensemble_models['satisfaction_ensemble'] = VotingClassifier(
            estimators=satisfaction_estimators,
            voting='soft',
            n_jobs=-1
        )
        print("  만족도 앙상블 훈련 중...")
        self.ensemble_models['satisfaction_ensemble'].fit(self.X_train, self.y_sat_train)
        
        print("  앙상블 모델 생성 완료!")
    
    def evaluate_all_models(self):
        """모든 모델 성능 평가 - MAE 포함"""
        print("\n모델 성능 평가 (RMSE, MAE, R², 정확도)...")
        
        self.results = {'individual': {}, 'ensemble': {}}
        
        # 개별 모델 평가
        print("  개별 모델 성능:")
        
        # 임금 예측 개별 모델
        wage_models = ['catboost_wage', 'xgb_wage', 'lgb_wage']
        for model_name in wage_models:
            model = self.individual_models[model_name]
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
            train_r2 = r2_score(self.y_wage_train, y_pred_train)
            test_r2 = r2_score(self.y_wage_test, y_pred_test)
            
            self.results['individual'][model_name] = {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_mae': train_mae, 'test_mae': test_mae,
                'train_r2': train_r2, 'test_r2': test_r2
            }
            
            print(f"    {model_name}: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
        
        # 만족도 예측 개별 모델
        sat_models = ['xgb_satisfaction', 'catboost_satisfaction', 'lgb_satisfaction']
        for model_name in sat_models:
            model = self.individual_models[model_name]
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_acc = accuracy_score(self.y_sat_train, y_pred_train)
            test_acc = accuracy_score(self.y_sat_test, y_pred_test)
            
            self.results['individual'][model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            
            print(f"    {model_name}: 정확도={test_acc:.4f}")
        
        # 앙상블 모델 평가
        print("\n  앙상블 모델 성능:")
        
        # 임금 앙상블
        y_pred_train = self.ensemble_models['wage_ensemble'].predict(self.X_train)
        y_pred_test = self.ensemble_models['wage_ensemble'].predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
        train_r2 = r2_score(self.y_wage_train, y_pred_train)
        test_r2 = r2_score(self.y_wage_test, y_pred_test)
        
        self.results['ensemble']['wage'] = {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
        
        print(f"    임금 앙상블: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
        
        # 만족도 앙상블
        y_pred_train = self.ensemble_models['satisfaction_ensemble'].predict(self.X_train)
        y_pred_test = self.ensemble_models['satisfaction_ensemble'].predict(self.X_test)
        
        train_acc = accuracy_score(self.y_sat_train, y_pred_train)
        test_acc = accuracy_score(self.y_sat_test, y_pred_test)
        
        self.results['ensemble']['satisfaction'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"    만족도 앙상블: 정확도={test_acc:.4f}")
    
    def create_comprehensive_visualization(self):
        """종합 성능 시각화 - MAE 포함"""
        print("\n종합 성능 비교 시각화 생성...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 임금 예측 RMSE 비교
        wage_models = ['CatBoost', 'XGBoost', 'LightGBM', 'Ensemble']
        wage_rmse = [
            self.results['individual']['catboost_wage']['test_rmse'],
            self.results['individual']['xgb_wage']['test_rmse'],
            self.results['individual']['lgb_wage']['test_rmse'],
            self.results['ensemble']['wage']['test_rmse']
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']
        bars = axes[0, 0].bar(wage_models, wage_rmse, color=colors)
        axes[0, 0].set_title('임금 예측 모델 RMSE 비교', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('RMSE (만원)')
        
        for bar, value in zip(bars, wage_rmse):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 임금 예측 MAE 비교
        wage_mae = [
            self.results['individual']['catboost_wage']['test_mae'],
            self.results['individual']['xgb_wage']['test_mae'],
            self.results['individual']['lgb_wage']['test_mae'],
            self.results['ensemble']['wage']['test_mae']
        ]
        
        bars = axes[0, 1].bar(wage_models, wage_mae, color=colors)
        axes[0, 1].set_title('임금 예측 모델 MAE 비교', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('MAE (만원)')
        
        for bar, value in zip(bars, wage_mae):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 3. 임금 예측 R² 비교
        wage_r2 = [
            self.results['individual']['catboost_wage']['test_r2'],
            self.results['individual']['xgb_wage']['test_r2'],
            self.results['individual']['lgb_wage']['test_r2'],
            self.results['ensemble']['wage']['test_r2']
        ]
        
        bars = axes[0, 2].bar(wage_models, wage_r2, color=colors)
        axes[0, 2].set_title('임금 예측 모델 R² 비교', fontweight='bold', fontsize=12)
        axes[0, 2].set_ylabel('R² Score')
        
        for bar, value in zip(bars, wage_r2):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 만족도 예측 정확도 비교
        sat_models = ['XGBoost', 'CatBoost', 'LightGBM', 'Ensemble']
        sat_acc = [
            self.results['individual']['xgb_satisfaction']['test_accuracy'],
            self.results['individual']['catboost_satisfaction']['test_accuracy'],
            self.results['individual']['lgb_satisfaction']['test_accuracy'],
            self.results['ensemble']['satisfaction']['test_accuracy']
        ]
        
        bars = axes[1, 0].bar(sat_models, sat_acc, color=colors)
        axes[1, 0].set_title('만족도 예측 모델 정확도 비교', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy')
        
        for bar, value in zip(bars, sat_acc):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 5. 베이스라인 대비 성능 (RMSE 중심)
        baseline_models = ['기존\n베이스라인', '전체 데이터\n앙상블']
        baseline_rmse = [115.92, self.results['ensemble']['wage']['test_rmse']]
        
        bars = axes[1, 1].bar(baseline_models, baseline_rmse, color=['lightgray', 'orange'])
        axes[1, 1].set_title('임금 예측 베이스라인 대비 성능', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('RMSE (만원)')
        
        for bar, value in zip(bars, baseline_rmse):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 6. 종합 성능 요약
        wage_result = self.results['ensemble']['wage']
        sat_result = self.results['ensemble']['satisfaction']
        
        summary_text = f'🎯 전체 데이터 최종 성과\n\n' + \
                      f'💰 임금 예측 앙상블\n' + \
                      f'   RMSE: {wage_result["test_rmse"]:.1f}만원\n' + \
                      f'   MAE: {wage_result["test_mae"]:.1f}만원\n' + \
                      f'   R²: {wage_result["test_r2"]:.3f}\n\n' + \
                      f'😊 만족도 예측 앙상블\n' + \
                      f'   정확도: {sat_result["test_accuracy"]:.3f}\n\n' + \
                      f'📊 데이터 규모\n' + \
                      f'   훈련: {len(self.X_train):,}개\n' + \
                      f'   테스트: {len(self.X_test):,}개'
        
        axes[1, 2].text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('최종 성과 요약', fontweight='bold', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/full_dataset_final_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_final_models(self):
        """최종 모델 저장"""
        print("\n최종 모델 저장 중...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        # 앙상블 모델 저장
        joblib.dump(self.ensemble_models['wage_ensemble'], 'models/full_dataset_wage_ensemble.pkl')
        joblib.dump(self.ensemble_models['satisfaction_ensemble'], 'models/full_dataset_satisfaction_ensemble.pkl')
        
        # 개별 모델도 저장
        for model_name, model in self.individual_models.items():
            joblib.dump(model, f'models/full_dataset_{model_name}.pkl')
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('model_results/full_dataset_model_results.csv')
        
        print("  [완료] 앙상블 모델 저장 완료")
        print("  [완료] 개별 모델 저장 완료")
        print("  [완료] 성능 결과 저장 완료")
    
    def generate_comprehensive_report(self):
        """종합 성능 보고서 - MAE 포함"""
        print("\n" + "="*80)
        print("전체 데이터셋 기반 최종 모델 성능 보고서")
        print("="*80)
        
        print(f"\n📊 데이터셋 정보:")
        print(f"   훈련 데이터: {len(self.X_train):,}개")
        print(f"   테스트 데이터: {len(self.X_test):,}개")
        print(f"   특성 개수: {len(self.feature_names)}개")
        print(f"   전체 활용률: 100% (전체 데이터셋 사용)")
        
        print(f"\n💰 임금 예측 결과:")
        wage_result = self.results['ensemble']['wage']
        print(f"   앙상블 모델 성능:")
        print(f"     - RMSE: {wage_result['test_rmse']:.2f}만원")
        print(f"     - MAE: {wage_result['test_mae']:.2f}만원")
        print(f"     - R²: {wage_result['test_r2']:.4f}")
        
        # 최고 개별 모델과 비교
        best_individual_wage = min(
            [(k, v) for k, v in self.results['individual'].items() if 'wage' in k], 
            key=lambda x: x[1]['test_rmse']
        )
        print(f"   최고 개별 모델: {best_individual_wage[0]}")
        print(f"     - RMSE: {best_individual_wage[1]['test_rmse']:.2f}만원")
        print(f"     - MAE: {best_individual_wage[1]['test_mae']:.2f}만원")
        
        rmse_improvement = best_individual_wage[1]['test_rmse'] - wage_result['test_rmse']
        mae_improvement = best_individual_wage[1]['test_mae'] - wage_result['test_mae']
        print(f"   앙상블 개선효과:")
        print(f"     - RMSE: {rmse_improvement:+.2f}만원")
        print(f"     - MAE: {mae_improvement:+.2f}만원")
        
        print(f"\n😊 만족도 예측 결과:")
        sat_result = self.results['ensemble']['satisfaction']
        print(f"   앙상블 모델 성능:")
        print(f"     - 정확도: {sat_result['test_accuracy']:.4f}")
        
        # 최고 개별 모델과 비교
        best_individual_sat = max(
            [(k, v) for k, v in self.results['individual'].items() if 'satisfaction' in k],
            key=lambda x: x[1]['test_accuracy']
        )
        print(f"   최고 개별 모델: {best_individual_sat[0]}")
        print(f"     - 정확도: {best_individual_sat[1]['test_accuracy']:.4f}")
        
        acc_improvement = sat_result['test_accuracy'] - best_individual_sat[1]['test_accuracy']
        print(f"   앙상블 개선효과: {acc_improvement:+.4f}")
        
        # 기존 베이스라인 대비 비교
        print(f"\n🔄 기존 베이스라인 대비 성능:")
        print(f"   임금 예측:")
        print(f"     - 기존 베이스라인: 115.92만원 (RMSE)")
        print(f"     - 전체 데이터 앙상블: {wage_result['test_rmse']:.2f}만원")
        baseline_diff = 115.92 - wage_result['test_rmse']
        print(f"     - 개선효과: {baseline_diff:+.2f}만원 ({'🎉 개선' if baseline_diff > 0 else '⚠️ 악화'})")
        
        print(f"   만족도 예측:")
        print(f"     - 기존 베이스라인: 0.694 (정확도)")
        print(f"     - 전체 데이터 앙상블: {sat_result['test_accuracy']:.4f}")
        baseline_diff_sat = sat_result['test_accuracy'] - 0.694
        print(f"     - 개선효과: {baseline_diff_sat:+.4f} ({'🎉 개선' if baseline_diff_sat > 0 else '⚠️ 악화'})")
        
        print(f"\n📁 생성된 결과물:")
        print(f"   - 임금 예측 앙상블: models/full_dataset_wage_ensemble.pkl")
        print(f"   - 만족도 예측 앙상블: models/full_dataset_satisfaction_ensemble.pkl")
        print(f"   - 성능 결과: model_results/full_dataset_model_results.csv")
        print(f"   - 시각화: visualizations/full_dataset_final_comparison.png")
        
        print(f"\n🚀 프로덕션 준비 상태:")
        print(f"   ✓ 전체 데이터셋으로 훈련 완료")
        print(f"   ✓ MAE 지표 포함 종합 평가")
        print(f"   ✓ 앙상블 모델 성능 검증")
        print(f"   ✓ 모델 파일 저장 완료")

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    print("전체 데이터셋 기반 최종 모델 구축을 시작합니다...")
    print("예상 소요 시간: 15-30분 (데이터 크기에 따라)")
    
    # 필요 폴더 생성
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 시스템 초기화
    system = FullDatasetModelSystem()
    
    # 실행
    system.load_and_prepare_data()
    system.train_individual_models()
    system.create_ensemble_models()
    system.evaluate_all_models()
    system.create_comprehensive_visualization()
    system.save_final_models()
    system.generate_comprehensive_report()
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 전체 분석 완료! 총 소요시간: {total_time:.1f}분")
    print("="*80)

if __name__ == "__main__":
    main()