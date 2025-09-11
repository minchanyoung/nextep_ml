import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
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

print("=== 최종 모델 구축 및 성능 분석 ===")

class FinalModelSystem:
    """베이스라인 모델 기반 최종 시스템"""
    
    def __init__(self):
        self.individual_models = {}
        self.ensemble_models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리 - 샘플링으로 속도 개선"""
        print("데이터 로드 및 샘플링...")
        
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
        
        # 샘플링으로 속도 개선 (전체 데이터의 30% 사용)
        sample_size = min(50000, len(df_final))  # 최대 50,000개 사용
        df_sample = df_final.sample(n=sample_size, random_state=42)
        
        print(f"원본 데이터: {df_final.shape[0]}개 → 샘플링: {df_sample.shape[0]}개")
        
        # 특성 선택
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_sample.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        for col in df_sample[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_sample[col] = le.fit_transform(df_sample[col].astype(str))
        
        # 결측값 처리
        df_sample[feature_cols] = df_sample[feature_cols].fillna(df_sample[feature_cols].median())
        
        # 시간 기반 분할
        train_mask = df_sample['year'] <= 2020
        test_mask = df_sample['year'] >= 2021
        
        self.X_train = df_sample[train_mask][feature_cols]
        self.X_test = df_sample[test_mask][feature_cols]
        self.y_wage_train = df_sample[train_mask]['next_wage']
        self.y_wage_test = df_sample[test_mask]['next_wage']
        # 만족도를 0부터 시작하도록 변환 (1-5 → 0-4)
        self.y_sat_train = (df_sample[train_mask]['next_satisfaction'] - 1).astype(int)
        self.y_sat_test = (df_sample[test_mask]['next_satisfaction'] - 1).astype(int)
        
        self.feature_names = feature_cols
        
        print(f"훈련 데이터: {self.X_train.shape[0]}개")
        print(f"테스트 데이터: {self.X_test.shape[0]}개")
        print(f"특성 개수: {len(self.feature_names)}개")
    
    def train_individual_models(self):
        """개별 모델 훈련"""
        print("\n개별 모델 훈련 중...")
        
        # 임금 예측 모델들
        print("  임금 예측 모델들...")
        
        # CatBoost (최고 성능 모델)
        self.individual_models['catboost_wage'] = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        self.individual_models['catboost_wage'].fit(self.X_train, self.y_wage_train)
        
        # XGBoost
        self.individual_models['xgb_wage'] = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        self.individual_models['xgb_wage'].fit(self.X_train, self.y_wage_train)
        
        # LightGBM
        self.individual_models['lgb_wage'] = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        self.individual_models['lgb_wage'].fit(self.X_train, self.y_wage_train)
        
        # 만족도 예측 모델들
        print("  만족도 예측 모델들...")
        
        # XGBoost (최고 성능 모델)
        self.individual_models['xgb_satisfaction'] = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )
        self.individual_models['xgb_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        # CatBoost
        self.individual_models['catboost_satisfaction'] = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        )
        self.individual_models['catboost_satisfaction'].fit(self.X_train, self.y_sat_train)
        
        # LightGBM
        self.individual_models['lgb_satisfaction'] = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
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
        self.ensemble_models['satisfaction_ensemble'].fit(self.X_train, self.y_sat_train)
        
        print("  앙상블 모델 생성 완료!")
    
    def evaluate_all_models(self):
        """모든 모델 성능 평가"""
        print("\n모델 성능 평가...")
        
        self.results = {'individual': {}, 'ensemble': {}}
        
        # 개별 모델 평가
        print("  개별 모델 성능:")
        
        # 임금 예측 개별 모델
        wage_models = ['catboost_wage', 'xgb_wage', 'lgb_wage']
        for model_name in wage_models:
            model = self.individual_models[model_name]
            y_pred = model.predict(self.X_test)
            
            rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred))
            mae = mean_absolute_error(self.y_wage_test, y_pred)
            r2 = r2_score(self.y_wage_test, y_pred)
            
            self.results['individual'][model_name] = {
                'rmse': rmse, 'mae': mae, 'r2': r2
            }
            
            print(f"    {model_name}: RMSE={rmse:.2f}, R²={r2:.4f}")
        
        # 만족도 예측 개별 모델
        sat_models = ['xgb_satisfaction', 'catboost_satisfaction', 'lgb_satisfaction']
        for model_name in sat_models:
            model = self.individual_models[model_name]
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_sat_test, y_pred)
            
            self.results['individual'][model_name] = {'accuracy': acc}
            
            print(f"    {model_name}: 정확도={acc:.4f}")
        
        # 앙상블 모델 평가
        print("\n  앙상블 모델 성능:")
        
        # 임금 앙상블
        y_pred_wage = self.ensemble_models['wage_ensemble'].predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_wage))
        mae = mean_absolute_error(self.y_wage_test, y_pred_wage)
        r2 = r2_score(self.y_wage_test, y_pred_wage)
        
        self.results['ensemble']['wage'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        print(f"    임금 앙상블: RMSE={rmse:.2f}, R²={r2:.4f}")
        
        # 만족도 앙상블
        y_pred_sat = self.ensemble_models['satisfaction_ensemble'].predict(self.X_test)
        acc = accuracy_score(self.y_sat_test, y_pred_sat)
        
        self.results['ensemble']['satisfaction'] = {'accuracy': acc}
        print(f"    만족도 앙상블: 정확도={acc:.4f}")
    
    def create_comparison_visualization(self):
        """성능 비교 시각화"""
        print("\n성능 비교 시각화 생성...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 임금 예측 RMSE 비교
        wage_models = ['catboost_wage', 'xgb_wage', 'lgb_wage', 'ensemble']
        wage_rmse = [
            self.results['individual']['catboost_wage']['rmse'],
            self.results['individual']['xgb_wage']['rmse'],
            self.results['individual']['lgb_wage']['rmse'],
            self.results['ensemble']['wage']['rmse']
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'orange']
        bars = axes[0, 0].bar(wage_models, wage_rmse, color=colors)
        axes[0, 0].set_title('임금 예측 모델 RMSE 비교', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE (만원)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, wage_rmse):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 임금 예측 R² 비교
        wage_r2 = [
            self.results['individual']['catboost_wage']['r2'],
            self.results['individual']['xgb_wage']['r2'],
            self.results['individual']['lgb_wage']['r2'],
            self.results['ensemble']['wage']['r2']
        ]
        
        bars = axes[0, 1].bar(wage_models, wage_r2, color=colors)
        axes[0, 1].set_title('임금 예측 모델 R² 비교', fontweight='bold')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, wage_r2):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 만족도 예측 정확도 비교
        sat_models = ['xgb_satisfaction', 'catboost_satisfaction', 'lgb_satisfaction', 'ensemble']
        sat_acc = [
            self.results['individual']['xgb_satisfaction']['accuracy'],
            self.results['individual']['catboost_satisfaction']['accuracy'],
            self.results['individual']['lgb_satisfaction']['accuracy'],
            self.results['ensemble']['satisfaction']['accuracy']
        ]
        
        bars = axes[1, 0].bar(sat_models, sat_acc, color=colors)
        axes[1, 0].set_title('만족도 예측 모델 정확도 비교', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, sat_acc):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 종합 성능 요약
        axes[1, 1].text(0.5, 0.7, f'🎯 최종 모델 성능\n\n' +
                       f'💰 임금 예측 (앙상블)\n' +
                       f'   RMSE: {self.results["ensemble"]["wage"]["rmse"]:.1f}만원\n' +
                       f'   R²: {self.results["ensemble"]["wage"]["r2"]:.3f}\n\n' +
                       f'😊 만족도 예측 (앙상블)\n' +
                       f'   정확도: {self.results["ensemble"]["satisfaction"]["accuracy"]:.3f}',
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('최종 성능 요약', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/final_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_final_models(self):
        """최종 모델 저장"""
        print("\n최종 모델 저장...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        # 앙상블 모델 저장
        joblib.dump(self.ensemble_models['wage_ensemble'], 'models/final_wage_ensemble.pkl')
        joblib.dump(self.ensemble_models['satisfaction_ensemble'], 'models/final_satisfaction_ensemble.pkl')
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('model_results/final_model_results.csv')
        
        print("  ✓ 앙상블 모델 저장 완료")
        print("  ✓ 성능 결과 저장 완료")
    
    def generate_final_report(self):
        """최종 성능 보고서"""
        print("\n" + "="*70)
        print("🏆 최종 모델 성능 보고서")
        print("="*70)
        
        print(f"\n📊 데이터셋 정보:")
        print(f"   훈련 데이터: {len(self.X_train):,}개")
        print(f"   테스트 데이터: {len(self.X_test):,}개")
        print(f"   특성 개수: {len(self.feature_names)}개")
        
        print(f"\n💰 임금 예측 결과:")
        wage_result = self.results['ensemble']['wage']
        print(f"   앙상블 모델 성능:")
        print(f"     - RMSE: {wage_result['rmse']:.2f}만원")
        print(f"     - MAE: {wage_result['mae']:.2f}만원")
        print(f"     - R²: {wage_result['r2']:.4f}")
        
        # 최고 개별 모델과 비교
        best_individual_wage = min(self.results['individual'].items(), 
                                 key=lambda x: x[1].get('rmse', float('inf')) if 'rmse' in x[1] else float('inf'))
        print(f"   최고 개별 모델: {best_individual_wage[0]} (RMSE: {best_individual_wage[1]['rmse']:.2f})")
        improvement = best_individual_wage[1]['rmse'] - wage_result['rmse']
        print(f"   앙상블 개선효과: {improvement:+.2f}만원")
        
        print(f"\n😊 만족도 예측 결과:")
        sat_result = self.results['ensemble']['satisfaction']
        print(f"   앙상블 모델 성능:")
        print(f"     - 정확도: {sat_result['accuracy']:.4f}")
        
        # 최고 개별 모델과 비교
        best_individual_sat = max(self.results['individual'].items(),
                                key=lambda x: x[1].get('accuracy', 0) if 'accuracy' in x[1] else 0)
        print(f"   최고 개별 모델: {best_individual_sat[0]} (정확도: {best_individual_sat[1]['accuracy']:.4f})")
        improvement = sat_result['accuracy'] - best_individual_sat[1]['accuracy']
        print(f"   앙상블 개선효과: {improvement:+.4f}")
        
        print(f"\n📁 결과 파일:")
        print(f"   - 임금 예측 모델: models/final_wage_ensemble.pkl")
        print(f"   - 만족도 예측 모델: models/final_satisfaction_ensemble.pkl")
        print(f"   - 성능 결과: model_results/final_model_results.csv")
        print(f"   - 시각화: visualizations/final_model_comparison.png")
        
        # 기존 베이스라인 대비 비교
        print(f"\n🔄 기존 베이스라인 대비 성능:")
        print(f"   임금 예측 (베이스라인 RMSE: 115.92만원):")
        print(f"     현재 성능: {wage_result['rmse']:.2f}만원")
        baseline_diff = 115.92 - wage_result['rmse']
        print(f"     차이: {baseline_diff:+.2f}만원 ({'개선' if baseline_diff > 0 else '악화'})")
        
        print(f"   만족도 예측 (베이스라인 정확도: 0.694):")
        print(f"     현재 성능: {sat_result['accuracy']:.4f}")
        baseline_diff = sat_result['accuracy'] - 0.694
        print(f"     차이: {baseline_diff:+.4f} ({'개선' if baseline_diff > 0 else '악화'})")

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    # 필요 폴더 생성
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 시스템 초기화
    system = FinalModelSystem()
    
    # 실행
    system.load_and_prepare_data()
    system.train_individual_models()
    system.create_ensemble_models()
    system.evaluate_all_models()
    system.create_comparison_visualization()
    system.save_final_models()
    system.generate_final_report()
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 전체 분석 완료! 소요시간: {total_time:.1f}분")

if __name__ == "__main__":
    main()