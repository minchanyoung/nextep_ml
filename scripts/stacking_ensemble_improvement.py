import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== Stacking 앙상블 성능 개선 시스템 ===")

class StackingEnsembleSystem:
    """Stacking 앙상블을 활용한 성능 개선 시스템"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.stacking_models = {}
        self.results = {}
        
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
        
        # 타겟이 모두 있는 케이스만 선별 및 만족도 -1 제거
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        df_final = df_consecutive[valid_mask].copy()
        
        print(f"전체 유효 데이터: {df_final.shape[0]:,}개")
        
        # 특성 선택
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in df_final.columns if col not in exclude_cols]
        
        # 범주형 변수 인코딩
        for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_final[col] = le.fit_transform(df_final[col].astype(str))
        
        # 결측값 처리 - 개선된 방법
        print("개선된 결측값 처리 중...")
        for col in feature_cols:
            if df_final[col].isna().sum() > 0:
                # 직업 관련 특성은 직업군별 중위수로 대체
                if any(keyword in col.lower() for keyword in ['job', 'ind', 'occupation']):
                    if 'occupation_group' in df_final.columns:
                        df_final[col] = df_final.groupby('occupation_group')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    else:
                        df_final[col] = df_final[col].fillna(df_final[col].median())
                else:
                    # 기타 특성은 전체 중위수로 대체
                    df_final[col] = df_final[col].fillna(df_final[col].median())
        
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
    
    def create_base_models(self):
        """베이스 모델들 생성"""
        print("\n베이스 모델 생성 중...")
        
        # 임금 예측용 베이스 모델들
        print("  임금 예측 베이스 모델들...")
        
        self.base_models['wage_catboost'] = cb.CatBoostRegressor(
            iterations=1500,
            learning_rate=0.08,
            depth=8,
            l2_leaf_reg=3,
            random_seed=self.random_state,
            verbose=False
        )
        
        self.base_models['wage_xgboost'] = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=1,
            random_state=self.random_state,
            verbosity=0
        )
        
        self.base_models['wage_lightgbm'] = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=1,
            random_state=self.random_state,
            verbose=-1
        )
        
        # 만족도 예측용 베이스 모델들
        print("  만족도 예측 베이스 모델들...")
        
        self.base_models['sat_xgboost'] = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=1,
            random_state=self.random_state,
            verbosity=0
        )
        
        self.base_models['sat_catboost'] = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.08,
            depth=8,
            random_seed=self.random_state,
            verbose=False
        )
        
        self.base_models['sat_lightgbm'] = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=1,
            random_state=self.random_state,
            verbose=-1
        )
        
        print("  베이스 모델 생성 완료!")
    
    def create_stacking_ensembles(self):
        """Stacking 앙상블 모델 생성"""
        print("\nStacking 앙상블 생성 중...")
        
        # 시계열 교차검증 설정
        cv = TimeSeriesSplit(n_splits=3)
        
        # 임금 예측 Stacking 앙상블
        print("  임금 예측 Stacking 앙상블...")
        
        wage_base_estimators = [
            ('catboost', self.base_models['wage_catboost']),
            ('xgboost', self.base_models['wage_xgboost']),
            ('lightgbm', self.base_models['wage_lightgbm'])
        ]
        
        # Meta-learner로 Ridge 회귀 사용 (선형 결합 + 정규화)
        from sklearn.linear_model import Ridge
        meta_regressor = Ridge(alpha=1.0, random_state=self.random_state)
        
        self.stacking_models['wage_stacking'] = StackingRegressor(
            estimators=wage_base_estimators,
            final_estimator=meta_regressor,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        print("    임금 Stacking 훈련 중...")
        self.stacking_models['wage_stacking'].fit(self.X_train, self.y_wage_train)
        
        # 만족도 예측 Stacking 앙상블
        print("  만족도 예측 Stacking 앙상블...")
        
        sat_base_estimators = [
            ('xgboost', self.base_models['sat_xgboost']),
            ('catboost', self.base_models['sat_catboost']),
            ('lightgbm', self.base_models['sat_lightgbm'])
        ]
        
        # Meta-learner로 LogisticRegression 사용
        meta_classifier = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            multi_class='multinomial'
        )
        
        self.stacking_models['sat_stacking'] = StackingClassifier(
            estimators=sat_base_estimators,
            final_estimator=meta_classifier,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        print("    만족도 Stacking 훈련 중...")
        self.stacking_models['sat_stacking'].fit(self.X_train, self.y_sat_train)
        
        print("  Stacking 앙상블 생성 완료!")
    
    def evaluate_models(self):
        """모든 모델 성능 평가"""
        print("\n모델 성능 평가...")
        
        self.results = {
            'base_models': {},
            'stacking_models': {},
            'baseline_comparison': {}
        }
        
        # 베이스 모델들 평가
        print("  베이스 모델 성능:")
        
        # 임금 예측 베이스 모델들
        for model_name in ['wage_catboost', 'wage_xgboost', 'wage_lightgbm']:
            model = self.base_models[model_name]
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
            train_r2 = r2_score(self.y_wage_train, y_pred_train)
            test_r2 = r2_score(self.y_wage_test, y_pred_test)
            
            self.results['base_models'][model_name] = {
                'train_rmse': train_rmse, 'test_rmse': test_rmse,
                'train_mae': train_mae, 'test_mae': test_mae,
                'train_r2': train_r2, 'test_r2': test_r2
            }
            
            print(f"    {model_name}: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
        
        # 만족도 예측 베이스 모델들
        for model_name in ['sat_xgboost', 'sat_catboost', 'sat_lightgbm']:
            model = self.base_models[model_name]
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            train_acc = accuracy_score(self.y_sat_train, y_pred_train)
            test_acc = accuracy_score(self.y_sat_test, y_pred_test)
            
            self.results['base_models'][model_name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            
            print(f"    {model_name}: 정확도={test_acc:.4f}")
        
        # Stacking 모델들 평가
        print("\n  Stacking 모델 성능:")
        
        # 임금 예측 Stacking
        y_pred_train = self.stacking_models['wage_stacking'].predict(self.X_train)
        y_pred_test = self.stacking_models['wage_stacking'].predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_wage_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_wage_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_wage_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_wage_test, y_pred_test)
        train_r2 = r2_score(self.y_wage_train, y_pred_train)
        test_r2 = r2_score(self.y_wage_test, y_pred_test)
        
        self.results['stacking_models']['wage_stacking'] = {
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
        
        print(f"    임금 Stacking: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
        
        # 만족도 예측 Stacking
        y_pred_train = self.stacking_models['sat_stacking'].predict(self.X_train)
        y_pred_test = self.stacking_models['sat_stacking'].predict(self.X_test)
        
        train_acc = accuracy_score(self.y_sat_train, y_pred_train)
        test_acc = accuracy_score(self.y_sat_test, y_pred_test)
        
        self.results['stacking_models']['sat_stacking'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        print(f"    만족도 Stacking: 정확도={test_acc:.4f}")
        
        # 베이스라인 대비 성능 비교
        self.compare_with_baseline()
    
    def compare_with_baseline(self):
        """베이스라인 및 기존 모델과 성능 비교"""
        print("\n베이스라인 대비 성능 비교:")
        print("-" * 60)
        
        # 기존 성능 (전체 데이터셋 Voting 앙상블)
        previous_wage_rmse = 118.89
        previous_wage_mae = 58.35
        previous_wage_r2 = 0.6776
        previous_sat_acc = 0.6716
        
        # 원본 베이스라인
        baseline_wage_rmse = 115.92
        baseline_sat_acc = 0.694
        
        # Stacking 성능
        stacking_wage = self.results['stacking_models']['wage_stacking']
        stacking_sat = self.results['stacking_models']['sat_stacking']
        
        # 성능 비교 계산
        self.results['baseline_comparison'] = {
            # vs 원본 베이스라인
            'vs_original_baseline': {
                'wage_rmse_diff': baseline_wage_rmse - stacking_wage['test_rmse'],
                'sat_acc_diff': stacking_sat['test_accuracy'] - baseline_sat_acc
            },
            # vs 이전 Voting 앙상블
            'vs_previous_voting': {
                'wage_rmse_diff': previous_wage_rmse - stacking_wage['test_rmse'],
                'wage_mae_diff': previous_wage_mae - stacking_wage['test_mae'],
                'wage_r2_diff': stacking_wage['test_r2'] - previous_wage_r2,
                'sat_acc_diff': stacking_sat['test_accuracy'] - previous_sat_acc
            }
        }
        
        # 결과 출력
        print("1. 원본 베이스라인 대비:")
        wage_vs_baseline = self.results['baseline_comparison']['vs_original_baseline']['wage_rmse_diff']
        sat_vs_baseline = self.results['baseline_comparison']['vs_original_baseline']['sat_acc_diff']
        
        print(f"   임금 RMSE: {baseline_wage_rmse:.2f} → {stacking_wage['test_rmse']:.2f} ({wage_vs_baseline:+.2f}만원)")
        print(f"   만족도 정확도: {baseline_sat_acc:.4f} → {stacking_sat['test_accuracy']:.4f} ({sat_vs_baseline:+.4f})")
        
        print("\n2. 이전 Voting 앙상블 대비:")
        comparisons = self.results['baseline_comparison']['vs_previous_voting']
        
        print(f"   임금 RMSE: {previous_wage_rmse:.2f} → {stacking_wage['test_rmse']:.2f} ({comparisons['wage_rmse_diff']:+.2f}만원)")
        print(f"   임금 MAE: {previous_wage_mae:.2f} → {stacking_wage['test_mae']:.2f} ({comparisons['wage_mae_diff']:+.2f}만원)")
        print(f"   임금 R²: {previous_wage_r2:.4f} → {stacking_wage['test_r2']:.4f} ({comparisons['wage_r2_diff']:+.4f})")
        print(f"   만족도 정확도: {previous_sat_acc:.4f} → {stacking_sat['test_accuracy']:.4f} ({comparisons['sat_acc_diff']:+.4f})")
    
    def create_performance_visualization(self):
        """성능 비교 시각화"""
        print("\n성능 비교 시각화 생성...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 임금 예측 RMSE 비교
        models = ['CatBoost', 'XGBoost', 'LightGBM', 'Stacking']
        wage_rmse = [
            self.results['base_models']['wage_catboost']['test_rmse'],
            self.results['base_models']['wage_xgboost']['test_rmse'],
            self.results['base_models']['wage_lightgbm']['test_rmse'],
            self.results['stacking_models']['wage_stacking']['test_rmse']
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        bars = axes[0, 0].bar(models, wage_rmse, color=colors)
        axes[0, 0].set_title('임금 예측 RMSE 비교', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE (만원)')
        
        for bar, value in zip(bars, wage_rmse):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 2. 임금 예측 MAE 비교
        wage_mae = [
            self.results['base_models']['wage_catboost']['test_mae'],
            self.results['base_models']['wage_xgboost']['test_mae'],
            self.results['base_models']['wage_lightgbm']['test_mae'],
            self.results['stacking_models']['wage_stacking']['test_mae']
        ]
        
        bars = axes[0, 1].bar(models, wage_mae, color=colors)
        axes[0, 1].set_title('임금 예측 MAE 비교', fontweight='bold')
        axes[0, 1].set_ylabel('MAE (만원)')
        
        for bar, value in zip(bars, wage_mae):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 3. 임금 예측 R² 비교
        wage_r2 = [
            self.results['base_models']['wage_catboost']['test_r2'],
            self.results['base_models']['wage_xgboost']['test_r2'],
            self.results['base_models']['wage_lightgbm']['test_r2'],
            self.results['stacking_models']['wage_stacking']['test_r2']
        ]
        
        bars = axes[0, 2].bar(models, wage_r2, color=colors)
        axes[0, 2].set_title('임금 예측 R² 비교', fontweight='bold')
        axes[0, 2].set_ylabel('R² Score')
        
        for bar, value in zip(bars, wage_r2):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 만족도 예측 정확도 비교
        sat_acc = [
            self.results['base_models']['sat_xgboost']['test_accuracy'],
            self.results['base_models']['sat_catboost']['test_accuracy'],
            self.results['base_models']['sat_lightgbm']['test_accuracy'],
            self.results['stacking_models']['sat_stacking']['test_accuracy']
        ]
        
        bars = axes[1, 0].bar(models, sat_acc, color=colors)
        axes[1, 0].set_title('만족도 예측 정확도 비교', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        
        for bar, value in zip(bars, sat_acc):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 5. 베이스라인 대비 성능 변화
        baseline_comparison = ['원본\n베이스라인', 'Stacking\n앙상블']
        
        # 임금 RMSE 베이스라인 비교
        baseline_rmse_values = [115.92, self.results['stacking_models']['wage_stacking']['test_rmse']]
        bars = axes[1, 1].bar(baseline_comparison, baseline_rmse_values, color=['lightgray', 'gold'])
        axes[1, 1].set_title('임금 RMSE: 베이스라인 vs Stacking', fontweight='bold')
        axes[1, 1].set_ylabel('RMSE (만원)')
        
        for bar, value in zip(bars, baseline_rmse_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 6. 종합 성능 요약
        stacking_wage = self.results['stacking_models']['wage_stacking']
        stacking_sat = self.results['stacking_models']['sat_stacking']
        
        improvement_text = f'🎯 Stacking 앙상블 성과\n\n'
        improvement_text += f'💰 임금 예측\n'
        improvement_text += f'   RMSE: {stacking_wage["test_rmse"]:.1f}만원\n'
        improvement_text += f'   MAE: {stacking_wage["test_mae"]:.1f}만원\n'
        improvement_text += f'   R²: {stacking_wage["test_r2"]:.3f}\n\n'
        improvement_text += f'😊 만족도 예측\n'
        improvement_text += f'   정확도: {stacking_sat["test_accuracy"]:.3f}\n\n'
        
        # 개선 효과
        vs_baseline = self.results['baseline_comparison']['vs_original_baseline']
        vs_voting = self.results['baseline_comparison']['vs_previous_voting']
        
        improvement_text += f'📈 개선 효과\n'
        improvement_text += f'   vs 베이스라인: {vs_baseline["wage_rmse_diff"]:+.1f}만원\n'
        improvement_text += f'   vs Voting: {vs_voting["wage_rmse_diff"]:+.1f}만원'
        
        axes[1, 2].text(0.5, 0.5, improvement_text, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('성과 요약', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/stacking_ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models_and_results(self):
        """모델 및 결과 저장"""
        print("\n모델 및 결과 저장...")
        
        import os
        os.makedirs('models', exist_ok=True)
        os.makedirs('model_results', exist_ok=True)
        
        # Stacking 모델 저장
        joblib.dump(self.stacking_models['wage_stacking'], 'models/stacking_wage_ensemble.pkl')
        joblib.dump(self.stacking_models['sat_stacking'], 'models/stacking_satisfaction_ensemble.pkl')
        
        # 베이스 모델들도 저장
        for model_name, model in self.base_models.items():
            joblib.dump(model, f'models/stacking_base_{model_name}.pkl')
        
        # 결과 저장
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('model_results/stacking_ensemble_results.csv')
        
        print("  [완료] Stacking 모델 저장 완료")
        print("  [완료] 성능 결과 저장 완료")
    
    def generate_final_report(self):
        """최종 성능 보고서"""
        print("\n" + "="*70)
        print("🚀 Stacking 앙상블 성능 개선 결과")
        print("="*70)
        
        stacking_wage = self.results['stacking_models']['wage_stacking']
        stacking_sat = self.results['stacking_models']['sat_stacking']
        
        print(f"\n💰 임금 예측 Stacking 성능:")
        print(f"   RMSE: {stacking_wage['test_rmse']:.2f}만원")
        print(f"   MAE: {stacking_wage['test_mae']:.2f}만원")
        print(f"   R²: {stacking_wage['test_r2']:.4f}")
        
        print(f"\n😊 만족도 예측 Stacking 성능:")
        print(f"   정확도: {stacking_sat['test_accuracy']:.4f}")
        
        # 개선 효과 분석
        comparisons = self.results['baseline_comparison']
        
        print(f"\n📊 성능 개선 분석:")
        print(f"   원본 베이스라인 대비:")
        print(f"     임금 RMSE: {comparisons['vs_original_baseline']['wage_rmse_diff']:+.2f}만원")
        print(f"     만족도 정확도: {comparisons['vs_original_baseline']['sat_acc_diff']:+.4f}")
        
        print(f"   이전 Voting 앙상블 대비:")
        print(f"     임금 RMSE: {comparisons['vs_previous_voting']['wage_rmse_diff']:+.2f}만원")
        print(f"     임금 MAE: {comparisons['vs_previous_voting']['wage_mae_diff']:+.2f}만원")
        print(f"     만족도 정확도: {comparisons['vs_previous_voting']['sat_acc_diff']:+.4f}")
        
        # 성공 여부 판단
        wage_success = comparisons['vs_original_baseline']['wage_rmse_diff'] > 0
        sat_success = comparisons['vs_original_baseline']['sat_acc_diff'] > 0
        
        print(f"\n🎯 개선 성공 여부:")
        print(f"   임금 예측: {'✅ 성공' if wage_success else '❌ 미달'}")
        print(f"   만족도 예측: {'✅ 성공' if sat_success else '❌ 미달'}")
        
        if wage_success or sat_success:
            print(f"\n🎉 Stacking 앙상블을 통한 성능 개선 달성!")
        else:
            print(f"\n⚠️ 추가 개선 방안 필요")
        
        print(f"\n📁 저장된 파일:")
        print(f"   - Stacking 모델: models/stacking_*_ensemble.pkl")
        print(f"   - 성능 결과: model_results/stacking_ensemble_results.csv")
        print(f"   - 시각화: visualizations/stacking_ensemble_comparison.png")

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    print("Stacking 앙상블을 통한 성능 개선을 시작합니다...")
    
    # 필요 폴더 생성
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('model_results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 시스템 초기화
    system = StackingEnsembleSystem(random_state=42)
    
    # 실행
    system.load_and_prepare_data()
    system.create_base_models()
    system.create_stacking_ensembles()
    system.evaluate_models()
    system.create_performance_visualization()
    system.save_models_and_results()
    system.generate_final_report()
    
    total_time = (time.time() - start_time) / 60
    print(f"\n🎉 Stacking 앙상블 구현 완료! 총 소요시간: {total_time:.1f}분")
    print("="*70)

if __name__ == "__main__":
    main()