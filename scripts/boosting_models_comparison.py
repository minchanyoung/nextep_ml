import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 부스팅 알고리즘 성능 비교: XGBoost vs CatBoost vs LightGBM ===")

class BoostingModelComparison:
    """부스팅 모델 비교 클래스"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.feature_importances = {}
        
    def load_and_prepare_data(self):
        """통합된 직업 데이터셋 로드 및 전처리"""
        
        print("데이터 로드 및 전처리...")
        
        # 통합 직업 분류가 적용된 데이터 로드
        df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
        print(f"전체 데이터 크기: {df.shape}")
        
        # 예측 가능한 케이스만 선별 (연속된 연도)
        df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
        
        # 다음 연도 타겟 생성
        df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
        df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
        df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
        
        # 연속된 연도만 필터링
        consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
        df_consecutive = df_sorted[consecutive_mask].copy()
        
        print(f"연속된 연도 데이터: {len(df_consecutive)}건")
        
        # 타겟 변수가 모두 있는 케이스만 선별
        complete_mask = (df_consecutive['next_wage'].notna() & 
                        df_consecutive['next_satisfaction'].notna() &
                        df_consecutive['p_wage'].notna() &
                        df_consecutive['p4321'].notna())
        
        df_ml = df_consecutive[complete_mask].copy()
        print(f"완전한 데이터: {len(df_ml)}건")
        
        return df_ml
    
    def prepare_features(self, df):
        """특성 준비 및 인코딩"""
        
        print("특성 준비 및 인코딩...")
        
        # 범주형 변수 인코딩
        categorical_features = ['occupation_group', 'occupation_subcategory', 'skill_level']
        label_encoders = {}
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].fillna('Unknown'))
                label_encoders[col] = le
        
        # 특성 선택
        feature_columns = [
            # 기본 개인 특성
            'p_age', 'p_edu', 'p_sex', 'experience_years',
            
            # 현재 상태
            'p_wage', 'p4321', 'time_trend',
            
            # 통합 직업 특성 (인코딩된 버전)
            'occupation_group_encoded', 'skill_level_encoded',
            
            # 직업 기반 파생 특성
            'job_stability', 'job_changes', 'occupation_tenure',
            'occupation_avg_wage', 'wage_vs_occupation_avg', 'wage_quartile_in_occupation',
            'occupation_avg_satisfaction', 'occupation_changed',
            
            # 산업/직업 정보 (숫자형으로 변환)
            'current_industry', 'current_job'
        ]
        
        # 실제 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        
        print(f"사용 가능한 특성: {len(available_features)}개")
        
        # 결측값 처리
        X = df[available_features].copy()
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
        
        # 타겟 변수
        y_wage = df['next_wage'].copy()
        y_satisfaction = df['next_satisfaction'].copy()
        
        # 만족도를 분류 문제로 변환 (1-5점을 3개 그룹으로)
        y_satisfaction_class = pd.cut(y_satisfaction, 
                                    bins=[0, 2, 3, 5], 
                                    labels=[0, 1, 2])  # 숫자 레이블 사용
        y_satisfaction_class = y_satisfaction_class.fillna(1).astype(int)  # NaN을 중간값으로 대체
        
        return X, y_wage, y_satisfaction, y_satisfaction_class, available_features, label_encoders
    
    def split_data_by_time(self, X, y_wage, y_satisfaction, y_satisfaction_class, df):
        """시간 기반 데이터 분할"""
        
        # 2020년 이전: 훈련, 2021년 이후: 테스트
        train_mask = df['year'] <= 2020
        test_mask = df['year'] >= 2021
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_wage_train, y_wage_test = y_wage[train_mask], y_wage[test_mask]
        y_sat_train, y_sat_test = y_satisfaction[train_mask], y_satisfaction[test_mask]
        y_sat_class_train, y_sat_class_test = y_satisfaction_class[train_mask], y_satisfaction_class[test_mask]
        
        print(f"훈련 데이터: {len(X_train)}개")
        print(f"테스트 데이터: {len(X_test)}개")
        
        return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, y_sat_class_train, y_sat_class_test
    
    def train_wage_models(self, X_train, y_wage_train, X_test, y_wage_test):
        """임금 예측 모델 훈련 (회귀)"""
        
        print("\n=== 임금 예측 모델 훈련 (회귀) ===")
        
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'CatBoost': cb.CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        wage_results = {}
        
        for name, model in models.items():
            print(f"\n{name} 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train, y_wage_train)
            
            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # 성능 평가
            train_rmse = np.sqrt(mean_squared_error(y_wage_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_wage_test, test_pred))
            train_mae = mean_absolute_error(y_wage_train, train_pred)
            test_mae = mean_absolute_error(y_wage_test, test_pred)
            train_r2 = r2_score(y_wage_train, train_pred)
            test_r2 = r2_score(y_wage_test, test_pred)
            
            wage_results[name] = {
                'model': model,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_pred': train_pred,
                'test_pred': test_pred
            }
            
            print(f"  훈련 RMSE: {train_rmse:.2f}, 테스트 RMSE: {test_rmse:.2f}")
            print(f"  훈련 R²: {train_r2:.3f}, 테스트 R²: {test_r2:.3f}")
        
        self.models['wage'] = {name: results['model'] for name, results in wage_results.items()}
        self.results['wage'] = wage_results
        
        return wage_results
    
    def train_satisfaction_models(self, X_train, y_sat_class_train, X_test, y_sat_class_test):
        """만족도 예측 모델 훈련 (분류)"""
        
        print("\n=== 만족도 예측 모델 훈련 (분류) ===")
        
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        satisfaction_results = {}
        
        for name, model in models.items():
            print(f"\n{name} 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train, y_sat_class_train)
            
            # 예측
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # 성능 평가
            train_acc = accuracy_score(y_sat_class_train, train_pred)
            test_acc = accuracy_score(y_sat_class_test, test_pred)
            
            satisfaction_results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'classification_report': classification_report(y_sat_class_test, test_pred, output_dict=True)
            }
            
            print(f"  훈련 정확도: {train_acc:.3f}, 테스트 정확도: {test_acc:.3f}")
        
        self.models['satisfaction'] = {name: results['model'] for name, results in satisfaction_results.items()}
        self.results['satisfaction'] = satisfaction_results
        
        return satisfaction_results
    
    def extract_feature_importance(self, available_features):
        """특성 중요도 추출"""
        
        print("\n특성 중요도 추출...")
        
        # 임금 예측 모델의 특성 중요도
        wage_importance = {}
        for name, model in self.models['wage'].items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                wage_importance[name] = pd.Series(importance, index=available_features)
        
        # 만족도 예측 모델의 특성 중요도  
        satisfaction_importance = {}
        for name, model in self.models['satisfaction'].items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                satisfaction_importance[name] = pd.Series(importance, index=available_features)
        
        self.feature_importances = {
            'wage': wage_importance,
            'satisfaction': satisfaction_importance
        }
        
        return wage_importance, satisfaction_importance
    
    def create_comparison_plots(self, y_wage_test, y_sat_class_test):
        """모델 비교 시각화"""
        
        print("비교 시각화 생성...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 임금 예측 성능 비교
        wage_metrics = []
        for name, results in self.results['wage'].items():
            wage_metrics.append({
                'Model': name,
                'Test_RMSE': results['test_rmse'],
                'Test_MAE': results['test_mae'], 
                'Test_R2': results['test_r2']
            })
        
        wage_df = pd.DataFrame(wage_metrics)
        
        # RMSE 비교
        wage_df.plot(x='Model', y='Test_RMSE', kind='bar', ax=axes[0,0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0,0].set_title('임금 예측 RMSE 비교')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R² 비교
        wage_df.plot(x='Model', y='Test_R2', kind='bar', ax=axes[0,1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0,1].set_title('임금 예측 R² 비교')
        axes[0,1].set_ylabel('R²')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 2. 만족도 예측 성능 비교
        satisfaction_metrics = []
        for name, results in self.results['satisfaction'].items():
            satisfaction_metrics.append({
                'Model': name,
                'Test_Accuracy': results['test_accuracy']
            })
        
        satisfaction_df = pd.DataFrame(satisfaction_metrics)
        satisfaction_df.plot(x='Model', y='Test_Accuracy', kind='bar', ax=axes[0,2], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0,2].set_title('만족도 예측 정확도 비교')
        axes[0,2].set_ylabel('정확도')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 3. 특성 중요도 비교 (임금)
        if self.feature_importances['wage']:
            importance_df = pd.DataFrame(self.feature_importances['wage'])
            top_features = importance_df.mean(axis=1).nlargest(10)
            top_features.plot(kind='barh', ax=axes[1,0])
            axes[1,0].set_title('임금 예측 상위 10개 특성 중요도')
            axes[1,0].set_xlabel('중요도')
        
        # 4. 특성 중요도 비교 (만족도)
        if self.feature_importances['satisfaction']:
            importance_df = pd.DataFrame(self.feature_importances['satisfaction'])
            top_features = importance_df.mean(axis=1).nlargest(10)
            top_features.plot(kind='barh', ax=axes[1,1])
            axes[1,1].set_title('만족도 예측 상위 10개 특성 중요도')
            axes[1,1].set_xlabel('중요도')
        
        # 5. 예측 vs 실제 산점도 (최고 성능 모델)
        best_wage_model = min(self.results['wage'].items(), key=lambda x: x[1]['test_rmse'])
        best_predictions = best_wage_model[1]['test_pred']
        
        axes[1,2].scatter(y_wage_test, best_predictions, alpha=0.5)
        axes[1,2].plot([y_wage_test.min(), y_wage_test.max()], [y_wage_test.min(), y_wage_test.max()], 'r--', lw=2)
        axes[1,2].set_title(f'예측 vs 실제 ({best_wage_model[0]})')
        axes[1,2].set_xlabel('실제 임금')
        axes[1,2].set_ylabel('예측 임금')
        
        plt.tight_layout()
        plt.savefig('visualizations/boosting_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """최종 결과 요약 출력"""
        
        print("\n" + "="*60)
        print("=== 부스팅 모델 성능 비교 결과 ===")
        print("="*60)
        
        # 임금 예측 결과
        print("\n[임금 예측 성능]")
        print("모델        | 테스트RMSE | 테스트MAE | 테스트R²")
        print("-" * 50)
        for name, results in self.results['wage'].items():
            print(f"{name:10s} | {results['test_rmse']:9.2f} | {results['test_mae']:8.2f} | {results['test_r2']:7.3f}")
        
        # 최고 성능 모델 식별
        best_wage_model = min(self.results['wage'].items(), key=lambda x: x[1]['test_rmse'])
        print(f"\n최고 성능 (임금): {best_wage_model[0]} (RMSE: {best_wage_model[1]['test_rmse']:.2f})")
        
        # 만족도 예측 결과
        print("\n[만족도 예측 성능]")
        print("모델        | 테스트 정확도")
        print("-" * 30)
        for name, results in self.results['satisfaction'].items():
            print(f"{name:10s} | {results['test_accuracy']:12.3f}")
        
        best_satisfaction_model = max(self.results['satisfaction'].items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n최고 성능 (만족도): {best_satisfaction_model[0]} (정확도: {best_satisfaction_model[1]['test_accuracy']:.3f})")
        
        # 특성 중요도 상위 5개
        if self.feature_importances['wage']:
            print("\n[임금 예측 주요 특성 (평균)]")
            wage_importance_avg = pd.DataFrame(self.feature_importances['wage']).mean(axis=1)
            for i, (feature, importance) in enumerate(wage_importance_avg.nlargest(5).items(), 1):
                print(f"  {i}. {feature}: {importance:.3f}")
        
        if self.feature_importances['satisfaction']:
            print("\n[만족도 예측 주요 특성 (평균)]")
            satisfaction_importance_avg = pd.DataFrame(self.feature_importances['satisfaction']).mean(axis=1)
            for i, (feature, importance) in enumerate(satisfaction_importance_avg.nlargest(5).items(), 1):
                print(f"  {i}. {feature}: {importance:.3f}")

def main():
    """메인 실행 함수"""
    
    # 비교 클래스 초기화
    comparison = BoostingModelComparison()
    
    # 1. 데이터 로드 및 준비
    df_ml = comparison.load_and_prepare_data()
    
    # 2. 특성 준비
    X, y_wage, y_satisfaction, y_satisfaction_class, available_features, label_encoders = comparison.prepare_features(df_ml)
    
    # 3. 데이터 분할
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, y_sat_class_train, y_sat_class_test = comparison.split_data_by_time(
        X, y_wage, y_satisfaction, y_satisfaction_class, df_ml
    )
    
    # 4. 임금 예측 모델 훈련
    wage_results = comparison.train_wage_models(X_train, y_wage_train, X_test, y_wage_test)
    
    # 5. 만족도 예측 모델 훈련  
    satisfaction_results = comparison.train_satisfaction_models(X_train, y_sat_class_train, X_test, y_sat_class_test)
    
    # 6. 특성 중요도 추출
    wage_importance, satisfaction_importance = comparison.extract_feature_importance(available_features)
    
    # 7. 시각화
    comparison.create_comparison_plots(y_wage_test, y_sat_class_test)
    
    # 8. 결과 요약
    comparison.print_summary()
    
    print("\n=== 부스팅 모델 비교 완료 ===")
    print("결과 시각화: visualizations/boosting_models_comparison.png")

if __name__ == "__main__":
    main()