import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import catboost as cb
import optuna
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("=== 빠른 하이퍼파라미터 최적화 ===")

def load_and_prepare_data():
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
    
    # 만족도 타겟 변수의 유효한 값(0 초과)만 필터링
    df_final = df_final[df_final['next_satisfaction'] > 0].copy()
    
    print(f"최종 데이터셋 크기: {df_final.shape}")
    
    # 특성 선택 (타겟 변수 제외)
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 
                   'p_wage', 'p4321']
    feature_cols = [col for col in df_final.columns if col not in exclude_cols]
    
    # 범주형 변수 인코딩
    for col in df_final[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
    
    # 결측값 처리
    df_final[feature_cols] = df_final[feature_cols].fillna(df_final[feature_cols].median())
    
    # 시간 기반 분할 (2000-2020: 훈련, 2021-2022: 테스트)
    train_mask = df_final['year'] <= 2020
    test_mask = df_final['year'] >= 2021
    
    X_train = df_final[train_mask][feature_cols]
    X_test = df_final[test_mask][feature_cols]
    y_wage_train = df_final[train_mask]['next_wage']
    y_wage_test = df_final[test_mask]['next_wage']
    y_sat_train = (df_final[train_mask]['next_satisfaction'] - 1).astype(int)
    y_sat_test = (df_final[test_mask]['next_satisfaction'] - 1).astype(int)
    
    print(f"훈련 데이터: {X_train.shape[0]}개")
    print(f"테스트 데이터: {X_test.shape[0]}개")
    print(f"특성 개수: {len(feature_cols)}개")
    
    return X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_cols

def optimize_catboost_wage(X_train, X_test, y_wage_train, y_wage_test, n_trials=20):
    """CatBoost 임금 예측 모델 빠른 최적화"""
    print(f"\nCatBoost 임금 예측 모델 최적화 시작 (시행 횟수: {n_trials})")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False
        }
        
        # 간단한 교차검증
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_wage_train.iloc[train_idx]
            y_fold_val = y_wage_train.iloc[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # 최적 모델 훈련
    best_params = study.best_params
    best_params['random_seed'] = 42
    best_params['verbose'] = False
    
    best_model = cb.CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_wage_train)
    
    # 성능 평가
    y_pred_test = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_wage_test, y_pred_test))
    test_r2 = r2_score(y_wage_test, y_pred_test)
    
    print(f"CatBoost 임금 예측 최적화 완료")
    print(f"최적 검증 RMSE: {study.best_value:.4f}")
    print(f"테스트 RMSE: {test_rmse:.4f}")
    print(f"테스트 R²: {test_r2:.4f}")
    print(f"최적 파라미터: {best_params}")
    
    return best_model, test_rmse, test_r2, best_params

def optimize_xgboost_satisfaction(X_train, X_test, y_sat_train, y_sat_test, n_trials=20):
    """XGBoost 만족도 예측 모델 빠른 최적화"""
    print(f"\nXGBoost 만족도 예측 모델 최적화 시작 (시행 횟수: {n_trials})")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'verbosity': 0
        }
        
        # 간단한 교차검증
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_sat_train.iloc[train_idx]
            y_fold_val = y_sat_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            accuracy = accuracy_score(y_fold_val, y_pred)
            scores.append(accuracy)
        
        return -np.mean(scores)  # 최소화를 위해 음수 사용
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # 최적 모델 훈련
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['objective'] = 'multi:softprob'
    best_params['eval_metric'] = 'mlogloss'
    best_params['verbosity'] = 0
    
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_sat_train)
    
    # 성능 평가
    y_pred_test = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_sat_test, y_pred_test)
    
    print(f"XGBoost 만족도 예측 최적화 완료")
    print(f"최적 검증 정확도: {-study.best_value:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"최적 파라미터: {best_params}")
    
    return best_model, test_accuracy, best_params

def main():
    start_time = time.time()
    
    # 데이터 로드
    X_train, X_test, y_wage_train, y_wage_test, y_sat_train, y_sat_test, feature_cols = load_and_prepare_data()
    
    # 결과 저장용 딕셔너리
    results = {}
    
    # CatBoost 임금 예측 최적화
    wage_model, wage_rmse, wage_r2, wage_params = optimize_catboost_wage(
        X_train, X_test, y_wage_train, y_wage_test, n_trials=20
    )
    results['catboost_wage'] = {
        'test_rmse': wage_rmse,
        'test_r2': wage_r2,
        'best_params': wage_params
    }
    
    # XGBoost 만족도 예측 최적화
    sat_model, sat_accuracy, sat_params = optimize_xgboost_satisfaction(
        X_train, X_test, y_sat_train, y_sat_test, n_trials=20
    )
    results['xgboost_satisfaction'] = {
        'test_accuracy': sat_accuracy,
        'best_params': sat_params
    }
    
    # 모델 저장
    joblib.dump(wage_model, 'models/quick_optimized_catboost_wage.pkl')
    joblib.dump(sat_model, 'models/quick_optimized_xgboost_satisfaction.pkl')
    
    # 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results/quick_optimization_results.csv')
    
    total_time = time.time() - start_time
    print(f"\n=== 빠른 최적화 완료 ===")
    print(f"총 소요시간: {total_time/60:.1f}분")
    print(f"CatBoost 임금 예측 RMSE: {wage_rmse:.2f}만원")
    print(f"XGBoost 만족도 예측 정확도: {sat_accuracy:.4f}")
    print("결과 파일:")
    print("- models/quick_optimized_*.pkl")
    print("- model_results/quick_optimization_results.csv")

if __name__ == "__main__":
    main()