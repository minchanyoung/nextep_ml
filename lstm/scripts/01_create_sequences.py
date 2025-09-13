# lstm/scripts/01_create_sequences.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import joblib

print("Running script: 01_create_sequences.py")

# --- 설정값 ---
SEQUENCE_LENGTH = 5  # 5년치 데이터를 기반으로 다음 해를 예측

# --- 경로 설정 ---
# 현재 스크립트의 위치를 기준으로 프로젝트 루트 경로를 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATA_PATH = PROJECT_ROOT / 'processed_data' / 'ml_dataset_engineered.csv'
OUTPUT_DIR = PROJECT_ROOT / 'lstm' / 'data'

# 출력 폴더 생성
OUTPUT_DIR.mkdir(exist_ok=True)

def create_sequences(data: pd.DataFrame):
    """Pandas DataFrame을 받아 시퀀스 데이터와 타겟을 생성합니다."""
    
    # 사용할 특성 선택 (타겟 및 ID 제외)
    exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features.")

    # 데이터 스케일링 (MinMax 스케일러 사용)
    scaler = MinMaxScaler()
    data_scaled = data.copy()
    data_scaled[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    # 스케일러 저장 (나중에 예측 시 동일하게 사용하기 위함)
    joblib.dump(scaler, OUTPUT_DIR / 'feature_scaler.pkl')
    print(f"Feature scaler saved to {OUTPUT_DIR / 'feature_scaler.pkl'}")

    X, y_wage, y_satisfaction, meta_info = [], [], [], []

    # 개인별(pid)로 데이터 그룹화
    grouped = data_scaled.groupby('pid')
    total_pids = len(grouped)

    for i, (pid, group) in enumerate(grouped):
        if i % 1000 == 0:
            print(f"Processing PID {i}/{total_pids}...")
            
        # 시간순으로 정렬
        group = group.sort_values('year')
        num_records = len(group)

        # 슬라이딩 윈도우로 시퀀스 생성
        for j in range(num_records - SEQUENCE_LENGTH):
            # 입력 시퀀스 (t, t+1, ..., t+4)
            sequence = group.iloc[j:j + SEQUENCE_LENGTH][feature_cols].values
            
            # 타겟 (t+5 시점의 값)
            target_record = group.iloc[j + SEQUENCE_LENGTH]
            target_wage = target_record['next_wage']
            target_satisfaction = target_record['next_satisfaction']
            
            # 유효한 타겟이 있는 경우에만 추가
            if pd.notna(target_wage) and pd.notna(target_satisfaction):
                X.append(sequence)
                y_wage.append(target_wage)
                # 만족도는 0-4 범위로 조정
                y_satisfaction.append(target_satisfaction - 1)
                
                # 나중에 훈련/테스트 분할을 위해 타겟의 연도 정보 저장
                meta_info.append({'pid': pid, 'target_year': target_record['year']})

    return np.array(X), np.array(y_wage), np.array(y_satisfaction), pd.DataFrame(meta_info)

def main():
    """메인 실행 함수"""
    print(f"Loading data from {INPUT_DATA_PATH}")
    if not INPUT_DATA_PATH.exists():
        raise FileNotFoundError(f"Input data not found at {INPUT_DATA_PATH}. Please ensure the main project data is available.")

    df = pd.read_csv(INPUT_DATA_PATH)

    print("Creating sequences...")
    X, y_wage, y_satisfaction, meta_df = create_sequences(df)

    print(f"Total sequences created: {len(X)}")

    # --- 훈련 / 테스트 데이터 분할 ---
    # 원본 프로젝트와 동일하게 2021년 이후 데이터를 테스트셋으로 사용
    test_indices = meta_df[meta_df['target_year'] >= 2021].index
    train_indices = meta_df[meta_df['target_year'] < 2021].index

    X_train, X_test = X[train_indices], X[test_indices]
    y_wage_train, y_wage_test = y_wage[train_indices], y_wage[test_indices]
    y_satisfaction_train, y_satisfaction_test = y_satisfaction[train_indices], y_satisfaction[test_indices]

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # --- 데이터 저장 ---
    print(f"Saving processed data to {OUTPUT_DIR}")
    np.save(OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(OUTPUT_DIR / 'y_wage_train.npy', y_wage_train)
    np.save(OUTPUT_DIR / 'y_wage_test.npy', y_wage_test)
    np.save(OUTPUT_DIR / 'y_satisfaction_train.npy', y_satisfaction_train)
    np.save(OUTPUT_DIR / 'y_satisfaction_test.npy', y_satisfaction_test)

    print("\nData processing complete. Files saved:")
    for filename in os.listdir(OUTPUT_DIR):
        print(f"- {filename}")

if __name__ == "__main__":
    main()
