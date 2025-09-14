# lstm/scripts/04_optimize_lstm.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import optuna
import joblib

print("Running script: 04_optimize_lstm.py")

# --- 설정값 ---
SEQUENCE_LENGTH = 5
EPOCHS = 25  # 최적화 시에는 Epoch를 약간 줄여서 빠르게 탐색
BATCH_SIZE = 128
N_TRIALS = 30 # Optuna 실행 횟수

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'lstm' / 'data'
RESULTS_DIR = PROJECT_ROOT / 'lstm' / 'results'

# 출력 폴더 생성
RESULTS_DIR.mkdir(exist_ok=True)
OPTUNA_RESULTS_PATH = RESULTS_DIR / 'lstm_optimization_results.pkl'

# --- 데이터 로딩 ---
print("Loading preprocessed sequence data...")
try:
    X_train = np.load(DATA_DIR / 'X_train.npy')
    X_test = np.load(DATA_DIR / 'X_test.npy')
    y_wage_train = np.load(DATA_DIR / 'y_wage_train.npy')
    y_wage_test = np.load(DATA_DIR / 'y_wage_test.npy')
    y_satisfaction_train = np.load(DATA_DIR / 'y_satisfaction_train.npy')
    y_satisfaction_test = np.load(DATA_DIR / 'y_satisfaction_test.npy')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("\nError: Processed data not found. Please run '01_create_sequences.py' first.")
    exit()

def build_model(trial):
    """Optuna trial 객체를 받아 하이퍼파라미터를 적용한 모델을 구축합니다."""
    num_features = X_train.shape[2]

    # --- 하이퍼파라미터 탐색 공간 정의 ---
    lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256, step=32)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    satisfaction_loss_weight = trial.suggest_float('satisfaction_loss_weight', 0.2, 1.0)

    # --- 모델 아키텍처 ---
    input_seq = Input(shape=(SEQUENCE_LENGTH, num_features), name='input_sequence')
    shared_lstm_1 = LSTM(lstm_units_1, return_sequences=True, name='shared_lstm_1')(input_seq)
    shared_lstm_1 = Dropout(dropout_rate)(shared_lstm_1)
    shared_lstm_2 = LSTM(lstm_units_2, name='shared_lstm_2')(shared_lstm_1)
    shared_lstm_2 = Dropout(dropout_rate)(shared_lstm_2)

    # 임금 예측 Head
    wage_head = Dense(32, activation='relu', name='wage_dense')(shared_lstm_2)
    wage_output = Dense(1, activation='linear', name='wage_output')(wage_head)

    # 만족도 예측 Head
    satisfaction_head = Dense(32, activation='relu', name='satisfaction_dense')(shared_lstm_2)
    satisfaction_output = Dense(5, activation='softmax', name='satisfaction_output')(satisfaction_head)

    model = Model(inputs=input_seq, outputs=[wage_output, satisfaction_output], name='multi_output_lstm_opt')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'wage_output': 'mean_squared_error',
            'satisfaction_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'wage_output': 1.0,
            'satisfaction_output': satisfaction_loss_weight
        },
        metrics={
            'wage_output': ['mae'],
            'satisfaction_output': ['accuracy']
        }
    )
    return model

def objective(trial):
    """Optuna가 최적화할 목적 함수입니다."""
    model = build_model(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(
        X_train,
        {'wage_output': y_wage_train, 'satisfaction_output': y_satisfaction_train},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, {'wage_output': y_wage_test, 'satisfaction_output': y_satisfaction_test}),
        callbacks=[early_stopping],
        verbose=0  # 로그를 최소화하여 Optuna 진행 상황을 잘 보이게 함
    )

    # 최적화의 기준이 될 검증 손실(validation loss)을 반환
    val_loss = min(history.history['val_loss'])
    return val_loss

def main():
    """메인 실행 함수"""
    study = optuna.create_study(direction='minimize')
    
    print(f"--- Starting Optuna Hyperparameter Optimization for {N_TRIALS} trials ---")
    study.optimize(objective, n_trials=N_TRIALS)
    print("--- Optimization Finished ---")

    # --- 결과 저장 및 출력 ---
    print("\n--- Optimization Results ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print(f"Best trial value (minimum validation loss): {best_trial.value:.4f}")
    
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    # 전체 스터디 결과 저장
    joblib.dump(study, OPTUNA_RESULTS_PATH)
    print(f"\nOptuna study saved to {OPTUNA_RESULTS_PATH}")
    print("You can now use these parameters to train the final model.")

if __name__ == "__main__":
    main()
