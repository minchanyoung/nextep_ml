# lstm/scripts/02_train_lstm.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import joblib

print("Running script: 02_train_lstm.py (v2)")

# --- 설정값 ---
SEQUENCE_LENGTH = 5
EPOCHS = 50
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'lstm' / 'data'
MODEL_DIR = PROJECT_ROOT / 'lstm' / 'models'
RESULTS_DIR = PROJECT_ROOT / 'lstm' / 'results' # 결과 저장용 폴더

# 출력 폴더 생성
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / 'best_lstm_model.keras'

def build_model(seq_length, num_features):
    """Multi-output LSTM 모델을 구축합니다."""
    input_seq = Input(shape=(seq_length, num_features), name='input_sequence')
    shared_lstm_1 = LSTM(128, return_sequences=True, name='shared_lstm_1')(input_seq)
    shared_lstm_1 = Dropout(0.3)(shared_lstm_1)
    shared_lstm_2 = LSTM(64, name='shared_lstm_2')(shared_lstm_1)
    shared_lstm_2 = Dropout(0.3)(shared_lstm_2)

    # 임금 예측 Head
    wage_head = Dense(32, activation='relu', name='wage_dense')(shared_lstm_2)
    wage_output = Dense(1, activation='linear', name='wage_output')(wage_head)

    # 만족도 예측 Head
    satisfaction_head = Dense(32, activation='relu', name='satisfaction_dense')(shared_lstm_2)
    satisfaction_output = Dense(5, activation='softmax', name='satisfaction_output')(satisfaction_head)

    model = Model(inputs=input_seq, outputs=[wage_output, satisfaction_output], name='multi_output_lstm')

    model.compile(
        optimizer='adam',
        loss={
            'wage_output': 'mean_squared_error',
            'satisfaction_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'wage_output': 1.0,
            'satisfaction_output': 0.5
        },
        metrics={
            'wage_output': ['mae'],
            'satisfaction_output': ['accuracy']
        }
    )
    return model

def main():
    """메인 실행 함수"""
    print("Loading preprocessed sequence data...")
    try:
        X_train = np.load(DATA_DIR / 'X_train.npy')
        X_test = np.load(DATA_DIR / 'X_test.npy')
        y_wage_train = np.load(DATA_DIR / 'y_wage_train.npy')
        y_wage_test = np.load(DATA_DIR / 'y_wage_test.npy')
        y_satisfaction_train = np.load(DATA_DIR / 'y_satisfaction_train.npy')
        y_satisfaction_test = np.load(DATA_DIR / 'y_satisfaction_test.npy')
    except FileNotFoundError:
        print("\nError: Processed data not found. Please run '01_create_sequences.py' first.")
        return

    num_features = X_train.shape[2]
    model = build_model(SEQUENCE_LENGTH, num_features)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(str(MODEL_SAVE_PATH), save_best_only=True, monitor='val_loss')

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train,
        {'wage_output': y_wage_train, 'satisfaction_output': y_satisfaction_train},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, {'wage_output': y_wage_test, 'satisfaction_output': y_satisfaction_test}),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    print("--- Model Training Finished ---")

    # --- 모델 평가 ---
    print("\n--- Evaluating Model Performance ---")
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    y_pred_wage, y_pred_satisfaction = best_model.predict(X_test)
    y_pred_satisfaction_class = np.argmax(y_pred_satisfaction, axis=1)

    # 성능 계산
    wage_rmse = np.sqrt(mean_squared_error(y_wage_test, y_pred_wage))
    wage_mae = mean_absolute_error(y_wage_test, y_pred_wage)
    wage_r2 = r2_score(y_wage_test, y_pred_wage)
    satisfaction_accuracy = np.mean(y_pred_satisfaction_class == y_satisfaction_test)

    print("\n--- Final Test Results ---")
    print(f"💰 Wage Prediction (임금 예측)")
    print(f"   - RMSE: {wage_rmse:.2f} 만원")
    print(f"   - MAE:  {wage_mae:.2f} 만원")
    print(f"   - R²:   {wage_r2:.4f}")
    print("\n😊 Satisfaction Prediction (만족도 예측)")
    print(f"   - Accuracy: {satisfaction_accuracy:.4f}")
    print("--------------------------")

    # --- 시각화를 위한 결과 저장 ---
    print("\nSaving results for visualization...")
    # 1. 훈련 히스토리 저장
    joblib.dump(history.history, RESULTS_DIR / 'training_history.pkl')
    # 2. 테스트셋 예측 결과 및 실제값 저장
    np.save(RESULTS_DIR / 'y_wage_test.npy', y_wage_test)
    np.save(RESULTS_DIR / 'y_pred_wage.npy', y_pred_wage)
    np.save(RESULTS_DIR / 'y_satisfaction_test.npy', y_satisfaction_test)
    np.save(RESULTS_DIR / 'y_pred_satisfaction.npy', y_pred_satisfaction_class)
    print("Results saved successfully.")

if __name__ == "__main__":
    main()