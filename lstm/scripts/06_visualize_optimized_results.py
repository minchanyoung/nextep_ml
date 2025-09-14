# lstm/scripts/06_visualize_optimized_results.py

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

print("Running script: 06_visualize_optimized_results.py")

# --- 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'lstm' / 'results' / 'optimized'
VISUALIZATIONS_DIR = PROJECT_ROOT / 'lstm' / 'visualizations' / 'optimized'

# 출력 폴더 생성
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history):
    """훈련 과정의 손실과 정확도를 시각화합니다."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. 전체 손실 (Total Loss)
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('최적화 모델 전체 손실 (Total Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 2. 임금 예측 손실 (Wage Loss)
    axes[1].plot(history['wage_output_loss'], label='Train Wage Loss')
    axes[1].plot(history['val_wage_output_loss'], label='Validation Wage Loss')
    axes[1].set_title('최적화 모델 임금 예측 손실 (Wage Prediction Loss)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)

    # 3. 만족도 예측 정확도 (Satisfaction Accuracy)
    axes[2].plot(history['satisfaction_output_accuracy'], label='Train Satisfaction Accuracy')
    axes[2].plot(history['val_satisfaction_output_accuracy'], label='Validation Satisfaction Accuracy')
    axes[2].set_title('최적화 모델 직업 만족도 예측 정확도')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = VISUALIZATIONS_DIR / 'lstm_optimized_training_history.png'
    plt.savefig(save_path)
    print(f"Optimized training history plot saved to {save_path}")
    plt.close()

def plot_wage_predictions(y_true, y_pred):
    """임금 예측 결과(실제값 vs 예측값)를 시각화합니다."""
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, label='Predictions')
    
    # y=x 라인 (완벽한 예측)
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title('최적화 모델 임금 예측 결과: 실제값 vs. 예측값', fontsize=16)
    plt.xlabel('실제 임금 (만원)', fontsize=12)
    plt.ylabel('예측 임금 (만원)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    save_path = VISUALIZATIONS_DIR / 'lstm_optimized_wage_prediction_scatter.png'
    plt.savefig(save_path)
    print(f"Optimized wage prediction scatter plot saved to {save_path}")
    plt.close()

def plot_satisfaction_confusion_matrix(y_true, y_pred):
    """직업 만족도 예측 결과(Confusion Matrix)를 시각화합니다."""
    labels = ['매우 불만족', '불만족', '보통', '만족', '매우 만족']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title('최적화 모델 직업 만족도 예측 Confusion Matrix', fontsize=16)
    plt.xlabel('예측된 클래스', fontsize=12)
    plt.ylabel('실제 클래스', fontsize=12)
    
    save_path = VISUALIZATIONS_DIR / 'lstm_optimized_satisfaction_confusion_matrix.png'
    plt.savefig(save_path)
    print(f"Optimized satisfaction confusion matrix saved to {save_path}")
    plt.close()

def main():
    """메인 실행 함수"""
    print("Loading optimized results for visualization...")
    try:
        history = joblib.load(RESULTS_DIR / 'training_history_optimized.pkl')
        y_wage_test = np.load(RESULTS_DIR / 'y_wage_test.npy', allow_pickle=True)
        y_pred_wage = np.load(RESULTS_DIR / 'y_pred_wage_optimized.npy', allow_pickle=True)
        y_satisfaction_test = np.load(RESULTS_DIR / 'y_satisfaction_test.npy', allow_pickle=True)
        y_pred_satisfaction = np.load(RESULTS_DIR / 'y_pred_satisfaction_optimized.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"\nError: Result file not found. {e}")
        print("Please run '05_train_optimized_lstm.py' first.")
        return

    # 시각화 함수 호출
    plot_training_history(history)
    plot_wage_predictions(y_wage_test, y_pred_wage.flatten())
    plot_satisfaction_confusion_matrix(y_satisfaction_test, y_pred_satisfaction)

    print("\nVisualization for optimized model complete.")

if __name__ == "__main__":
    main()
