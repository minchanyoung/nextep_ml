import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import json

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class PerformanceMonitor:
    """모델 성능 모니터링 및 종합 리포트 생성 시스템"""
    
    def __init__(self):
        self.results = {}
        self.models_info = {}
        
    def check_optimization_status(self):
        """최적화 작업 상태 확인"""
        print("=== 모델 최적화 상태 확인 ===")
        
        # 모델 파일들 확인
        model_files = [
            'models/best_catboost_wage.pkl',
            'models/best_xgboost_satisfaction.pkl'
        ]
        
        result_files = [
            'model_results/optimization_results.csv'
        ]
        
        print("\n생성된 파일 확인:")
        for file_path in model_files + result_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"[OK] {file_path} ({size:.1f}KB, 수정: {modified.strftime('%H:%M:%S')})")
            else:
                print(f"[X] {file_path} - 파일 없음")
        
        return all(os.path.exists(f) for f in model_files)
    
    def load_optimization_results(self):
        """최적화 결과 로드"""
        try:
            if os.path.exists('model_results/optimization_results.csv'):
                results_df = pd.read_csv('model_results/optimization_results.csv', index_col=0)
                self.results = results_df.to_dict('index')
                print("[OK] 최적화 결과 로드 완료")
                return True
            else:
                print("[주의] 최적화 결과 파일이 아직 생성되지 않았습니다.")
                return False
        except Exception as e:
            print(f"[오류] 결과 로드 중 오류: {e}")
            return False
    
    def compare_with_baseline(self):
        """기존 베이스라인과 성능 비교"""
        print("\n=== 베이스라인 대비 성능 향상 분석 ===")
        
        # 기존 부스팅 모델 비교 결과 (README에서 확인된 수치)
        baseline_results = {
            'catboost_wage': {
                'test_rmse': 115.92,
                'test_r2': 0.704,
                'test_mae': 54.77
            },
            'xgboost_satisfaction': {
                'test_accuracy': 0.694
            }
        }
        
        print("📊 성능 비교:")
        print("-" * 60)
        
        if 'catboost_wage' in self.results:
            current = self.results['catboost_wage']
            baseline = baseline_results['catboost_wage']
            
            rmse_improvement = baseline['test_rmse'] - current.get('test_rmse', baseline['test_rmse'])
            r2_improvement = current.get('test_r2', baseline['test_r2']) - baseline['test_r2']
            mae_improvement = baseline['test_mae'] - current.get('test_mae', baseline['test_mae'])
            
            print(f"CatBoost 임금 예측 모델:")
            print(f"   RMSE: {baseline['test_rmse']:.2f} → {current.get('test_rmse', 'N/A')} ({rmse_improvement:+.2f})")
            print(f"   R²:   {baseline['test_r2']:.4f} → {current.get('test_r2', 'N/A')} ({r2_improvement:+.4f})")
            print(f"   MAE:  {baseline['test_mae']:.2f} → {current.get('test_mae', 'N/A')} ({mae_improvement:+.2f})")
        
        if 'xgboost_satisfaction' in self.results:
            current = self.results['xgboost_satisfaction']
            baseline = baseline_results['xgboost_satisfaction']
            
            acc_improvement = current.get('test_accuracy', baseline['test_accuracy']) - baseline['test_accuracy']
            
            print(f"\nXGBoost 만족도 예측 모델:")
            print(f"   정확도: {baseline['test_accuracy']:.4f} → {current.get('test_accuracy', 'N/A')} ({acc_improvement:+.4f})")
    
    def create_progress_visualization(self):
        """최적화 진행 상황 시각화"""
        print("\n시각화 생성 중...")
        
        if not self.results:
            print("[주의] 최적화 결과가 없어 시각화를 생성할 수 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 모델 성능 비교 (기존 vs 최적화)
        models = []
        baseline_scores = []
        optimized_scores = []
        metrics = []
        
        if 'catboost_wage' in self.results:
            models.append('CatBoost\n(임금 예측)')
            baseline_scores.append(0.704)  # 기존 R²
            optimized_scores.append(self.results['catboost_wage'].get('test_r2', 0.704))
            metrics.append('R²')
        
        if 'xgboost_satisfaction' in self.results:
            models.append('XGBoost\n(만족도 예측)')
            baseline_scores.append(0.694)  # 기존 정확도
            optimized_scores.append(self.results['xgboost_satisfaction'].get('test_accuracy', 0.694))
            metrics.append('정확도')
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, baseline_scores, width, label='기존 모델', alpha=0.8, color='lightblue')
            axes[0, 0].bar(x + width/2, optimized_scores, width, label='최적화 모델', alpha=0.8, color='orange')
            axes[0, 0].set_xlabel('모델')
            axes[0, 0].set_ylabel('성능 점수')
            axes[0, 0].set_title('기존 vs 최적화 모델 성능 비교', fontweight='bold')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(models)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 최적화 진행 시간 정보 (예상)
        axes[0, 1].text(0.5, 0.5, '최적화 진행 상황\n\n✅ 데이터 전처리\n✅ 베이스라인 모델\n🔄 하이퍼파라미터 최적화\n⏳ SHAP 분석\n⏳ 앙상블 모델', 
                       ha='center', va='center', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[0, 1].set_xlim(0, 1)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('작업 진행 상태', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 성능 메트릭 상세 분석
        if 'catboost_wage' in self.results:
            metrics_data = {
                'RMSE (만원)': self.results['catboost_wage'].get('test_rmse', 115.92),
                'MAE (만원)': self.results['catboost_wage'].get('test_mae', 54.77),
                'R² Score': self.results['catboost_wage'].get('test_r2', 0.704)
            }
            
            colors = ['lightcoral' if 'RMSE' in k or 'MAE' in k else 'lightblue' for k in metrics_data.keys()]
            bars = axes[1, 0].bar(metrics_data.keys(), metrics_data.values(), color=colors, alpha=0.8)
            axes[1, 0].set_title('CatBoost 임금 예측 모델 상세 성능', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, metrics_data.values()):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # 프로젝트 타임라인
        timeline_data = {
            '데이터 전처리': 100,
            '탐색적 분석': 100,
            '특성 엔지니어링': 100,
            '직업 분류 통합': 100,
            '베이스라인 모델': 100,
            '모델 최적화': 60,  # 진행 중
            'SHAP 분석': 0,
            '앙상블 모델': 0,
            '최종 배포': 0
        }
        
        colors = ['green' if v == 100 else 'orange' if v > 0 else 'lightgray' for v in timeline_data.values()]
        bars = axes[1, 1].barh(list(timeline_data.keys()), list(timeline_data.values()), color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('진행률 (%)')
        axes[1, 1].set_title('프로젝트 전체 진행 상황', fontweight='bold')
        axes[1, 1].set_xlim(0, 100)
        
        # 진행률 텍스트 표시
        for i, (task, progress) in enumerate(timeline_data.items()):
            axes[1, 1].text(progress + 2, i, f'{progress}%', va='center')
        
        plt.tight_layout()
        plt.savefig('visualizations/optimization_progress_monitor.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_intermediate_report(self):
        """중간 진행 상황 보고서"""
        print("\n" + "="*60)
        print("모델 최적화 중간 보고서")
        print("="*60)
        
        print(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"작업 상태: 하이퍼파라미터 최적화 진행 중")
        
        # 파일 상태 체크
        files_status = self.check_optimization_status()
        
        if files_status:
            print("\n[완료] 최적화 작업이 완료되었습니다!")
            self.load_optimization_results()
            self.compare_with_baseline()
        else:
            print("\n[진행중] 최적화 작업이 아직 진행 중입니다...")
            print("\n예상 완료 시간: 약 10-20분")
            print("\n현재까지의 진행 상황:")
            print("   [완료] 1단계: 데이터 로드 및 전처리 완료")
            print("   [진행] 2단계: CatBoost 임금 예측 모델 최적화 진행 중")
            print("   [대기] 3단계: XGBoost 만족도 예측 모델 최적화 대기")
            print("   [대기] 4단계: SHAP 분석 대기")
            print("   [대기] 5단계: 결과 저장 및 시각화 대기")
        
        # 시각화 생성
        self.create_progress_visualization()
        
        print(f"\n시각화 결과: visualizations/optimization_progress_monitor.png")
        print("\n다음 단계:")
        if not files_status:
            print("   1. 현재 최적화 작업 완료 대기")
            print("   2. SHAP 분석 실행")
            print("   3. 앙상블 모델 구축")
            print("   4. 최종 성능 평가 및 보고서 생성")
        else:
            print("   1. SHAP 분석 실행")
            print("   2. 앙상블 모델 구축")
            print("   3. 최종 성능 평가 및 보고서 생성")

def main():
    """메인 실행 함수"""
    monitor = PerformanceMonitor()
    monitor.generate_intermediate_report()

if __name__ == "__main__":
    main()