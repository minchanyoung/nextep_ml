import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 성능 개선 분석 및 제안 ===")

class PerformanceImprovementAnalysis:
    """성능 개선 방안 분석 및 실험 시스템"""
    
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        
    def load_and_analyze_data(self):
        """데이터 로드 및 기본 분석"""
        print("데이터 로드 및 분석...")
        
        # 기존 결과 로드
        if os.path.exists('model_results/full_dataset_model_results.csv'):
            results_df = pd.read_csv('model_results/full_dataset_model_results.csv', index_col=0)
            print("기존 모델 성능:")
            print(results_df.head())
        
        # 원본 데이터 로드
        df = pd.read_csv('processed_data/dataset_with_unified_occupations.csv')
        
        # 예측 데이터셋 생성
        df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)
        df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
        df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
        df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
        
        consecutive_mask = (df_sorted['next_year'] == df_sorted['year'] + 1)
        df_consecutive = df_sorted[consecutive_mask].copy()
        
        valid_mask = (~df_consecutive['next_wage'].isna()) & (~df_consecutive['next_satisfaction'].isna()) & (df_consecutive['next_satisfaction'] > 0)
        self.data = df_consecutive[valid_mask].copy()
        
        print(f"분석 데이터: {self.data.shape[0]:,}개 관측값")
        
        return self.data
    
    def analyze_data_quality_issues(self):
        """데이터 품질 이슈 분석"""
        print("\n1. 데이터 품질 이슈 분석")
        print("-" * 50)
        
        # 결측값 패턴 분석
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        missing_stats = []
        for col in feature_cols:
            missing_count = self.data[col].isna().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            missing_stats.append({
                'feature': col,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'dtype': str(self.data[col].dtype)
            })
        
        missing_df = pd.DataFrame(missing_stats)
        missing_df = missing_df[missing_df['missing_pct'] > 0].sort_values('missing_pct', ascending=False)
        
        print(f"결측값이 있는 특성: {len(missing_df)}개")
        if len(missing_df) > 0:
            print("상위 5개 결측값 특성:")
            print(missing_df.head()[['feature', 'missing_pct', 'dtype']])
        else:
            print("결측값 없음 - 이미 처리됨")
        
        # 타겟 변수 분포 분석
        print(f"\n타겟 변수 분포:")
        print(f"임금 - 평균: {self.data['next_wage'].mean():.1f}, 표준편차: {self.data['next_wage'].std():.1f}")
        print(f"만족도 - 분포: {self.data['next_satisfaction'].value_counts().sort_index().to_dict()}")
        
        # 이상치 분석
        wage_q99 = self.data['next_wage'].quantile(0.99)
        wage_outliers = (self.data['next_wage'] > wage_q99).sum()
        print(f"임금 이상치 (99% 초과): {wage_outliers}개 ({wage_outliers/len(self.data)*100:.1f}%)")
        
        self.analysis_results['data_quality'] = {
            'missing_features': len(missing_df),
            'wage_outliers_pct': wage_outliers/len(self.data)*100
        }
    
    def analyze_feature_importance(self):
        """특성 중요도 분석"""
        print("\n2. 특성 중요도 분석")
        print("-" * 50)
        
        from sklearn.preprocessing import LabelEncoder
        
        # 데이터 전처리
        exclude_cols = ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction', 'p_wage', 'p4321']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        df_features = self.data[feature_cols].copy()
        
        # 범주형 변수 인코딩
        for col in df_features.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
        
        # 결측값 처리
        df_features = df_features.fillna(df_features.median())
        
        # 임금 예측용 특성 중요도
        y_wage = self.data['next_wage']
        selector_wage = SelectKBest(score_func=f_regression, k='all')
        selector_wage.fit(df_features, y_wage)
        
        wage_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector_wage.scores_
        }).sort_values('score', ascending=False)
        
        print("임금 예측 중요 특성 (상위 10개):")
        print(wage_scores.head(10))
        
        # 만족도 예측용 특성 중요도
        y_sat = (self.data['next_satisfaction'] - 1).astype(int)
        selector_sat = SelectKBest(score_func=f_classif, k='all')
        selector_sat.fit(df_features, y_sat)
        
        sat_scores = pd.DataFrame({
            'feature': feature_cols,
            'score': selector_sat.scores_
        }).sort_values('score', ascending=False)
        
        print("\n만족도 예측 중요 특성 (상위 10개):")
        print(sat_scores.head(10))
        
        self.analysis_results['feature_importance'] = {
            'wage_top_features': wage_scores.head(20)['feature'].tolist(),
            'sat_top_features': sat_scores.head(20)['feature'].tolist()
        }
    
    def analyze_model_performance_gaps(self):
        """모델 성능 갭 분석"""
        print("\n3. 모델 성능 갭 분석")
        print("-" * 50)
        
        # 현재 성능
        current_performance = {
            'wage_rmse': 118.89,
            'wage_mae': 58.35,
            'wage_r2': 0.6776,
            'sat_accuracy': 0.6716
        }
        
        # 베이스라인 성능
        baseline_performance = {
            'wage_rmse': 115.92,
            'sat_accuracy': 0.694
        }
        
        # 성능 갭 계산
        wage_rmse_gap = current_performance['wage_rmse'] - baseline_performance['wage_rmse']
        sat_acc_gap = baseline_performance['sat_accuracy'] - current_performance['sat_accuracy']
        
        print(f"성능 갭 분석:")
        print(f"  임금 RMSE 갭: +{wage_rmse_gap:.2f}만원 ({wage_rmse_gap/baseline_performance['wage_rmse']*100:+.1f}%)")
        print(f"  만족도 정확도 갭: -{sat_acc_gap:.4f} ({sat_acc_gap/baseline_performance['sat_accuracy']*100:-.1f}%)")
        
        # 이론적 최대 성능 추정 (Cross-validation 기반)
        print(f"\n현재 모델 교차검증 분석이 필요합니다.")
        
        self.analysis_results['performance_gaps'] = {
            'wage_rmse_gap': wage_rmse_gap,
            'sat_accuracy_gap': sat_acc_gap
        }
    
    def suggest_improvement_strategies(self):
        """개선 전략 제안"""
        print("\n4. 개선 전략 제안")
        print("-" * 50)
        
        strategies = []
        
        # 데이터 품질 기반 제안
        if self.analysis_results.get('data_quality', {}).get('wage_outliers_pct', 0) > 1:
            strategies.append({
                'category': '데이터 품질',
                'strategy': '이상치 처리 개선',
                'description': f"임금 이상치 {self.analysis_results['data_quality']['wage_outliers_pct']:.1f}% 존재, Robust Scaling 적용",
                'expected_improvement': '2-3% RMSE 개선'
            })
        
        # 특성 엔지니어링 제안
        strategies.append({
            'category': '특성 엔지니어링',
            'strategy': '고차원 상호작용 특성 생성',
            'description': '중요 특성들의 곱셈/나눗셈 조합, 다항식 특성',
            'expected_improvement': '3-5% 성능 향상'
        })
        
        # 모델링 기법 제안
        strategies.append({
            'category': '모델링',
            'strategy': 'Stacking 앙상블 적용',
            'description': '현재 Voting → Stacking으로 변경, Meta-learner 추가',
            'expected_improvement': '2-4% 성능 향상'
        })
        
        # 하이퍼파라미터 최적화
        strategies.append({
            'category': '최적화',
            'strategy': 'Bayesian Optimization',
            'description': 'Optuna를 활용한 고급 하이퍼파라미터 최적화',
            'expected_improvement': '1-3% 성능 향상'
        })
        
        # 시간 시리즈 특화
        strategies.append({
            'category': '시계열',
            'strategy': '개인별 시간 트렌드 모델링',
            'description': '개인별 고정효과 + 시간 트렌드 결합',
            'expected_improvement': '3-6% 성능 향상'
        })
        
        # 결과 출력
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. [{strategy['category']}] {strategy['strategy']}")
            print(f"   설명: {strategy['description']}")
            print(f"   예상 효과: {strategy['expected_improvement']}")
            print()
        
        # 우선순위 제안
        print("권장 우선순위:")
        print("1순위: Stacking 앙상블 (즉시 적용 가능, 확실한 효과)")
        print("2순위: 고차원 특성 엔지니어링 (중간 노력, 높은 효과)")
        print("3순위: Bayesian 최적화 (높은 노력, 확실한 효과)")
        print("4순위: 시간 트렌드 모델링 (높은 노력, 잠재적 고효과)")
        
        return strategies
    
    def create_improvement_roadmap(self):
        """개선 로드맵 생성"""
        print("\n5. 개선 로드맵")
        print("-" * 50)
        
        phases = [
            {
                'phase': 'Phase 1: 빠른 개선 (1-2일)',
                'tasks': [
                    'Stacking 앙상블 구현',
                    '이상치 Robust Scaling',
                    '교차검증 전략 개선'
                ],
                'expected_gain': '3-5% 성능 향상'
            },
            {
                'phase': 'Phase 2: 특성 고도화 (3-5일)',
                'tasks': [
                    '고차원 상호작용 특성 생성',
                    '시간 윈도우 확장 특성',
                    '업종별 벤치마킹 특성'
                ],
                'expected_gain': '2-4% 추가 향상'
            },
            {
                'phase': 'Phase 3: 모델 고도화 (5-7일)',
                'tasks': [
                    'Bayesian 하이퍼파라미터 최적화',
                    'Neural Network 모델 추가',
                    '개인별 고정효과 모델'
                ],
                'expected_gain': '2-3% 추가 향상'
            },
            {
                'phase': 'Phase 4: 고급 기법 (7-10일)',
                'tasks': [
                    '다중 과제 학습 (Multi-task)',
                    '도메인 적응 기법',
                    '온라인 학습 시스템'
                ],
                'expected_gain': '1-2% 추가 향상'
            }
        ]
        
        total_expected = 0
        for phase in phases:
            print(f"\n{phase['phase']}")
            for task in phase['tasks']:
                print(f"  • {task}")
            print(f"  예상 효과: {phase['expected_gain']}")
            
            # 수치 추출 (간단한 파싱)
            gains = phase['expected_gain'].split('-')
            if len(gains) >= 2:
                avg_gain = (int(gains[0]) + int(gains[1].split('%')[0])) / 2
                total_expected += avg_gain
        
        print(f"\n총 예상 성능 향상: {total_expected:.1f}%")
        print(f"목표: 베이스라인 대비 동등 또는 우수한 성능 달성")

def main():
    """메인 실행 함수"""
    import os
    
    analyzer = PerformanceImprovementAnalysis()
    
    # 분석 실행
    analyzer.load_and_analyze_data()
    analyzer.analyze_data_quality_issues()
    analyzer.analyze_feature_importance()
    analyzer.analyze_model_performance_gaps()
    
    # 개선 전략 제안
    strategies = analyzer.suggest_improvement_strategies()
    analyzer.create_improvement_roadmap()
    
    print(f"\n" + "="*60)
    print("성능 개선 분석 완료!")
    print("다음 단계: 위 제안 중 우선순위에 따라 구현 시작")
    print("="*60)

if __name__ == "__main__":
    main()