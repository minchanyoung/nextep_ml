# 프로젝트 폴더 구조

```
ML/
├── README_DATA_PREPROCESSING.md         # 메인 프로젝트 문서 (루트)
├── data/                               # 원본 및 메타데이터
│   ├── raw_data/                       # 원본 데이터
│   │   └── nextep_dataset.csv         # 원본 KLIPS 데이터 (369,307 rows)
│   │   └── nextep_dataset_codebook.csv # 데이터 코드북
│   └── feature_info_final.csv         # 생성된 특성 정보 메타데이터
├── processed_data/                     # 전처리된 ML 데이터셋
│   ├── ml_dataset_engineered.csv      # 최종 ML 데이터셋 (166,507 × 38)
│   ├── train_engineered.csv           # 훈련 데이터 (2000-2020, 143,113개)
│   ├── test_engineered.csv            # 테스트 데이터 (2021-2022, 23,394개)
│   ├── combined_prediction_clean.csv  # 통합 예측용 정제 데이터
│   ├── wage_prediction_clean.csv      # 임금 예측용 데이터
│   ├── satisfaction_prediction_clean.csv # 만족도 예측용 데이터
│   └── prediction_sample.csv          # 예측 샘플 데이터
├── scripts/                           # 데이터 전처리 스크립트
│   ├── data_exploration.py           # 기본 탐색적 데이터 분석
│   ├── target_analysis.py            # 타겟 변수 세부 분석
│   ├── panel_analysis.py             # 패널 데이터 구조 분석
│   ├── data_preprocessing.py         # 초기 전처리 스크립트
│   ├── data_preprocessing_efficient.py # 효율적인 전처리 스크립트
│   ├── feature_engineering.py        # 특성 엔지니어링 (초기)
│   └── feature_engineering_fixed.py  # 특성 엔지니어링 (최종)
├── visualizations/                    # 분석 결과 시각화
│   ├── data_exploration_plots.png    # 기본 데이터 탐색 결과
│   ├── target_analysis_plots.png     # 타겟 변수 분석 결과
│   ├── panel_analysis_plots.png      # 패널 데이터 분석 결과
│   ├── preprocessing_summary.png     # 전처리 요약 결과
│   └── feature_engineering_final.png # 특성 엔지니어링 결과
└── documentation/                     # 프로젝트 문서
    ├── README_DATA_PREPROCESSING.md  # 데이터 전처리 상세 보고서
    └── PROJECT_STRUCTURE.md         # 이 파일
```

## 📁 폴더별 상세 설명

### 🗂️ data/ - 원본 및 메타데이터
- **raw_data/**: 한국노동패널조사 원본 데이터
- **feature_info_final.csv**: 33개 생성 특성의 상세 정보 (타입, 결측값, 통계량 등)

### 🔄 processed_data/ - 머신러닝용 데이터셋
- **ml_dataset_engineered.csv**: 33개 특성으로 구성된 최종 ML 데이터셋
- **train/test_engineered.csv**: 시간 기반 분할된 훈련/테스트 데이터
- **prediction_clean.csv**: 타겟별 정제된 예측용 데이터

### 🐍 scripts/ - 데이터 전처리 코드
각 스크립트는 순차적 실행 가능하며, 다음 단계를 포함:
1. **탐색적 분석**: data_exploration.py → target_analysis.py → panel_analysis.py
2. **전처리**: data_preprocessing_efficient.py 
3. **특성 생성**: feature_engineering_fixed.py

### 📊 visualizations/ - 분석 결과 그래프
- 데이터 분포, 시계열 패턴, 상관관계, 특성 중요도 등 시각화 결과

### 📚 documentation/ - 프로젝트 문서
- 전처리 과정 상세 보고서 및 프로젝트 구조 안내

## 🎯 다음 단계 작업

### 1. 머신러닝 모델링
```
ML/
├── models/                    # 모델 구현 코드
├── model_results/            # 모델 성능 결과
└── predictions/              # 예측 결과
```

### 2. 권장 작업 순서
1. `processed_data/train_engineered.csv` 로드
2. Random Forest/XGBoost 베이스라인 모델 구축
3. 시계열 특성 활용 LSTM 모델 개발  
4. 모델 성능 비교 및 앙상블
5. `processed_data/test_engineered.csv`로 최종 평가

## 💡 사용법

### 데이터 로드 예제
```python
import pandas as pd

# 최종 ML 데이터셋 로드
df = pd.read_csv('processed_data/ml_dataset_engineered.csv')

# 훈련/테스트 데이터 로드  
train = pd.read_csv('processed_data/train_engineered.csv')
test = pd.read_csv('processed_data/test_engineered.csv')

# 특성과 타겟 분리
features = [col for col in train.columns if col not in ['pid', 'year', 'next_year', 'next_wage', 'next_satisfaction']]
X_train = train[features]
y_wage = train['next_wage']
y_satisfaction = train['next_satisfaction']
```

### 특성 정보 확인
```python
# 특성 메타데이터 로드
feature_info = pd.read_csv('data/feature_info_final.csv')
print(feature_info.head())
```