# 프로젝트 폴더 구조 (최종)

이 문서는 KLIPS 데이터 분석 및 예측 모델 개발 프로젝트의 최종 폴더 구조를 설명합니다.

```
ML/
├── .gitignore
├── MODEL_INTEGRATION_GUIDE.md      # 외부 프로젝트 통합 가이드
├── PROJECT_PROGRESS_UPDATE.md      # 전체 프로젝트 진행 과정 및 최종 보고서 (마스터 문서)
├── README_DATA_PREPROCESSING.md    # 데이터 전처리 및 초기 모델링 상세 보고서
└── STACKING_SUCCESS_REPORT.md      # Stacking 앙상블 실험 결과 보고서

├── data/                           # 원본 및 메타데이터
│   ├── raw_data/                   # 원본 KLIPS 데이터
│   └── ... (직업 매핑 json 등)
│
├── processed_data/                 # 전처리 및 특성 엔지니어링이 완료된 데이터
│   ├── ml_dataset_engineered.csv   # 최종 ML 훈련용 데이터셋
│   ├── train_engineered.csv        # 훈련 데이터 (시간 분할)
│   └── test_engineered.csv         # 테스트 데이터 (시간 분할)
│
├── scripts/                        # 모든 실행 스크립트
│   ├── 1_data_processing/
│   │   ├── data_exploration.py
│   │   ├── data_preprocessing_efficient.py
│   │   └── feature_engineering_fixed.py
│   │
│   ├── 2_modeling_and_optimization/
│   │   ├── boosting_models_comparison.py
│   │   ├── model_optimization_and_shap.py
│   │   ├── optimized_ensemble_reconstruction.py
│   │   └── xgboost_satisfaction_optimization.py
│   │
│   ├── 3_analysis_and_reporting/
│   │   ├── final_report.py
│   │   ├── shap_analysis_final.py
│   │   └── optimization_results_analysis.py
│   │
│   └── archive/ (실험용 스크립트)
│       ├── quick_stacking_test.py
│       ├── simple_xgboost_satisfaction_test.py
│       └── ...
│
├── models/                         # 훈련된 모델 파일 (.pkl)
│   ├── optimized_wage_voting_ensemble.pkl      # 최종 임금 예측 앙상블 모델
│   ├── optimized_satisfaction_voting_ensemble.pkl # 최종 만족도 예측 앙상블 모델
│   ├── final_optimized_catboost_wage.pkl       # 최적화된 CatBoost 임금 모델
│   └── optimized_xgboost_satisfaction_final.pkl # 최적화된 XGBoost 만족도 모델
│
├── model_results/                  # 모델 성능 및 분석 결과 (.csv)
│   ├── optimized_ensemble_results.csv          # 최종 앙상블 성능
│   ├── final_optimization_results.csv          # 개별 모델 최적화 결과
│   ├── shap_wage_feature_importance.csv        # 임금 예측 모델 SHAP 결과
│   └── shap_satisfaction_feature_importance.csv # 만족도 예측 모델 SHAP 결과
│
├── visualizations/                 # 분석 및 결과 시각화 자료 (.png)
│   ├── shap_analysis_wage_optimized.png
│   └── shap_analysis_satisfaction_optimized.png
│
└── documentation/                  # 프로젝트 문서
    └── PROJECT_STRUCTURE.md        # (현재 파일)

```

## 📁 폴더별 상세 설명

- **루트**: 프로젝트의 핵심 보고서 및 가이드 파일이 위치합니다.
- **data/**: 가공되지 않은 원본 데이터 및 분석에 필요한 각종 메타데이터를 포함합니다.
- **processed_data/**: 원본 데이터를 정제하고 특성 엔지니어링을 거쳐, 모델 훈련에 직접 사용되는 최종 데이터셋이 저장됩니다.
- **scripts/**: 프로젝트의 모든 실행 코드를 기능별로 분류하여 관리합니다.
    - `1_data_processing`: 데이터 탐색, 전처리, 특성 생성 스크립트.
    - `2_modeling_and_optimization`: 모델 훈련, 하이퍼파라미터 최적화, 앙상블 구성 스크립트.
    - `3_analysis_and_reporting`: 최종 결과 분석, SHAP 분석, 보고서 생성용 스크립트.
    - `archive`: 개발 과정에서 사용된 간단한 테스트 및 실험용 스크립트 보관소.
- **models/**: 훈련이 완료되어 저장된 최종 모델 파일(`.pkl`)이 위치합니다. 웹 서비스 통합 시 이 폴더의 모델을 사용합니다.
- **model_results/**: 모델의 성능 평가 결과(RMSE, Accuracy 등)와 SHAP 분석 결과 등 수치로 된 결과물을 `.csv` 파일로 저장합니다.
- **visualizations/**: SHAP 요약 플롯, 성능 비교 차트 등 분석 결과를 시각화한 이미지 파일을 저장합니다.
- **documentation/**: 프로젝트의 구조를 설명하는 문서가 위치합니다.
