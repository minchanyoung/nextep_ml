# KLIPS 데이터 기반 임금 및 직업만족도 예측 모델

이 프로젝트는 한국노동패널조사(KLIPS) 데이터를 사용하여 개인의 다음 연도 임금과 직업 만족도를 예측하는 머신러닝 모델을 개발하고, 그 과정을 상세히 기록한 것입니다.

---

## 🏆 최종 모델 성능

수많은 실험과 하이퍼파라미터 최적화, 앙상블 재구성을 거쳐 완성된 최종 모델의 성능은 다음과 같습니다.

| 예측 대상 | 모델 | 주요 메트릭 | 성능 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **임금 예측** | Voting Ensemble | **RMSE** | **116.42 만원** | 낮을수록 좋음 |
| | (CatBoost, XGBoost, LightGBM) | **R²** | **0.691** | 높을수록 좋음 |
| **만족도 예측** | Voting Ensemble | **Accuracy** | **0.6753** | 높을수록 좋음 |
| | (CatBoost, XGBoost, LightGBM) | **F1-Score** | **0.655** | 높을수록 좋음 |


## 🛠️ 주요 기술 스택

- **언어**: Python 3.11
- **데이터 처리**: Pandas, NumPy
- **머신러닝**: Scikit-learn, CatBoost, XGBoost, LightGBM
- **하이퍼파라미터 최적화**: Optuna
- **모델 해석**: SHAP

---

## 📂 프로젝트 구조

프로젝트는 데이터, 스크립트, 모델, 결과 등 기능별로 폴더가 명확하게 분리되어 있습니다.

상세한 전체 폴더 구조는 아래 문서를 참고하세요.

- **[📄 전체 프로젝트 구조 보기](./documentation/PROJECT_STRUCTURE.md)**

---

## 🚀 모델 사용 및 통합 방법

훈련된 최종 모델은 다른 웹 프로젝트(Flask 등)에 통합하여 실제 예측 서비스로 활용할 수 있습니다. 통합에 필요한 모든 절차는 아래 가이드에 상세히 설명되어 있습니다.

- **[🚀 웹 프로젝트 통합 가이드 보기](./MODEL_INTEGRATION_GUIDE.md)**

---

## 📚 상세 문서

프로젝트의 각 단계에 대한 더 자세한 정보는 아래 문서들에서 확인할 수 있습니다.

1.  **[PROJECT_PROGRESS_UPDATE.md](./PROJECT_PROGRESS_UPDATE.md)**
    - 프로젝트의 시작부터 끝까지 모든 진행 상황, 의사결정, 실험 결과를 기록한 **마스터 문서**입니다.

2.  **[README_DATA_PREPROCESSING.md](./README_DATA_PREPROCESSING.md)**
    - 데이터 전처리, 특성 엔지니어링, 초기 베이스라인 모델링 과정에 대한 상세한 기술 보고서입니다.

3.  **[STACKING_SUCCESS_REPORT.md](./STACKING_SUCCESS_REPORT.md)**
    - 최종 모델로 채택되지는 않았으나, 개발 과정에서 높은 성능을 보였던 Stacking 앙상블 실험에 대한 결과 보고서입니다.
