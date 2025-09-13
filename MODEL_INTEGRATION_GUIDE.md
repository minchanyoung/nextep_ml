# ML 모델 웹 프로젝트 통합 가이드

이 문서는 `KLIPS 데이터 기반 임금 및 직업만족도 예측 모델`을 기존 웹 프로젝트(Flask 기준)에 통합하는 방법을 안내합니다.

## 1. 사전 준비: 필수 파일 복사

예측 모델을 구동하기 위해 필요한 핵심 파일들을 웹 프로젝트의 적절한 위치로 복사합니다.

### 1.1. 모델 파일 (`.pkl`)

최종적으로 최적화된 앙상블 모델 2개를 복사합니다.

- **원본 위치**:
  - `models/optimized_wage_voting_ensemble.pkl`
  - `models/optimized_satisfaction_voting_ensemble.pkl`
- **대상 위치**:
  - `(웹 프로젝트 폴더)/models/`
  (※ `models` 폴더가 없다면 새로 생성하세요.)

### 1.2. 특성 생성용 데이터

모델이 예측을 수행하려면, 훈련 시 사용했던 것과 동일한 방법으로 특성을 생성해야 합니다. 이때 필요한 통계 데이터 파일을 복사합니다.

- **원본 위치**:
  - `processed_data/dataset_with_unified_occupations.csv`
  - `data/occupation_mapping_complete.json`
- **대상 위치**:
  - `(웹 프로젝트 폴더)/data/`
  (※ `data` 폴더가 없다면 새로 생성하세요.)

---

## 2. 통합 단계

### 2.1. 예측 서비스 모듈 생성 (`prediction_service.py`)

웹 프로젝트 폴더에 `prediction_service.py` 파일을 새로 만들고 아래 코드를 붙여넣습니다. 이 모듈은 모델 로딩, 데이터 전처리, 특성 생성, 예측 등 모든 머신러닝 관련 작업을 캡슐화합니다.

```python
# prediction_service.py

import joblib
import pandas as pd
import numpy as np

# 앱 구동 시 한 번만 로드하기 위한 전역 변수
WAGE_MODEL = None
SATISFACTION_MODEL = None
OCCUPATION_STATS = None
MODEL_FEATURES = None

def load_models_and_data():
    """앱 시작 시 모델과 특성 생성에 필요한 데이터를 로드하는 함수"""
    global WAGE_MODEL, SATISFACTION_MODEL, OCCUPATION_STATS, MODEL_FEATURES
    
    print("Loading models and data for prediction service...")
    
    # 1. 모델 로드
    WAGE_MODEL = joblib.load('models/optimized_wage_voting_ensemble.pkl')
    SATISFACTION_MODEL = joblib.load('models/optimized_satisfaction_voting_ensemble.pkl')
    
    # 2. 모델이 학습한 특성 이름 저장
    MODEL_FEATURES = WAGE_MODEL.feature_names_in_
    
    # 3. 특성 생성에 필요한 통계 데이터 사전 계산
    df = pd.read_csv('data/dataset_with_unified_occupations.csv')
    OCCUPATION_STATS = df.groupby('occupation_code').agg(
        avg_wage=('p_wage', 'mean'),
        med_wage=('p_wage', 'median')
    ).reset_index()
    
    print("Models and data loaded successfully.")

def preprocess_input(user_input: dict):
    """
    사용자 입력을 받아 전처리하고, 훈련 때 사용한 특성을 생성하는 함수.
    이 부분이 프로젝트의 핵심 로직이며, 훈련 과정과 100% 동일해야 합니다.
    """
    # 1. 사용자 입력을 DataFrame으로 변환
    input_df = pd.DataFrame([user_input])
    
    # 2. 특성 엔지니어링 (훈련 과정과 동일하게 수행)
    # 예시: 직업군 평균 대비 임금 비율 생성
    if OCCUPATION_STATS is not None:
        input_df = pd.merge(input_df, OCCUPATION_STATS, on='occupation_code', how='left')
        input_df['wage_vs_occupation_avg'] = input_df['p_wage'] / input_df['avg_wage']
    
    # =================================================================================
    # TODO: 여기에 훈련 과정에서 사용된 모든 특성 생성 로직을 추가해야 합니다.
    # 예: wage_quartile_in_occupation, p_age_squared, wave 등 40개 특성
    # 이 부분은 `scripts/feature_engineering_fixed.py` 파일의 로직을 참고하세요.
    # =================================================================================
    
    # 3. 결측값 처리 (훈련 때와 동일한 중위값/최빈값 등으로 대체)
    # 예: input_df.fillna(훈련데이터의_중위값, inplace=True)
    
    # 4. 모델이 기대하는 최종 특성들만 선택 및 순서 정렬
    # 누락된 특성은 0이나 적절한 기본값으로 채웁니다.
    for col in MODEL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0 
            
    return input_df[MODEL_FEATURES]

def predict(user_input: dict):
    """사용자 입력을 받아 최종 예측 결과를 반환하는 메인 함수"""
    if WAGE_MODEL is None:
        raise RuntimeError("Models are not loaded. Please run load_models_and_data() first.")
        
    # 1. 입력 데이터 전처리 및 특성 생성
    processed_df = preprocess_input(user_input)
    
    # 2. 모델을 사용해 예측 수행
    wage_prediction = WAGE_MODEL.predict(processed_df)
    satisfaction_prediction = SATISFACTION_MODEL.predict(processed_df)
    
    # 3. 예측 결과를 가공하여 반환
    # 만족도 예측값(0~4)을 실제 의미(1~5)로 변환하거나, 텍스트로 매핑할 수 있습니다.
    satisfaction_map = {0: '매우 불만족', 1: '불만족', 2: '보통', 3: '만족', 4: '매우 만족'}
    
    return {
        'predicted_wage': round(wage_prediction[0], 2),
        'predicted_satisfaction_code': int(satisfaction_prediction[0]),
        'predicted_satisfaction_label': satisfaction_map.get(int(satisfaction_prediction[0]), '알 수 없음')
    }
```

### 2.2. Flask 앱에 API 엔드포인트 추가

기존 웹 프로젝트의 메인 Flask 파일(예: `app.py`)에 아래 코드를 추가하여, `/predict` 주소로 예측 요청을 처리하는 API를 만듭니다.

```python
# app.py (기존 Flask 앱)

from flask import Flask, request, jsonify
import prediction_service # 방금 만든 예측 서비스 모듈 임포트

app = Flask(__name__)

# 앱이 처음 구동될 때, 예측에 필요한 모델과 데이터를 미리 로드합니다.
# 이렇게 하면 매 요청마다 파일을 읽지 않아 성능이 향상됩니다.
with app.app_context():
    prediction_service.load_models_and_data()

@app.route('/')
def index():
    return "My Flask App with ML Model is running!"

# '/predict' 라는 주소로 POST 요청을 처리하는 API 엔드포인트
@app.route('/predict', methods=['POST'])
def handle_prediction():
    # 요청이 JSON 형태인지 확인
    if not request.json:
        return jsonify({"error": "JSON payload is required"}), 400
    
    try:
        # 클라이언트로부터 받은 JSON 데이터를 추출
        user_input = request.json
        
        # 예측 서비스 모듈의 predict 함수를 호출하여 결과 받기
        predictions = prediction_service.predict(user_input)
        
        # 예측 결과를 JSON 형태로 클라이언트에게 반환
        return jsonify(predictions)
    
    except Exception as e:
        # 예측 과정에서 오류 발생 시 서버 로그에 오류를 기록하고 클라이언트에게 에러 메시지 전송
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 3. API 사용 예시

Flask 앱을 실행한 후, 아래와 같이 `curl` 이나 API 테스트 도구를 사용하여 예측을 요청할 수 있습니다.

### 요청 (Request)

`p_wage`는 현재 임금으로, `wage_vs_occupation_avg` 같은 특성을 만드는 데 사용됩니다.

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "p_age": 40,
    "p_sex": 1,
    "p_edu": 7,
    "occupation_code": "C20",
    "p_wage": 4500 
}'
```

### 응답 (Response)

```json
{
  "predicted_satisfaction_code": 2,
  "predicted_satisfaction_label": "보통",
  "predicted_wage": 4850.75
}
```

---

## 4. 최종 폴더 구조 예시

모든 단계를 완료하면 웹 프로젝트의 폴더 구조는 아래와 유사한 형태가 됩니다.

```
my_flask_project/
├── app.py                 # 메인 Flask 애플리케이션
├── prediction_service.py  # 예측 로직 모듈
│
├── models/                # 모델 저장 폴더
│   ├── optimized_wage_voting_ensemble.pkl
│   └── optimized_satisfaction_voting_ensemble.pkl
│
├── data/                  # 특성 생성용 데이터 폴더
│   └── dataset_with_unified_occupations.csv
│
├── templates/             # (기존 웹 프로젝트의 템플릿 폴더)
│   └── ...
│
└── static/                # (기존 웹 프로젝트의 정적 파일 폴더)
    └── ...
```
