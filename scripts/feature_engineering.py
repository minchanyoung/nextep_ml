import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 고급 특성 엔지니어링 ===")

# 전처리된 데이터 로드
print("전처리된 데이터 로딩...")
df = pd.read_csv('combined_prediction_clean.csv')
print(f"데이터 크기: {df.shape}")

# 1. 이력 기반 특성 생성
print("\n1. 개인 이력 기반 특성 생성")

# 개인별 과거 정보 집계
person_history = df.groupby('pid').agg({
    'p_wage': ['mean', 'std', 'count'],
    'p4321': ['mean', 'std', 'count'],
    'year': ['min', 'max'],
    'p_age': 'first',
    'p_sex': 'first',
    'p_edu': 'first'
}).round(3)

# 컬럼명 평면화
person_history.columns = ['_'.join(col).strip() if col[1] else col[0] for col in person_history.columns.values]
person_history = person_history.reset_index()

print(f"개인 이력 특성 생성: {len(person_history)}명")

# 2. 시계열 특성 생성 (lag features)
print("\n2. 시계열 lag 특성 생성")

# 개인별로 정렬하여 lag features 생성
df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)

# 1년, 2년, 3년 lag features
for lag in [1, 2, 3]:
    df_sorted[f'wage_lag_{lag}'] = df_sorted.groupby('pid')['p_wage'].shift(lag)
    df_sorted[f'satisfaction_lag_{lag}'] = df_sorted.groupby('pid')['p4321'].shift(lag)
    df_sorted[f'age_lag_{lag}'] = df_sorted.groupby('pid')['p_age'].shift(lag)

# 변화율 특성
df_sorted['wage_change_1y'] = (df_sorted['p_wage'] - df_sorted['wage_lag_1']) / (df_sorted['wage_lag_1'] + 1)
df_sorted['satisfaction_change_1y'] = df_sorted['p4321'] - df_sorted['satisfaction_lag_1']

# 이동평균 특성
df_sorted['wage_ma_3y'] = df_sorted.groupby('pid')['p_wage'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
df_sorted['satisfaction_ma_3y'] = df_sorted.groupby('pid')['p4321'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

print("lag features 생성 완료")

# 3. 상호작용 특성
print("\n3. 상호작용 특성 생성")

# 나이와 교육 수준 상호작용
df_sorted['age_edu_interaction'] = df_sorted['p_age'] * df_sorted['p_edu'].fillna(df_sorted['p_edu'].median())

# 임금과 만족도 상호작용
df_sorted['wage_satisfaction_ratio'] = df_sorted['p_wage'] / (df_sorted['p4321'] + 1)

# 경력과 임금 상호작용
df_sorted['experience_wage_ratio'] = df_sorted['p_wage'] / (df_sorted['experience_years'] + 1)

# 4. 업종/직업 특성 처리
print("\n4. 업종/직업 특성 처리")

# 업종 코드 처리 (가장 최근 연도 기준)
industry_cols = ['p_ind2000', 'p_ind2007', 'p_ind2017']
job_cols = ['p_jobfam2000', 'p_jobfam2007', 'p_jobfam2017']

# 최신 업종/직업 정보 사용
df_sorted['current_industry'] = df_sorted[industry_cols].bfill(axis=1).iloc[:, 0]
df_sorted['current_job'] = df_sorted[job_cols].bfill(axis=1).iloc[:, 0]

# 업종/직업 변경 여부
df_sorted['industry_changed'] = (df_sorted.groupby('pid')['current_industry'].diff() != 0).astype(int)
df_sorted['job_changed'] = (df_sorted.groupby('pid')['current_job'].diff() != 0).astype(int)

# 5. 시간 기반 특성
print("\n5. 시간 기반 특성 생성")

# 계절성 (웨이브 기반)
df_sorted['wave_sin'] = np.sin(2 * np.pi * df_sorted['wave'] / df_sorted['wave'].max())
df_sorted['wave_cos'] = np.cos(2 * np.pi * df_sorted['wave'] / df_sorted['wave'].max())

# 경제 사이클 (2008 금융위기, 2020 코로나 등)
df_sorted['financial_crisis'] = (df_sorted['year'].between(2008, 2009)).astype(int)
df_sorted['covid_period'] = (df_sorted['year'] >= 2020).astype(int)

# 연령 구간별 더미 변수
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['20s', '30s', '40s', '50s', '60plus']
df_sorted['age_group'] = pd.cut(df_sorted['p_age'], bins=age_bins, labels=age_labels)

# 6. 통계적 특성
print("\n6. 통계적 특성 생성")

# 개인별 변동성 지표
person_volatility = df_sorted.groupby('pid').agg({
    'p_wage': 'std',
    'p4321': 'std'
}).fillna(0)
person_volatility.columns = ['wage_volatility', 'satisfaction_volatility']
person_volatility = person_volatility.reset_index()

# 메인 데이터에 병합
df_final = df_sorted.merge(person_volatility, on='pid', how='left')

# 7. 특성 선택 및 정리
print("\n7. 최종 특성 선택")

# 예측에 사용할 특성들
feature_columns = [
    # 기본 개인 특성
    'p_age', 'p_edu', 'p_sex', 'experience_years',
    
    # 현재 상태
    'p_wage', 'p4321', 'time_trend',
    
    # lag features
    'wage_lag_1', 'wage_lag_2', 'wage_lag_3',
    'satisfaction_lag_1', 'satisfaction_lag_2', 'satisfaction_lag_3',
    
    # 변화율 특성
    'wage_change_1y', 'satisfaction_change_1y',
    
    # 이동평균
    'wage_ma_3y', 'satisfaction_ma_3y',
    
    # 상호작용
    'age_edu_interaction', 'wage_satisfaction_ratio', 'experience_wage_ratio',
    
    # 업종/직업
    'current_industry', 'current_job', 'industry_changed', 'job_changed',
    
    # 시간 특성
    'wave_sin', 'wave_cos', 'financial_crisis', 'covid_period',
    
    # 통계적 특성
    'wage_volatility', 'satisfaction_volatility'
]

# 타겟 변수
target_columns = ['next_wage', 'next_satisfaction']

# 필수 메타 정보
meta_columns = ['pid', 'year', 'next_year']

# 모든 컬럼 확인 및 누락된 컬럼 제외
available_features = [col for col in feature_columns if col in df_final.columns]
missing_features = [col for col in feature_columns if col not in df_final.columns]

if missing_features:
    print(f"누락된 특성들: {missing_features}")

print(f"사용 가능한 특성 수: {len(available_features)}")

# 8. 최종 데이터셋 생성
print("\n8. 최종 머신러닝용 데이터셋 생성")

# 유효한 샘플만 선택 (타겟 변수가 있는 경우)
ml_dataset = df_final[
    df_final['next_wage'].notna() & 
    df_final['next_satisfaction'].notna()
][meta_columns + available_features + target_columns].copy()

print(f"최종 ML 데이터셋 크기: {ml_dataset.shape}")

# 결측값 처리 (간단한 전진 충전 + 중위값)
for col in available_features:
    if ml_dataset[col].dtype in ['float64', 'int64']:
        ml_dataset[col] = ml_dataset[col].fillna(ml_dataset[col].median())
    else:
        ml_dataset[col] = ml_dataset[col].fillna(ml_dataset[col].mode()[0] if len(ml_dataset[col].mode()) > 0 else 0)

# 9. 데이터 분할
print("\n9. 훈련/검증 데이터 분할")

# 시간 기반 분할 (최근 2년을 테스트셋으로)
train_years = ml_dataset['year'] <= 2020
test_years = ml_dataset['year'] > 2020

train_data = ml_dataset[train_years].copy()
test_data = ml_dataset[test_years].copy()

print(f"훈련 데이터: {len(train_data)}개 ({train_data['year'].min()}-{train_data['year'].max()})")
print(f"테스트 데이터: {len(test_data)}개 ({test_data['year'].min()}-{test_data['year'].max()})")

# 10. 데이터 저장
print("\n10. 최종 데이터셋 저장")

# 전체 데이터셋
ml_dataset.to_csv('ml_dataset_final.csv', index=False, encoding='utf-8-sig')

# 훈련/테스트 분할
train_data.to_csv('train_dataset.csv', index=False, encoding='utf-8-sig')
test_data.to_csv('test_dataset.csv', index=False, encoding='utf-8-sig')

# 특성 정보 저장
feature_info = pd.DataFrame({
    'feature_name': available_features,
    'feature_type': [str(ml_dataset[col].dtype) for col in available_features],
    'missing_count': [ml_dataset[col].isnull().sum() for col in available_features],
    'unique_values': [ml_dataset[col].nunique() for col in available_features]
})
feature_info.to_csv('feature_information.csv', index=False, encoding='utf-8-sig')

print("저장된 파일:")
print("  - ml_dataset_final.csv: 전체 ML 데이터셋")
print("  - train_dataset.csv: 훈련용 데이터")
print("  - test_dataset.csv: 테스트용 데이터")
print("  - feature_information.csv: 특성 정보")

# 11. 특성 중요도 시각화
print("\n11. 특성 분석 및 시각화")

# 상관관계 분석
numeric_features = [col for col in available_features if ml_dataset[col].dtype in ['float64', 'int64']]
correlation_matrix = ml_dataset[numeric_features + target_columns].corr()

# 타겟 변수와의 상관관계
target_corr = correlation_matrix[target_columns].drop(target_columns).abs().mean(axis=1).sort_values(ascending=False)

print("타겟 변수와 가장 상관관계가 높은 특성들:")
print(target_corr.head(10))

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 특성 개수 및 타입
feature_types = feature_info['feature_type'].value_counts()
feature_types.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('특성 타입별 분포')
axes[0, 0].set_xlabel('데이터 타입')
axes[0, 0].set_ylabel('특성 수')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. 연도별 데이터 분포
yearly_dist = ml_dataset['year'].value_counts().sort_index()
yearly_dist.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('연도별 샘플 수')
axes[0, 1].set_xlabel('연도')
axes[0, 1].set_ylabel('샘플 수')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. 타겟 변수 상관관계 히트맵
top_features = target_corr.head(10).index.tolist()
corr_subset = correlation_matrix.loc[top_features + target_columns, top_features + target_columns]
sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('주요 특성-타겟 상관관계')

# 4. 결측값 현황
missing_counts = feature_info[feature_info['missing_count'] > 0]
if len(missing_counts) > 0:
    missing_counts.plot(x='feature_name', y='missing_count', kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('특성별 결측값 수')
    axes[1, 1].set_xlabel('특성명')
    axes[1, 1].set_ylabel('결측값 수')
    axes[1, 1].tick_params(axis='x', rotation=45)
else:
    axes[1, 1].text(0.5, 0.5, '결측값 없음', ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('결측값 현황')

plt.tight_layout()
plt.savefig('feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 특성 엔지니어링 완료 ===")
print(f"최종 특성 수: {len(available_features)}")
print(f"최종 샘플 수: {len(ml_dataset)}")
print(f"타겟 변수: next_wage, next_satisfaction")
print("머신러닝 모델링 준비 완료!")