import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 고급 특성 엔지니어링 (수정 버전) ===")

# 전처리된 데이터 로드
df = pd.read_csv('combined_prediction_clean.csv')
print(f"데이터 크기: {df.shape}")
print(f"컬럼들: {list(df.columns)}")

# 1. 기본 데이터 정제
print("\n1. 기본 데이터 정제")
df = df.dropna(subset=['next_wage', 'next_satisfaction']).reset_index(drop=True)
print(f"타겟 변수 결측값 제거 후: {len(df)}개")

# 2. 시계열 특성 생성 (lag features)
print("\n2. 시계열 lag 특성 생성")

# 개인별로 정렬
df_sorted = df.sort_values(['pid', 'year']).reset_index(drop=True)

# 1년, 2년 lag features (사용 가능한 변수만)
lag_vars = ['p_wage', 'p4321', 'p_age']
for var in lag_vars:
    if var in df_sorted.columns:
        for lag in [1, 2]:
            df_sorted[f'{var}_lag_{lag}'] = df_sorted.groupby('pid')[var].shift(lag)

# 변화율 특성
if 'p_wage_lag_1' in df_sorted.columns:
    df_sorted['wage_change_1y'] = (df_sorted['p_wage'] - df_sorted['p_wage_lag_1']) / (df_sorted['p_wage_lag_1'] + 1)
    df_sorted['wage_change_1y'] = df_sorted['wage_change_1y'].fillna(0)

if 'p4321_lag_1' in df_sorted.columns:
    df_sorted['satisfaction_change_1y'] = df_sorted['p4321'] - df_sorted['p4321_lag_1']
    df_sorted['satisfaction_change_1y'] = df_sorted['satisfaction_change_1y'].fillna(0)

# 이동평균 (3년)
if 'p_wage' in df_sorted.columns:
    df_sorted['wage_ma_3y'] = df_sorted.groupby('pid')['p_wage'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

if 'p4321' in df_sorted.columns:
    df_sorted['satisfaction_ma_3y'] = df_sorted.groupby('pid')['p4321'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

print("lag features 생성 완료")

# 3. 상호작용 특성
print("\n3. 상호작용 특성 생성")

# 나이와 교육 상호작용
if 'p_edu' in df_sorted.columns:
    df_sorted['age_edu_interaction'] = df_sorted['p_age'] * df_sorted['p_edu'].fillna(df_sorted['p_edu'].median())

# 임금과 만족도 상호작용
if 'p_wage' in df_sorted.columns and 'p4321' in df_sorted.columns:
    df_sorted['wage_satisfaction_ratio'] = df_sorted['p_wage'] / (df_sorted['p4321'] + 1)
    df_sorted['wage_satisfaction_ratio'] = df_sorted['wage_satisfaction_ratio'].fillna(0)

# 경력과 임금 상호작용
if 'experience_years' in df_sorted.columns and 'p_wage' in df_sorted.columns:
    df_sorted['experience_wage_ratio'] = df_sorted['p_wage'] / (df_sorted['experience_years'] + 1)

# 4. 업종 특성 처리 (사용 가능한 것만)
print("\n4. 업종 특성 처리")

industry_cols = [col for col in ['p_ind2000', 'p_ind2007', 'p_ind2017'] if col in df_sorted.columns]
if industry_cols:
    df_sorted['current_industry'] = df_sorted[industry_cols].bfill(axis=1).iloc[:, 0]
    # 업종 변경 여부
    df_sorted['industry_changed'] = (df_sorted.groupby('pid')['current_industry'].diff() != 0).astype(int)
else:
    print("업종 관련 컬럼을 찾을 수 없음")

# 5. 시간 기반 특성
print("\n5. 시간 기반 특성 생성")

# 계절성 (웨이브 기반)
if 'wave' in df_sorted.columns:
    df_sorted['wave_sin'] = np.sin(2 * np.pi * df_sorted['wave'] / df_sorted['wave'].max())
    df_sorted['wave_cos'] = np.cos(2 * np.pi * df_sorted['wave'] / df_sorted['wave'].max())

# 경제 사이클
df_sorted['financial_crisis'] = (df_sorted['year'].between(2008, 2009)).astype(int)
df_sorted['covid_period'] = (df_sorted['year'] >= 2020).astype(int)

# 연령대 더미 변수
age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['20s', '30s', '40s', '50s', '60plus']
df_sorted['age_category'] = pd.cut(df_sorted['p_age'], bins=age_bins, labels=age_labels)

# 6. 개인별 통계적 특성
print("\n6. 개인별 통계 특성 생성")

# 개인별 변동성
person_stats = df_sorted.groupby('pid').agg({
    'p_wage': ['std', 'mean', 'count'],
    'p4321': ['std', 'mean', 'count']
}).fillna(0)

person_stats.columns = ['_'.join(col).strip() for col in person_stats.columns.values]
person_stats = person_stats.reset_index()

# 메인 데이터에 병합
df_final = df_sorted.merge(person_stats, on='pid', how='left')

# 7. 특성 선택
print("\n7. 최종 특성 선택")

# 기본 특성들
base_features = ['p_age', 'p_sex', 'time_trend']
if 'p_edu' in df_final.columns:
    base_features.append('p_edu')
if 'experience_years' in df_final.columns:
    base_features.append('experience_years')

# 현재 상태 특성
current_features = []
if 'p_wage' in df_final.columns:
    current_features.append('p_wage')
if 'p4321' in df_final.columns:
    current_features.append('p4321')

# lag 특성들
lag_features = [col for col in df_final.columns if 'lag' in col]

# 파생 특성들
derived_features = [col for col in df_final.columns if any(keyword in col for keyword in 
    ['change', 'ma_', 'ratio', 'interaction', 'sin', 'cos', 'crisis', 'covid', '_std', '_mean', '_count'])]

# 업종 특성
industry_features = [col for col in df_final.columns if 'industry' in col or 'current_ind' in col]

# 모든 특성 결합
all_features = base_features + current_features + lag_features + derived_features + industry_features

# 실제로 존재하는 특성만 선택
available_features = [col for col in all_features if col in df_final.columns]

print(f"사용 가능한 특성 수: {len(available_features)}")
print("특성 목록:")
for i, feature in enumerate(available_features, 1):
    print(f"  {i:2d}. {feature}")

# 8. 최종 ML 데이터셋 생성
print("\n8. 최종 ML 데이터셋 생성")

# 메타 정보
meta_cols = ['pid', 'year', 'next_year']
target_cols = ['next_wage', 'next_satisfaction']

# 전체 컬럼
ml_columns = meta_cols + available_features + target_cols

# 존재하는 컬럼만 선택
final_columns = [col for col in ml_columns if col in df_final.columns]
ml_dataset = df_final[final_columns].copy()

# 결측값 처리
for col in available_features:
    if col in ml_dataset.columns:
        if pd.api.types.is_numeric_dtype(ml_dataset[col]):
            ml_dataset[col] = ml_dataset[col].fillna(ml_dataset[col].median())
        else:
            mode_val = ml_dataset[col].mode()
            if len(mode_val) > 0:
                fill_val = mode_val.iloc[0]
            else:
                fill_val = 0
            ml_dataset[col] = ml_dataset[col].fillna(fill_val)

print(f"최종 ML 데이터셋 크기: {ml_dataset.shape}")

# 9. 데이터 분할 (시간 기반)
print("\n9. 시간 기반 데이터 분할")

# 2020년 이전: 훈련, 2021년 이후: 테스트
train_mask = ml_dataset['year'] <= 2020
test_mask = ml_dataset['year'] >= 2021

train_data = ml_dataset[train_mask].copy()
test_data = ml_dataset[test_mask].copy()

print(f"훈련 데이터: {len(train_data)}개 ({train_data['year'].min()}-{train_data['year'].max()})")
print(f"테스트 데이터: {len(test_data)}개 ({test_data['year'].min()}-{test_data['year'].max()})")

# 10. 데이터 저장
print("\n10. 데이터 저장")

ml_dataset.to_csv('ml_dataset_engineered.csv', index=False, encoding='utf-8-sig')
train_data.to_csv('train_engineered.csv', index=False, encoding='utf-8-sig')
test_data.to_csv('test_engineered.csv', index=False, encoding='utf-8-sig')

# 특성 정보
feature_info = pd.DataFrame({
    'feature_name': available_features,
    'data_type': [str(ml_dataset[col].dtypes) for col in available_features],
    'missing_count': [ml_dataset[col].isnull().sum() for col in available_features],
    'unique_values': [ml_dataset[col].nunique() for col in available_features],
    'mean': [ml_dataset[col].mean() if pd.api.types.is_numeric_dtype(ml_dataset[col]) else np.nan for col in available_features],
    'std': [ml_dataset[col].std() if pd.api.types.is_numeric_dtype(ml_dataset[col]) else np.nan for col in available_features]
})
feature_info.to_csv('feature_info_final.csv', index=False, encoding='utf-8-sig')

# 11. 상관관계 분석
print("\n11. 특성 중요도 분석")

numeric_features = [col for col in available_features if pd.api.types.is_numeric_dtype(ml_dataset[col])]
if len(numeric_features) > 0:
    corr_with_target = ml_dataset[numeric_features + target_cols].corr()
    target_correlations = corr_with_target[target_cols].drop(target_cols).abs().mean(axis=1).sort_values(ascending=False)
    
    print("타겟 변수와 상관관계가 높은 상위 10개 특성:")
    for i, (feature, corr) in enumerate(target_correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature}: {corr:.3f}")

# 12. 시각화
print("\n12. 특성 분석 시각화")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 연도별 데이터 분포
yearly_counts = ml_dataset['year'].value_counts().sort_index()
yearly_counts.plot(kind='line', marker='o', ax=axes[0, 0])
axes[0, 0].set_title('연도별 샘플 수')
axes[0, 0].set_xlabel('연도')
axes[0, 0].set_ylabel('샘플 수')
axes[0, 0].grid(True, alpha=0.3)

# 타겟 변수 분포
axes[0, 1].hist(ml_dataset['next_wage'], bins=50, alpha=0.7, label='임금')
axes[0, 1].set_title('다음 연도 임금 분포')
axes[0, 1].set_xlabel('임금')
axes[0, 1].set_ylabel('빈도')

# 만족도 분포
ml_dataset['next_satisfaction'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('다음 연도 만족도 분포')
axes[1, 0].set_xlabel('만족도')
axes[1, 0].set_ylabel('빈도')

# 상관관계 히트맵 (상위 10개 특성)
if len(numeric_features) >= 10:
    top_features = target_correlations.head(8).index.tolist()
    corr_subset = corr_with_target.loc[top_features + target_cols, top_features + target_cols]
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1], fmt='.2f')
    axes[1, 1].set_title('주요 특성-타겟 상관관계')

plt.tight_layout()
plt.savefig('feature_engineering_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 특성 엔지니어링 완료 ===")
print("생성된 파일:")
print("  - ml_dataset_engineered.csv: 전체 ML 데이터셋")
print("  - train_engineered.csv: 훈련 데이터")
print("  - test_engineered.csv: 테스트 데이터")
print("  - feature_info_final.csv: 특성 상세 정보")
print("  - feature_engineering_final.png: 분석 결과 시각화")

print(f"\n최종 통계:")
print(f"  특성 수: {len(available_features)}")
print(f"  전체 샘플: {len(ml_dataset)}")
print(f"  훈련 샘플: {len(train_data)}")
print(f"  테스트 샘플: {len(test_data)}")
print(f"  대상 개인 수: {ml_dataset['pid'].nunique()}")

print("\n다음 단계: 머신러닝 모델 학습 및 예측 준비 완료!")