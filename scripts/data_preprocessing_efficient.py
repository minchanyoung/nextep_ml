import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 효율적인 데이터 전처리 ===")

# 데이터 로드 (청크 단위로 처리하지 않고 직접 로드)
print("데이터 로딩 중...")
df = pd.read_csv('dateset/nextep_dataset.csv')
print(f"원본 데이터 크기: {df.shape}")

# 1. 기본 정제
print("\n1. 기본 데이터 정제")
df_clean = df.drop_duplicates(subset=['pid', 'year']).reset_index(drop=True)
print(f"중복 제거 후: {len(df_clean)}개")

# 2. 타겟 변수 정제
print("\n2. 타겟 변수 정제")
# -1을 NaN으로 처리
df_clean.loc[df_clean['p4321'] == -1, 'p4321'] = np.nan

# 극값 처리 (상위 1% 캡핑)
wage_99 = df_clean['p_wage'].quantile(0.99)
df_clean['p_wage'] = df_clean['p_wage'].clip(upper=wage_99)

print(f"임금 99% 분위수로 캡핑: {wage_99}")

# 3. 예측용 데이터 생성 (벡터화 방식 사용)
print("\n3. 예측용 데이터셋 생성")

# 개인별로 연도 정렬
df_sorted = df_clean.sort_values(['pid', 'year']).reset_index(drop=True)

# 다음 연도 데이터 생성 (shift 사용)
df_sorted['next_year'] = df_sorted.groupby('pid')['year'].shift(-1)
df_sorted['next_wage'] = df_sorted.groupby('pid')['p_wage'].shift(-1)
df_sorted['next_satisfaction'] = df_sorted.groupby('pid')['p4321'].shift(-1)
df_sorted['next_age'] = df_sorted.groupby('pid')['p_age'].shift(-1)

# 연속된 연도만 필터링
consecutive_years = (df_sorted['next_year'] == df_sorted['year'] + 1)
pred_df = df_sorted[consecutive_years].copy()

print(f"연속된 연도 쌍: {len(pred_df)}개")

# 4. 특성 생성
print("\n4. 파생 변수 생성")

# 나이 그룹
pred_df['age_group'] = pd.cut(pred_df['p_age'], 
                             bins=[0, 30, 40, 50, 60, 100], 
                             labels=['20s', '30s', '40s', '50s', '60+'])

# 경력 연차
pred_df['experience_years'] = np.maximum(pred_df['p_age'] - 22, 0)

# 시간 트렌드
pred_df['time_trend'] = pred_df['year'] - pred_df['year'].min()

# 임금 변화율 (유효한 경우만)
pred_df['wage_available'] = pred_df['p_wage'].notna() & pred_df['next_wage'].notna()

print("파생 변수 생성 완료")

# 5. 데이터셋 분할
print("\n5. 데이터셋 분할")

# 예측 가능한 케이스 필터링
wage_data = pred_df[pred_df['next_wage'].notna()].copy()
satisfaction_data = pred_df[pred_df['next_satisfaction'].notna()].copy()
combined_data = pred_df[pred_df['next_wage'].notna() & pred_df['next_satisfaction'].notna()].copy()

print(f"임금 예측 데이터: {len(wage_data)}개")
print(f"만족도 예측 데이터: {len(satisfaction_data)}개") 
print(f"통합 예측 데이터: {len(combined_data)}개")

# 6. 주요 특성 선택 및 정리
features_to_keep = [
    'pid', 'year', 'next_year', 'wave',
    'p_age', 'p_edu', 'p_sex',
    'p_wage', 'p4321',
    'p_ind2000', 'p_ind2007', 'p_ind2017',
    'next_wage', 'next_satisfaction', 'next_age',
    'age_group', 'experience_years', 'time_trend'
]

# 컬럼이 존재하는 것만 선택
available_features = [col for col in features_to_keep if col in pred_df.columns]
clean_pred_df = pred_df[available_features].copy()

# 7. 결측값 현황 분석
print("\n6. 결측값 현황")
missing_summary = clean_pred_df.isnull().sum()
for col in missing_summary[missing_summary > 0].index:
    count = missing_summary[col]
    pct = (count / len(clean_pred_df)) * 100
    print(f"  {col}: {count}개 ({pct:.1f}%)")

# 8. 최종 데이터셋 저장
print("\n7. 데이터셋 저장")

# 임금 예측용
wage_final = clean_pred_df[clean_pred_df['next_wage'].notna()].copy()
wage_final.to_csv('wage_prediction_clean.csv', index=False, encoding='utf-8-sig')
print(f"임금 예측 데이터 저장: {len(wage_final)}개")

# 만족도 예측용
satisfaction_final = clean_pred_df[clean_pred_df['next_satisfaction'].notna()].copy()
satisfaction_final.to_csv('satisfaction_prediction_clean.csv', index=False, encoding='utf-8-sig')
print(f"만족도 예측 데이터 저장: {len(satisfaction_final)}개")

# 통합 예측용
combined_final = clean_pred_df[clean_pred_df['next_wage'].notna() & 
                               clean_pred_df['next_satisfaction'].notna()].copy()
combined_final.to_csv('combined_prediction_clean.csv', index=False, encoding='utf-8-sig')
print(f"통합 예측 데이터 저장: {len(combined_final)}개")

# 9. 기본 통계 및 시각화
print("\n8. 기본 분석 결과")

# 연도별 분포
yearly_dist = wage_final['year'].value_counts().sort_index()
print(f"연도 범위: {yearly_dist.index.min()} - {yearly_dist.index.max()}")
print(f"연도별 평균 샘플 수: {yearly_dist.mean():.0f}")

# 개인별 관측 수
person_obs = wage_final.groupby('pid').size()
print(f"개인별 평균 관측 수: {person_obs.mean():.1f}")
print(f"최대 관측 수: {person_obs.max()}")

# 타겟 변수 기본 통계
print(f"\n임금 통계 (다음 연도):")
print(f"  평균: {wage_final['next_wage'].mean():.1f}")
print(f"  중위수: {wage_final['next_wage'].median():.1f}")
print(f"  표준편차: {wage_final['next_wage'].std():.1f}")

if len(satisfaction_final) > 0:
    print(f"\n만족도 통계 (다음 연도):")
    print(f"  평균: {satisfaction_final['next_satisfaction'].mean():.2f}")
    print(f"  분포: {satisfaction_final['next_satisfaction'].value_counts().sort_index().to_dict()}")

# 간단한 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 연도별 샘플 수
yearly_dist.plot(kind='bar', ax=axes[0])
axes[0].set_title('연도별 예측 샘플 수')
axes[0].set_xlabel('연도')
axes[0].set_ylabel('샘플 수')
axes[0].tick_params(axis='x', rotation=45)

# 임금 분포
wage_final['next_wage'].hist(bins=50, ax=axes[1])
axes[1].set_title('다음 연도 임금 분포')
axes[1].set_xlabel('임금')
axes[1].set_ylabel('빈도')

# 나이 그룹 분포
if 'age_group' in wage_final.columns:
    wage_final['age_group'].value_counts().plot(kind='bar', ax=axes[2])
    axes[2].set_title('연령 그룹별 분포')
    axes[2].set_xlabel('연령 그룹')
    axes[2].set_ylabel('샘플 수')
    axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('preprocessing_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 전처리 완료 ===")
print("생성된 파일:")
print("  - wage_prediction_clean.csv")
print("  - satisfaction_prediction_clean.csv") 
print("  - combined_prediction_clean.csv")
print("  - preprocessing_summary.png")

print(f"\n최종 요약:")
print(f"  고유 개인 수: {clean_pred_df['pid'].nunique()}")
print(f"  전체 예측 쌍: {len(clean_pred_df)}")
print(f"  연도 범위: {clean_pred_df['year'].min()}-{clean_pred_df['year'].max()}")