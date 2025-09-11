import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 데이터 전처리 시작 ===")

# 데이터 로드
df = pd.read_csv('dateset/nextep_dataset.csv')
print(f"원본 데이터 크기: {df.shape}")

# 1. 기본 데이터 정제
print("\n1. 기본 데이터 정제")

# 연도와 웨이브 정보 확인
print(f"연도 범위: {df['year'].min()} ~ {df['year'].max()}")
print(f"웨이브 범위: {df['wave'].min()} ~ {df['wave'].max()}")

# 중복 제거 (혹시 있을 경우)
original_len = len(df)
df_clean = df.drop_duplicates(subset=['pid', 'year'])
print(f"중복 제거 후: {original_len} -> {len(df_clean)} (제거: {original_len - len(df_clean)}개)")

# 2. 타겟 변수 전처리
print("\n2. 타겟 변수 전처리")

# p_wage 전처리
print("p_wage (임금) 전처리:")
print(f"  - 결측값: {df_clean['p_wage'].isna().sum()}개")
print(f"  - 0값: {(df_clean['p_wage'] == 0).sum()}개")

# 극값 처리 (99.5% 분위수 기준)
wage_99_5 = df_clean['p_wage'].quantile(0.995)
extreme_wage = (df_clean['p_wage'] > wage_99_5).sum()
print(f"  - 극값(상위 0.5%): {extreme_wage}개, 기준값: {wage_99_5}")

# p4321 전처리
print("\np4321 (만족도) 전처리:")
print(f"  - 결측값: {df_clean['p4321'].isna().sum()}개")
print(f"  - -1값 (무응답): {(df_clean['p4321'] == -1).sum()}개")

# -1을 결측값으로 처리
df_clean.loc[df_clean['p4321'] == -1, 'p4321'] = np.nan
print(f"  - -1을 결측값 처리 후 결측값: {df_clean['p4321'].isna().sum()}개")

# 3. 예측용 데이터셋 생성
print("\n3. 예측용 데이터셋 생성")

# 연속된 연도 쌍 식별
prediction_pairs = []
for pid in df_clean['pid'].unique():
    person_data = df_clean[df_clean['pid'] == pid].sort_values('year')
    
    for i in range(len(person_data) - 1):
        current_row = person_data.iloc[i]
        next_row = person_data.iloc[i + 1]
        
        # 연속된 연도인지 확인
        if next_row['year'] == current_row['year'] + 1:
            pair_data = {
                'pid': pid,
                'current_year': current_row['year'],
                'next_year': next_row['year'],
                
                # 현재 연도 특성들
                'current_age': current_row['p_age'],
                'current_edu': current_row['p_edu'],
                'current_sex': current_row['p_sex'],
                'current_wage': current_row['p_wage'],
                'current_satisfaction': current_row['p4321'],
                'current_ind2000': current_row['p_ind2000'],
                'current_ind2007': current_row['p_ind2007'],
                'current_ind2017': current_row['p_ind2017'],
                'current_jobfam2000': current_row['p_jobfam2000'],
                'current_jobfam2007': current_row['p_jobfam2007'],
                'current_jobfam2017': current_row['p_jobfam2017'],
                
                # 다음 연도 타겟 (예측 대상)
                'next_wage': next_row['p_wage'],
                'next_satisfaction': next_row['p4321'],
                
                # 추가 특성들
                'next_age': next_row['p_age'],
                'wave': current_row['wave']
            }
            prediction_pairs.append(pair_data)

pred_df = pd.DataFrame(prediction_pairs)
print(f"총 예측 가능한 쌍: {len(pred_df)}개")

# 유효한 예측 케이스 필터링
wage_pred_cases = pred_df[pred_df['next_wage'].notna()]
satisfaction_pred_cases = pred_df[pred_df['next_satisfaction'].notna()]
both_pred_cases = pred_df[pred_df['next_wage'].notna() & pred_df['next_satisfaction'].notna()]

print(f"임금 예측 가능: {len(wage_pred_cases)}개")
print(f"만족도 예측 가능: {len(satisfaction_pred_cases)}개")
print(f"둘 다 예측 가능: {len(both_pred_cases)}개")

# 4. 특성 엔지니어링
print("\n4. 기본 특성 엔지니어링")

# 나이 그룹 생성
pred_df['age_group'] = pd.cut(pred_df['current_age'], 
                             bins=[0, 30, 40, 50, 60, 100], 
                             labels=['20s', '30s', '40s', '50s', '60+'])

# 경력 연차 (근사치)
pred_df['experience_years'] = pred_df['current_age'] - 22  # 대학 졸업 나이를 22로 가정
pred_df['experience_years'] = pred_df['experience_years'].clip(lower=0)

# 시간 트렌드
pred_df['time_trend'] = pred_df['current_year'] - pred_df['current_year'].min()

# 교육 수준 정리
education_mapping = {1: '초졸', 2: '중졸', 3: '고졸', 4: '대졸', 5: '대학원졸', 6: '기타'}
pred_df['education_label'] = pred_df['current_edu'].map(education_mapping)

print("생성된 파생 변수:")
print("  - age_group: 연령 구간")
print("  - experience_years: 경력 연차")
print("  - time_trend: 시간 트렌드")
print("  - education_label: 교육 수준")

# 5. 결측값 처리 전략
print("\n5. 결측값 처리")

# 각 변수별 결측값 현황
missing_info = pred_df.isnull().sum()
print("변수별 결측값 현황:")
for col in missing_info[missing_info > 0].index:
    count = missing_info[col]
    pct = (count / len(pred_df)) * 100
    print(f"  {col}: {count}개 ({pct:.1f}%)")

# 결측값 처리 방법별 데이터셋 생성
# 방법 1: 완전한 케이스만 사용 (listwise deletion)
complete_cases = pred_df.dropna(subset=['current_wage', 'current_satisfaction', 
                                       'next_wage', 'next_satisfaction'])
print(f"\n완전 케이스 데이터: {len(complete_cases)}개")

# 방법 2: 개인별 평균으로 대체
def fill_missing_by_person(df, column):
    """개인별 평균으로 결측값 채우기"""
    person_means = df.groupby('pid')[column].mean()
    df[column + '_filled'] = df.apply(
        lambda row: person_means.get(row['pid'], df[column].median()) 
        if pd.isna(row[column]) else row[column], axis=1)
    return df

# 임금과 만족도에 대해 개인별 평균 대체
for col in ['current_wage', 'current_satisfaction']:
    pred_df = fill_missing_by_person(pred_df, col)

print("개인별 평균 대체 완료")

# 6. 이상값 처리
print("\n6. 이상값 처리")

# 임금 이상값 (상위 1% 제거)
wage_99 = pred_df['current_wage'].quantile(0.99)
wage_outliers = (pred_df['current_wage'] > wage_99).sum()
print(f"임금 이상값 (상위 1%): {wage_outliers}개, 기준: {wage_99}")

# 이상값을 99% 분위수로 대체
pred_df['current_wage_capped'] = pred_df['current_wage'].clip(upper=wage_99)
pred_df['next_wage_capped'] = pred_df['next_wage'].clip(upper=wage_99)

# 7. 최종 데이터셋 저장
print("\n7. 최종 데이터셋 저장")

# 임금 예측용 데이터셋
wage_dataset = pred_df[pred_df['next_wage'].notna()].copy()
wage_dataset.to_csv('wage_prediction_dataset.csv', index=False, encoding='utf-8-sig')
print(f"임금 예측 데이터셋 저장: {len(wage_dataset)}개 샘플")

# 만족도 예측용 데이터셋
satisfaction_dataset = pred_df[pred_df['next_satisfaction'].notna()].copy()
satisfaction_dataset.to_csv('satisfaction_prediction_dataset.csv', index=False, encoding='utf-8-sig')
print(f"만족도 예측 데이터셋 저장: {len(satisfaction_dataset)}개 샘플")

# 통합 데이터셋 (둘 다 있는 경우)
combined_dataset = pred_df[pred_df['next_wage'].notna() & pred_df['next_satisfaction'].notna()].copy()
combined_dataset.to_csv('combined_prediction_dataset.csv', index=False, encoding='utf-8-sig')
print(f"통합 예측 데이터셋 저장: {len(combined_dataset)}개 샘플")

# 8. 전처리 결과 시각화
print("\n8. 전처리 결과 시각화")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 연도별 데이터 수
yearly_counts = pred_df['current_year'].value_counts().sort_index()
yearly_counts.plot(kind='line', marker='o', ax=axes[0, 0])
axes[0, 0].set_title('연도별 예측 데이터 수')
axes[0, 0].set_xlabel('연도')
axes[0, 0].set_ylabel('샘플 수')
axes[0, 0].grid(True, alpha=0.3)

# 나이 그룹별 분포
pred_df['age_group'].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('연령 그룹별 분포')
axes[0, 1].set_xlabel('연령 그룹')
axes[0, 1].set_ylabel('샘플 수')
axes[0, 1].tick_params(axis='x', rotation=45)

# 교육 수준별 분포
pred_df['education_label'].value_counts().plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('교육 수준별 분포')
axes[0, 2].set_xlabel('교육 수준')
axes[0, 2].set_ylabel('샘플 수')
axes[0, 2].tick_params(axis='x', rotation=45)

# 임금 분포 (전후 비교)
axes[1, 0].hist(pred_df['current_wage'].dropna(), bins=50, alpha=0.7, label='원본')
axes[1, 0].hist(pred_df['current_wage_capped'].dropna(), bins=50, alpha=0.7, label='이상값처리')
axes[1, 0].set_title('임금 분포 (이상값 처리 전후)')
axes[1, 0].set_xlabel('임금')
axes[1, 0].set_ylabel('빈도')
axes[1, 0].legend()

# 만족도 분포
pred_df['current_satisfaction'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('현재 만족도 분포')
axes[1, 1].set_xlabel('만족도')
axes[1, 1].set_ylabel('빈도')

# 결측값 히트맵
missing_matrix = pred_df[['current_wage', 'current_satisfaction', 'next_wage', 'next_satisfaction']].isnull()
sns.heatmap(missing_matrix.astype(int), ax=axes[1, 2], cmap='Reds', cbar=True)
axes[1, 2].set_title('결측값 패턴')
axes[1, 2].set_xlabel('변수')
axes[1, 2].set_ylabel('샘플')

plt.tight_layout()
plt.savefig('preprocessing_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 데이터 전처리 완료 ===")
print("결과 파일들:")
print("  - wage_prediction_dataset.csv: 임금 예측용 데이터")
print("  - satisfaction_prediction_dataset.csv: 만족도 예측용 데이터")  
print("  - combined_prediction_dataset.csv: 통합 예측용 데이터")
print("  - preprocessing_results.png: 전처리 결과 시각화")

# 요약 통계
print(f"\n최종 데이터셋 요약:")
print(f"  - 전체 예측 쌍: {len(pred_df)}")
print(f"  - 임금 예측 가능: {len(wage_dataset)}")
print(f"  - 만족도 예측 가능: {len(satisfaction_dataset)}")
print(f"  - 통합 예측 가능: {len(combined_dataset)}")
print(f"  - 고유 개인 수: {pred_df['pid'].nunique()}")
print(f"  - 연도 범위: {pred_df['current_year'].min()} ~ {pred_df['current_year'].max()}")