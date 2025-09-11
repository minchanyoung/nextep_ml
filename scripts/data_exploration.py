import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
data_path = Path('dateset/nextep_dataset.csv')
df = pd.read_csv(data_path)

print("=== 데이터셋 기본 정보 ===")
print(f"데이터 형태: {df.shape}")
print(f"총 행 수: {len(df)}")
print(f"열 수: {len(df.columns)}")

print("\n=== 컬럼 정보 ===")
print(df.info())

print("\n=== 기본 통계량 ===")
print(df.describe())

print("\n=== 타겟 변수 분석 ===")
print("p_wage (임금) 통계:")
print(f"  - 결측값: {df['p_wage'].isna().sum()}")
print(f"  - 유효값: {df['p_wage'].notna().sum()}")
print(f"  - 최솟값: {df['p_wage'].min()}")
print(f"  - 최댓값: {df['p_wage'].max()}")
print(f"  - 평균: {df['p_wage'].mean():.2f}")
print(f"  - 중위값: {df['p_wage'].median():.2f}")

print("\np4321 (만족도) 통계:")
print(f"  - 결측값: {df['p4321'].isna().sum()}")
print(f"  - 유효값: {df['p4321'].notna().sum()}")
print(f"  - 고유값: {sorted(df['p4321'].dropna().unique())}")
print(f"  - 값별 분포:")
print(df['p4321'].value_counts().sort_index())

print("\n=== 연도별 데이터 분포 ===")
year_counts = df['year'].value_counts().sort_index()
print(year_counts)

print("\n=== 패널 데이터 구조 분석 ===")
print(f"고유한 개인 ID 수: {df['pid'].nunique()}")
print(f"고유한 가구 ID 수: {df['hhid'].nunique()}")
print(f"웨이브 범위: {df['wave'].min()} ~ {df['wave'].max()}")
print(f"연도 범위: {df['year'].min()} ~ {df['year'].max()}")

# 개인별 관측 수 분포
person_obs = df.groupby('pid').size()
print(f"\n개인별 관측 수 분포:")
print(person_obs.describe())

print("\n=== 주요 변수 결측값 현황 ===")
key_vars = ['p_wage', 'p4321', 'p_age', 'p_edu', 'p_sex']
missing_info = {}
for var in key_vars:
    missing_count = df[var].isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_info[var] = {'count': missing_count, 'percentage': missing_pct}
    print(f"{var}: {missing_count}개 ({missing_pct:.1f}%)")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# p_wage 분포
axes[0, 0].hist(df['p_wage'].dropna(), bins=50, alpha=0.7)
axes[0, 0].set_title('p_wage 분포')
axes[0, 0].set_xlabel('임금')
axes[0, 0].set_ylabel('빈도')

# p4321 분포
df['p4321'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('p4321 분포')
axes[0, 1].set_xlabel('만족도')
axes[0, 1].set_ylabel('빈도')

# 연도별 데이터 수
year_counts.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('연도별 관측값 수')
axes[1, 0].set_xlabel('연도')
axes[1, 0].set_ylabel('관측값 수')

# 연도별 평균 임금
yearly_wage = df.groupby('year')['p_wage'].mean()
yearly_wage.plot(kind='line', marker='o', ax=axes[1, 1])
axes[1, 1].set_title('연도별 평균 임금 추이')
axes[1, 1].set_xlabel('연도')
axes[1, 1].set_ylabel('평균 임금')

plt.tight_layout()
plt.savefig('data_exploration_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 분석 완료 ===")
print("시각화 결과가 'data_exploration_plots.png'로 저장되었습니다.")