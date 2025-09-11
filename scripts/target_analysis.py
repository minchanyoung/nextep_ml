import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('dateset/nextep_dataset.csv')

print("=== 타겟 변수 세부 분석 ===")

# p_wage 분석
print("\n1. p_wage (임금) 심화 분석")
wage_stats = df['p_wage'].describe()
print("기본 통계량:")
print(wage_stats)

# 임금 0인 경우 분석
zero_wage = (df['p_wage'] == 0).sum()
print(f"\n임금이 0인 관측값: {zero_wage}개")

# 극값 분석
wage_99 = df['p_wage'].quantile(0.99)
wage_95 = df['p_wage'].quantile(0.95)
print(f"95% 분위수: {wage_95}")
print(f"99% 분위수: {wage_99}")

# 연도별 임금 분포
print("\n연도별 평균 임금:")
yearly_wage = df.groupby('year')['p_wage'].agg(['mean', 'median', 'count']).round(2)
print(yearly_wage)

# p4321 분석
print("\n2. p4321 (만족도) 심화 분석")
satisfaction_counts = df['p4321'].value_counts().sort_index()
print("만족도 분포:")
print(satisfaction_counts)

# -1 값의 의미 파악
negative_satisfaction = (df['p4321'] == -1).sum()
print(f"\n만족도가 -1인 관측값: {negative_satisfaction}개 (무응답 또는 해당없음으로 추정)")

# 연도별 만족도 분포
print("\n연도별 평균 만족도:")
yearly_satisfaction = df.groupby('year')['p4321'].agg(['mean', 'count']).round(3)
print(yearly_satisfaction)

# 임금과 만족도의 관계
print("\n3. 임금과 만족도 상관관계")
# 유효한 값들만으로 상관계수 계산
valid_data = df[(df['p_wage'].notna()) & (df['p4321'].notna()) & (df['p4321'] > 0)]
correlation = valid_data['p_wage'].corr(valid_data['p4321'])
print(f"임금과 만족도 상관계수: {correlation:.3f}")

# 만족도별 평균 임금
print("\n만족도별 평균 임금:")
satisfaction_wage = valid_data.groupby('p4321')['p_wage'].agg(['mean', 'count']).round(2)
print(satisfaction_wage)

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 임금 분포 (히스토그램)
axes[0, 0].hist(df['p_wage'].dropna(), bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('임금 분포')
axes[0, 0].set_xlabel('임금')
axes[0, 0].set_ylabel('빈도')

# 2. 임금 분포 (박스플롯, 극값 제외)
wage_filtered = df['p_wage'][(df['p_wage'] <= wage_95) & (df['p_wage'] > 0)]
axes[0, 1].boxplot(wage_filtered.dropna())
axes[0, 1].set_title('임금 분포 (95% 이하)')
axes[0, 1].set_ylabel('임금')

# 3. 만족도 분포
satisfaction_counts.plot(kind='bar', ax=axes[0, 2], color='skyblue', edgecolor='black')
axes[0, 2].set_title('만족도 분포')
axes[0, 2].set_xlabel('만족도')
axes[0, 2].set_ylabel('빈도')
axes[0, 2].tick_params(axis='x', rotation=0)

# 4. 연도별 평균 임금 추이
yearly_wage['mean'].plot(kind='line', marker='o', ax=axes[1, 0], linewidth=2)
axes[1, 0].set_title('연도별 평균 임금 추이')
axes[1, 0].set_xlabel('연도')
axes[1, 0].set_ylabel('평균 임금')
axes[1, 0].grid(True, alpha=0.3)

# 5. 연도별 평균 만족도 추이
yearly_satisfaction['mean'].plot(kind='line', marker='s', ax=axes[1, 1], linewidth=2, color='green')
axes[1, 1].set_title('연도별 평균 만족도 추이')
axes[1, 1].set_xlabel('연도')
axes[1, 1].set_ylabel('평균 만족도')
axes[1, 1].grid(True, alpha=0.3)

# 6. 만족도별 평균 임금 (박스플롯)
valid_data_viz = valid_data[valid_data['p_wage'] <= wage_95]
sns.boxplot(data=valid_data_viz, x='p4321', y='p_wage', ax=axes[1, 2])
axes[1, 2].set_title('만족도별 임금 분포')
axes[1, 2].set_xlabel('만족도')
axes[1, 2].set_ylabel('임금')

plt.tight_layout()
plt.savefig('target_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 결측값 패턴 분석
print("\n4. 결측값 패턴 분석")
missing_pattern = pd.DataFrame({
    'p_wage_missing': df['p_wage'].isna(),
    'p4321_missing': df['p4321'].isna()
})

pattern_counts = missing_pattern.groupby(['p_wage_missing', 'p4321_missing']).size()
print("결측값 패턴:")
print(pattern_counts)

# 개인별 관측 패턴
print("\n5. 개인별 타겟 변수 관측 패턴")
person_target_obs = df.groupby('pid').agg({
    'p_wage': lambda x: x.notna().sum(),
    'p4321': lambda x: x.notna().sum(),
    'year': 'count'
}).rename(columns={'year': 'total_obs'})

print("개인별 임금 관측 수 분포:")
print(person_target_obs['p_wage'].describe())
print("\n개인별 만족도 관측 수 분포:")
print(person_target_obs['p4321'].describe())

print("\n=== 타겟 변수 분석 완료 ===")
print("결과 시각화가 'target_analysis_plots.png'에 저장되었습니다.")