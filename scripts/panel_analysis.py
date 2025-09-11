import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('dateset/nextep_dataset.csv')

print("=== 패널 데이터 구조 심화 분석 ===")

# 1. 개인별 관측 패턴 분석
print("\n1. 개인별 관측 패턴")
person_years = df.groupby('pid')['year'].apply(lambda x: sorted(x.unique()))
person_obs_count = df.groupby('pid').size()

print(f"전체 개인 수: {len(person_years)}")
print(f"개인별 관측 수 분포:")
print(person_obs_count.describe())

# 연속 관측 vs 비연속 관측 분석
def check_consecutive_years(years):
    if len(years) <= 1:
        return True
    return all(years[i] == years[i-1] + 1 for i in range(1, len(years)))

consecutive_obs = person_years.apply(check_consecutive_years)
print(f"\n연속 관측 개인 수: {consecutive_obs.sum()}")
print(f"비연속 관측 개인 수: {(~consecutive_obs).sum()}")

# 2. 데이터 균형성 분석
print("\n2. 패널 데이터 균형성 분석")
complete_panel = df.groupby('pid').agg({
    'year': lambda x: len(x.unique()),
    'p_wage': lambda x: x.notna().sum(),
    'p4321': lambda x: x.notna().sum()
})

# 완전한 패널 (모든 연도 관측)
max_years = df['year'].nunique()
complete_individuals = (complete_panel['year'] == max_years).sum()
print(f"전체 연도({max_years}년) 모두 관측된 개인: {complete_individuals}명")

# 3. 타겟 변수의 시간적 패턴
print("\n3. 타겟 변수의 시간적 연속성")
# 각 개인의 연속적인 타겟 변수 관측 패턴
target_continuity = df.groupby('pid').apply(lambda x: {
    'years_span': x['year'].max() - x['year'].min() + 1,
    'actual_obs': len(x),
    'wage_obs': x['p_wage'].notna().sum(),
    'satisfaction_obs': x['p4321'].notna().sum(),
    'wage_ratio': x['p_wage'].notna().sum() / len(x) if len(x) > 0 else 0,
    'satisfaction_ratio': x['p4321'].notna().sum() / len(x) if len(x) > 0 else 0
})

target_df = pd.DataFrame(target_continuity.tolist())
print("개인별 타겟 변수 관측 비율:")
print(target_df[['wage_ratio', 'satisfaction_ratio']].describe())

# 4. 예측을 위한 데이터 구조 분석
print("\n4. 예측 가능성 분석")
# 다음 해 예측이 가능한 케이스 식별
prediction_candidates = []

for pid in df['pid'].unique():
    person_data = df[df['pid'] == pid].sort_values('year')
    
    for i in range(len(person_data) - 1):
        current_year = person_data.iloc[i]['year']
        next_year = person_data.iloc[i + 1]['year']
        
        # 연속된 연도이고, 다음 해에 타겟 변수가 있는 경우
        if next_year == current_year + 1:
            current_wage = person_data.iloc[i]['p_wage']
            current_satisfaction = person_data.iloc[i]['p4321']
            next_wage = person_data.iloc[i + 1]['p_wage']
            next_satisfaction = person_data.iloc[i + 1]['p4321']
            
            prediction_candidates.append({
                'pid': pid,
                'current_year': current_year,
                'next_year': next_year,
                'has_current_wage': not pd.isna(current_wage),
                'has_current_satisfaction': not pd.isna(current_satisfaction),
                'has_next_wage': not pd.isna(next_wage),
                'has_next_satisfaction': not pd.isna(next_satisfaction)
            })

pred_df = pd.DataFrame(prediction_candidates)
print(f"연속된 연도 관측 쌍: {len(pred_df)}개")

# 예측 가능한 케이스 분류
wage_predictable = pred_df['has_next_wage'].sum()
satisfaction_predictable = pred_df['has_next_satisfaction'].sum()
both_predictable = (pred_df['has_next_wage'] & pred_df['has_next_satisfaction']).sum()

print(f"임금 예측 가능 케이스: {wage_predictable}개")
print(f"만족도 예측 가능 케이스: {satisfaction_predictable}개")
print(f"둘 다 예측 가능한 케이스: {both_predictable}개")

# 5. 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 개인별 관측 수 분포
axes[0, 0].hist(person_obs_count, bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('개인별 관측 수 분포')
axes[0, 0].set_xlabel('관측 수')
axes[0, 0].set_ylabel('개인 수')

# 개인별 타겟 변수 관측 비율
axes[0, 1].hist(target_df['wage_ratio'], bins=20, alpha=0.7, label='임금', edgecolor='black')
axes[0, 1].hist(target_df['satisfaction_ratio'], bins=20, alpha=0.7, label='만족도', edgecolor='black')
axes[0, 1].set_title('개인별 타겟 변수 관측 비율')
axes[0, 1].set_xlabel('관측 비율')
axes[0, 1].set_ylabel('개인 수')
axes[0, 1].legend()

# 연도별 데이터 가용성
yearly_availability = df.groupby('year').agg({
    'pid': 'count',
    'p_wage': lambda x: x.notna().sum(),
    'p4321': lambda x: x.notna().sum()
})

yearly_availability.plot(kind='line', ax=axes[0, 2], marker='o')
axes[0, 2].set_title('연도별 데이터 가용성')
axes[0, 2].set_xlabel('연도')
axes[0, 2].set_ylabel('관측 수')
axes[0, 2].legend(['전체', '임금', '만족도'])
axes[0, 2].grid(True, alpha=0.3)

# 예측 가능성 분석 시각화
pred_summary = pd.DataFrame({
    '전체': [len(pred_df)],
    '임금_예측가능': [wage_predictable],
    '만족도_예측가능': [satisfaction_predictable],
    '둘다_예측가능': [both_predictable]
})
pred_summary.T.plot(kind='bar', ax=axes[1, 0], legend=False)
axes[1, 0].set_title('예측 가능한 케이스 수')
axes[1, 0].set_ylabel('케이스 수')
axes[1, 0].tick_params(axis='x', rotation=45)

# 연도별 예측 가능 케이스
yearly_pred = pred_df.groupby('current_year').agg({
    'has_next_wage': 'sum',
    'has_next_satisfaction': 'sum'
})
yearly_pred.plot(kind='line', ax=axes[1, 1], marker='o')
axes[1, 1].set_title('연도별 예측 가능 케이스')
axes[1, 1].set_xlabel('기준 연도')
axes[1, 1].set_ylabel('예측 가능 케이스 수')
axes[1, 1].legend(['임금', '만족도'])
axes[1, 1].grid(True, alpha=0.3)

# 데이터 균형성 히트맵
balance_matrix = df.pivot_table(
    values='pid', 
    index='year', 
    columns='wave', 
    aggfunc='count', 
    fill_value=0
)
sns.heatmap(balance_matrix, ax=axes[1, 2], cmap='YlOrRd', annot=False)
axes[1, 2].set_title('연도-웨이브 데이터 분포')
axes[1, 2].set_xlabel('웨이브')
axes[1, 2].set_ylabel('연도')

plt.tight_layout()
plt.savefig('panel_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 최종 예측 데이터셋 생성을 위한 권고사항
print("\n5. 데이터 전처리 권고사항")
print("="*50)
print("1. 타겟 변수:")
print(f"   - 임금 예측 가능 샘플: {wage_predictable}개")
print(f"   - 만족도 예측 가능 샘플: {satisfaction_predictable}개")
print(f"   - 결측값 비율이 높으므로 결측값 처리 전략 필요")

print("\n2. 시간적 구조:")
print("   - 불균형 패널 데이터 (개인별 관측 수 상이)")
print("   - 연속 관측이 제한적이므로 시계열 모델링 시 주의")

print("\n3. 예측 모델링 전략:")
print("   - 개인별 고정효과 고려")
print("   - 시간 트렌드 변수 생성")
print("   - 라그 변수 활용 (1년, 2년, 3년 등)")
print("   - 개인 특성 변수 활용 (나이, 교육, 성별 등)")

# 예측 데이터셋 샘플 저장
sample_pred_data = pred_df[
    (pred_df['has_next_wage']) | (pred_df['has_next_satisfaction'])
].head(1000)

print(f"\n예측용 샘플 데이터 {len(sample_pred_data)}개를 'prediction_sample.csv'로 저장합니다.")
sample_pred_data.to_csv('prediction_sample.csv', index=False)

print("\n=== 패널 데이터 분석 완료 ===")
print("결과 시각화가 'panel_analysis_plots.png'에 저장되었습니다.")