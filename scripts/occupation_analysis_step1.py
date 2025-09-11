import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 1단계: 원본 데이터 직업 분류 현황 분석 ===")

# 원본 데이터 로드
df = pd.read_csv('data/raw_data/nextep_dataset.csv')
print(f"전체 데이터 크기: {df.shape}")

# 직업 관련 변수들 확인
occupation_cols = ['p_ind2000', 'p_ind2007', 'p_ind2017', 'p_jobfam2000', 'p_jobfam2007', 'p_jobfam2017']

print("\n=== 직업 관련 변수들의 기본 현황 ===")
for col in occupation_cols:
    if col in df.columns:
        total_count = len(df)
        non_null_count = df[col].notna().sum()
        unique_count = df[col].nunique()
        print(f"{col}:")
        print(f"  - 전체: {total_count:,}개")
        print(f"  - 유효값: {non_null_count:,}개 ({non_null_count/total_count*100:.1f}%)")
        print(f"  - 고유값: {unique_count}개")
        print(f"  - 범위: {df[col].min()} ~ {df[col].max()}")
        print()

# 연도별 데이터 가용성 분석
print("=== 연도별 직업 데이터 가용성 ===")
year_analysis = df.groupby('year').agg({
    'p_ind2000': lambda x: x.notna().sum(),
    'p_ind2007': lambda x: x.notna().sum(), 
    'p_ind2017': lambda x: x.notna().sum(),
    'p_jobfam2000': lambda x: x.notna().sum(),
    'p_jobfam2007': lambda x: x.notna().sum(),
    'p_jobfam2017': lambda x: x.notna().sum(),
    'pid': 'count'
}).round(0).astype(int)

year_analysis.columns = ['산업2000', '산업2007', '산업2017', '직업2000', '직업2007', '직업2017', '전체']
print(year_analysis.head(10))

# 가장 빈번한 산업/직업 코드들 분석
print("\n=== 주요 산업 분류 코드 (상위 20개) ===")
all_industries = []
for col in ['p_ind2000', 'p_ind2007', 'p_ind2017']:
    if col in df.columns:
        valid_industries = df[col].dropna().astype(int)
        all_industries.extend(valid_industries.tolist())

if all_industries:
    industry_counts = Counter(all_industries)
    print("산업코드 | 빈도 | 비율")
    print("-" * 30)
    for code, count in industry_counts.most_common(20):
        pct = count / len(all_industries) * 100
        print(f"{code:8d} | {count:4d} | {pct:5.1f}%")

print("\n=== 주요 직업 분류 코드 (상위 20개) ===")
all_jobs = []
for col in ['p_jobfam2000', 'p_jobfam2007', 'p_jobfam2017']:
    if col in df.columns:
        valid_jobs = df[col].dropna().astype(int)
        all_jobs.extend(valid_jobs.tolist())

if all_jobs:
    job_counts = Counter(all_jobs)
    print("직업코드 | 빈도 | 비율")
    print("-" * 30) 
    for code, count in job_counts.most_common(20):
        pct = count / len(all_jobs) * 100
        print(f"{code:8d} | {count:4d} | {pct:5.1f}%")

# 산업-직업 조합 분석
print("\n=== 산업-직업 조합 패턴 분석 ===")

# 각 시기별로 가장 흔한 조합들
combinations_analysis = []

for year_suffix in ['2000', '2007', '2017']:
    ind_col = f'p_ind{year_suffix}'
    job_col = f'p_jobfam{year_suffix}'
    
    if ind_col in df.columns and job_col in df.columns:
        # 둘 다 유효한 값이 있는 케이스만
        valid_data = df[(df[ind_col].notna()) & (df[job_col].notna())].copy()
        
        if len(valid_data) > 0:
            valid_data['combination'] = valid_data[ind_col].astype(int).astype(str) + '_' + valid_data[job_col].astype(int).astype(str)
            combo_counts = valid_data['combination'].value_counts()
            
            print(f"\n{year_suffix}년 기준 상위 10개 산업-직업 조합:")
            print("산업_직업 | 빈도")
            print("-" * 25)
            for combo, count in combo_counts.head(10).items():
                print(f"{combo:12s} | {count:4d}")
                
            combinations_analysis.append({
                'year': year_suffix,
                'total_combinations': len(combo_counts),
                'total_records': len(valid_data),
                'top_combination': combo_counts.index[0],
                'top_count': combo_counts.iloc[0]
            })

# 시간에 따른 변화 패턴 분석
print("\n=== 개인별 직업 변화 패턴 ===")
person_changes = []

# 각 개인의 직업 변화 추적
for pid in df['pid'].unique()[:1000]:  # 샘플로 1000명만 분석
    person_data = df[df['pid'] == pid].sort_values('year')
    
    # 산업 변화
    industries = []
    for col in ['p_ind2000', 'p_ind2007', 'p_ind2017']:
        if col in person_data.columns:
            valid_ind = person_data[col].dropna()
            if len(valid_ind) > 0:
                industries.extend(valid_ind.astype(int).tolist())
    
    # 직업 변화  
    jobs = []
    for col in ['p_jobfam2000', 'p_jobfam2007', 'p_jobfam2017']:
        if col in person_data.columns:
            valid_job = person_data[col].dropna()
            if len(valid_job) > 0:
                jobs.extend(valid_job.astype(int).tolist())
    
    if industries and jobs:
        person_changes.append({
            'pid': pid,
            'industry_changes': len(set(industries)),
            'job_changes': len(set(jobs)),
            'total_records': len(person_data)
        })

if person_changes:
    changes_df = pd.DataFrame(person_changes)
    print(f"분석 대상: {len(changes_df)}명")
    print(f"산업 변경 평균: {changes_df['industry_changes'].mean():.2f}개")
    print(f"직업 변경 평균: {changes_df['job_changes'].mean():.2f}개")
    print(f"산업 변경 없음: {(changes_df['industry_changes'] == 1).sum()}명 ({(changes_df['industry_changes'] == 1).mean()*100:.1f}%)")
    print(f"직업 변경 없음: {(changes_df['job_changes'] == 1).sum()}명 ({(changes_df['job_changes'] == 1).mean()*100:.1f}%)")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 연도별 데이터 가용성
year_subset = year_analysis.loc[2000:2020]  # 최근 20년만
industry_availability = year_subset[['산업2000', '산업2007', '산업2017']]
job_availability = year_subset[['직업2000', '직업2007', '직업2017']]

industry_availability.plot(kind='line', ax=axes[0,0], marker='o')
axes[0,0].set_title('연도별 산업 분류 데이터 가용성')
axes[0,0].set_xlabel('연도')
axes[0,0].set_ylabel('유효 데이터 수')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

job_availability.plot(kind='line', ax=axes[0,1], marker='s')
axes[0,1].set_title('연도별 직업 분류 데이터 가용성')
axes[0,1].set_xlabel('연도')
axes[0,1].set_ylabel('유효 데이터 수')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 2. 상위 산업 분류 분포
if all_industries:
    top_industries = dict(Counter(all_industries).most_common(15))
    axes[1,0].bar(range(len(top_industries)), list(top_industries.values()))
    axes[1,0].set_title('상위 15개 산업 분류 분포')
    axes[1,0].set_xlabel('산업 코드 순위')
    axes[1,0].set_ylabel('빈도')
    
    # x축 레이블 설정
    axes[1,0].set_xticks(range(len(top_industries)))
    axes[1,0].set_xticklabels([str(k) for k in top_industries.keys()], rotation=45)

# 3. 상위 직업 분류 분포
if all_jobs:
    top_jobs = dict(Counter(all_jobs).most_common(15))
    axes[1,1].bar(range(len(top_jobs)), list(top_jobs.values()))
    axes[1,1].set_title('상위 15개 직업 분류 분포')
    axes[1,1].set_xlabel('직업 코드 순위')
    axes[1,1].set_ylabel('빈도')
    
    # x축 레이블 설정
    axes[1,1].set_xticks(range(len(top_jobs)))
    axes[1,1].set_xticklabels([str(k) for k in top_jobs.keys()], rotation=45)

plt.tight_layout()
plt.savefig('visualizations/occupation_analysis_step1.png', dpi=300, bbox_inches='tight')
plt.show()

# 분석 결과 요약
print("\n" + "="*60)
print("=== 1단계 분석 결과 요약 ===")
print("="*60)

print("\n[데이터 가용성]")
print(f"- 전체 관측값: {len(df):,}개")
for col in occupation_cols:
    if col in df.columns:
        pct = df[col].notna().mean() * 100
        print(f"- {col}: {pct:.1f}% 가용")

if all_industries and all_jobs:
    print(f"\n[산업 분류]")
    print(f"- 총 {len(set(all_industries))}개 고유 산업 코드")
    print(f"- 가장 많은 산업: {Counter(all_industries).most_common(1)[0]}")
    
    print(f"\n[직업 분류]")
    print(f"- 총 {len(set(all_jobs))}개 고유 직업 코드") 
    print(f"- 가장 많은 직업: {Counter(all_jobs).most_common(1)[0]}")

if combinations_analysis:
    print(f"\n[산업-직업 조합]")
    for analysis in combinations_analysis:
        print(f"- {analysis['year']}년: {analysis['total_combinations']}개 조합, 최다 조합: {analysis['top_combination']} ({analysis['top_count']}개)")

print("\n[다음 단계 준비]")
print("- 주요 산업/직업 코드 파악 완료")
print("- 시간별 분류체계 변화 패턴 확인")
print("- 사용자 친화적 매핑 테이블 설계 준비")

print("\n=== 1단계 분석 완료 ===")