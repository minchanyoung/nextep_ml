import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 2단계: 주요 직업 코드 의미 파악 및 그룹핑 ===")

# 1단계 결과를 바탕으로 주요 코드들의 의미 추론
# 실제 한국표준산업분류/한국표준직업분류를 참고하여 매핑

# 주요 산업 코드 매핑 (상위 20개 기준)
INDUSTRY_CODE_MAPPING = {
    # 농림업
    11: "농림업",
    
    # 제조업 관련
    15: "음식료품제조업", 16: "담배제조업", 17: "섬유제품제조업",
    18: "의복및모피제품제조업", 19: "가죽및신발제조업", 20: "목재및나무제품제조업",
    21: "펄프및종이제품제조업", 22: "출판인쇄매체복제업", 23: "코크스석유정제품제조업",
    24: "화학제품제조업", 25: "고무및플라스틱제품제조업", 26: "비금속광물제품제조업",
    27: "1차금속산업", 28: "조립금속제품제조업", 29: "기타기계장비제조업",
    30: "사무용기계제조업", 31: "전기기계제조업", 32: "영상음향통신장비제조업",
    33: "의료정밀광학기기제조업", 34: "자동차제조업", 35: "기타운송장비제조업",
    36: "가구제조업", 37: "재생용가공원료생산업",
    
    # 서비스업
    411: "전력가스수도업",
    452: "건설업",
    463: "자동차판매업", 464: "연료소매업",
    471: "도매업",
    492: "소매업", 493: "소매업",
    521: "숙박업", 522: "음식점업",
    552: "교육서비스업",
    561: "보건업",
    602: "육상운송업", 603: "수상운송업",
    641: "통신업",
    651: "금융업", 659: "금융업",
    682: "부동산업",
    702: "컴퓨터관련서비스업",
    721: "전문서비스업",
    741: "전문서비스업",
    801: "공공행정",
    809: "공공행정", 
    841: "교육서비스업",
    851: "보건업", 852: "보건업", 855: "보건업",
    861: "사회복지서비스업", 862: "사회복지서비스업",
    871: "영화방송업", 872: "방송업",
    900: "환경정화서비스업", 930: "기타서비스업", 931: "기타서비스업",
    961: "기타서비스업"
}

# 주요 직업 코드 매핑 (상위 20개 기준)  
JOB_CODE_MAPPING = {
    # 관리자
    111: "기업고위임원", 112: "정부고위공무원", 113: "정치관련직",
    121: "행정관리자", 122: "생산운영관리자", 123: "전문서비스관리자",
    131: "일반관리자",
    
    # 전문가
    211: "물리학관련전문가", 212: "수학관련전문가", 213: "컴퓨터전문가",
    214: "건축기술자", 215: "전기전자기술자", 216: "기계기술자",
    221: "생명과학전문가", 222: "의료진료전문가", 223: "간호전문가",
    231: "교육전문가", 232: "교육전문가",
    241: "경영전문가", 242: "법률전문가", 243: "사회과학전문가",
    244: "종교전문가", 245: "문화예술전문가", 246: "언론전문가",
    247: "사회복지전문가", 248: "기타전문가",
    251: "정보처리전문가", 252: "간호조무사", 254: "기타보건의료전문가",
    261: "법무관련종사자", 262: "사회복지종사자",
    271: "경영관리자", 
    
    # 기술자 및 준전문가
    311: "물리과학기술자", 312: "컴퓨터기술자", 313: "광학전자기술자",
    314: "선박항공기기술자", 315: "건축기술자", 316: "제도원",
    321: "의료기사", 322: "간호조무사",
    331: "금융관련종사자", 332: "상품중개인", 333: "보험관련종사자",
    341: "법무관련종사자", 342: "사회복지관련종사자",
    343: "종교관련종사자", 344: "예술관련종사자",
    
    # 사무종사자
    411: "일반사무종사자", 412: "고객정보사무종사자", 413: "수치기록사무종사자",
    414: "자료입력원", 415: "우편관련사무종사자",
    421: "출납창구사무종사자", 422: "금고사무종사자",
    431: "통계관련사무종사자",
    441: "행정사무원", 442: "경찰관련사무원",
    
    # 서비스 종사자  
    511: "여행안내원", 512: "요리사", 513: "웨이터",
    514: "바텐더", 515: "이미용서비스종사자", 516: "혼례서비스종사자",
    521: "가사도우미", 522: "보육교사",
    
    # 판매종사자
    611: "농어업관련단순노무직", 612: "임업관련종사자",
    
    # 농업관련 종사자
    611: "작물재배종사자", 612: "원예관련종사자", 613: "축산업종사자",
    
    # 기능원 및 관련기능종사자
    711: "광업종사자", 712: "건설관련종사자",
    721: "금속관련종사자", 722: "기계관련종사자",
    731: "전기전자관련종사자", 732: "정보통신관련종사자",
    741: "식품가공관련종사자", 742: "섬유의복관련종사자",
    
    # 장치기계조작원 및 조립종사자
    811: "광업관련장치조작원", 812: "제조업관련장치조작원",
    821: "금속관련장치조작원", 822: "기계관련장치조작원",
    831: "운송장비조작원", 832: "건설장비조작원",
    833: "농업관련장비조작원",
    841: "제조업관련조립원", 842: "운송장비조립원",
    
    # 단순노무종사자
    911: "건설관련단순노무종사자", 912: "운송관련단순노무종사자",
    913: "제조업관련단순노무종사자", 914: "청소관련단순노무종사자",
    921: "농림어업관련단순노무종사자", 922: "광업관련단순노무종사자",
    931: "배달원", 932: "경비원", 933: "청소원",
    941: "가사도우미", 942: "기타서비스단순종사자",
    
    # 기타
    873: "스포츠관련종사자", 156: "기타관리자"
}

# 사용자 친화적 대분류 그룹 정의
USER_FRIENDLY_GROUPS = {
    "농림어업": {
        "industries": [11, 12, 13, 14],
        "jobs": [611, 612, 613, 614, 615, 921],
        "icon": "[농업]",
        "description": "농업, 임업, 어업"
    },
    "제조업": {
        "industries": list(range(15, 38)) + [36, 37],
        "jobs": [711, 712, 721, 722, 723, 724, 731, 732, 741, 742, 
                811, 812, 821, 822, 831, 832, 841, 842, 913],
        "icon": "[제조]", 
        "description": "각종 제조업 및 생산직"
    },
    "IT·정보통신": {
        "industries": [32, 641, 642, 702, 721, 722],
        "jobs": [213, 312, 313, 251, 732],
        "icon": "[IT]",
        "description": "프로그래머, 시스템 관리자, IT 기획자"
    },
    "금융·보험": {
        "industries": [651, 652, 659, 660, 671, 672],
        "jobs": [241, 331, 332, 333, 421, 422],
        "icon": "[금융]",
        "description": "은행원, 보험설계사, 투자상담사"
    },
    "교육": {
        "industries": [801, 841, 842, 843, 849, 552],
        "jobs": [231, 232, 233, 234, 235, 236, 237],
        "icon": "[교육]",
        "description": "교사, 교수, 교육 관련직"
    },
    "의료·보건": {
        "industries": [851, 852, 853, 854, 855],
        "jobs": [222, 223, 224, 225, 226, 227, 321, 322, 323],
        "icon": "[의료]",
        "description": "의사, 간호사, 의료기사"
    },
    "건설·건축": {
        "industries": [451, 452, 453, 454, 455],
        "jobs": [214, 215, 216, 711, 712, 713, 832, 911],
        "icon": "[건설]",
        "description": "건축설계, 시공, 건설장비 운영"
    },
    "유통·판매": {
        "industries": [471, 472, 473, 474, 492, 493, 494, 495, 496],
        "jobs": [331, 332, 520, 521, 522, 523, 524],
        "icon": "[유통]", 
        "description": "도소매업, 영업, 판매"
    },
    "음식·숙박": {
        "industries": [521, 522, 551, 552],
        "jobs": [512, 513, 514, 515, 516],
        "icon": "[음식]",
        "description": "요리사, 서빙, 호텔업"
    },
    "운송·물류": {
        "industries": [601, 602, 603, 611, 621, 622, 623, 631, 634],
        "jobs": [831, 832, 833, 912],
        "icon": "[운송]",
        "description": "택시, 버스, 화물, 배송"
    },
    "공공행정": {
        "industries": [751, 752, 753, 801, 802, 803, 804, 805, 806, 807, 808, 809],
        "jobs": [111, 112, 113, 121, 122, 123, 131, 441, 442],
        "icon": "[공공]",
        "description": "공무원, 공공기관"
    },
    "기타서비스": {
        "industries": [900, 910, 920, 930, 931, 932, 960, 961],
        "jobs": [511, 516, 517, 518, 521, 522, 531, 532, 533, 534, 
                611, 612, 613, 621, 622, 623, 914, 931, 932, 933, 941, 942],
        "icon": "[서비스]",
        "description": "개인서비스, 사회서비스 등"
    }
}

def analyze_code_distribution():
    """주요 코드들의 분포와 매핑 가능성 분석"""
    
    df = pd.read_csv('data/raw_data/nextep_dataset.csv')
    
    # 모든 산업/직업 코드 수집
    all_industries = []
    all_jobs = []
    
    for col in ['p_ind2000', 'p_ind2007', 'p_ind2017']:
        if col in df.columns:
            valid_data = df[col].dropna().astype(int)
            all_industries.extend(valid_data.tolist())
    
    for col in ['p_jobfam2000', 'p_jobfam2007', 'p_jobfam2017']:
        if col in df.columns:
            valid_data = df[col].dropna().astype(int) 
            all_jobs.extend(valid_data.tolist())
    
    industry_counts = Counter(all_industries)
    job_counts = Counter(all_jobs)
    
    print("=== 상위 30개 산업 코드 매핑 ===")
    print("코드 | 빈도 | 매핑된 이름")
    print("-" * 50)
    
    mapped_industries = 0
    total_industry_records = 0
    
    for code, count in industry_counts.most_common(30):
        total_industry_records += count
        mapped_name = INDUSTRY_CODE_MAPPING.get(code, "미매핑")
        if mapped_name != "미매핑":
            mapped_industries += count
        print(f"{code:4d} | {count:5d} | {mapped_name}")
    
    print(f"\n매핑 커버리지: {mapped_industries/total_industry_records*100:.1f}% ({mapped_industries:,}/{total_industry_records:,})")
    
    print("\n=== 상위 30개 직업 코드 매핑 ===")
    print("코드 | 빈도 | 매핑된 이름")
    print("-" * 50)
    
    mapped_jobs = 0
    total_job_records = 0
    
    for code, count in job_counts.most_common(30):
        total_job_records += count
        mapped_name = JOB_CODE_MAPPING.get(code, "미매핑")
        if mapped_name != "미매핑":
            mapped_jobs += count
        print(f"{code:4d} | {count:5d} | {mapped_name}")
    
    print(f"\n매핑 커버리지: {mapped_jobs/total_job_records*100:.1f}% ({mapped_jobs:,}/{total_job_records:,})")
    
    return industry_counts, job_counts

def create_user_friendly_mapping():
    """사용자 친화적 그룹으로 매핑"""
    
    print("\n=== 사용자 친화적 직업 그룹 설계 ===")
    
    for group_name, group_info in USER_FRIENDLY_GROUPS.items():
        print(f"\n{group_info['icon']} {group_name}")
        print(f"   설명: {group_info['description']}")
        print(f"   산업코드: {len(group_info['industries'])}개")
        print(f"   직업코드: {len(group_info['jobs'])}개")
        
        # 샘플 코드 몇 개 표시
        sample_industries = group_info['industries'][:5]
        sample_jobs = group_info['jobs'][:5]
        
        industry_names = [INDUSTRY_CODE_MAPPING.get(code, f"코드{code}") for code in sample_industries]
        job_names = [JOB_CODE_MAPPING.get(code, f"코드{code}") for code in sample_jobs]
        
        print(f"   산업 예시: {', '.join(industry_names)}")
        print(f"   직업 예시: {', '.join(job_names)}")

def analyze_group_coverage():
    """각 그룹이 실제 데이터를 얼마나 커버하는지 분석"""
    
    df = pd.read_csv('data/raw_data/nextep_dataset.csv')
    
    # 2017년 기준 데이터로 분석 (가장 최신)
    df_2017 = df[(df['p_ind2017'].notna()) & (df['p_jobfam2017'].notna())].copy()
    df_2017['p_ind2017'] = df_2017['p_ind2017'].astype(int)
    df_2017['p_jobfam2017'] = df_2017['p_jobfam2017'].astype(int)
    
    print(f"\n=== 그룹별 데이터 커버리지 분석 (2017년 기준, N={len(df_2017):,}) ===")
    
    group_coverage = {}
    total_covered = 0
    
    for group_name, group_info in USER_FRIENDLY_GROUPS.items():
        # 해당 그룹에 속하는 데이터 찾기
        in_group = df_2017[
            df_2017['p_ind2017'].isin(group_info['industries']) |
            df_2017['p_jobfam2017'].isin(group_info['jobs'])
        ]
        
        count = len(in_group)
        percentage = count / len(df_2017) * 100
        
        group_coverage[group_name] = {
            'count': count,
            'percentage': percentage
        }
        
        total_covered += count
        
        print(f"{group_info['icon']} {group_name:12s}: {count:5d}명 ({percentage:5.1f}%)")
    
    uncovered = len(df_2017) - total_covered
    print(f"[미분류] 미분류:        {uncovered:5d}명 ({uncovered/len(df_2017)*100:5.1f}%)")
    print(f"[전체] 총 커버리지:   {total_covered:5d}명 ({total_covered/len(df_2017)*100:5.1f}%)")
    
    return group_coverage

# 실행
if __name__ == "__main__":
    print("2단계 분석 시작...")
    
    # 코드 분포 분석
    industry_counts, job_counts = analyze_code_distribution()
    
    # 사용자 친화적 매핑 생성
    create_user_friendly_mapping()
    
    # 그룹 커버리지 분석
    group_coverage = analyze_group_coverage()
    
    # 시각화 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 사용자 그룹별 분포
    group_names = list(group_coverage.keys())
    group_counts = [group_coverage[name]['count'] for name in group_names]
    
    axes[0, 0].pie(group_counts, labels=[name.split('·')[0] for name in group_names], autopct='%1.1f%%')
    axes[0, 0].set_title('사용자 친화적 직업 그룹 분포')
    
    # 2. 상위 산업 코드
    top_industries = dict(Counter(industry_counts).most_common(10))
    axes[0, 1].bar(range(len(top_industries)), list(top_industries.values()))
    axes[0, 1].set_title('상위 10개 산업 코드')
    axes[0, 1].set_xticks(range(len(top_industries)))
    axes[0, 1].set_xticklabels([str(k) for k in top_industries.keys()], rotation=45)
    
    # 3. 상위 직업 코드  
    top_jobs = dict(Counter(job_counts).most_common(10))
    axes[1, 0].bar(range(len(top_jobs)), list(top_jobs.values()))
    axes[1, 0].set_title('상위 10개 직업 코드')
    axes[1, 0].set_xticks(range(len(top_jobs)))
    axes[1, 0].set_xticklabels([str(k) for k in top_jobs.keys()], rotation=45)
    
    # 4. 그룹별 커버리지
    axes[1, 1].bar(range(len(group_names)), [group_coverage[name]['percentage'] for name in group_names])
    axes[1, 1].set_title('사용자 그룹별 커버리지')
    axes[1, 1].set_xticks(range(len(group_names)))
    axes[1, 1].set_xticklabels([name.split('·')[0] for name in group_names], rotation=45)
    axes[1, 1].set_ylabel('비율 (%)')
    
    plt.tight_layout()
    plt.savefig('visualizations/occupation_analysis_step2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 2단계 분석 완료 ===")
    print("다음 단계: 매핑 테이블 정교화 및 통합 코드 구현")