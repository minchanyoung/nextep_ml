import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import percentileofscore
import warnings
warnings.filterwarnings('ignore')

print("=== 4단계: 최종 통합 직업 분류 코드 구현 ===")

class OccupationMapper:
    """직업 분류 통합 매핑 클래스"""
    
    def __init__(self):
        self.load_mapping_tables()
    
    def load_mapping_tables(self):
        """저장된 매핑 테이블 로드"""
        try:
            with open('data/occupation_mapping_api.json', 'r', encoding='utf-8') as f:
                self.api_data = json.load(f)
            
            self.industry_to_group = self.api_data['code_mapping']['industry_to_group']
            self.job_to_group = self.api_data['code_mapping']['job_to_group']
            self.industry_to_subcat = self.api_data['code_mapping']['industry_to_subcat']
            self.job_to_subcat = self.api_data['code_mapping']['job_to_subcat']
            
            # 문자열 키를 정수로 변환
            self.industry_to_group = {int(k): v for k, v in self.industry_to_group.items()}
            self.job_to_group = {int(k): v for k, v in self.job_to_group.items()}
            self.industry_to_subcat = {int(k): v for k, v in self.industry_to_subcat.items()}
            self.job_to_subcat = {int(k): v for k, v in self.job_to_subcat.items()}
            
            print("매핑 테이블 로드 완료")
            
        except FileNotFoundError:
            print("매핑 테이블 파일을 찾을 수 없습니다. 3단계를 먼저 실행하세요.")
            self.industry_to_group = {}
            self.job_to_group = {}
            self.industry_to_subcat = {}
            self.job_to_subcat = {}
    
    def get_unified_occupation(self, industry_code, job_code):
        """산업코드와 직업코드를 통합 직업으로 매핑"""
        
        # NaN 처리
        if pd.isna(industry_code) and pd.isna(job_code):
            return "미분류"
        
        # 정수 변환
        try:
            ind_code = int(industry_code) if not pd.isna(industry_code) else None
            job_code = int(job_code) if not pd.isna(job_code) else None
        except (ValueError, TypeError):
            return "미분류"
        
        # 우선순위: 직업코드 > 산업코드
        if job_code and job_code in self.job_to_group:
            return self.job_to_group[job_code]
        elif ind_code and ind_code in self.industry_to_group:
            return self.industry_to_group[ind_code]
        else:
            return "미분류"
    
    def get_occupation_subcategory(self, industry_code, job_code):
        """세부 직업 카테고리 매핑"""
        
        if pd.isna(industry_code) and pd.isna(job_code):
            return "미분류"
        
        try:
            ind_code = int(industry_code) if not pd.isna(industry_code) else None
            job_code = int(job_code) if not pd.isna(job_code) else None
        except (ValueError, TypeError):
            return "미분류"
        
        # 우선순위: 직업코드 > 산업코드
        if job_code and job_code in self.job_to_subcat:
            return self.job_to_subcat[job_code]
        elif ind_code and ind_code in self.industry_to_subcat:
            return self.industry_to_subcat[ind_code]
        else:
            return "미분류"
    
    def create_occupation_code(self, industry_code, job_code):
        """원본 코드를 유지하면서 통합 식별자 생성"""
        
        # 가능한 한 원본 정보 보존
        ind_str = str(int(industry_code)) if not pd.isna(industry_code) else "0"
        job_str = str(int(job_code)) if not pd.isna(job_code) else "0"
        
        return f"{ind_str}_{job_str}"

def apply_occupation_mapping(df):
    """데이터프레임에 통합 직업 매핑 적용"""
    
    mapper = OccupationMapper()
    
    print(f"매핑 적용 전 데이터 크기: {df.shape}")
    
    # 시간순으로 가장 최신 분류 사용
    industry_cols = ['p_ind2017', 'p_ind2007', 'p_ind2000']
    job_cols = ['p_jobfam2017', 'p_jobfam2007', 'p_jobfam2000']
    
    # 최신 분류를 우선으로 통합 컬럼 생성
    df['current_industry'] = df[industry_cols].bfill(axis=1).iloc[:, 0]
    df['current_job'] = df[job_cols].bfill(axis=1).iloc[:, 0]
    
    # 통합 직업 분류 생성
    print("통합 직업 분류 생성 중...")
    df['occupation_group'] = df.apply(
        lambda row: mapper.get_unified_occupation(row['current_industry'], row['current_job']), 
        axis=1
    )
    
    df['occupation_subcategory'] = df.apply(
        lambda row: mapper.get_occupation_subcategory(row['current_industry'], row['current_job']),
        axis=1
    )
    
    df['occupation_code'] = df.apply(
        lambda row: mapper.create_occupation_code(row['current_industry'], row['current_job']),
        axis=1
    )
    
    # 매핑 결과 분석
    print("\n매핑 결과 분석:")
    group_counts = df['occupation_group'].value_counts()
    print("직업 그룹별 분포:")
    for group, count in group_counts.items():
        pct = count / len(df) * 100
        print(f"  {group}: {count:,}건 ({pct:.1f}%)")
    
    return df

def create_enhanced_features(df):
    """통합 직업 분류 기반 추가 특성 생성"""
    
    print("\n고급 직업 기반 특성 생성 중...")
    
    # 1. 직업 안정성 지표
    df['job_stability'] = df.groupby('pid')['occupation_group'].transform(lambda x: 1 if x.nunique() == 1 else 0)
    df['job_changes'] = df.groupby('pid')['occupation_group'].transform('nunique') - 1
    
    # 2. 직업군별 임금 통계
    occupation_wage_stats = df.groupby('occupation_group')['p_wage'].agg(['mean', 'median', 'std']).round(2)
    occupation_wage_stats.columns = ['occupation_avg_wage', 'occupation_med_wage', 'occupation_std_wage']
    
    # 임금 통계 병합
    df = df.merge(occupation_wage_stats, left_on='occupation_group', right_index=True, how='left')
    
    # 3. 개인 임금 vs 직업군 평균 비교
    df['wage_vs_occupation_avg'] = df['p_wage'] - df['occupation_avg_wage']
    
    # 단순화된 percentile 계산 (quartile 기반)
    df['wage_quartile_in_occupation'] = df.groupby('occupation_group')['p_wage'].rank(pct=True) * 100
    
    # 4. 직업군별 만족도 통계
    occupation_satisfaction_stats = df.groupby('occupation_group')['p4321'].agg(['mean', 'std']).round(3)
    occupation_satisfaction_stats.columns = ['occupation_avg_satisfaction', 'occupation_std_satisfaction']
    
    df = df.merge(occupation_satisfaction_stats, left_on='occupation_group', right_index=True, how='left')
    
    # 5. 시간 기반 직업 특성
    df['occupation_tenure'] = df.groupby(['pid', 'occupation_group']).cumcount() + 1  # 동일 직업군 근속
    df['years_since_job_change'] = df.groupby('pid')['occupation_group'].apply(
        lambda x: (x != x.shift()).cumsum()
    ).reset_index(level=0, drop=True)
    
    # 6. 직업 이동 패턴
    df['occupation_prev'] = df.groupby('pid')['occupation_group'].shift(1)
    df['occupation_changed'] = (df['occupation_group'] != df['occupation_prev']).astype(int)
    
    # 7. 스킬 레벨 매핑 (대략적)
    skill_level_mapping = {
        'IT·정보통신': 'high_skill',
        '의료·보건': 'high_skill', 
        '교육': 'high_skill',
        '금융·보험': 'mid_skill',
        '공공행정': 'mid_skill',
        '제조업': 'mid_skill',
        '건설·건축': 'mid_skill',
        '유통·판매': 'low_skill',
        '운송·물류': 'low_skill',
        '음식·숙박': 'low_skill',
        '농림어업': 'low_skill',
        '미분류': 'unknown'
    }
    
    df['skill_level'] = df['occupation_group'].map(skill_level_mapping)
    
    print("고급 특성 생성 완료")
    return df

def save_processed_dataset(df, output_path):
    """처리된 데이터셋 저장"""
    
    print(f"\n최종 데이터셋 저장: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 통계 정보 출력
    print(f"최종 데이터 크기: {df.shape}")
    print(f"추가된 직업 관련 컬럼 수: {len([col for col in df.columns if 'occupation' in col])}")
    
    # 새로 생성된 컬럼들
    new_columns = [
        'current_industry', 'current_job', 'occupation_group', 'occupation_subcategory', 
        'occupation_code', 'job_stability', 'job_changes', 'occupation_avg_wage',
        'occupation_med_wage', 'occupation_std_wage', 'wage_vs_occupation_avg',
        'wage_quartile_in_occupation', 'occupation_avg_satisfaction', 
        'occupation_std_satisfaction', 'occupation_tenure', 'years_since_job_change',
        'occupation_prev', 'occupation_changed', 'skill_level'
    ]
    
    print(f"\n생성된 직업 관련 특성들:")
    for i, col in enumerate(new_columns, 1):
        if col in df.columns:
            print(f"  {i:2d}. {col}")

def create_user_service_functions():
    """사용자 서비스를 위한 헬퍼 함수들"""
    
    service_code = '''
# 사용자 서비스용 헬퍼 함수들

class UserOccupationService:
    """사용자 입력을 처리하는 서비스 클래스"""
    
    def __init__(self):
        self.mapper = OccupationMapper()
        with open('data/occupation_mapping_api.json', 'r', encoding='utf-8') as f:
            self.api_data = json.load(f)
    
    def search_jobs(self, keyword):
        """직업 검색 함수"""
        results = []
        search_index = self.api_data['search_index']
        
        for key, jobs in search_index.items():
            if keyword.lower() in key.lower():
                results.extend(jobs)
        
        # 중복 제거
        unique_results = []
        seen = set()
        for job in results:
            job_key = (job['job_name'], job['group'])
            if job_key not in seen:
                unique_results.append(job)
                seen.add(job_key)
        
        return unique_results
    
    def get_job_suggestions(self, group=None, subcategory=None):
        """그룹/서브카테고리별 직업 제안"""
        suggestions = []
        
        for group_data in self.api_data['job_selection']['groups']:
            if group and group != group_data['group_name']:
                continue
                
            for subcat_data in group_data['subcategories']:
                if subcategory and subcategory != subcat_data['subcategory_name']:
                    continue
                    
                for job in subcat_data['jobs']:
                    suggestions.append({
                        'job_name': job,
                        'group': group_data['group_name'],
                        'subcategory': subcat_data['subcategory_name'],
                        'icon': group_data['icon']
                    })
        
        return suggestions
    
    def predict_user_profile(self, user_input):
        """사용자 입력 기반 특성 생성"""
        
        # 사용자 입력 예시
        # user_input = {
        #     'age': 30,
        #     'education': '대학교 졸업',
        #     'job_name': '소프트웨어 개발자',
        #     'current_salary': 5000,
        #     'job_satisfaction': 4,
        #     'experience_years': 5
        # }
        
        # 직업명을 그룹으로 매핑
        job_suggestions = self.search_jobs(user_input['job_name'])
        if job_suggestions:
            occupation_group = job_suggestions[0]['group']
        else:
            occupation_group = '미분류'
        
        # 특성 생성
        features = {
            'p_age': user_input['age'],
            'p_edu': self._map_education(user_input['education']),
            'experience_years': user_input['experience_years'],
            'p_wage': user_input['current_salary'],
            'p4321': user_input['job_satisfaction'],
            'occupation_group': occupation_group,
            'skill_level': self._get_skill_level(occupation_group)
        }
        
        return features
    
    def _map_education(self, education_str):
        """교육 수준 매핑"""
        mapping = {
            '고등학교 졸업': 3,
            '전문대 졸업': 4, 
            '대학교 졸업': 5,
            '대학원 졸업': 6
        }
        return mapping.get(education_str, 5)
    
    def _get_skill_level(self, occupation_group):
        """스킬 레벨 매핑"""
        skill_mapping = {
            'IT·정보통신': 'high_skill',
            '의료·보건': 'high_skill',
            '교육': 'high_skill',
            '금융·보험': 'mid_skill',
            '공공행정': 'mid_skill'
        }
        return skill_mapping.get(occupation_group, 'mid_skill')
'''
    
    # 서비스 코드를 파일로 저장
    with open('scripts/user_occupation_service.py', 'w', encoding='utf-8') as f:
        f.write("import json\nimport pandas as pd\nfrom occupation_integration_final import OccupationMapper\n\n")
        f.write(service_code)
    
    print("사용자 서비스 함수 생성 완료: scripts/user_occupation_service.py")

# 메인 실행
def main():
    """메인 실행 함수"""
    
    print("통합 직업 분류 시스템 구현 시작...")
    
    # 1. 원본 데이터 로드
    df = pd.read_csv('data/raw_data/nextep_dataset.csv')
    print(f"원본 데이터 로드: {df.shape}")
    
    # 2. 직업 매핑 적용
    df_mapped = apply_occupation_mapping(df)
    
    # 3. 고급 특성 생성
    df_enhanced = create_enhanced_features(df_mapped)
    
    # 4. 최종 데이터셋 저장
    save_processed_dataset(df_enhanced, 'processed_data/dataset_with_unified_occupations.csv')
    
    # 5. 사용자 서비스 함수 생성
    create_user_service_functions()
    
    print("\n=== 4단계 완료 ===")
    print("통합 직업 분류 시스템 구현 완료!")
    print("생성된 파일:")
    print("  - processed_data/dataset_with_unified_occupations.csv")
    print("  - scripts/user_occupation_service.py")

if __name__ == "__main__":
    main()