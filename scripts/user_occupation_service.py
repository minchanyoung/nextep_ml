import json
import pandas as pd
from occupation_integration_final import OccupationMapper


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
