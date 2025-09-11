import pandas as pd
import json
from pathlib import Path

print("=== 3단계: 정교한 사용자 친화적 매핑 테이블 설계 ===")

# 실제 서비스에서 사용할 완전한 매핑 테이블
COMPLETE_OCCUPATION_MAPPING = {
    # 대분류 -> 중분류 -> 세분류 구조
    "IT·정보통신": {
        "icon": "💻",
        "description": "정보기술, 소프트웨어, 통신 관련 직업",
        "subcategories": {
            "소프트웨어개발": {
                "jobs": ["소프트웨어 개발자", "웹 개발자", "앱 개발자", "게임 개발자"],
                "industry_codes": [702, 721, 32],
                "job_codes": [213, 312, 313]
            },
            "IT기획관리": {
                "jobs": ["IT 기획자", "프로젝트 매니저", "시스템 분석가"],
                "industry_codes": [702],
                "job_codes": [241, 251]
            },
            "네트워크시스템": {
                "jobs": ["시스템 관리자", "네트워크 관리자", "보안 관리자"],
                "industry_codes": [641, 642],
                "job_codes": [312, 313]
            },
            "IT지원": {
                "jobs": ["기술지원", "QA 엔지니어", "IT 컨설턴트"],
                "industry_codes": [702],
                "job_codes": [312, 251]
            }
        }
    },
    
    "금융·보험": {
        "icon": "🏦", 
        "description": "은행, 보험, 증권, 투자 관련 직업",
        "subcategories": {
            "은행업무": {
                "jobs": ["은행원", "대출 상담사", "외환 담당자"],
                "industry_codes": [651, 659],
                "job_codes": [421, 422, 331]
            },
            "보험업무": {
                "jobs": ["보험설계사", "손해사정사", "보험 언더라이터"],
                "industry_codes": [660, 671, 672],
                "job_codes": [333, 332]
            },
            "증권투자": {
                "jobs": ["증권 브로커", "투자상담사", "펀드매니저", "애널리스트"],
                "industry_codes": [652],
                "job_codes": [241, 331, 332]
            },
            "금융기획": {
                "jobs": ["금융 기획자", "리스크 관리자", "신용분석가"],
                "industry_codes": [651, 659],
                "job_codes": [241]
            }
        }
    },
    
    "의료·보건": {
        "icon": "🏥",
        "description": "의료진, 간호, 의료기사, 보건 관련 직업", 
        "subcategories": {
            "의료진": {
                "jobs": ["의사", "치과의사", "한의사", "수의사"],
                "industry_codes": [851, 852],
                "job_codes": [222]
            },
            "간호": {
                "jobs": ["간호사", "간호조무사"],
                "industry_codes": [851, 852, 855],
                "job_codes": [223, 252, 322]
            },
            "의료기사": {
                "jobs": ["방사선사", "임상병리사", "물리치료사", "작업치료사"],
                "industry_codes": [851, 852],
                "job_codes": [321, 224, 225, 226]
            },
            "약료": {
                "jobs": ["약사", "한약사"],
                "industry_codes": [853],
                "job_codes": [227]
            },
            "보건관리": {
                "jobs": ["보건관리자", "영양사", "위생사"],
                "industry_codes": [854, 855],
                "job_codes": [254, 322]
            }
        }
    },
    
    "교육": {
        "icon": "📚",
        "description": "교사, 교수, 강사, 교육 관련 직업",
        "subcategories": {
            "초중고교육": {
                "jobs": ["초등교사", "중학교사", "고등학교사", "특수교사"],
                "industry_codes": [801, 841, 842],
                "job_codes": [231, 232, 233]
            },
            "고등교육": {
                "jobs": ["대학교수", "전문대 교수", "연구교수"],
                "industry_codes": [843],
                "job_codes": [231]
            },
            "직업교육": {
                "jobs": ["직업훈련교사", "기술교육교사", "평생교육사"],
                "industry_codes": [849, 552],
                "job_codes": [234, 235]
            },
            "교육지원": {
                "jobs": ["교육행정가", "상담교사", "사서"],
                "industry_codes": [841, 842],
                "job_codes": [236, 237]
            }
        }
    },
    
    "제조업": {
        "icon": "🏭",
        "description": "제조, 생산, 품질관리, 기술 관련 직업",
        "subcategories": {
            "생산관리": {
                "jobs": ["생산관리자", "품질관리자", "공장장"],
                "industry_codes": list(range(15, 38)),
                "job_codes": [122, 131]
            },
            "기술연구": {
                "jobs": ["연구개발자", "기술자", "설계자"],
                "industry_codes": list(range(15, 38)),
                "job_codes": [214, 215, 216, 311, 315, 316]
            },
            "생산직": {
                "jobs": ["조립원", "기계조작원", "검사원"],
                "industry_codes": list(range(15, 38)),
                "job_codes": [721, 722, 811, 812, 821, 822, 841, 842]
            },
            "기능직": {
                "jobs": ["용접공", "선반공", "도장공", "전기공"],
                "industry_codes": list(range(15, 38)),
                "job_codes": [711, 712, 721, 722, 731]
            }
        }
    },
    
    "건설·건축": {
        "icon": "🏗️", 
        "description": "건설, 건축, 토목, 설비 관련 직업",
        "subcategories": {
            "건축설계": {
                "jobs": ["건축사", "건축설계사", "구조설계사"],
                "industry_codes": [741],
                "job_codes": [214]
            },
            "토목설계": {
                "jobs": ["토목기술자", "측량기술자", "도시계획가"],
                "industry_codes": [741],
                "job_codes": [214, 215]
            },
            "건설시공": {
                "jobs": ["현장소장", "공사감독", "안전관리자"],
                "industry_codes": [452, 453, 454, 455],
                "job_codes": [122, 131, 216]
            },
            "건설기능": {
                "jobs": ["목수", "철근공", "타일공", "미장공", "도배공"],
                "industry_codes": [452, 453, 454],
                "job_codes": [711, 712, 713]
            },
            "건설장비": {
                "jobs": ["굴삭기 운전원", "크레인 운전원", "덤프트럭 운전원"],
                "industry_codes": [452, 453],
                "job_codes": [832]
            }
        }
    },
    
    "유통·판매": {
        "icon": "🛒",
        "description": "도소매, 영업, 마케팅, 판매 관련 직업",
        "subcategories": {
            "영업": {
                "jobs": ["영업사원", "영업관리자", "세ール즈"],
                "industry_codes": [471, 472, 473, 474],
                "job_codes": [331, 332]
            },
            "판매": {
                "jobs": ["판매원", "매장관리자", "캐셔"],
                "industry_codes": [492, 493, 494, 495, 496],
                "job_codes": [520, 521, 522, 523, 524]
            },
            "마케팅": {
                "jobs": ["마케팅 기획자", "광고 기획자", "브랜드 매니저"],
                "industry_codes": [471, 741],
                "job_codes": [241, 245]
            },
            "유통관리": {
                "jobs": ["유통관리자", "물류관리자", "구매담당자"],
                "industry_codes": [471, 631],
                "job_codes": [122, 131]
            }
        }
    },
    
    "공공행정": {
        "icon": "🏛️",
        "description": "공무원, 공공기관, 정부 관련 직업",
        "subcategories": {
            "행정공무원": {
                "jobs": ["일반행정직", "재무직", "인사직"],
                "industry_codes": [751, 752, 753, 801, 802, 803],
                "job_codes": [121, 441, 442]
            },
            "기술공무원": {
                "jobs": ["토목직", "건축직", "전산직", "환경직"],
                "industry_codes": [751, 804, 805],
                "job_codes": [214, 215, 312]
            },
            "전문공무원": {
                "jobs": ["외교관", "검사", "판사", "경찰관"],
                "industry_codes": [806, 807, 808],
                "job_codes": [111, 112, 242, 341]
            },
            "공공서비스": {
                "jobs": ["사회복지공무원", "보건공무원", "교육공무원"],
                "industry_codes": [809],
                "job_codes": [247, 254, 234]
            }
        }
    },
    
    "운송·물류": {
        "icon": "🚚",
        "description": "운송, 배송, 물류, 교통 관련 직업",
        "subcategories": {
            "육상운송": {
                "jobs": ["택시기사", "버스기사", "트럭기사"],
                "industry_codes": [602, 603],
                "job_codes": [831]
            },
            "물류창고": {
                "jobs": ["물류관리자", "창고관리자", "하역작업원"],
                "industry_codes": [631, 634],
                "job_codes": [122, 912, 913]
            },
            "배송서비스": {
                "jobs": ["택배기사", "배달원", "퀵서비스"],
                "industry_codes": [634],
                "job_codes": [912, 931]
            },
            "교통관제": {
                "jobs": ["교통관제사", "운송기획자", "물류기획자"],
                "industry_codes": [601, 631],
                "job_codes": [241, 122]
            }
        }
    },
    
    "음식·숙박": {
        "icon": "🍴",
        "description": "요리, 서빙, 호텔, 관광 관련 직업",
        "subcategories": {
            "조리": {
                "jobs": ["한식조리사", "양식조리사", "중식조리사", "일식조리사"],
                "industry_codes": [522],
                "job_codes": [512]
            },
            "서빙": {
                "jobs": ["웨이터", "바리스타", "바텐더", "홀매니저"],
                "industry_codes": [522],
                "job_codes": [513, 514]
            },
            "호텔": {
                "jobs": ["호텔리어", "프론트", "하우스키핑", "컨시어지"],
                "industry_codes": [521, 551],
                "job_codes": [511, 516, 941]
            },
            "관광": {
                "jobs": ["여행가이드", "관광기획자", "여행상품기획자"],
                "industry_codes": [633],
                "job_codes": [511, 241]
            }
        }
    },
    
    "농림어업": {
        "icon": "🌾",
        "description": "농업, 임업, 어업, 축산업 관련 직업",
        "subcategories": {
            "농업": {
                "jobs": ["농업인", "농장관리자", "작물재배자"],
                "industry_codes": [11],
                "job_codes": [611, 612]
            },
            "축산업": {
                "jobs": ["축산업자", "목장관리자", "사육사"],
                "industry_codes": [12],
                "job_codes": [613]
            },
            "임업": {
                "jobs": ["임업인", "산림관리자", "조경사"],
                "industry_codes": [13],
                "job_codes": [612]
            },
            "어업": {
                "jobs": ["어업인", "양식업자", "선원"],
                "industry_codes": [14],
                "job_codes": [614, 615]
            }
        }
    }
}

# 역매핑 테이블 생성 (코드 -> 사용자 친화적 이름)
def create_reverse_mapping_tables():
    """코드에서 사용자 친화적 이름으로의 역매핑 테이블 생성"""
    
    industry_to_group = {}
    job_to_group = {}
    industry_to_subcat = {}
    job_to_subcat = {}
    
    for group_name, group_data in COMPLETE_OCCUPATION_MAPPING.items():
        for subcat_name, subcat_data in group_data["subcategories"].items():
            # 산업 코드 매핑
            for ind_code in subcat_data["industry_codes"]:
                industry_to_group[ind_code] = group_name
                industry_to_subcat[ind_code] = subcat_name
            
            # 직업 코드 매핑  
            for job_code in subcat_data["job_codes"]:
                job_to_group[job_code] = group_name
                job_to_subcat[job_code] = subcat_name
    
    return {
        "industry_to_group": industry_to_group,
        "job_to_group": job_to_group, 
        "industry_to_subcat": industry_to_subcat,
        "job_to_subcat": job_to_subcat
    }

# 사용자 입력 매핑 테이블 생성
def create_user_input_mapping():
    """사용자가 선택할 수 있는 직업 리스트 생성"""
    
    user_job_list = []
    
    for group_name, group_data in COMPLETE_OCCUPATION_MAPPING.items():
        for subcat_name, subcat_data in group_data["subcategories"].items():
            for job_name in subcat_data["jobs"]:
                user_job_list.append({
                    "job_name": job_name,
                    "group": group_name,
                    "subcategory": subcat_name,
                    "icon": group_data["icon"],
                    "description": group_data["description"]
                })
    
    # 검색을 위한 키워드 매핑도 생성
    search_keywords = {}
    for item in user_job_list:
        job_name = item["job_name"]
        keywords = job_name.split() + [item["group"], item["subcategory"]]
        for keyword in keywords:
            if keyword not in search_keywords:
                search_keywords[keyword] = []
            search_keywords[keyword].append(item)
    
    return user_job_list, search_keywords

# 매핑 테이블 검증
def validate_mapping_coverage():
    """생성된 매핑 테이블이 실제 데이터를 얼마나 커버하는지 검증"""
    
    df = pd.read_csv('data/raw_data/nextep_dataset.csv')
    
    # 2017년 기준으로 검증
    df_2017 = df[(df['p_ind2017'].notna()) & (df['p_jobfam2017'].notna())].copy()
    df_2017['p_ind2017'] = df_2017['p_ind2017'].astype(int)
    df_2017['p_jobfam2017'] = df_2017['p_jobfam2017'].astype(int)
    
    reverse_mapping = create_reverse_mapping_tables()
    
    # 커버리지 계산
    covered_by_industry = 0
    covered_by_job = 0
    covered_by_both = 0
    
    for _, row in df_2017.iterrows():
        ind_code = row['p_ind2017']
        job_code = row['p_jobfam2017']
        
        has_industry_mapping = ind_code in reverse_mapping["industry_to_group"]
        has_job_mapping = job_code in reverse_mapping["job_to_group"]
        
        if has_industry_mapping:
            covered_by_industry += 1
        if has_job_mapping:
            covered_by_job += 1
        if has_industry_mapping or has_job_mapping:
            covered_by_both += 1
    
    total = len(df_2017)
    
    print(f"매핑 테이블 커버리지 검증 (2017년 기준, N={total:,})")
    print(f"산업 코드로 매핑 가능: {covered_by_industry:,}건 ({covered_by_industry/total*100:.1f}%)")
    print(f"직업 코드로 매핑 가능: {covered_by_job:,}건 ({covered_by_job/total*100:.1f}%)")
    print(f"둘 중 하나로 매핑 가능: {covered_by_both:,}건 ({covered_by_both/total*100:.1f}%)")
    
    return {
        "total": total,
        "covered_by_industry": covered_by_industry,
        "covered_by_job": covered_by_job,
        "covered_by_both": covered_by_both,
        "coverage_rate": covered_by_both / total
    }

# 서비스용 API 데이터 생성
def generate_service_api_data():
    """실제 서비스에서 사용할 API용 데이터 구조 생성"""
    
    user_job_list, search_keywords = create_user_input_mapping()
    reverse_mapping = create_reverse_mapping_tables()
    
    # 1. 사용자 선택용 직업 리스트 (드롭다운/검색용)
    job_selection_data = {
        "groups": []
    }
    
    for group_name, group_data in COMPLETE_OCCUPATION_MAPPING.items():
        group_info = {
            "group_id": group_name.replace("·", "_").replace(" ", "_"),
            "group_name": group_name,
            "icon": group_data["icon"],
            "description": group_data["description"],
            "subcategories": []
        }
        
        for subcat_name, subcat_data in group_data["subcategories"].items():
            subcat_info = {
                "subcategory_id": subcat_name,
                "subcategory_name": subcat_name,
                "jobs": subcat_data["jobs"]
            }
            group_info["subcategories"].append(subcat_info)
        
        job_selection_data["groups"].append(group_info)
    
    # 2. 검색용 키워드 인덱스
    search_index = {}
    for keyword, jobs in search_keywords.items():
        search_index[keyword.lower()] = [
            {
                "job_name": job["job_name"],
                "group": job["group"],
                "subcategory": job["subcategory"]
            } for job in jobs
        ]
    
    # 3. 코드 변환용 매핑 테이블
    code_mapping = {
        "industry_to_group": reverse_mapping["industry_to_group"],
        "job_to_group": reverse_mapping["job_to_group"],
        "industry_to_subcat": reverse_mapping["industry_to_subcat"],
        "job_to_subcat": reverse_mapping["job_to_subcat"]
    }
    
    return {
        "job_selection": job_selection_data,
        "search_index": search_index,
        "code_mapping": code_mapping
    }

# 파일 저장
def save_mapping_tables():
    """생성된 매핑 테이블들을 파일로 저장"""
    
    # 데이터 생성
    api_data = generate_service_api_data()
    validation_result = validate_mapping_coverage()
    
    # 1. 전체 매핑 정보 저장
    with open('data/occupation_mapping_complete.json', 'w', encoding='utf-8') as f:
        json.dump(COMPLETE_OCCUPATION_MAPPING, f, ensure_ascii=False, indent=2)
    
    # 2. 서비스 API용 데이터 저장
    with open('data/occupation_mapping_api.json', 'w', encoding='utf-8') as f:
        json.dump(api_data, f, ensure_ascii=False, indent=2)
    
    # 3. 검증 결과 저장
    with open('data/occupation_mapping_validation.json', 'w', encoding='utf-8') as f:
        json.dump(validation_result, f, ensure_ascii=False, indent=2)
    
    # 4. CSV 형태로도 저장 (Excel에서 확인용)
    user_job_list, _ = create_user_input_mapping()
    df_jobs = pd.DataFrame(user_job_list)
    df_jobs.to_csv('data/occupation_job_list.csv', index=False, encoding='utf-8-sig')
    
    print("매핑 테이블 파일 저장 완료:")
    print("  - occupation_mapping_complete.json: 전체 매핑 정보")
    print("  - occupation_mapping_api.json: 서비스 API용 데이터")
    print("  - occupation_mapping_validation.json: 검증 결과")
    print("  - occupation_job_list.csv: 직업 리스트 (Excel용)")
    
    return api_data, validation_result

# 실행
if __name__ == "__main__":
    print("3단계 매핑 테이블 생성 시작...")
    
    # 역매핑 테이블 생성
    reverse_mapping = create_reverse_mapping_tables()
    print(f"역매핑 테이블 생성 완료:")
    print(f"  - 산업코드 매핑: {len(reverse_mapping['industry_to_group'])}개")
    print(f"  - 직업코드 매핑: {len(reverse_mapping['job_to_group'])}개")
    
    # 사용자 입력 매핑 생성
    user_job_list, search_keywords = create_user_input_mapping()
    print(f"\n사용자 입력 매핑 생성 완료:")
    print(f"  - 전체 직업: {len(user_job_list)}개")
    print(f"  - 검색 키워드: {len(search_keywords)}개")
    
    # 샘플 직업 출력
    print(f"\n샘플 직업 리스트:")
    for i, job in enumerate(user_job_list[:10], 1):
        print(f"  {i:2d}. {job['job_name']} ({job['group']} > {job['subcategory']})")
    
    # 매핑 테이블 검증
    validation_result = validate_mapping_coverage()
    
    # 파일 저장
    api_data, validation_result = save_mapping_tables()
    
    print("\n=== 3단계 완료 ===")
    print(f"최종 커버리지: {validation_result['coverage_rate']*100:.1f}%")
    print("다음 단계: 통합 직업 분류 코드 구현")