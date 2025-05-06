import pandas as pd

# 파일 경로 설정
all_data_path = 'log_all.csv'
critical_data_path = 'log_all_critical.csv'
output_path = 'log_all_filtered.csv'

def remove_duplicate_by_text_column():
    print(f"{all_data_path} 파일 읽는 중...")
    all_data = pd.read_csv(all_data_path)
    print(f"원본 데이터 크기: {len(all_data)} 행")
    
    print(f"{critical_data_path} 파일 읽는 중...")
    critical_data = pd.read_csv(critical_data_path)
    print(f"제거할 데이터 크기: {len(critical_data)} 행")
    
    # text 컬럼이 존재하는지 확인
    if 'text' not in all_data.columns or 'text' not in critical_data.columns:
        print("오류: 두 CSV 파일 모두 'text' 컬럼이 있어야 합니다.")
        return
    
    print("'text' 컬럼 기준으로 중복 데이터 제거 중...")
    
    # critical_data의 text 컬럼 값들을 set으로 변환 (빠른 검색을 위해)
    critical_text_set = set(critical_data['text'].values)
    
    # all_data에서 text 값이 critical_text_set에 없는 행만 필터링
    filtered_data = all_data[~all_data['text'].isin(critical_text_set)]
    
    print(f"제거 후 데이터 크기: {len(filtered_data)} 행")
    print(f"총 {len(all_data) - len(filtered_data)} 개의 중복 행 제거됨")
    
    # 결과 저장
    filtered_data.to_csv(output_path, index=False)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    remove_duplicate_by_text_column()