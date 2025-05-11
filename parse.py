import os
import glob

# 특정 폴더 경로 (실제 경로로 수정해주세요)
folder_path = '03_results_osr_multi_oe_experiments'

# 결과를 저장할 파일 경로
output_path = os.path.join(folder_path, 'combined.txt')

# 찾을 문자열
target_string = "--- Overall Metrics ---"

# 폴더 내 모든 txt 파일 찾기
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))

# 파일 내용 모으기
combined_content = []

# 각 파일 처리
for file_path in txt_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # "--- Overall Metrics ---" 문자열이 있는지 확인
            if target_string in content:
                # 파일 이름 가져오기
                file_name = os.path.basename(file_path)
                
                # 해당 문자열 위치 찾기
                start_index = content.find(target_string)
                
                # 그 부분부터 파일 끝까지 추출
                extracted_content = content[start_index:]
                
                # 파일 이름을 헤더로 추가하고 내용 합치기
                file_content = f"\n===== {file_name} =====\n\n{extracted_content}\n\n"
                combined_content.append(file_content)
    except Exception as e:
        print(f"파일 {file_path} 처리 중 오류 발생: {e}")

# 결과 파일에 저장
if combined_content:
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(''.join(combined_content).strip())
    print(f"{output_path}에 파일이 성공적으로 생성되었습니다.")
    print(f"총 {len(combined_content)}개의 파일이 합쳐졌습니다.")
else:
    print("조건에 맞는 파일을 찾지 못했습니다.")