import pandas as pd

def remove_unknown_class(input_file, output_file):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_file)
        
        # 원본 데이터 정보 출력
        print(f"원본 데이터 행 수: {len(df)}")
        print(f"클래스 분포:\n{df['class'].value_counts()}")
        
        # 'unknown' 클래스 행 필터링
        df_filtered = df[df['class'] != 'unknown']
        
        # 필터링된 데이터 정보 출력
        print(f"\n필터링 후 데이터 행 수: {len(df_filtered)}")
        print(f"필터링 후 클래스 분포:\n{df_filtered['class'].value_counts()}")
        print(f"제거된 행 수: {len(df) - len(df_filtered)}")
        
        # 결과 저장
        df_filtered.to_csv(output_file, index=False)
        print(f"\n필터링된 데이터가 {output_file}에 성공적으로 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    input_file = "masked_for_oe_without_unknown.csv"
    output_file = "masked_for_oe_without_unknown2.csv"
    
    remove_unknown_class(input_file, output_file)