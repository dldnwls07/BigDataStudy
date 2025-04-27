import pandas as pd


def preprocess_important_factors(input_file, output_file):
    # CSV 파일 읽기
    data = pd.read_csv(input_file)

    # 영향을 많이 준다고 판단된 열만 선택 (영어 열 이름으로 수정)
    important_columns = [
        "Hours_Studied", "Exam_Score", "Parental_Involvement", "Motivation_Level", 
        "Previous_Scores", "Attendance", "Access_to_Resources", "Learning_Disabilities"
    ]

    # 선택된 열로 데이터 필터링
    filtered_data = data[important_columns]

    # 결측치 처리 (필요에 따라 수정 가능)
    filtered_data = filtered_data.dropna()

    # 전처리된 데이터를 새로운 CSV 파일로 저장
    filtered_data.to_csv(output_file, index=False)
    print(f"전처리된 데이터가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    input_file = "data/original_data/StudentPerformanceFactors.csv"  
    output_file = "data/processed_data/ImportantFactors.csv" 
    preprocess_important_factors(input_file, output_file)