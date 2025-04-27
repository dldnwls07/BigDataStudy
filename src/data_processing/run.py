import os
import pandas as pd
from main import main
from preprocess_important_factors import preprocess_important_factors
from remove_outliers import remove_outliers
from split_data import split_data

if __name__ == "__main__":
    # 기본 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 입출력 파일 경로 설정
    original_file = os.path.join(base_dir, "data", "original_data", "StudentPerformanceFactors.csv")
    no_outliers_file = os.path.join(base_dir, "data", "processed_data", "StudentPerformanceFactors_no_outliers.csv")
    important_factors_file = os.path.join(base_dir, "data", "processed_data", "ImportantFactors.csv")
    train_file = os.path.join(base_dir, "data", "processed_data", "TrainFactors.csv")
    test_file = os.path.join(base_dir, "data", "processed_data", "TestFactors.csv")

    try:
        # 1. 이상치 제거
        print("이상치 제거 중...")
        data = pd.read_csv(original_file)
        cleaned_data = remove_outliers(data, 'Exam_Score', method='iqr')
        cleaned_data.to_csv(no_outliers_file, index=False)
        
        # 2. 중요 요인 추출
        print("중요 요인 추출 중...")
        preprocess_important_factors(no_outliers_file, important_factors_file)
        
        # 3. 학습/테스트 데이터 분할
        print("데이터 분할 중...")
        split_data(important_factors_file, train_file, test_file)
        
        # 4. 메인 처리
        print("메인 처리 시작...")
        output_dir = os.path.join(base_dir, "output", "picture", "picture_High_impact")
        main(output_dir)
        
        print("모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"오류 발생: {e}")