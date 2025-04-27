#원본 데이터 전처리 데이터 비교

import pandas as pd

def compare_datasets(raw_path, processed_path):
    # 원본 데이터 로드
    raw_data = pd.read_csv(raw_path)
    processed_data = pd.read_csv(processed_path)

    # 컬럼 비교
    print("원본 데이터 컬럼:", raw_data.columns.tolist())
    print("전처리된 데이터 컬럼:", processed_data.columns.tolist())

    # 데이터 크기 비교
    print("원본 데이터 크기:", raw_data.shape)
    print("전처리된 데이터 크기:", processed_data.shape)

    # 샘플 데이터 비교
    print("원본 데이터 샘플:")
    print(raw_data.head())
    print("전처리된 데이터 샘플:")
    print(processed_data.head())

if __name__ == "__main__":
    raw_path = "data/original_data/StudentPerformanceFactors.csv"  
    processed_path = "data/processed_data/ImportantFactors.csv" 
    compare_datasets(raw_path, processed_path)