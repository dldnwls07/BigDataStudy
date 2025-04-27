import os
from data_processing import setup_font, load_data, preprocess_data, encode_categorical_data, get_data_info
from visualization import create_visualization_sets

def main(output_dir=None):
    # BASE_DIR과 DATA_DIR 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # BigDataStudy 폴더
    input_file = os.path.join(BASE_DIR, "data", "original_data", "StudentPerformanceFactors.csv")
    
    # 한글 폰트 설정
    setup_font()
    
    print("데이터 로드 중...")
    data = load_data(input_file)
    
    print("데이터 전처리 중...")
    data = preprocess_data(data)
 
    print("범주형 데이터 인코딩 중...")
    data_encoded = encode_categorical_data(data)

    print("\n데이터 기본 정보:")
    info = get_data_info(data)
    
    # 상관관계 분석 출력
    print("\n수정된 상관관계 분석 (범주형 데이터 인코딩 후):")
    print(data_encoded.corr())
    
    # 시각화 파일 저장 경로
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 생성
    print(f"\n시각화 생성 및 저장 중... 저장 경로: {output_dir}")
    create_visualization_sets(data, data_encoded, output_dir)
    
    print("\n모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
