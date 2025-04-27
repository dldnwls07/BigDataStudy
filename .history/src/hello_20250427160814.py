"""
이 파일은 이제 main.py를 import하여 실행합니다.
모든 코드는 기능별로 모듈화되었습니다.

- data_processing.py: 데이터 로딩, 전처리, 인코딩 기능
- visualization.py: 데이터 시각화 관련 기능 
- main.py: 전체 워크플로우 관리
"""

import os
from main import main

# 출력 경로 설정
OUTPUT_DIR = "C:\Users\feca1\Desktop\공부(빅분)\output\picture"

if __name__ == "__main__":
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"학생 성적 데이터 분석 프로그램 시작...")
    print(f"시각화 결과는 다음 경로에 저장됩니다: {OUTPUT_DIR}")
    
    # main 함수 호출 시 출력 경로 전달
    main(output_dir=OUTPUT_DIR)
    
    print("프로그램 종료.")