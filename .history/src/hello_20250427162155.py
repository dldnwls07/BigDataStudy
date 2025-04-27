"""
- data_processing.py: 데이터 로딩, 전처리, 인코딩 기능
- visualization.py: 데이터 시각화 관련 기능 
- main.py: 전체 워크플로우 관리
"""

import os
import sys
from main import main
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
OUTPUT_DIR = os.path.join(parent_dir, "output", "picture")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"학생 성적 데이터 분석 프로그램 시작...")
    print(f"시각화 결과는 다음 경로에 저장됩니다: {OUTPUT_DIR}")
  
    main(output_dir=OUTPUT_DIR)
    
    print("프로그램 종료.")