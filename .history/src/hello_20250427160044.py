"""
이 파일은 이제 main.py를 import하여 실행합니다.
모든 코드는 기능별로 모듈화되었습니다.

- data_processing.py: 데이터 로딩, 전처리, 인코딩 기능 (데이터 경로 설정 포함)
- visualization.py: 데이터 시각화 관련 기능 (출력 디렉토리 지원)
- main.py: 전체 워크플로우 관리
"""

from main import main

if __name__ == "__main__":
    print("학생 성적 데이터 분석 프로그램 시작...")
    main()
    print("프로그램 종료.")