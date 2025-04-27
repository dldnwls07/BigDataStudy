import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# 상대 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src의 상위 폴더 (프로젝트 루트)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_FILE_PATH = os.path.join(DATA_DIR, "StudentPerformanceFactors.csv")

# 필요한 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)

def setup_font():
    """한글 폰트 설정"""
    mpl.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def load_data(file_path=None):
    """CSV 파일 로드"""
    if file_path is None:
        file_path = DEFAULT_FILE_PATH
    
    print(f"파일을 불러오는 중: {file_path}")
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig')
        return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        raise
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        raise

def preprocess_data(data):
    """데이터 전처리: 결측값 처리 및 컬럼명 변경"""
    # 결측값 처리
    data.fillna('Unknown', inplace=True)
    
    # 영어 컬럼명을 한글로 변경하기 위한 딕셔너리
    column_mapping = {
        'Hours_Studied': '공부한 시간',
        'Attendance': '학교 출석',
        'Parental_Involvement': '부모의 참여도',
        'Access_to_Resources': '학습 자원에 대한 접근성',
        'Extracurricular_Activities': '방과 후 활동 참여도',
        'Sleep_Hours': '수면 시간',
        'Previous_Scores': '이전 시험 성적',
        'Motivation_Level': '학습의욕 수준',
        'Internet_Access': '인터넷 접근 여부',
        'Tutoring_Sessions': '개인 지도 세션 수',
        'Family_Income': '가정 소득',
        'Teacher_Quality': '교사의 수준',
        'School_Type': '학교 종류',
        'Peer_Influence': '또래의 영향',
        'Physical_Activity': '신체 활동 수준',
        'Learning_Disabilities': '학습 장애 여부',
        'Parental_Education_Level': '부모의 교육 수준',
        'Distance_from_Home': '집에서 학교까지의 거리',
        'Gender': '성별',
        'Exam_Score': '시험 점수'
    }
    
    # 데이터 컬럼명 변경
    data = data.rename(columns=column_mapping)
    return data

def encode_categorical_data(data):
    """범주형 변수를 숫자로 변환"""
    # 문자열 변수를 숫자로 매핑하는 딕셔너리 정의
    mapping_dicts = {
        '학습의욕 수준': {'Low': 0, 'Medium': 1, 'High': 2},
        '부모의 참여도': {'Low': 0, 'Medium': 1, 'High': 2},
        '학습 자원에 대한 접근성': {'Low': 0, 'Medium': 1, 'High': 2},
        '학습 장애 여부': {'Yes': 1, 'No': 0},
        '인터넷 접근 여부': {'Yes': 1, 'No': 0},
        '방과 후 활동 참여도': {'Yes': 1, 'No': 0},
        '가정 소득': {'Low': 0, 'Medium': 1, 'High': 2},
        '교사의 수준': {'Low': 0, 'Medium': 1, 'High': 2},
        '학교 종류': {'Public': 0, 'Private': 1},
        '또래의 영향': {'Negative': -1, 'Neutral': 0, 'Positive': 1},
        '부모의 교육 수준': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        '집에서 학교까지의 거리': {'Near': 0, 'Moderate': 1, 'Far': 2},
        '성별': {'Male': 0, 'Female': 1}
    }
    
    # 데이터 전처리: 범주형 변수를 숫자로 변환
    data_encoded = data.copy()
    
    for column, mapping in mapping_dicts.items():
        if column in data.columns:
            data_encoded[column] = data[column].map(mapping)
    
    return data_encoded

def get_data_info(data):
    """데이터 기본 정보 출력"""
    info = {
        "기본 정보": data.info(),
        "첫 5줄": data.head(),
        "통계 요약": data.describe(),
        "결측값": data.isnull().sum()
    }
    return info
