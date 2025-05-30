import pandas as pd

def load_and_preprocess_data(file_path):
    # 데이터 로드
    data = pd.read_csv(file_path, encoding='utf-8-sig')

    # 결측값 처리
    data.fillna('Unknown', inplace=True)

    # 컬럼명 변경
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
    data.rename(columns=column_mapping, inplace=True)

    return data