import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    # 데이터 로드
    data = pd.read_csv(file_path, encoding='utf-8-sig')

    # 결측값 처리
    data['Distance_from_Home'].fillna('Unknown', inplace=True)  # 범주형 결측값 대체
    data['Exam_Score'].fillna(data['Exam_Score'].mean(), inplace=True)  # 연속형 결측값 평균으로 대체

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

    # 범주형 데이터 인코딩
    encoding_maps = {
        '부모의 참여도': {'Low': 0, 'Medium': 1, 'High': 2},
        '학습 자원에 대한 접근성': {'Low': 0, 'Medium': 1, 'High': 2},
        '학습의욕 수준': {'Low': 0, 'Medium': 1, 'High': 2},
        '인터넷 접근 여부': {'No': 0, 'Yes': 1},
        '방과 후 활동 참여도': {'No': 0, 'Yes': 1},
        '가정 소득': {'Low': 0, 'Medium': 1, 'High': 2},
        '교사의 수준': {'Low': 0, 'Medium': 1, 'High': 2},
        '학교 종류': {'Public': 0, 'Private': 1},
        '또래의 영향': {'Negative': -1, 'Neutral': 0, 'Positive': 1},
        '집에서 학교까지의 거리': {'Near': 0, 'Moderate': 1, 'Far': 2}
    }
    for column, mapping in encoding_maps.items():
        data[column] = data[column].map(mapping)

    # 데이터 정규화
    scaler = MinMaxScaler()
    continuous_columns = ['공부한 시간', '수면 시간', '이전 시험 성적', '학교 출석', '시험 점수']
    data[continuous_columns] = scaler.fit_transform(data[continuous_columns])

    return data