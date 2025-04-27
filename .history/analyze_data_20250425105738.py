import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 한글 폰트 설정 (Windows의 경우)
mpl.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# CSV 파일 로드
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\StudentPerformanceFactors.csv"
data = pd.read_csv(file_path)

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

# 원본 데이터 백업
data_original = data.copy()

# 데이터 컬럼명 변경 (한 번만 실행)
data = data.rename(columns=column_mapping)

# 데이터의 기본 정보와 첫 몇 줄 출력
print("데이터 기본 정보:")
print(data.info())
print("\n데이터 첫 5줄:")
print(data.head())

# 통계 요약
print("\n통계 요약:")
print(data.describe())

# 결측값 확인
print("\n결측값 확인:")
print(data.isnull().sum())

# 문자열 변수를 숫자로 매핑하는 딕셔너리 정의
motivation_map = {'Low': 0, 'Medium': 1, 'High': 2}
parental_inv_map = {'Low': 0, 'Medium': 1, 'High': 2}
resources_map = {'Low': 0, 'Medium': 1, 'High': 2}
learning_dis_map = {'Yes': 1, 'No': 0}
internet_access_map = {'Yes': 1, 'No': 0}
extracurricular_map = {'Yes': 1, 'No': 0}
income_map = {'Low': 0, 'Medium': 1, 'High': 2}
teacher_map = {'Low': 0, 'Medium': 1, 'High': 2}
school_map = {'Public': 0, 'Private': 1}
peer_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
edu_level_map = {'High School': 0, 'College': 1, 'Postgraduate': 2}
distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
gender_map = {'Male': 0, 'Female': 1}

# 데이터 전처리: 범주형 변수를 숫자로 변환
data_encoded = data.copy()
data_encoded['학습의욕 수준'] = data['학습의욕 수준'].map(motivation_map)
data_encoded['부모의 참여도'] = data['부모의 참여도'].map(parental_inv_map)
data_encoded['학습 자원에 대한 접근성'] = data['학습 자원에 대한 접근성'].map(resources_map)
data_encoded['학습 장애 여부'] = data['학습 장애 여부'].map(learning_dis_map)
data_encoded['인터넷 접근 여부'] = data['인터넷 접근 여부'].map(internet_access_map)
data_encoded['방과 후 활동 참여도'] = data['방과 후 활동 참여도'].map(extracurricular_map)
data_encoded['가정 소득'] = data['가정 소득'].map(income_map)
data_encoded['교사의 수준'] = data['교사의 수준'].map(teacher_map)
data_encoded['학교 종류'] = data['학교 종류'].map(school_map)
data_encoded['또래의 영향'] = data['또래의 영향'].map(peer_map)
data_encoded['부모의 교육 수준'] = data['부모의 교육 수준'].map(edu_level_map)
data_encoded['집에서 학교까지의 거리'] = data['집에서 학교까지의 거리'].map(distance_map)
data_encoded['성별'] = data['성별'].map(gender_map)

# 상관관계 분석 수정: 인코딩된 데이터 사용
print("\n수정된 상관관계 분석 (범주형 데이터 인코딩 후):")
print(data_encoded.corr())

# 특정 열의 분포 시각화 (예: 시험 점수)
plt.hist(data['시험 점수'], bins=20, color='blue', alpha=0.7)
plt.title('시험 점수 분포')
plt.xlabel('시험 점수')
plt.ylabel('빈도')
plt.show()

# 공부한 시간과 시험 점수 간의 관계 분석
plt.scatter(data['공부한 시간'], data['시험 점수'], alpha=0.5)
plt.title('공부한 시간과 시험 점수의 관계')
plt.xlabel('공부한 시간')
plt.ylabel('시험 점수')
plt.show()

# 학생의 성적과 스트레스 간의 상관관계 분석
if '스트레스 수준' in data.columns:
    print("\n성적과 스트레스 상관관계:")
    print(data[['시험 점수', '스트레스 수준']].corr())
else:
    print("\n'스트레스 수준' 열이 데이터에 없습니다.")

# 시험 성적과 스트레스 간접 요인 상관 분석 (인코딩된 데이터 사용)
stress_factors = ['수면 시간', '공부한 시간', '학습의욕 수준', '학습 장애 여부']
print("\n시험 성적과 스트레스 간접 요인 상관 분석:")
print(data_encoded[stress_factors + ['시험 점수']].corr())

# 그룹별 비교 (인코딩 전 원본 데이터 사용)
# 동기 수준이 낮을수록 성적이 낮은지 확인
low_motivation = data[data['학습의욕 수준'] == 'Low']
medium_motivation = data[data['학습의욕 수준'] == 'Medium']
high_motivation = data[data['학습의욕 수준'] == 'High']

print("\n동기 수준별 평균 성적:")
print("낮은 동기 그룹 평균 성적:", low_motivation['시험 점수'].mean())
print("중간 동기 그룹 평균 성적:", medium_motivation['시험 점수'].mean())
print("높은 동기 그룹 평균 성적:", high_motivation['시험 점수'].mean())

# 부모 참여도에 따른 성적 비교
low_parental = data[data['부모의 참여도'] == 'Low']
medium_parental = data[data['부모의 참여도'] == 'Medium']
high_parental = data[data['부모의 참여도'] == 'High']

print("\n부모 참여도별 평균 성적:")
print("낮은 부모 참여 그룹 평균 성적:", low_parental['시험 점수'].mean())
print("중간 부모 참여 그룹 평균 성적:", medium_parental['시험 점수'].mean())
print("높은 부모 참여 그룹 평균 성적:", high_parental['시험 점수'].mean())

# 학습의욕 수준별 시험 점수 박스플롯
plt.figure(figsize=(10, 6))
sns.boxplot(x='학습의욕 수준', y='시험 점수', data=data, order=['Low', 'Medium', 'High'])
plt.title('학습의욕 수준별 시험 점수')
plt.xlabel('학습의욕 수준')
plt.ylabel('시험 점수')
plt.savefig('학습의욕_수준별_시험점수_박스플롯.png')
plt.show()

# 수면 시간과 시험 점수 산점도 (색상을 학습의욕 수준으로 구분)
plt.figure(figsize=(10, 6))
for level in ['Low', 'Medium', 'High']:
    subset = data[data['학습의욕 수준'] == level]
    plt.scatter(subset['수면 시간'], subset['시험 점수'], 
                alpha=0.7, label=f'학습의욕 수준: {level}')
plt.title('수면 시간과 시험 점수의 관계 (학습의욕 수준별)')
plt.xlabel('수면 시간')
plt.ylabel('시험 점수')
plt.legend()
plt.savefig('수면시간_시험점수_산점도.png')
plt.show()

# 상관 행렬 히트맵 (인코딩된 데이터 사용)
plt.figure(figsize=(14, 12))
corr_matrix = data_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('상관 관계 히트맵')
plt.tight_layout()
plt.savefig('상관관계_히트맵.png')
plt.show()

# 공부한 시간과 시험 점수 간의 산점도 (추가 시각화)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='공부한 시간', y='시험 점수', hue='학습의욕 수준', 
                data=data, palette='viridis', s=100, alpha=0.7)
plt.title('공부한 시간과 시험 점수의 관계 (학습의욕 수준별)')
plt.xlabel('공부한 시간')
plt.ylabel('시험 점수')
plt.savefig('공부시간_시험점수_학습의욕수준_산점도.png')
plt.show()

# 변수와 시험 점수 간의 관계 시각화 함수
def visualize_relationship(data, columns, target_column='시험 점수'):
    for column in columns:
        plt.figure(figsize=(8, 6))
        if data[column].dtype == 'object':  # 범주형 데이터
            data.groupby(column)[target_column].mean().plot(kind='bar', color='skyblue')
            plt.title(f'{column}에 따른 {target_column} 평균')
            plt.ylabel(f'{target_column} 평균')
            plt.xlabel(column)
        else:  # 연속형 데이터
            plt.scatter(data[column], data[target_column], alpha=0.5, color='skyblue')
            plt.title(f'{column}와 {target_column}의 관계')
            plt.ylabel(target_column)
            plt.xlabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 변수와 시험 점수 간의 관계 시각화 실행
analysis_columns = list(column_mapping.values())[:-1]  # '시험 점수' 제외
visualize_relationship(data, analysis_columns)

# AI 기반 진로 유형 분류 모델 생성 함수
def train_career_model(data, feature_columns, target_column='진로 유형'):
    if target_column in data.columns:
        # 데이터 전처리
        features = data[feature_columns]
        labels = data[target_column]
        features = pd.get_dummies(features, drop_first=True)  # 범주형 데이터 인코딩

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # 모델 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_test)
        print("\n진로 유형 분류 모델 평가:")
        print(classification_report(y_test, y_pred))
    else:
        print(f"\n'{target_column}' 열이 데이터에 없습니다.")

# 진로 유형 분류 모델 학습 실행
train_career_model(data, ['공부한 시간', '학습의욕 수준', '이전 시험 성적'])
