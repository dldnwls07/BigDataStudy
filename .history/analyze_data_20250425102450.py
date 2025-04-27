import pandas as pd

# CSV 파일 로드
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\StudentPerformanceFactors.csv"
data = pd.read_csv(file_path)

# 데이터의 기본 정보와 첫 몇 줄 출력
print("데이터 기본 정보:")
print(data.info())
print("\n데이터 첫 5줄:")
print(data.head())

# 데이터의 기본 정보 출력
print("데이터 정보:")
print(data.info())

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
data_encoded['Motivation_Level'] = data['Motivation_Level'].map(motivation_map)
data_encoded['Parental_Involvement'] = data['Parental_Involvement'].map(parental_inv_map)
data_encoded['Access_to_Resources'] = data['Access_to_Resources'].map(resources_map)
data_encoded['Learning_Disabilities'] = data['Learning_Disabilities'].map(learning_dis_map)
data_encoded['Internet_Access'] = data['Internet_Access'].map(internet_access_map)
data_encoded['Extracurricular_Activities'] = data['Extracurricular_Activities'].map(extracurricular_map)
data_encoded['Family_Income'] = data['Family_Income'].map(income_map)
data_encoded['Teacher_Quality'] = data['Teacher_Quality'].map(teacher_map)
data_encoded['School_Type'] = data['School_Type'].map(school_map)
data_encoded['Peer_Influence'] = data['Peer_Influence'].map(peer_map)
data_encoded['Parental_Education_Level'] = data['Parental_Education_Level'].map(edu_level_map)
data_encoded['Distance_from_Home'] = data['Distance_from_Home'].map(distance_map)
data_encoded['Gender'] = data['Gender'].map(gender_map)

# 상관관계 분석 수정: 인코딩된 데이터 사용
print("\n수정된 상관관계 분석 (범주형 데이터 인코딩 후):")
print(data_encoded.corr())

# 특정 열의 분포 시각화 (예: Exam_Score)
import matplotlib.pyplot as plt
plt.hist(data['Exam_Score'], bins=20, color='blue', alpha=0.7)
plt.title('Exam Score Distribution')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.show()

# 수능 점수와 공부 시간 간의 관계 분석
plt.scatter(data['Hours_Studied'], data['Exam_Score'], alpha=0.5)
plt.title('Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.show()

# 학생의 성적과 스트레스 간의 상관관계 분석
if 'Stress_Level' in data.columns:
    print("\n성적과 스트레스 상관관계:")
    print(data[['Exam_Score', 'Stress_Level']].corr())
else:
    print("\n'Stress_Level' 열이 데이터에 없습니다.")

# 시험 성적과 스트레스 간접 요인 상관 분석 (인코딩된 데이터 사용)
stress_factors = ['Sleep_Hours', 'Hours_Studied', 'Motivation_Level', 'Learning_Disabilities']
print("\n시험 성적과 스트레스 간접 요인 상관 분석:")
print(data_encoded[stress_factors + ['Exam_Score']].corr())

# 그룹별 비교 (인코딩 전 원본 데이터 사용)
# Motivation_Level이 낮을수록 성적이 낮은지 확인
low_motivation = data[data['Motivation_Level'] == 'Low']
medium_motivation = data[data['Motivation_Level'] == 'Medium']
high_motivation = data[data['Motivation_Level'] == 'High']

print("\n동기 수준별 평균 성적:")
print("낮은 동기 그룹 평균 성적:", low_motivation['Exam_Score'].mean())
print("중간 동기 그룹 평균 성적:", medium_motivation['Exam_Score'].mean())
print("높은 동기 그룹 평균 성적:", high_motivation['Exam_Score'].mean())

# 부모 참여도에 따른 성적 비교
low_parental = data[data['Parental_Involvement'] == 'Low']
medium_parental = data[data['Parental_Involvement'] == 'Medium']
high_parental = data[data['Parental_Involvement'] == 'High']

print("\n부모 참여도별 평균 성적:")
print("낮은 부모 참여 그룹 평균 성적:", low_parental['Exam_Score'].mean())
print("중간 부모 참여 그룹 평균 성적:", medium_parental['Exam_Score'].mean())
print("높은 부모 참여 그룹 평균 성적:", high_parental['Exam_Score'].mean())

# 시각화 코드 수정
import seaborn as sns

# Motivation_Level별 Exam_Score 박스플롯
plt.figure(figsize=(10, 6))
sns.boxplot(x='Motivation_Level', y='Exam_Score', data=data, order=['Low', 'Medium', 'High'])
plt.title('Exam Score by Motivation Level')
plt.xlabel('Motivation Level')
plt.ylabel('Exam Score')
plt.savefig('motivation_boxplot.png')
plt.show()

# Sleep_Hours와 Exam_Score 산점도 (색상을 Motivation_Level로 구분)
plt.figure(figsize=(10, 6))
for level in ['Low', 'Medium', 'High']:
    subset = data[data['Motivation_Level'] == level]
    plt.scatter(subset['Sleep_Hours'], subset['Exam_Score'], 
                alpha=0.7, label=f'Motivation: {level}')
plt.title('Sleep Hours vs Exam Score by Motivation Level')
plt.xlabel('Sleep Hours')
plt.ylabel('Exam Score')
plt.legend()
plt.savefig('sleep_vs_exam_scatter.png')
plt.show()

# 상관 행렬 히트맵 (인코딩된 데이터 사용)
plt.figure(figsize=(14, 12))
corr_matrix = data_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Hours_Studied와 Exam_Score 간의 산점도 (추가 시각화)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', hue='Motivation_Level', 
                data=data, palette='viridis', s=100, alpha=0.7)
plt.title('Hours Studied vs Exam Score by Motivation Level')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.savefig('hours_vs_exam_scatter.png')
plt.show()

# AI 기반 진로 유형 분류 모델 생성
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if 'Career_Type' in data.columns:
    # 데이터 전처리
    features = data[['Hours_Studied', 'Motivation_Level', 'Previous_Scores']]  # 주요 특징 선택
    labels = data['Career_Type'] 
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
    print("\n'Career_Type' 열이 데이터에 없습니다.")
