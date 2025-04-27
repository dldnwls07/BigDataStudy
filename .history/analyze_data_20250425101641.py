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

# 상관관계 분석
print("\n상관관계 분석:")
print(data.corr())

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

# 시험 성적과 스트레스 간접 요인 상관 분석
stress_factors = ['Sleep_Hours', 'Hours_Studied', 'Motivation_Level', 'Learning_Disabilities']
if all(factor in data.columns for factor in stress_factors):
    print("\n시험 성적과 스트레스 간접 요인 상관 분석:")
    print(data[stress_factors + ['Exam_Score']].corr())
else:
    print("\n일부 스트레스 요인 열이 데이터에 없습니다.")

# 그룹별 비교
if 'Motivation_Level' in data.columns and 'Parental_Involvement' in data.columns:
    low_motivation = data[data['Motivation_Level'] < data['Motivation_Level'].median()]
    high_motivation = data[data['Motivation_Level'] >= data['Motivation_Level'].median()]

    print("\n낮은 동기 그룹 평균 성적:", low_motivation['Exam_Score'].mean())
    print("높은 동기 그룹 평균 성적:", high_motivation['Exam_Score'].mean())

    low_parental = data[data['Parental_Involvement'] < data['Parental_Involvement'].median()]
    high_parental = data[data['Parental_Involvement'] >= data['Parental_Involvement'].median()]

    print("\n낮은 부모 참여 그룹 평균 성적:", low_parental['Exam_Score'].mean())
    print("높은 부모 참여 그룹 평균 성적:", high_parental['Exam_Score'].mean())
else:
    print("\n'Motivation_Level' 또는 'Parental_Involvement' 열이 데이터에 없습니다.")

# 시각화
import seaborn as sns

# Motivation_Level별 Exam_Score 박스플롯
if 'Motivation_Level' in data.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Motivation_Level', y='Exam_Score', data=data)
    plt.title('Exam Score by Motivation Level')
    plt.xlabel('Motivation Level')
    plt.ylabel('Exam Score')
    plt.show()

# Sleep_Hours와 Exam_Score 산점도
if 'Sleep_Hours' in data.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Sleep_Hours'], data['Exam_Score'], alpha=0.5)
    plt.title('Sleep Hours vs Exam Score')
    plt.xlabel('Sleep Hours')
    plt.ylabel('Exam Score')
    plt.show()

# 상관 행렬 히트맵
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
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
