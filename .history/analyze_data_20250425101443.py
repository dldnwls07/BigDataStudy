import pandas as pd

# CSV 파일 로드
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\StudentPerformanceFactors.csv"
data = pd.read_csv(file_path)

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
