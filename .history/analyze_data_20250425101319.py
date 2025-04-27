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
