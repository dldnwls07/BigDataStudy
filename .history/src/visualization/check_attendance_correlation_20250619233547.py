import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 절대 경로 사용
data_path = r"c:\Users\feca1\Desktop\BigDataStudy\data\original_data\StudentPerformanceFactors.csv"
output_dir = r"c:\Users\feca1\Desktop\BigDataStudy\output\picture\presentation"

# 출력 폴더가 존재하는지 확인
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
df = pd.read_csv(data_path)

# 수치형 변수만 선택
numeric_df = df.select_dtypes(include=['number'])

# 상관관계 계산
correlation_matrix = numeric_df.corr()

# 시험 점수와의 상관관계를 내림차순으로 정렬
exam_score_correlations = correlation_matrix['Exam_Score'].sort_values(ascending=False)

print("시험 점수와의 상관관계 (내림차순):")
print(exam_score_correlations)

# 출석과 시험 점수의 산점도 그리기
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Attendance', y='Exam_Score', data=df)
plt.title('출석과 시험 점수의 관계')
plt.grid(True)
plt.savefig('../../output/picture/presentation/attendance_exam_score_scatter.png')

# 상관계수 히트맵 그리기
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('변수 간 상관관계 히트맵')
plt.tight_layout()
plt.savefig('../../output/picture/presentation/correlation_heatmap_detailed.png')

print("분석 완료 - 이미지가 output/picture/presentation 폴더에 저장되었습니다.")
