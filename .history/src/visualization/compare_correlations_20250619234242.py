import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 오류 출력이 보이도록 설정
sys.stderr.write = print

# 출력 디렉토리 생성
output_dir = r"c:\Users\feca1\Desktop\BigDataStudy\output\picture\presentation"
os.makedirs(output_dir, exist_ok=True)

# 데이터 불러오기
original_data_path = r"c:\Users\feca1\Desktop\BigDataStudy\data\original_data\StudentPerformanceFactors.csv"
train_data_path = r"c:\Users\feca1\Desktop\BigDataStudy\data\processed_data\TrainFactors.csv"

original_df = pd.read_csv(original_data_path)
train_df = pd.read_csv(train_data_path)

# 데이터 차이점 출력
print("원본 데이터 크기:", original_df.shape)
print("학습 데이터 크기:", train_df.shape)

# 두 데이터셋의 상관계수 비교
print("\n원본 데이터 상관계수:")
original_numeric = original_df.select_dtypes(include=['number'])
original_corr = original_numeric.corr()
original_exam_corr = original_corr['Exam_Score'].sort_values(ascending=False)
print(original_exam_corr[:10])  # 상위 10개 변수만 출력

print("\n학습 데이터 상관계수:")
train_numeric = train_df.select_dtypes(include=['number'])
# 학습 데이터에 Exam_Score 컬럼이 있는지 확인
if 'Exam_Score' in train_numeric.columns:
    train_corr = train_numeric.corr()
    train_exam_corr = train_corr['Exam_Score'].sort_values(ascending=False)
    print(train_exam_corr[:10])  # 상위 10개 변수만 출력
else:
    # 목표 변수 이름이 다를 수 있음
    print("학습 데이터의 열 목록:")
    print(train_numeric.columns.tolist())

# 산점도 비교 (출석 vs 시험 점수)
plt.figure(figsize=(15, 6))

# 원본 데이터 산점도
plt.subplot(1, 2, 1)
sns.scatterplot(x='Attendance', y='Exam_Score', data=original_df, alpha=0.5)
plt.title(f'원본 데이터 (상관계수: {original_corr["Exam_Score"]["Attendance"]:.2f})')
plt.grid(True)

# 학습 데이터 산점도 (학습 데이터에 해당 변수들이 있다면)
plt.subplot(1, 2, 2)
if 'Attendance' in train_df.columns and 'Exam_Score' in train_df.columns:
    sns.scatterplot(x='Attendance', y='Exam_Score', data=train_df, alpha=0.5)
    plt.title(f'학습 데이터 (상관계수: {train_corr["Exam_Score"]["Attendance"]:.2f})')
else:
    # 학습 데이터에 변수가 없을 경우 메시지 출력
    plt.text(0.5, 0.5, '학습 데이터에 해당 변수가 없음', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.title('학습 데이터 비교 불가')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_comparison.png'))
plt.close()

print("\n산점도 비교 이미지가 저장되었습니다:", os.path.join(output_dir, 'correlation_comparison.png'))
