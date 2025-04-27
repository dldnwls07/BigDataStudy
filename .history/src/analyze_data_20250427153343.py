import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data

# 한글 폰트 설정 (Windows의 경우)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 로드 및 전처리
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\data\\raw\\StudentPerformanceFactors.csv"
data = load_and_preprocess_data(file_path)

# 시각화 함수
def save_plot(plot_func, filename, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    plot_func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. 시험 점수 분포 시각화
save_plot(sns.histplot, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\시험점수_분포_개선.png', data['시험 점수'], kde=True, bins=20, color='blue')

# 2. 학교 출석과 시험 점수의 관계
save_plot(sns.boxplot, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\학교출석_시험점수_개선.png', x='학교 출석', y='시험 점수', data=data, palette='coolwarm')

# 3. 공부한 시간과 시험 점수의 관계 (산점도)
save_plot(sns.scatterplot, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\공부시간_시험점수_개선.png', x='공부한 시간', y='시험 점수', hue='학습의욕 수준', data=data, palette='viridis', s=100, alpha=0.7)

# 4. 상관 행렬 히트맵
corr = data.corr()
save_plot(sns.heatmap, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\상관관계_히트맵_개선.png', corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)