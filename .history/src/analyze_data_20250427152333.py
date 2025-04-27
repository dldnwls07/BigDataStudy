import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_preprocess_data

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

# 예시: 시험 점수 분포 시각화
save_plot(plt.hist, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\시험점수_분포.png', data['시험 점수'], bins=20, color='blue', alpha=0.7)

# 추가 시각화 코드 작성 가능