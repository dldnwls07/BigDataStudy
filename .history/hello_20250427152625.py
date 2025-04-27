from src.preprocess import load_and_preprocess_data
from src.analyze_data import save_plot
from src.train_model import train_and_evaluate_model

# 데이터 로드 및 전처리
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\data\\raw\\StudentPerformanceFactors.csv"
data = load_and_preprocess_data(file_path)

# 데이터 분석 및 시각화
# 예시: 시험 점수 분포 시각화
save_plot(plt.hist, 'c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\plot\\시험점수_분포.png', data['시험 점수'], bins=20, color='blue', alpha=0.7)

# 모델 학습 및 평가
train_and_evaluate_model(
    data,
    target_column="진로 유형",
    feature_columns=["공부한 시간", "학습의욕 수준", "이전 시험 성적"],
    model_path="c:\\Users\\feca1\\Desktop\\공부(빅분)\\output\\model\\model.pkl"
)