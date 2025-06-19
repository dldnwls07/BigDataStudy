# -*- coding: utf-8 -*-
# 모델 테스트

import os
import sys
import io

# 표준 출력 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 필요한 클래스만 직접 재정의하여 사용
class TestStudentDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # 범주형 변수 처리
        categorical_cols = ['Parental_Involvement', 'Motivation_Level', 'Access_to_Resources', 'Learning_Disabilities']
        
        # 범주형 데이터 처리
        for col in categorical_cols:
            if col in self.data.columns:
                # 범주형 변수 처리
                if col == 'Learning_Disabilities':
                    self.data[col] = self.data[col].map({'Yes': 1, 'No': 0})
                else:
                    self.data[col] = self.data[col].map({'High': 2, 'Medium': 1, 'Low': 0})
        
        # 수치형 데이터 결측치 처리
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # 타겟 변수 분리
        self.X = self.data.drop('Exam_Score', axis=1)
        self.y = self.data['Exam_Score']
        
        # 특성 스케일링
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
            
        # 데이터 타입 변환
        self.X_scaled = self.X_scaled.astype('float32')
        self.y = self.y.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.X_scaled[idx], dtype=torch.float32)
        label = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        return features, label

# 모델 정의
class ImprovedStudentPerformanceModel(torch.nn.Module):
    def __init__(self, input_size):
        super(ImprovedStudentPerformanceModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def plot_predictions_vs_actuals(actuals, predictions, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.6, color='blue', label='Predictions')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--', label='Ideal Fit')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)

def test_model(test_csv, model_path):
    try:
        # 데이터 로드
        test_dataset = TestStudentDataset(test_csv)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 모델 로드
        input_size = test_dataset.X_scaled.shape[1]
        model = ImprovedStudentPerformanceModel(input_size)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        model.load_state_dict(torch.load(model_path, weights_only=True))

        # GPU 사용 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # 예측 및 성능 평가
        predictions = []
        actuals = []
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(labels.cpu().numpy())

        # 다양한 평가지표 계산
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions)
        print(f"테스트 데이터 평균 제곱 오차(MSE): {mse:.4f}")
        print(f"테스트 데이터 R² 점수: {r2:.4f}")
        print(f"테스트 데이터 평균 절대 오차(MAE): {mae:.4f}")
        print(f"테스트 데이터 평균 절대 백분율 오차(MAPE): {mape:.2f}%")

        # 예측값과 실제값 비교 시각화
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(output_dir, "picture", "train_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "predictions_vs_actuals.png")
        
        plot_predictions_vs_actuals(actuals, predictions, output_path)
        print(f"예측값과 실제값 비교 그래프가 저장되었습니다: {output_path}")
        
    except Exception as e:
        print(f"모델 테스트 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 기본 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    test_csv = os.path.join(base_dir, "data", "processed_data", "TestFactors.csv")
    model_path = os.path.join(base_dir, "src", "model_training", "refactored", "model", "student_performance_model_improved.pth")
    
    test_model(test_csv, model_path)