#모델 학습습

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os

# 전역 스케일러 객체 정의
global_scaler = MinMaxScaler()
is_scaler_fitted = False

# 디렉토리 생성 함수
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 시각화 결과 저장 경로
vis_dir = "output/picture/train_results"  # 시각화 결과 저장 경로
ensure_dir(vis_dir)  # 디렉토리가 없으면 생성

# 향상된 데이터셋 정의
class ImprovedStudentDataset(Dataset):
    def __init__(self, csv_file, is_train=True):
        global global_scaler, is_scaler_fitted
        
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
        
        # 범주형 변수 처리 개선
        categorical_maps = {
            'Parental_Involvement': {'High': 2, 'Medium': 1, 'Low': 0},
            'Motivation_Level': {'High': 2, 'Medium': 1, 'Low': 0},
            'Access_to_Resources': {'High': 2, 'Medium': 1, 'Low': 0},
            'Learning_Disabilities': {'Yes': 1, 'No': 0}
        }

        for col, value_map in categorical_maps.items():
            if col in self.data.columns:
                # 누락된 값을 가장 빈번한 값으로 대체
                mode_value = self.data[col].mode()[0]
                self.data[col] = self.data[col].fillna(mode_value)

                # 매핑 적용 전 유효성 검사
                invalid_values = set(self.data[col].unique()) - set(value_map.keys())
                if invalid_values:
                    print(f"Warning: Found invalid values in {col}: {invalid_values}")
                    # 잘못된 값을 가장 빈번한 값으로 대체
                    self.data.loc[self.data[col].isin(invalid_values), col] = mode_value

                # 매핑 적용
                self.data[col] = self.data[col].map(value_map)
        
        # 수치형 데이터 결측치 처리
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # 타겟 변수 분리
        self.X = self.data.drop('Exam_Score', axis=1)
        self.y = self.data['Exam_Score']
        
        # 특성 스케일링
        if is_train:
            # 학습 데이터인 경우 스케일러 학습
            self.X_scaled = global_scaler.fit_transform(self.X)
            is_scaler_fitted = True
        else:
            # 테스트 데이터인 경우 학습된 스케일러 사용
            self.X_scaled = global_scaler.transform(self.X)
            
        # 데이터 타입 변환
        self.X_scaled = self.X_scaled.astype('float32')
        self.y = self.y.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.X_scaled[idx], dtype=torch.float32)
        label = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        return features, label

# 향상된 모델 정의
class ImprovedStudentPerformanceModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedStudentPerformanceModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# 향상된 학습 및 평가 함수
def improved_train_and_evaluate(train_csv, test_csv, epochs=50, batch_size=16, learning_rate=0.0005):
    # 데이터 로드
    train_dataset = ImprovedStudentDataset(train_csv, is_train=True)
    test_dataset = ImprovedStudentDataset(test_csv, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    input_size = train_dataset.X_scaled.shape[1]
    model = ImprovedStudentPerformanceModel(input_size)
    
    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Early Stopping 설정
    early_stopping_patience = 15
    min_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        avg_train_loss = running_loss/len(train_loader)
        avg_val_loss = val_loss/len(test_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 학습률 조정
        scheduler.step(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            patience_counter = 0
            # 최적 가중치를 불러와 테스트 데이터 평가 또는 예측에 사용
            torch.save(model.state_dict(), "src/model_training/refactored/model/best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 최적 모델 로드
    try:
        model.load_state_dict(torch.load("src/model_training/refactored/model/best_model.pth"))
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
    
    # 테스트
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    print(f"테스트 데이터 평균 제곱 오차(MSE): {mse:.4f}")
    print(f"테스트 데이터 R² 점수: {r2:.4f}")
    
    # 모델 저장 디렉토리 확인 및 생성
    model_dir = "src/model_training/refactored/model"
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_model.pth")
    final_model_path = os.path.join(model_dir, "student_performance_model_improved.pth")
    
    # 최종 모델 저장
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"개선된 모델이 저장되었습니다: {final_model_path}")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
    
    # 학습 및 검증 손실 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig(f"{vis_dir}/loss_curve.png")
    
    return model, r2

if __name__ == "__main__":
    train_csv = "data/processed_data/TrainFactors.csv"  
    test_csv = "data/processed_data/TestFactors.csv"  
    model, r2 = improved_train_and_evaluate(train_csv, test_csv)