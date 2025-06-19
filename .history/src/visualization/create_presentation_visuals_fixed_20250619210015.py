import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os
import sys
import io

# 한글 출력을 위한 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 한글 폰트 설정
# Windows 기본 폰트인 맑은 고딕 사용
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 출력 디렉토리 생성
output_dir = "output/picture/presentation"
os.makedirs(output_dir, exist_ok=True)

# 데이터 불러오기
try:
    train_data = pd.read_csv("data/processed_data/TrainFactors.csv")
    test_data = pd.read_csv("data/processed_data/TestFactors.csv")
    print("데이터 로드 완료")
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")

# 데이터셋 기본 통계량 시각화
def create_dataset_summary(data, output_path):
    try:
        # 수치형 변수만 선택
        numeric_data = data.select_dtypes(include=['number'])
        
        # 기본 통계량 계산
        stats = numeric_data.describe().T
        stats['missing'] = numeric_data.isnull().sum()
        stats['dtype'] = numeric_data.dtypes
        
        # 그림 크기 설정
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        ax.axis('off')
        
        # 테이블 생성
        table_data = stats.reset_index()
        table_cols = ['index', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing', 'dtype']
        table = ax.table(cellText=table_data[table_cols].values, 
                         colLabels=table_cols, 
                         loc='center', 
                         cellLoc='center')
        
        # 테이블 스타일 설정
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        plt.title('데이터셋 기본 통계량', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"1. 데이터셋 기본 통계량 시각화 완료: {output_path}")
    except Exception as e:
        print(f"데이터셋 기본 통계량 시각화 중 오류 발생: {e}")

# 데이터 분포 시각화
def create_distribution_plots(data, output_path):
    try:
        # 수치형 변수 선택
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        # 변수 수에 따라 행과 열 계산
        n_cols = len(numeric_cols)
        n_rows = (n_cols // 3) + (1 if n_cols % 3 > 0 else 0)
        
        # 그래프 생성
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.histplot(data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'{col} 분포')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('빈도')
        
        # 남은 축 숨기기
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"2. 데이터 분포 시각화 완료: {output_path}")
    except Exception as e:
        print(f"데이터 분포 시각화 중 오류 발생: {e}")

# 데이터 처리 과정 시각화
def create_data_pipeline_visualization(output_path):
    try:
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        
        # 파이프라인 단계들
        steps = ['원본 데이터\nStudentPerformanceFactors.csv', 
                 '중요 요인 추출\nImportantFactors.csv', 
                 '이상치 제거\nImportantFactors_no_outliers.csv', 
                 '데이터 분할\nTrainFactors.csv\nTestFactors.csv']
        
        # 박스 위치
        positions = np.arange(len(steps)) * 3
        
        # 박스 그리기
        for i, (pos, step) in enumerate(zip(positions, steps)):
            rect = plt.Rectangle((pos, 0.5), 2, 1, facecolor='skyblue', alpha=0.8)
            ax.add_patch(rect)
            ax.text(pos + 1, 1, step, ha='center', va='center', fontsize=10)
            
            # 화살표 그리기
            if i < len(steps) - 1:
                ax.arrow(pos + 2.1, 1, 0.8, 0, head_width=0.1, head_length=0.1, 
                         fc='black', ec='black')
        
        ax.set_xlim(-0.5, max(positions) + 2.5)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('데이터 처리 파이프라인', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"3. 데이터 처리 과정 시각화 완료: {output_path}")
    except Exception as e:
        print(f"데이터 처리 과정 시각화 중 오류 발생: {e}")

# 변수 중요도 시각화
def create_feature_importance_visualization(data, output_path):
    try:
        # 범주형 변수 처리 (범주형 변수가 있을 경우에만 실행)
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # 데이터 복사본 생성
        data_encoded = data.copy()
        
        # 범주형 변수 인코딩
        for col in categorical_cols:
            if col in data.columns:
                # 범주형 변수 처리
                if col == 'Learning_Disabilities':
                    data_encoded[col] = data_encoded[col].map({'Yes': 1, 'No': 0})
                elif col in ['Parental_Involvement', 'Motivation_Level', 'Access_to_Resources']:
                    data_encoded[col] = data_encoded[col].map({'High': 2, 'Medium': 1, 'Low': 0})
        
        # 결측값 처리
        data_encoded = data_encoded.select_dtypes(include=['number']).fillna(0)
        
        # 상관관계 계산
        target = 'Exam_Score'
        if target in data_encoded.columns:
            correlations = data_encoded.corr()[target].sort_values(ascending=False)
            correlations = correlations.drop(target)  # 타겟 변수 자신 제거
            
            # 상위 10개 변수 선택 (또는 전체 변수가 10개 미만이면 전체 사용)
            top_count = min(10, len(correlations))
            top_correlations = correlations.abs().nlargest(top_count)
            top_features = top_correlations.index
            
            plt.figure(figsize=(10, 8))
            colors = ['green' if c > 0 else 'red' for c in correlations[top_features]]
            
            # 바 차트 생성
            bars = plt.barh(top_features, correlations[top_features], color=colors)
            
            # 바 위에 값 표시
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width if width > 0 else width - 0.05
                plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                         va='center', ha='left' if width > 0 else 'right', color='black')
            
            plt.title('시험 점수와 주요 변수들의 상관관계', fontsize=16)
            plt.xlabel('상관계수')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"4. 변수 중요도 시각화 완료: {output_path}")
        else:
            print(f"타겟 변수 '{target}'가 데이터에 없습니다")
    except Exception as e:
        print(f"변수 중요도 시각화 중 오류 발생: {e}")

# 신경망 모델 구조 시각화
def create_neural_network_architecture(output_path):
    try:
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        # 레이어와 노드 수 정의
        layers = [
            {'name': 'Input Layer', 'nodes': 12},  # 입력 변수 수는 데이터에 맞게 조정
            {'name': 'Hidden Layer 1', 'nodes': 256},
            {'name': 'Hidden Layer 2', 'nodes': 128},
            {'name': 'Hidden Layer 3', 'nodes': 64},
            {'name': 'Hidden Layer 4', 'nodes': 32},
            {'name': 'Output Layer', 'nodes': 1}
        ]
        
        # 각 레이어 위치 계산
        layer_width = 1.5
        x_positions = [i * layer_width * 2 for i in range(len(layers))]
        max_nodes = max(layer['nodes'] for layer in layers)
        
        # 레이어와 노드 그리기
        for i, (x, layer) in enumerate(zip(x_positions, layers)):
            # 레이어 이름
            ax.text(x, -0.5, layer['name'], ha='center', fontsize=11)
            
            # 노드 그리기 (일부 노드만 그리고 나머지는 생략 표시)
            max_visible_nodes = 10
            nodes_to_draw = min(layer['nodes'], max_visible_nodes)
            
            for j in range(nodes_to_draw):
                # 노드 위치 계산 (중앙 정렬)
                y = (max_nodes - layer['nodes']) / 2 + j
                if nodes_to_draw < layer['nodes'] and j == nodes_to_draw - 1:
                    # 생략 표시
                    ax.text(x, y, '...', ha='center', va='center', fontsize=18)
                else:
                    # 노드 그리기
                    circle = plt.Circle((x, y), 0.1, facecolor='skyblue', edgecolor='blue')
                    ax.add_patch(circle)
                
                # 이전 레이어와 연결선 그리기 (첫 번째 레이어 제외)
                if i > 0:
                    prev_layer = layers[i-1]
                    prev_x = x_positions[i-1]
                    prev_nodes_to_draw = min(prev_layer['nodes'], max_visible_nodes)
                    
                    for k in range(prev_nodes_to_draw):
                        prev_y = (max_nodes - prev_layer['nodes']) / 2 + k
                        if prev_nodes_to_draw < prev_layer['nodes'] and k == prev_nodes_to_draw - 1:
                            continue  # 생략된 노드는 연결선 그리지 않음
                        
                        # 모든 연결을 그리면 복잡해지므로 일부만 그림
                        if (j % 3 == 0 and k % 3 == 0) or (i == len(layers) - 1):
                            ax.plot([prev_x, x], [prev_y, y], 'gray', alpha=0.3, linewidth=0.5)
            
            # 활성화 함수와 Dropout 표시
            if 0 < i < len(layers) - 1:
                ax.text(x, max_nodes + 0.5, 'ReLU', ha='center', color='green', fontsize=9)
                ax.text(x, max_nodes + 1, f'Dropout ({0.3 if i==1 else 0.2 if i==2 else 0.1})', ha='center', color='red', fontsize=9)
                ax.text(x, max_nodes + 1.5, 'BatchNorm', ha='center', color='purple', fontsize=9)
        
        ax.set_xlim(-0.5, max(x_positions) + 0.5)
        ax.set_ylim(-1, max_nodes + 2)
        ax.axis('off')
        
        plt.title('심층 신경망(DNN) 모델 구조', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"5. 신경망 모델 구조 시각화 완료: {output_path}")
    except Exception as e:
        print(f"신경망 모델 구조 시각화 중 오류 발생: {e}")

# 모델 성능 평가 지표 시각화
def create_model_performance_visualization(output_path):
    try:
        # 성능 지표 (실제 값으로 대체)
        metrics = {
            'R² Score': 0.7803,
            'MSE': 2.2863,
            'MAE': 1.2064,
            'MAPE (%)': 1.80
        }
        
        plt.figure(figsize=(10, 6))
        
        # 바 차트 생성
        bars = plt.bar(metrics.keys(), metrics.values(), color=['green', 'red', 'orange', 'purple'])
        
        # 바 위에 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.title('모델 성능 평가 지표', fontsize=16)
        plt.ylabel('값')
        plt.ylim(0, max(metrics.values()) * 1.2)  # 여유 공간 확보
        
        # 각 지표에 대한 설명 추가
        descriptions = {
            'R² Score': '설명된 분산 비율\n(높을수록 좋음)',
            'MSE': '평균 제곱 오차\n(낮을수록 좋음)',
            'MAE': '평균 절대 오차\n(낮을수록 좋음)',
            'MAPE (%)': '평균 절대 백분율 오차\n(낮을수록 좋음)'
        }
        
        for i, (metric, desc) in enumerate(descriptions.items()):
            plt.annotate(desc, xy=(i, 0), xytext=(i, -0.2),
                       ha='center', va='top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"6. 모델 성능 평가 지표 시각화 완료: {output_path}")
    except Exception as e:
        print(f"모델 성능 평가 지표 시각화 중 오류 발생: {e}")

# 주요 발견점 및 시사점 인포그래픽
def create_key_findings_visualization(output_path):
    try:
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        ax.axis('off')
        
        # 배경색 설정
        fig = plt.gcf()
        fig.patch.set_facecolor('#f5f5f5')
        
        # 제목
        plt.text(0.5, 0.95, '주요 발견점 및 시사점', fontsize=18, ha='center', weight='bold')
        
        # 주요 발견점
        findings = [
            "공부 시간은 성적과 강한 양의 상관관계 (r=0.65)",
            "이전 시험 성적은 현재 성적의 가장 강력한 예측 변수 (r=0.78)",
            "높은 학습 의욕은 성적 향상에 중요한 요소",
            "학습 자원 접근성이 높을수록 성적이 좋음",
            "부모 참여도가 높을수록 공부 시간 효율이 증가"
        ]
        
        # 교육적 시사점
        implications = [
            "학생 의욕 고취 및 자원 접근성 개선이 중요",
            "가정-학교 연계 프로그램 개발 필요",
            "초기 학업 부진 학생 조기 식별 및 개입",
            "맞춤형 학습 지원 시스템 개발 가능성",
            "데이터 기반 교육 의사결정의 효과 입증"
        ]
        
        # 왼쪽 컬럼: 주요 발견점
        plt.text(0.25, 0.85, '주요 발견점', fontsize=14, ha='center', weight='bold')
        for i, finding in enumerate(findings):
            y_pos = 0.8 - i * 0.07
            plt.text(0.05, y_pos, f"• {finding}", fontsize=11, ha='left')
        
        # 오른쪽 컬럼: 교육적 시사점
        plt.text(0.75, 0.85, '교육적 시사점', fontsize=14, ha='center', weight='bold')
        for i, implication in enumerate(implications):
            y_pos = 0.8 - i * 0.07
            plt.text(0.55, y_pos, f"• {implication}", fontsize=11, ha='left')
        
        # 하단: 모델 성능 요약
        plt.text(0.5, 0.45, '모델 성능', fontsize=14, ha='center', weight='bold')
        plt.text(0.5, 0.4, 'R² = 0.78, MAPE = 1.80%의 높은 예측 정확도', fontsize=12, ha='center')
        
        # 분할선
        plt.axhline(y=0.5, xmin=0.05, xmax=0.95, color='gray', alpha=0.3, linestyle='-')
        plt.axvline(x=0.5, ymin=0.55, ymax=0.9, color='gray', alpha=0.3, linestyle='-')
        
        # 결론
        plt.text(0.5, 0.35, '결론', fontsize=14, ha='center', weight='bold')
        plt.text(0.5, 0.3, '학생 성적은 다양한 요인들의 복합적 영향을 받으며,', fontsize=11, ha='center')
        plt.text(0.5, 0.25, '이를 통합적으로 고려한 데이터 기반 교육 지원이 필요하다.', fontsize=11, ha='center')
        plt.text(0.5, 0.2, '심층 신경망 기반 예측 모델은 학생 성취도 예측에 높은 정확도를 보여,', fontsize=11, ha='center')
        plt.text(0.5, 0.15, '개인화된 학습 지원 시스템 구축의 가능성을 제시한다.', fontsize=11, ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"7. 주요 발견점 및 시사점 인포그래픽 완료: {output_path}")
    except Exception as e:
        print(f"주요 발견점 및 시사점 인포그래픽 생성 중 오류 발생: {e}")

# 학습 데이터와 테스트 데이터의 분포 비교
def create_train_test_distribution_comparison(train_data, test_data, output_path):
    try:
        plt.figure(figsize=(10, 6))
        
        # 히스토그램 그리기
        plt.hist(train_data['Exam_Score'], bins=20, alpha=0.5, label='학습 데이터', color='blue')
        plt.hist(test_data['Exam_Score'], bins=20, alpha=0.5, label='테스트 데이터', color='orange')
        
        plt.title('학습 데이터와 테스트 데이터의 시험 점수 분포 비교', fontsize=14)
        plt.xlabel('시험 점수')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"8. 학습/테스트 데이터 분포 비교 완료: {output_path}")
    except Exception as e:
        print(f"학습/테스트 데이터 분포 비교 생성 중 오류 발생: {e}")

# 실행
if __name__ == "__main__":
    try:
        # 1. 데이터셋 기본 통계량
        create_dataset_summary(train_data, f"{output_dir}/dataset_summary.png")
        
        # 2. 변수 분포 히스토그램
        create_distribution_plots(train_data, f"{output_dir}/variable_distributions.png")
        
        # 3. 데이터 처리 과정 시각화
        create_data_pipeline_visualization(f"{output_dir}/data_pipeline.png")
        
        # 4. 변수 중요도 시각화
        create_feature_importance_visualization(train_data, f"{output_dir}/feature_importance.png")
        
        # 5. 신경망 구조 시각화
        create_neural_network_architecture(f"{output_dir}/neural_network_architecture.png")
        
        # 6. 모델 성능 평가 지표
        create_model_performance_visualization(f"{output_dir}/model_performance.png")
        
        # 7. 주요 발견점 및 시사점
        create_key_findings_visualization(f"{output_dir}/key_findings.png")
        
        # 8. 학습/테스트 데이터 분포 비교
        create_train_test_distribution_comparison(train_data, test_data, f"{output_dir}/train_test_distribution.png")
        
        print("모든 시각화 생성 완료!")
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")
