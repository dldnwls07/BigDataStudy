import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# 출력 디렉토리 생성
output_dir = "../../output/picture/presentation"
os.makedirs(output_dir, exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def create_data_sample_table():
    """
    원본 데이터의 샘플을 표 형태로 시각화하여 저장합니다.
    """
    # 원본 데이터 불러오기
    data_path = "../../data/original_data/StudentPerformanceFactors.csv"
    df = pd.read_csv(data_path)
    
    # 상위 10개 행만 사용
    sample_data = df.head(10)
    
    # 1. 간단한 테이블 미리보기 (일부 컬럼만)
    # 표시할 중요 컬럼 선택 (모든 컬럼을 표시하면 너무 복잡해짐)
    columns_to_display = ['Hours_Studied', 'Attendance', 'Previous_Scores', 
                         'Sleep_Hours', 'Family_Income', 'School_Type', 'Gender', 'Exam_Score']
    sample_preview = sample_data[columns_to_display].head(5)
    
    # 그림 크기와 스타일 설정
    plt.figure(figsize=(12, 4))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # 테이블 생성 - 더 보기 좋게 하기 위해 seaborn 사용
    table = plt.table(
        cellText=sample_preview.values,
        colLabels=sample_preview.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 테이블 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # 테이블 크기 조정
    
    # 헤더 행 스타일 조정
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 헤더 행
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')  # 파워포인트 스타일에 맞춤
        else:
            if col % 2 == 1:  # 홀수 열에 약간의 색상 추가
                cell.set_facecolor('#E6F0FF')
    
    # 저장
    plt.savefig(f"{output_dir}/data_sample_table.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("데이터 샘플 테이블 이미지 생성 완료")

    # 2. 데이터셋 요약 정보 테이블 (변수 수, 샘플 수, 목표변수 등)
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # 요약 정보 생성
    summary_info = [
        ["데이터셋 크기", f"{df.shape[0]} 행 × {df.shape[1]} 열"],
        ["독립 변수 수", f"{df.shape[1] - 1}개"],
        ["종속 변수", "Exam_Score (시험 점수)"],
        ["수치형 변수", f"{df.select_dtypes(include=['number']).shape[1]}개"],
        ["범주형 변수", f"{df.select_dtypes(include=['object']).shape[1]}개"]
    ]
    
    # 테이블 생성
    table = plt.table(
        cellText=summary_info,
        colLabels=["항목", "값"],
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 테이블 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 헤더 행 스타일 조정
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 헤더 행
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            if row % 2 == 0:  # 짝수 행에 약간의 색상 추가
                cell.set_facecolor('#E6F0FF')
    
    # 저장
    plt.savefig(f"{output_dir}/dataset_summary.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("데이터셋 요약 테이블 이미지 생성 완료")
      # 3. 변수 영향도 요약 테이블
    # 상관관계를 기준으로 상위 변수들의 영향도를 보여주는 표
    # 수치형 변수만 선택하여 상관관계 계산
    numeric_df = df.select_dtypes(include=['number'])
    correlations = numeric_df.corr()['Exam_Score'].sort_values(ascending=False)
    
    # 자기 자신(Exam_Score)을 제외하고 상위 5개 변수 선택
    top_correlations = correlations[1:6].to_frame().reset_index()
    top_correlations.columns = ['변수명', '상관계수']
    top_correlations['상관계수'] = top_correlations['상관계수'].round(3)
    
    plt.figure(figsize=(8, 4))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # 테이블 생성
    table = plt.table(
        cellText=top_correlations.values,
        colLabels=top_correlations.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 테이블 스타일 설정
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # 헤더 행 스타일 조정
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 헤더 행
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#E6F0FF')
    
    # 저장
    plt.savefig(f"{output_dir}/variable_importance_table.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print("변수 영향도 테이블 이미지 생성 완료")

if __name__ == "__main__":
    create_data_sample_table()
    print("모든 테이블 이미지 생성 완료")
