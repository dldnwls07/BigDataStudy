import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_plot(plot_func, filename, output_dir=None, *args, **kwargs):
    """시각화 플롯을 생성하고 저장하는 함수"""
    plt.figure(figsize=(10, 6))
    plot_func(*args, **kwargs)
    plt.tight_layout()
    
    # 출력 디렉토리가 지정되면 해당 경로에 파일 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        full_path = filename
        
    plt.savefig(full_path)
    print(f"그래프 저장: {full_path}")
    plt.close()  
    
def create_histogram(data, column, filename, output_dir=None, bins=20, color='blue', alpha=0.7):
    """히스토그램 생성 및 저장"""
    save_plot(plt.hist, filename, output_dir, data[column], bins=bins, color=color, alpha=alpha)
    
def create_boxplot(data, x_col, y_col, filename, output_dir=None, title=None, hue=None, palette='coolwarm', dodge=False):
    """박스플롯 생성 및 저장"""
    plt.figure(figsize=(10, 6))
    
    if hue is None:
        hue = x_col
        
    sns.boxplot(x=x_col, y=y_col, data=data, hue=hue, palette=palette, dodge=dodge)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{x_col}과 {y_col}의 관계')
        
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        full_path = filename
    
    plt.savefig(full_path)
    print(f"그래프 저장: {full_path}")
    plt.close()

def create_scatterplot(data, x_col, y_col, filename, output_dir=None, title=None, 
                      hue=None, palette='viridis', color=None, s=100, alpha=0.7):
    """산점도 생성 및 저장"""
    plt.figure(figsize=(10, 6))
    
    # hue가 지정되었을 때는 palette를 사용, 그렇지 않으면 color 사용
    if hue is not None:
        sns.scatterplot(x=x_col, y=y_col, data=data, hue=hue, palette=palette, s=s, alpha=alpha)
    else:
        # color 파라미터가 지정되지 않았다면 기본 색상 사용
        if color is None:
            color = 'blue'
        sns.scatterplot(x=x_col, y=y_col, data=data, color=color, s=s, alpha=alpha)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{x_col}과 {y_col}의 관계')
        
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    if hue:
        plt.legend(title=hue)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
    else:
        full_path = filename
    
    plt.savefig(full_path)
    print(f"그래프 저장: {full_path}")
    plt.close()

def create_heatmap(data, filename, output_dir=None, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5):
    """상관관계 히트맵 생성 및 저장"""
    save_plot(sns.heatmap, filename, output_dir, data, annot=annot, cmap=cmap, fmt=fmt, linewidths=linewidths)

def create_visualization_sets(data, data_encoded, output_dir=None):
    """기본적인 시각화 세트 생성"""
    # 시험 점수 분포 히스토그램
    create_histogram(data, '시험 점수', '시험점수_분포.png', output_dir)
    
    # 박스플롯
    boxplot_columns = [
        '학교 출석', '학습 자원에 대한 접근성', '방과 후 활동 참여도', 
        '학습의욕 수준', '인터넷 접근 여부', '가정 소득', '교사의 수준',
        '학교 종류', '또래의 영향', '신체 활동 수준', '학습 장애 여부',
        '부모의 교육 수준', '집에서 학교까지의 거리', '성별'
    ]
    
    for col in boxplot_columns:
        if col in data.columns:
            create_boxplot(data, col, '시험 점수', f'{col}_시험점수_박스플롯.png', output_dir,
                          title=f'{col}과 시험 점수의 관계')
    
    # 산점도
    create_scatterplot(data, '공부한 시간', '시험 점수', '공부시간_시험점수_산점도.png', output_dir)
    create_scatterplot(data, '수면 시간', '시험 점수', '수면시간_시험점수_산점도.png', output_dir, color='green')
    create_scatterplot(data, '이전 시험 성적', '시험 점수', '이전시험성적_시험점수_산점도.png', output_dir, color='purple')
    create_scatterplot(data, '공부한 시간', '시험 점수', '공부시간_시험점수_부모참여도별_산점도.png', output_dir,
                      title='공부한 시간과 시험 점수의 관계 (부모의 참여도별)', 
                      hue='부모의 참여도')
    create_scatterplot(data, '공부한 시간', '시험 점수', '공부시간_시험점수_학습의욕수준별_산점도.png', output_dir,
                      title='공부한 시간과 시험 점수의 관계 (학습의욕 수준별)',
                      hue='학습의욕 수준')
    
    # 상관관계 히트맵
    create_heatmap(data_encoded.corr(), '상관관계_히트맵.png', output_dir)
