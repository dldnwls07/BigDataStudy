import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 한글 폰트 설정 (Windows의 경우)
mpl.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# CSV 파일 로드 (인코딩 수정)
file_path = "c:\\Users\\feca1\\Desktop\\공부(빅분)\\StudentPerformanceFactors.csv"
data = pd.read_csv(file_path, encoding='utf-8-sig')

# 결측값 처리 (결측값을 'Unknown'으로 대체)
data.fillna('Unknown', inplace=True)

# 영어 컬럼명을 한글로 변경하기 위한 딕셔너리
column_mapping = {
    'Hours_Studied': '공부한 시간',
    'Attendance': '학교 출석',
    'Parental_Involvement': '부모의 참여도',
    'Access_to_Resources': '학습 자원에 대한 접근성',
    'Extracurricular_Activities': '방과 후 활동 참여도',
    'Sleep_Hours': '수면 시간',
    'Previous_Scores': '이전 시험 성적',
    'Motivation_Level': '학습의욕 수준',
    'Internet_Access': '인터넷 접근 여부',
    'Tutoring_Sessions': '개인 지도 세션 수',
    'Family_Income': '가정 소득',
    'Teacher_Quality': '교사의 수준',
    'School_Type': '학교 종류',
    'Peer_Influence': '또래의 영향',
    'Physical_Activity': '신체 활동 수준',
    'Learning_Disabilities': '학습 장애 여부',
    'Parental_Education_Level': '부모의 교육 수준',
    'Distance_from_Home': '집에서 학교까지의 거리',
    'Gender': '성별',
    'Exam_Score': '시험 점수'
}

# 원본 데이터 백업
data_original = data.copy()

# 데이터 컬럼명 변경
data = data.rename(columns=column_mapping)

# 데이터의 기본 정보와 첫 몇 줄 출력
print("데이터 기본 정보:")
print(data.info())
print("\n데이터 첫 5줄:")
print(data.head())

# 통계 요약
print("\n통계 요약:")
print(data.describe())

# 결측값 확인
print("\n결측값 확인:")
print(data.isnull().sum())

# 데이터 전처리: 범주형 변수를 숫자로 변환
encoder = LabelEncoder()

# 데이터 타입이 'object'인 컬럼만 인코딩
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

# 상관관계 분석 수정: 인코딩된 데이터 사용
print("\n수정된 상관관계 분석 (범주형 데이터 인코딩 후):")
print(data.corr())

# 시각화와 저장을 처리하는 함수
def save_plot(plot_func, filename, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    plot_func(*args, **kwargs)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # 플롯을 닫아 메모리 사용을 줄임

# 특정 열의 분포 시각화 (예: 시험 점수)
save_plot(plt.hist, '시험점수_분포.png', data['시험 점수'], bins=20, color='blue', alpha=0.7)

# 학교 출석과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='학교 출석', y='시험 점수', data=data, hue='학교 출석', palette='coolwarm', dodge=False)
plt.title('학교 출석과 시험 점수의 관계')
plt.xlabel('학교 출석')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('학교출석_시험점수_박스플롯.png')
plt.show()

# 부모의 참여도와 시험 점수의 관계 (공부 시간 기준)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='공부한 시간', y='시험 점수', hue='부모의 참여도', data=data, palette='coolwarm', s=100, alpha=0.7)
plt.title('공부한 시간과 시험 점수의 관계 (부모의 참여도별)')
plt.xlabel('공부한 시간')
plt.ylabel('시험 점수')
plt.legend(title='부모의 참여도')
plt.tight_layout()
plt.savefig('공부시간_시험점수_부모참여도별_산점도.png')
plt.show()

# 학습 자원에 대한 접근성과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='학습 자원에 대한 접근성', y='시험 점수', data=data, hue='학습 자원에 대한 접근성', palette='viridis', dodge=False)
plt.title('학습 자원에 대한 접근성과 시험 점수의 관계')
plt.xlabel('학습 자원에 대한 접근성')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('학습자원접근성_시험점수_박스플롯.png')
plt.show()

# 방과 후 활동 참여도와 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='방과 후 활동 참여도', y='시험 점수', data=data, hue='방과 후 활동 참여도', palette='coolwarm', dodge=False)
plt.title('방과 후 활동 참여도와 시험 점수의 관계')
plt.xlabel('방과 후 활동 참여도')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('방과후활동참여도_시험점수_박스플롯.png')
plt.show()

# 수면 시간과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.scatterplot(x='수면 시간', y='시험 점수', data=data, color='green', s=100, alpha=0.7)
plt.title('수면 시간과 시험 점수의 관계')
plt.xlabel('수면 시간')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('수면시간_시험점수_산점도.png')
plt.show()

# 이전 시험 성적과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.scatterplot(x='이전 시험 성적', y='시험 점수', data=data, color='purple', s=100, alpha=0.7)
plt.title('이전 시험 성적과 시험 점수의 관계')
plt.xlabel('이전 시험 성적')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('이전시험성적_시험점수_산점도.png')
plt.show()

# 학습의욕 수준과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='학습의욕 수준', y='시험 점수', data=data, hue='학습의욕 수준', palette='coolwarm', dodge=False)
plt.title('학습의욕 수준과 시험 점수의 관계')
plt.xlabel('학습의욕 수준')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('학습의욕수준_시험점수_박스플롯.png')
plt.show()

# 인터넷 접근 여부와 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='인터넷 접근 여부', y='시험 점수', data=data, hue='인터넷 접근 여부', palette='viridis', dodge=False)
plt.title('인터넷 접근 여부와 시험 점수의 관계')
plt.xlabel('인터넷 접근 여부')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('인터넷접근여부_시험점수_박스플롯.png')
plt.show()

# 가정 소득과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='가정 소득', y='시험 점수', data=data, hue='가정 소득', palette='coolwarm', dodge=False)
plt.title('가정 소득과 시험 점수의 관계')
plt.xlabel('가정 소득')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('가정소득_시험점수_박스플롯.png')
plt.show()

# 교사의 수준과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='교사의 수준', y='시험 점수', data=data, hue='교사의 수준', palette='viridis', dodge=False)
plt.title('교사의 수준과 시험 점수의 관계')
plt.xlabel('교사의 수준')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('교사수준_시험점수_박스플롯.png')
plt.show()

# 학교 종류와 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='학교 종류', y='시험 점수', data=data, hue='학교 종류', palette='coolwarm', dodge=False)
plt.title('학교 종류와 시험 점수의 관계')
plt.xlabel('학교 종류')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('학교종류_시험점수_박스플롯.png')
plt.show()

# 또래의 영향과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='또래의 영향', y='시험 점수', data=data, hue='또래의 영향', palette='viridis', dodge=False)
plt.title('또래의 영향과 시험 점수의 관계')
plt.xlabel('또래의 영향')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('또래영향_시험점수_박스플롯.png')
plt.show()

# 신체 활동 수준과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='신체 활동 수준', y='시험 점수', data=data, hue='신체 활동 수준', palette='coolwarm', dodge=False)
plt.title('신체 활동 수준과 시험 점수의 관계')
plt.xlabel('신체 활동 수준')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('신체활동수준_시험점수_박스플롯.png')
plt.show()

# 학습 장애 여부와 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='학습 장애 여부', y='시험 점수', data=data, hue='학습 장애 여부', palette='viridis', dodge=False)
plt.title('학습 장애 여부와 시험 점수의 관계')
plt.xlabel('학습 장애 여부')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('학습장애여부_시험점수_박스플롯.png')
plt.show()

# 부모의 교육 수준과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='부모의 교육 수준', y='시험 점수', data=data, hue='부모의 교육 수준', palette='coolwarm', dodge=False)
plt.title('부모의 교육 수준과 시험 점수의 관계')
plt.xlabel('부모의 교육 수준')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('부모교육수준_시험점수_박스플롯.png')
plt.show()

# 집에서 학교까지의 거리와 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='집에서 학교까지의 거리', y='시험 점수', data=data, hue='집에서 학교까지의 거리', palette='viridis', dodge=False)
plt.title('집에서 학교까지의 거리와 시험 점수의 관계')
plt.xlabel('집에서 학교까지의 거리')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('집에서학교까지거리_시험점수_박스플롯.png')
plt.show()

# 성별과 시험 점수의 관계
plt.figure(figsize=(10, 6))
sns.boxplot(x='성별', y='시험 점수', data=data, hue='성별', palette='coolwarm', dodge=False)
plt.title('성별과 시험 점수의 관계')
plt.xlabel('성별')
plt.ylabel('시험 점수')
plt.tight_layout()
plt.savefig('성별_시험점수_박스플롯.png')
plt.show()

# 상관 행렬 히트맵 (인코딩된 데이터 사용)
save_plot(sns.heatmap, '상관관계_히트맵.png', data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# 공부한 시간과 시험 점수 간의 산점도 (추가 시각화 - hue 사용)
save_plot(sns.scatterplot, '공부시간_시험점수_학습의욕수준별_산점도.png', x='공부한 시간', y='시험 점수', hue='학습의욕 수준', data=data, palette='viridis', s=100, alpha=0.7)

# AI 기반 진로 유형 분류 모델 생성
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if '진로 유형' in data.columns:
    # 데이터 전처리
    features = data[['공부한 시간', '학습의욕 수준', '이전 시험 성적']]  # 주요 특징 선택
    labels = data['진로 유형']
    features = pd.get_dummies(features, drop_first=True)  # 범주형 데이터 인코딩

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    print("\n진로 유형 분류 모델 평가:")
    print(classification_report(y_test, y_pred))
else:
    print("\n'진로 유형' 열이 데이터에 없습니다.")
