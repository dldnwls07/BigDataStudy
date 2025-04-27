# 학생 성적 영향 요인 분석 프로젝트

이 프로젝트는 다양한 요인들이 학생들의 시험 성적에 미치는 영향을 분석하고 예측하는 데이터 분석/머신러닝 프로젝트입니다.

## 프로젝트 구조

```
├── data/
│   ├── original_data/          # 원본 데이터
│   └── processed_data/         # 전처리된 데이터
├── output/
│   └── picture/               # 데이터 시각화 결과
│       ├── picture_High_impact/  # 주요 영향 요인 시각화
│       └── picture_main/         # 전체 요인 시각화
└── src/
    ├── data_processing/       # 데이터 전처리 관련 코드
    └── model_training/        # 모델 학습 관련 코드
```

## 주요 기능

1. 데이터 전처리
   - 이상치 제거
   - 중요 요인 추출
   - 학습/테스트 데이터 분할

2. 데이터 시각화
   - 상관관계 분석
   - 요인별 영향도 시각화
   - 산점도 및 박스플롯

3. 모델 학습
   - 성적 예측 모델 학습
   - 모델 성능 평가

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 데이터 전처리 및 시각화
```bash
python src/data_processing/run.py
```

2. 모델 학습
```bash
python src/model_training/refactored/train_model_refactored.py
```

## 요구사항

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn
- pytorch

## 라이선스

MIT License