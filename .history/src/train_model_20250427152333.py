from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_and_evaluate_model(data, target_column, feature_columns, model_path):
    # 데이터 분할
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(model, model_path)

    # 평가
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))