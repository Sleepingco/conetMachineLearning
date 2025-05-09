# 필요한 라이브러리 임포트
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# 사용자 입력을 통해 단계별로 진행되도록 설정
def wait_for_enter(step_name="다음 단계로 진행하려면 Enter 키를 누르세요."):
    input(f"\n {step_name} (Enter)")

# 1단계: 와인 데이터셋 로드 및 샘플 확인
print("\n[1단계] 데이터 로드 중...")
wine = load_wine()  # 와인 데이터셋 로드
X = pd.DataFrame(wine.data, columns=wine.feature_names)  # 특성 데이터프레임 생성
y = pd.Series(wine.target, name='target')  # 타겟 시리즈 생성

print(" 데이터 샘플 예시:")
print(X.head())  # 데이터 일부 출력
print("\n 타겟 값 예시:")
print(y.head())  # 타겟 값 일부 출력

wait_for_enter("2단계: 데이터 전처리 샘플 보기")

# 2단계: 데이터 전처리 - 결측치 및 통계량 확인
print("\n[2단계] 데이터 전처리 점검...")
print(" 결측치 여부 확인:")
print(X.isnull().sum())  # 각 열의 결측치 수 출력
print("\n 기본 통계량 확인:")
print(X.describe())  # 특성들의 기초 통계량 출력

wait_for_enter("3단계: 훈련/테스트 데이터 분할")

# 3단계: 훈련용 / 테스트용 데이터 분리
print("\n[3단계] 데이터 분할 (train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42  # 클래스 비율 유지하며 분할
)
print(f" 훈련 데이터 크기: {X_train.shape}")
print(f" 테스트 데이터 크기: {X_test.shape}")
print("\n 훈련 데이터 예시:")
print(X_train.head())  # 훈련 데이터 샘플 출력

wait_for_enter("4단계: 모델 학습 시작")

# 4단계: 랜덤 포레스트 모델 학습
print("\n[4단계] 모델 학습(RandomForestClassifier)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)  # 모델 정의
model.fit(X_train, y_train)  # 모델 학습 수행
print(" 모델 학습 완료.")

wait_for_enter("5단계: 모델 저장 및 로딩")

# 5단계: 모델 저장 및 불러오기
print("\n[5단계] 학습된 모델 저장 및 로딩...")
joblib.dump(model, 'wine_rf_model.pkl')  # 학습된 모델을 파일로 저장
print(" 모델 저장 완료 (wine_rf_model.pkl)")
loaded_model = joblib.load('wine_rf_model.pkl')  # 저장된 모델 불러오기
print(" 모델 로딩 완료")

wait_for_enter("6단계: 예측 수행")

# 6단계: 테스트 데이터에 대해 예측 수행
print("\n[6단계] 테스트 데이터 예측 수행...")
print(" 예측에 사용될 테스트 데이터 샘플:")
print(X_test.head())  # 테스트 데이터 일부 출력
y_pred = loaded_model.predict(X_test)  # 예측 수행
print(" 예측 완료")

wait_for_enter("7단계: 결과 평가")

# 7단계: 모델 성능 평가
print("\n[7단계] 결과 평가")
accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
print(f" 정확도: {accuracy:.4f}\n")
print(" 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))  # 리포트 출력

# 8단계: 임의 데이터로 예측 수행 (wine 데이터셋 구조에 맞춤)
print("\n[8단계] 임의 데이터로 예측 수행...")
wait_for_enter("8단계: 결과 평가")

# Wine 데이터는 총 13개의 특성(feature)을 가짐
# 아래는 3개의 임의 샘플 (범위는 실제 wine 데이터셋 참고하여 설정)
random_data = np.array([
    [13.2, 2.77, 2.51, 18.0, 98.0, 2.1, 1.5, 0.3, 1.6, 5.5, 1.05, 3.4, 1050.0],
    [12.4, 1.9, 2.3, 16.0, 100.0, 1.8, 1.1, 0.2, 1.2, 4.2, 0.9, 3.2, 850.0],
    [14.1, 3.1, 2.7, 20.0, 110.0, 2.5, 2.0, 0.4, 1.8, 6.0, 1.1, 3.6, 1300.0]
])

# DataFrame으로 변환 (열 이름 포함)
random_df = pd.DataFrame(random_data, columns=wine.feature_names)

print(" 입력된 임의 데이터 샘플:")
print(random_df)

# 예측 수행
predictions = loaded_model.predict(random_df)

# 클래스 번호를 품종 이름으로 매핑
predicted_classes = [wine.target_names[pred] for pred in predictions]

print("\n 예측 결과:")
for i, pred in enumerate(predicted_classes):
    print(f"샘플 {i+1}: {pred}")
