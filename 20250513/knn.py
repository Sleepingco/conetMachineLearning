# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# %%
# IRIS 데이터 셋 로드
iris = load_iris()

# 특성과 타겟 데이터 분할
X = iris.data
y = iris.target

# 학습 데이터와 테스트 데이터로 분할
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# %%
# KNN 모델 초기화 및 파라미터 설정
knn = KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',n_jobs=1)

# 모델 학습
knn.fit(X_train,y_train)

# 분류 결과 예측
y_pred = knn.predict(X_test)

# 분류 결과 평가
report = classification_report(y_test,y_pred)
print(report)


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 탐색할 파라미터 그리드 정의
param_grid = {
    'n_neighbors' : [1,3,5], # 이웃의 수
    'weights':['unform','distance'],
    'algorithm':['ball_tree','kd_tree','brute'] # 가중치 함수
}
# GridSearchCV를 사용하여 최적 파라미터 탐색
grid_search = GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(X,y)


# %%
# 최적 파라미터 확인
best_params = grid_search.best_params_
print('Best Paramethers:',best_params)

# 최적 파라미터로 훈련된 모델 사용하여 예측
y_pred = grid_search.predict(X_test)

# 분류 결과 평가
report = classification_report(y_test,y_pred)
print("classification_report")
print(report)


